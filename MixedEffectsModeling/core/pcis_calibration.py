import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix
from statsmodels.nonparametric.smoothers_lowess import lowess

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import ENGINE_MIXED_DIR, FIT_PARAMS, PCIS_CAL_DIR, PCIS_CAL_FIG_DIR
from viz_style import apply_style

FWER_TARGETS = [0.05, 0.01]
OBS_TARGETS = [1e-3, 1e-4]
N_DECILES = 10
N_OBS_BINS = 20
B_SPLINE_DF = 6
GRID_N = 200
GRID_PCT = (0.1, 99.9)


def load_null(path):
    d = pd.read_csv(path)
    return d[np.isfinite(d.pcis)].copy()


def gene_table(d):
    g = d[d["rank"] == 1].copy()
    g["max_pcis"] = g.pcis
    return g.drop(columns=["rank", "pcis"])


def current_rates(d, g, real_summary=None):
    n = int(d.n_obs.iloc[0])
    n_genes = int(g.gene.nunique())
    ex = d.pcis > d.cut_current
    per_gene = ex.groupby(d.gene).sum()
    out = {
        "n_genes": n_genes,
        "n_obs_per_gene": n,
        "n_obs_total": n_genes * n,
        "pcis_f_q": FIT_PARAMS["pcis_f_q"],
        "cut_current_q01": float(np.percentile(g.cut_current, 1)),
        "cut_current_median": float(np.median(g.cut_current)),
        "cut_current_q99": float(np.percentile(g.cut_current, 99)),
        "null_per_obs_rate": float(ex.sum() / (n_genes * n)),
        "null_per_gene_fwer": float((g.max_pcis > g.cut_current).mean()),
        "null_mean_removed_per_gene": float(per_gene.mean()),
        "null_genes_with_any": float((per_gene > 0).mean()),
        "topk_saturated_genes": int((per_gene >= 50).sum()),
    }
    if real_summary is not None:
        r = real_summary[real_summary.ok.astype(str).str.upper() == "TRUE"]
        out["real_mean_removed_per_gene"] = float(r.n_outliers.mean())
        out["real_genes_with_any"] = float((r.n_outliers > 0).mean())
        out["excess_removed_per_gene"] = out["real_mean_removed_per_gene"] - out["null_mean_removed_per_gene"]
        out["null_share_of_removals"] = out["null_mean_removed_per_gene"] / out["real_mean_removed_per_gene"]
    return out


def threshold_a(g):
    return {f"fwer_{a:g}": float(np.quantile(g.max_pcis, 1 - a)) for a in FWER_TARGETS}


def threshold_a_obs(d, n_genes, n):
    p = np.sort(d.pcis.values)[::-1]
    out = {}
    for r in OBS_TARGETS:
        k = int(round(r * n_genes * n))
        out[f"per_obs_{r:g}"] = float(p[k - 1]) if 1 <= k <= len(p) else np.nan
    return out


def fit_b_fwer(g, alpha):
    y = np.log(np.maximum(g.max_pcis.values, 1e-12))
    des = dmatrix(f"bs(log_mu, df={B_SPLINE_DF})", g, return_type="dataframe")
    m = sm.QuantReg(y, des.values).fit(q=1 - alpha, max_iter=5000)
    pred = np.exp(m.predict(des.values))
    lo, hi = np.percentile(g.log_mu, GRID_PCT)
    grid = pd.DataFrame({"log_mu": np.linspace(lo, hi, GRID_N)})
    gd = build_design_matrices([des.design_info], grid)[0]
    return {
        "params": m.params.tolist(),
        "design_info": f"bs(log_mu, df={B_SPLINE_DF})",
        "realized_fwer": float((g.max_pcis.values > pred).mean()),
        "grid_log_mu_range": [float(lo), float(hi)],
        "log_mu_data_range": [float(g.log_mu.min()), float(g.log_mu.max())],
        "cut_at_fitted_genes_min": float(pred.min()),
        "cut_at_fitted_genes_max": float(pred.max()),
        "n_genes_below_grid": int((g.log_mu < lo).sum()),
        "n_genes_above_grid": int((g.log_mu > hi).sum()),
        "grid_log_mu": grid.log_mu.tolist(),
        "grid_cut": np.exp(m.predict(np.asarray(gd))).tolist(),
    }, pred


def fit_b_obs(d, rate, n):
    lm = d.groupby("gene").log_mu.first()
    bins = pd.qcut(lm, N_OBS_BINS, labels=False)
    d = d.assign(bin=d.gene.map(bins))
    rows = []
    for b, x in d.groupby("bin"):
        ng = x.gene.nunique()
        p = np.sort(x.pcis.values)[::-1]
        k = int(round(rate * ng * n))
        rows.append({
            "bin": int(b),
            "log_mu": float(x.log_mu.median()),
            "n_genes": ng,
            "cut_empirical": float(p[k - 1]) if 1 <= k <= len(p) else np.nan,
            "cut_current_median": float(x.cut_current.median()),
            "current_per_obs_rate": float((x.pcis > x.cut_current).sum() / (ng * n)),
        })
    t = pd.DataFrame(rows)
    ok = np.isfinite(t.cut_empirical)
    sm_fit = lowess(np.log(t.cut_empirical[ok]), t.log_mu[ok], frac=0.5, it=3, return_sorted=True)
    t["cut_smooth"] = np.exp(np.interp(t.log_mu, sm_fit[:, 0], sm_fit[:, 1]))
    grid = np.linspace(t.log_mu.min(), t.log_mu.max(), GRID_N)
    curve = pd.DataFrame({"log_mu": grid, "cut": np.exp(np.interp(grid, sm_fit[:, 0], sm_fit[:, 1]))})
    return t, curve


def decile_table(g, a_cut, b_pred):
    g = g.assign(decile=pd.qcut(g.log_mu, N_DECILES, labels=False), b_cut=b_pred)
    return g.groupby("decile").apply(lambda x: pd.Series({
        "log_mu_median": x.log_mu.median(),
        "trend_alpha_median": x.trend_alpha.median(),
        "tau2_median": x.tau2.median(),
        "p_eff_median": x.p_eff.median(),
        "n_genes": len(x),
        "null_max_q95": np.quantile(x.max_pcis, 0.95),
        "null_max_q99": np.quantile(x.max_pcis, 0.99),
        "cut_current_median": x.cut_current.median(),
        "fwer_current": (x.max_pcis > x.cut_current).mean(),
        "cut_A": a_cut,
        "fwer_A": (x.max_pcis > a_cut).mean(),
        "cut_B_median": x.b_cut.median(),
        "fwer_B": (x.max_pcis > x.b_cut).mean(),
    }), include_groups=False).reset_index()


def covariate_diagnostics(g):
    rows = []
    lp = np.log(np.maximum(g.max_pcis, 1e-12))
    for c in ["log_mu", "trend_alpha", "tau2", "p_eff"]:
        rows.append({"covariate": c, "spearman_vs_log_max_pcis": g[c].corr(lp, method="spearman")})
    return pd.DataFrame(rows)


def b_model_comparison(g, alpha):
    forms = {
        "bs(log_mu)": f"bs(log_mu, df={B_SPLINE_DF})",
        "bs(log_mu)+tau2": f"bs(log_mu, df={B_SPLINE_DF})+tau2",
        "bs(log_mu)+bs(log p_eff)": f"bs(log_mu, df={B_SPLINE_DF})+bs(np.log(p_eff), df=3)",
        "bs(log_mu)+tau2+bs(log p_eff)": f"bs(log_mu, df={B_SPLINE_DF})+tau2+bs(np.log(p_eff), df=3)",
    }
    y = np.log(np.maximum(g.max_pcis.values, 1e-12))
    rows = []
    for name, f in forms.items():
        des = dmatrix(f, g, return_type="dataframe")
        m = sm.QuantReg(y, des.values).fit(q=1 - alpha, max_iter=5000)
        pred = np.exp(m.predict(des.values))
        rows.append({"formula": name, "n_params": des.shape[1],
                     "realized_fwer": (g.max_pcis.values > pred).mean(),
                     "pinball_loss": _pinball(y, np.log(pred), 1 - alpha),
                     "cut_min": pred.min(), "cut_max": pred.max()})
    return pd.DataFrame(rows)


def _pinball(y, q, tau):
    e = y - q
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def figures(g, dec, obs_tables, b_curves, outdir):
    apply_style()
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    lp = np.log10(np.maximum(g.max_pcis, 1e-12))
    ax[0].hist(lp, bins=120, color="0.6")
    for v, lab, c in [(np.median(g.cut_current), "current (median)", "C3"),
                      (np.quantile(g.max_pcis, 0.95), "A, FWER 0.05", "C0")]:
        ax[0].axvline(np.log10(v), color=c, ls="--", label=lab)
    ax[0].set_xlabel("log10 null max PCIS per gene"); ax[0].set_ylabel("genes"); ax[0].legend()
    ax[1].scatter(g.log_mu, lp, s=1, alpha=0.15, color="0.5", rasterized=True)
    ax[1].set_xlim(*np.percentile(g.log_mu, GRID_PCT))
    o = np.argsort(dec.log_mu_median)
    ax[1].plot(dec.log_mu_median.values[o], np.log10(dec.null_max_q95.values[o]), "o-", color="k", label="null q95")
    ax[1].plot(dec.log_mu_median.values[o], np.log10(dec.cut_current_median.values[o]), "s-", color="C3", label="current cut")
    ax[1].plot(b_curves[FWER_TARGETS[0]]["grid_log_mu"], np.log10(b_curves[FWER_TARGETS[0]]["grid_cut"]),
               color="C0", label=f"B, FWER {FWER_TARGETS[0]:g}")
    ax[1].set_xlabel("log mean expression"); ax[1].set_ylabel("log10 PCIS"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "null_max_pcis.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for col, lab, c in [("fwer_current", "current", "C3"), ("fwer_A", "A (constant)", "C1"), ("fwer_B", "B (smooth)", "C0")]:
        ax.plot(dec.log_mu_median, dec[col], "o-", color=c, label=lab)
    ax.axhline(FWER_TARGETS[0], color="k", ls=":", label=f"target {FWER_TARGETS[0]:g}")
    ax.set_xlabel("log mean expression"); ax.set_ylabel("realized null FWER"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fwer_by_expression.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(1, len(OBS_TARGETS), figsize=(5.5 * len(OBS_TARGETS), 4), squeeze=False)
    for i, r in enumerate(OBS_TARGETS):
        t, curve = obs_tables[r]
        a = ax[0][i]
        a.plot(t.log_mu, t.cut_empirical, "o", color="0.4", label="empirical bin")
        a.plot(curve.log_mu, curve.cut, "-", color="C0", label="B (lowess)")
        a.plot(t.log_mu, t.cut_current_median, "s-", color="C3", label="current cut")
        a.set_yscale("log"); a.set_xlabel("log mean expression"); a.set_ylabel("PCIS cut")
        a.set_title(f"per-observation rate {r:g}"); a.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "per_obs_thresholds.png"), dpi=200); plt.close(fig)


def run_all(null_path, outdir=None, fig_dir=None, real_summary_path=None):
    outdir = str(PCIS_CAL_DIR if outdir is None else outdir)
    fig_dir = str(PCIS_CAL_FIG_DIR if fig_dir is None else fig_dir)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    d = load_null(null_path)
    g = gene_table(d)
    n = int(d.n_obs.iloc[0])
    n_genes = int(g.gene.nunique())

    real = None
    rsp = str(ENGINE_MIXED_DIR / "training_summary.csv") if real_summary_path is None else str(real_summary_path)
    if os.path.isfile(rsp):
        real = pd.read_csv(rsp)

    summary = current_rates(d, g, real)
    summary["threshold_A_fwer"] = threshold_a(g)
    summary["threshold_A_per_obs"] = threshold_a_obs(d, n_genes, n)

    b_curves, b_preds = {}, {}
    for a in FWER_TARGETS:
        b_curves[a], b_preds[a] = fit_b_fwer(g, a)
    summary["threshold_B_fwer"] = {f"{a:g}": {k: v for k, v in b_curves[a].items() if not k.startswith("grid_")}
                                   for a in FWER_TARGETS}

    obs_tables = {r: fit_b_obs(d, r, n) for r in OBS_TARGETS}

    a05 = summary["threshold_A_fwer"][f"fwer_{FWER_TARGETS[0]:g}"]
    dec = decile_table(g, a05, b_preds[FWER_TARGETS[0]])

    g.assign(**{f"cut_B_fwer_{a:g}": b_preds[a] for a in FWER_TARGETS}).to_csv(
        os.path.join(outdir, "null_max_pcis_per_gene.csv.gz"), index=False)
    dec.to_csv(os.path.join(outdir, "fwer_by_expression_decile.csv"), index=False)
    covariate_diagnostics(g).to_csv(os.path.join(outdir, "covariate_dependence.csv"), index=False)
    b_model_comparison(g, FWER_TARGETS[0]).to_csv(os.path.join(outdir, "threshold_B_model_comparison.csv"), index=False)
    for r in OBS_TARGETS:
        t, curve = obs_tables[r]
        t.to_csv(os.path.join(outdir, f"per_obs_bins_rate{r:g}.csv"), index=False)
        curve.to_csv(os.path.join(outdir, f"threshold_B_curve_per_obs_rate{r:g}.csv"), index=False)
    for a in FWER_TARGETS:
        pd.DataFrame({"log_mu": b_curves[a]["grid_log_mu"], "cut": b_curves[a]["grid_cut"]}).to_csv(
            os.path.join(outdir, f"threshold_B_curve_fwer{a:g}.csv"), index=False)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    figures(g, dec, obs_tables, b_curves, fig_dir)
    return summary


if __name__ == "__main__":
    s = run_all(sys.argv[1] if len(sys.argv) > 1 else "/tmp/null_pcis.csv")
    print(json.dumps(s, indent=2))
