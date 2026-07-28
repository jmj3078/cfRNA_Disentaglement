import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.dispersion_trend import load_trend
from MixedEffectsModeling.core.marginal_rqr import _poisson_rqr, marginal_nb_rqr
from MixedEffectsModeling.core.model_engine_mixed import NormativeModelEngineMixed

MP = config.SPIKE_PARAMS
# Copied from Modeling/model_engine.py's _w1_normal (mean abs deviation vs
# theoretical N(0,1) quantiles) -- duplicate, not an import, per isolation.
def _w1_normal(z):
    v = z[np.isfinite(z)]
    n = len(v)
    if n < 8:
        return np.nan
    ref = norm.ppf(np.linspace(1 / (2 * n), 1 - 1 / (2 * n), n))
    return float(np.mean(np.abs(np.sort(v) - ref)))


def cv_model_route(e2, model_genes, stage_of, folds, tmp):
    """Route 'model': refit the already-assigned demotion stage per fold via
    glmm_fit.R --mode fixed_stage, score the held-out fold marginally."""
    if not model_genes:
        return {}, {}
    rows = []
    for fi, (tr, te) in enumerate(folds):
        pd.DataFrame(e2.X_hc_scaled[tr], columns=config.BIAS_COLUMNS).to_csv(f"{tmp}/X_{fi}.csv.gz")
        Y_tr = e2.Y_hc[tr][:, [e2._gene_col[g] for g in model_genes]]
        pd.DataFrame(Y_tr, columns=model_genes).to_csv(f"{tmp}/Y_{fi}.csv.gz")
        pd.DataFrame({"Batch_ID": e2.batch[tr]}).to_csv(f"{tmp}/batch_{fi}.csv.gz")
        pd.DataFrame({"gene": model_genes, "stage": [stage_of[g] for g in model_genes]}).to_csv(
            f"{tmp}/genes_{fi}.csv", index=False)
        subprocess.run([
            "Rscript", str(config.GLMM_FIT_R), "--x", f"{tmp}/X_{fi}.csv.gz", "--y", f"{tmp}/Y_{fi}.csv.gz",
            "--batch", f"{tmp}/batch_{fi}.csv.gz", "--genes", f"{tmp}/genes_{fi}.csv",
            "--trend", str(config.DISPERSION_TREND_PATH), "--mode", "fixed_stage", "--out", f"{tmp}/res_{fi}.csv",
        ], check=True, cwd=str(config.GLMM_FIT_R.parent))

        fold_fits = pd.read_csv(f"{tmp}/res_{fi}.csv").set_index("gene")
        Xa_te = np.column_stack([np.ones(len(te)), e2.X_hc_scaled[te]])
        for g in model_genes:
            if g not in fold_fits.index or not bool(fold_fits.loc[g, "ok"]):
                continue
            row = fold_fits.loc[g]
            mu_coef = row[[c for c in fold_fits.columns if c.startswith("mu_coef_")]].values.astype(float)
            disp_coef = row[[c for c in fold_fits.columns if c.startswith("disp_coef_")]].values.astype(float)
            mu = np.clip(np.exp(Xa_te @ np.nan_to_num(mu_coef, nan=0.0)), 1e-6, 1e8)
            # Full per-sample linear predictor (intercept + covariate slopes),
            # matching model_engine_mixed.py's score() fix -- NOT disp_coef[0]
            # alone, which silently flattens dispersion for stage nbi genes
            # (see task-7-report.md's "Finding 1" post-review fix).
            if not np.all(np.isnan(disp_coef)):
                alpha = np.exp(-Xa_te @ np.nan_to_num(disp_coef, nan=0.0))
            elif "fixed_alpha" in row.index and not pd.isna(row["fixed_alpha"]):
                # nb_fixed/intercept: reuse the fold-training-mean fixed dispersion
                # from this fold's refit, not the held-out fold's own mean -- matches
                # model_engine_mixed.py's score() fix (same bug class).
                alpha = np.full(len(te), float(row["fixed_alpha"]))
            else:
                alpha = np.full(len(te), e2.alpha_fn(float(mu.mean())))
            y_te = e2.Y_hc[te, e2._gene_col[g]]
            tau2 = float(row["tau2"])
            z = marginal_nb_rqr(y_te, mu, alpha, tau2, seed=42 + fi)
            # y/mu/alpha/tau2 kept per held-out sample (not just z) so a later PPC
            # step can simulate without rerunning the ~3h R fit. tau2 broadcast per
            # fold since it's a fold-level refit, not per-sample.
            rows.append(dict(gene=g, fold=fi, y=y_te.astype(np.float32), mu=mu.astype(np.float32),
                             alpha=np.asarray(alpha, dtype=np.float32),
                             tau2=np.full(len(te), tau2, dtype=np.float32), z=z))

    zdict, ppc_dict = {}, {}
    for g in model_genes:
        grecs = [r for r in rows if r["gene"] == g]
        if not grecs:
            continue
        zdict[g] = np.concatenate([r["z"] for r in grecs])
        ppc_dict[g] = dict(
            y=np.concatenate([r["y"] for r in grecs]),
            mu=np.concatenate([r["mu"] for r in grecs]),
            alpha=np.concatenate([r["alpha"] for r in grecs]),
            tau2=np.concatenate([r["tau2"] for r in grecs]),
            family="negbin", stage=stage_of[g])
    return zdict, ppc_dict


def cv_pool_route(e2, pool_genes, folds, tmp):
    """Route 'pool': refit the shared-beta pooled GLM per fold (train-fold only,
    NOT the full-data fit -- unlike Modeling/cv_model_engine.py's cv_pool(), which
    reuses the full-HC rare_glm for every fold and is therefore not truly held-out).
    Held-out fold scored via the same marginal_nb_rqr/_poisson_rqr as the model
    route, using this fold's own tau2/mult clip."""
    if not pool_genes:
        return {}, {}
    rows = []
    for fi, (tr, te) in enumerate(folds):
        pd.DataFrame(e2.X_hc_scaled[tr], columns=config.BIAS_COLUMNS).to_csv(f"{tmp}/Xp_{fi}.csv.gz")
        Y_tr = e2.Y_hc[tr][:, [e2._gene_col[g] for g in pool_genes]]
        pd.DataFrame(Y_tr, columns=pool_genes).to_csv(f"{tmp}/Yp_{fi}.csv.gz")
        pd.DataFrame({"Batch_ID": e2.batch[tr]}).to_csv(f"{tmp}/batchp_{fi}.csv.gz")
        pd.DataFrame({"gene": pool_genes}).to_csv(f"{tmp}/genesp_{fi}.csv", index=False)
        subprocess.run([
            "Rscript", str(config.GLMM_FIT_POOL_R), "--x", f"{tmp}/Xp_{fi}.csv.gz", "--y", f"{tmp}/Yp_{fi}.csv.gz",
            "--batch", f"{tmp}/batchp_{fi}.csv.gz", "--genes", f"{tmp}/genesp_{fi}.csv",
            "--rare-overdisp-thr", str(MP["rare_overdisp_thr"]), "--out", f"{tmp}/resp_{fi}.json",
        ], check=True, cwd=str(config.GLMM_FIT_POOL_R.parent))
        with open(f"{tmp}/resp_{fi}.json") as f:
            fit = json.load(f)
        if not fit["ok"]:
            continue

        beta = np.asarray(fit["beta"])
        tau2 = float(fit["tau2"]) if fit.get("tau2") is not None else 0.0
        eps = 1.0 / (2 * len(tr))
        mean_hc = dict(zip(fit["gene"], fit["mean_hc"]))
        Xte = e2.X_hc_scaled[te]
        mult = np.exp(Xte @ beta[1:])
        if fit.get("mult_lo") is not None:
            mult = np.clip(mult, fit["mult_lo"], fit["mult_hi"])
        for g in pool_genes:
            mu = np.clip((mean_hc[g] + eps) * np.exp(beta[0]) * mult, 1e-6, 1e8)
            y_te = e2.Y_hc[te, e2._gene_col[g]]
            if fit["family"] == "poisson":
                z = _poisson_rqr(y_te, mu, seed=42 + fi)
                alpha_arr = np.zeros(len(te), dtype=np.float32)
            else:
                z = marginal_nb_rqr(y_te, mu, fit["alpha"], tau2, seed=42 + fi)
                alpha_arr = np.full(len(te), fit["alpha"], dtype=np.float32)
            rows.append(dict(gene=g, fold=fi, y=y_te.astype(np.float32), mu=mu.astype(np.float32),
                             alpha=alpha_arr, tau2=np.full(len(te), tau2, dtype=np.float32),
                             z=z, family=fit["family"]))

    zdict, ppc_dict = {}, {}
    for g in pool_genes:
        grecs = [r for r in rows if r["gene"] == g]
        if not grecs:
            continue
        zdict[g] = np.concatenate([r["z"] for r in grecs])
        ppc_dict[g] = dict(
            y=np.concatenate([r["y"] for r in grecs]),
            mu=np.concatenate([r["mu"] for r in grecs]),
            alpha=np.concatenate([r["alpha"] for r in grecs]),
            tau2=np.concatenate([r["tau2"] for r in grecs]),
            family=grecs[0]["family"], stage="pool")
    return zdict, ppc_dict


def main():
    out_dir = config.CV_MIXED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full gene-set finalization: both routes, whatever assign_routes() decided
    # (pool if nz < nz_a_max, model otherwise) -- CV must cover the actual
    # deployed gene set, not just the model route.
    summary = pd.read_csv(config.ENGINE_MIXED_DIR / "training_summary.csv", index_col="gene")
    summary = summary[summary["ok"]]
    model_genes = summary.index[summary["route"] == "model"].tolist()
    pool_genes = summary.index[summary["route"] == "pool"].tolist()
    stage_of = summary["stage"].to_dict()

    e2 = NormativeModelEngineMixed()
    e2.load_hc_data()
    e2.alpha_fn = load_trend()  # needed for nb_fixed/intercept genes' fixed dispersion fallback
    n_hc = e2.X_hc_scaled.shape[0]
    folds = list(StratifiedKFold(MP["n_splits"], shuffle=True, random_state=42).split(np.zeros(n_hc), e2.batch))

    tmp = "/tmp/cv_glmm"
    Path(tmp).mkdir(exist_ok=True)

    print(f"CV: {len(model_genes)} model-route genes, {len(pool_genes)} pool-route genes")
    zdict_m, ppc_m = cv_model_route(e2, model_genes, stage_of, folds, tmp)
    zdict_p, ppc_p = cv_pool_route(e2, pool_genes, folds, tmp)
    zdict = {**zdict_m, **zdict_p}
    ppc_dict = {**ppc_m, **ppc_p}

    stats = []
    for g, z in zdict.items():
        v = z[np.isfinite(z)]
        if len(v) < 8:
            continue
        nz = int((e2.Y_hc[:, e2._gene_col[g]] > 0).sum())
        route = "model" if g in zdict_m else "pool"
        stage = summary.loc[g, "stage"]
        stats.append(dict(gene=g, route=route, stage=stage, nz=nz,
                          w1=_w1_normal(v), mean_z=float(v.mean()), std_z=float(v.std()),
                          skew_z=float(skew(v)), kurt_z=float(kurtosis(v)), n_valid=len(v)))
    df = pd.DataFrame(stats)
    df.to_csv(out_dir / "cv_stats.csv", index=False)
    with open(out_dir / "cv_zscores.pkl", "wb") as f:
        pickle.dump(zdict, f)
    with open(out_dir / "cv_ppc.pkl", "wb") as f:
        pickle.dump(ppc_dict, f)
    print(df.groupby("stage")[["w1", "mean_z", "std_z"]].median().to_string())
    print(f"Saved -> {out_dir}/cv_stats.csv, cv_zscores.pkl, cv_ppc.pkl")


if __name__ == "__main__":
    main()
