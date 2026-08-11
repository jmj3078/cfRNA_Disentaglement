import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.calibration import gene_shash_calibration
from MixedEffectsModeling.core.dispersion_trend import load_trend
from MixedEffectsModeling.core.eb_shrinkage import squeeze_log_theta
from MixedEffectsModeling.core.marginal_rqr import (
    _poisson_rqr, marginal_nb_loglik, marginal_nb_rqr, nb_marginal_mean_var, nb_marginal_pmf0,
)
from MixedEffectsModeling.core.model_engine_mixed import NormativeModelEngineMixed

MP = config.SPIKE_PARAMS
SHASH_MAX_N = 3000
POISSON_ALPHA_EPS = 1e-8  # alpha -> 0 limit of NB reduces to Poisson; reused so PPC math needs no family branch


# Every (gene, fold) pair is recorded here regardless of convergence -- the
# v1 CV script silently dropped non-converging folds, which hid exactly the
# fold-level failure information this module exists to keep.
def squeeze_fold(fits):
    """Apply the same EB dispersion-intercept squeeze the deployed engine applies,
    re-estimating tau_d from this fold's own fits so held-out Z-scores reflect the
    deployed model rather than the unshrunk MLEs."""
    ok = fits["ok"].astype(bool) & fits["trend_alpha"].notna() & fits["disp_coef_0"].notna()
    if not ok.any():
        return fits
    sub = fits[ok]
    post, _ = squeeze_log_theta(sub["disp_coef_0"].to_numpy(dtype=float),
                               sub["disp_se_0"].to_numpy(dtype=float),
                               -np.log(sub["trend_alpha"].to_numpy(dtype=float)))
    fits.loc[ok, "disp_coef_0"] = post
    return fits


def cv_model_route(e2, model_genes, stage_of, folds, tmp, disp_prior_path=None):
    if not model_genes:
        return {}, {}, []
    rows, fold_stat_rows = [], []
    for fi, (tr, te) in enumerate(folds):
        scaler = StandardScaler().fit(e2.X_hc_raw[tr])
        pd.DataFrame(scaler.transform(e2.X_hc_raw[tr]), columns=config.BIAS_COLUMNS).to_csv(f"{tmp}/X_{fi}.csv.gz")
        Y_tr = e2.Y_hc[tr][:, [e2._gene_col[g] for g in model_genes]]
        pd.DataFrame(Y_tr, columns=model_genes).to_csv(f"{tmp}/Y_{fi}.csv.gz")
        pd.DataFrame({"Batch_ID": e2.batch[tr]}).to_csv(f"{tmp}/batch_{fi}.csv.gz")
        pd.DataFrame({"gene": model_genes, "stage": [stage_of[g] for g in model_genes]}).to_csv(
            f"{tmp}/genes_{fi}.csv", index=False)
        fit_params = Path(tmp) / "fit_params.json"
        fit_params.write_text(json.dumps(config.FIT_PARAMS))
        cmd = [
            "Rscript", str(config.GLMM_FIT_R), "--x", f"{tmp}/X_{fi}.csv.gz", "--y", f"{tmp}/Y_{fi}.csv.gz",
            "--batch", f"{tmp}/batch_{fi}.csv.gz", "--genes", f"{tmp}/genes_{fi}.csv",
            "--trend", str(config.DISPERSION_TREND_PATH), "--fit-params", str(fit_params),
            "--mode", "fixed_stage", "--out", f"{tmp}/res_{fi}.csv",
        ]
        if disp_prior_path is not None:
            cmd += ["--disp-prior", str(disp_prior_path)]
        subprocess.run(cmd, check=True, cwd=str(config.GLMM_FIT_R.parent))

        fold_fits = squeeze_fold(pd.read_csv(f"{tmp}/res_{fi}.csv").set_index("gene"))
        Xa_te = np.column_stack([np.ones(len(te)), scaler.transform(e2.X_hc_raw[te])])
        for g in model_genes:
            if g not in fold_fits.index:
                fold_stat_rows.append(dict(gene=g, fold=fi, stage=stage_of[g], ok=False,
                                           singular=None, tau2=np.nan, fail_reason="fold_output_missing", n_test=len(te)))
                continue
            row = fold_fits.loc[g]
            ok = bool(row["ok"])
            fold_stat_rows.append(dict(gene=g, fold=fi, stage=row["stage"], ok=ok,
                                       singular=bool(row["singular"]) if not pd.isna(row["singular"]) else None,
                                       tau2=float(row["tau2"]) if not pd.isna(row["tau2"]) else np.nan,
                                       fail_reason=row["fail_reason"] if not pd.isna(row["fail_reason"]) else "",
                                       n_outliers=int(row["n_outliers"]) if not pd.isna(row["n_outliers"]) else 0,
                                       n_test=len(te)))
            if not ok:
                continue
            mu_coef = row[[c for c in fold_fits.columns if c.startswith("mu_coef_")]].values.astype(float)
            disp_coef = row[[c for c in fold_fits.columns if c.startswith("disp_coef_")]].values.astype(float)
            mu = np.clip(np.exp(Xa_te @ np.nan_to_num(mu_coef, nan=0.0)), 1e-6, 1e8)
            if not np.all(np.isnan(disp_coef)):
                alpha = np.exp(-Xa_te @ np.nan_to_num(disp_coef, nan=0.0))
            elif "trend_alpha" in row.index and not pd.isna(row["trend_alpha"]):
                alpha = np.full(len(te), float(row["trend_alpha"]))
            else:
                alpha = np.full(len(te), e2.alpha_fn(float(mu.mean())))
            y_te = e2.Y_hc[te, e2._gene_col[g]]
            tau2 = float(row["tau2"])
            z = marginal_nb_rqr(y_te, mu, alpha, tau2, seed=42 + fi)
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
    return zdict, ppc_dict, fold_stat_rows


# Shared-beta pooled GLM CV -- one fit per fold across all pool-route genes at once,
# so fold_stats here are per-fold (fit succeeded or not for the whole bundle), not per-gene.
def cv_pool_route(e2, genes_t, folds, tmp):
    if not genes_t:
        return {}, {}, []
    rows, fold_stats = [], []
    for fi, (tr, te) in enumerate(folds):
        scaler = StandardScaler().fit(e2.X_hc_raw[tr])
        pd.DataFrame(scaler.transform(e2.X_hc_raw[tr]), columns=config.BIAS_COLUMNS).to_csv(f"{tmp}/Xp_{fi}.csv.gz")
        Y_tr = e2.Y_hc[tr][:, [e2._gene_col[g] for g in genes_t]]
        pd.DataFrame(Y_tr, columns=genes_t).to_csv(f"{tmp}/Yp_{fi}.csv.gz")
        pd.DataFrame({"Batch_ID": e2.batch[tr]}).to_csv(f"{tmp}/batchp_{fi}.csv.gz")
        pd.DataFrame({"gene": genes_t}).to_csv(f"{tmp}/genesp_{fi}.csv", index=False)

        subprocess.run([
            "Rscript", str(config.GLMM_FIT_POOL_R), "--x", f"{tmp}/Xp_{fi}.csv.gz", "--y", f"{tmp}/Yp_{fi}.csv.gz",
            "--batch", f"{tmp}/batchp_{fi}.csv.gz", "--genes", f"{tmp}/genesp_{fi}.csv",
            "--rare-overdisp-thr", str(MP["rare_overdisp_thr"]), "--out", f"{tmp}/pool_res_{fi}.json",
        ], check=True, cwd=str(config.GLMM_FIT_POOL_R.parent))

        with open(f"{tmp}/pool_res_{fi}.json") as f:
            fit = json.load(f)
        if not fit["ok"]:
            fold_stats.append(dict(fold=fi, ok=False, family=None, fail_reason="pooled_glmm_fit_error"))
            continue
        fold_stats.append(dict(fold=fi, ok=True, family=fit["family"], fail_reason=""))

        beta = np.asarray(fit["beta"])
        tau2 = float(fit["tau2"]) if fit.get("tau2") is not None else 0.0
        alpha_eff = fit["alpha"] if fit["family"] == "negbin" else POISSON_ALPHA_EPS
        mean_hc = dict(zip(fit["gene"], fit["mean_hc"]))
        eps = fit["eps"]
        X_te = scaler.transform(e2.X_hc_raw[te])
        mult = np.exp(X_te @ beta[1:])
        if fit.get("mult_lo") is not None:
            mult = np.clip(mult, fit["mult_lo"], fit["mult_hi"])

        for g in genes_t:
            mu = np.clip((mean_hc[g] + eps) * np.exp(beta[0]) * mult, 1e-6, 1e8)
            y_te = e2.Y_hc[te, e2._gene_col[g]].astype(np.float64)
            if fit["family"] == "poisson":
                z = _poisson_rqr(y_te, mu, seed=42 + fi)
            else:
                z = marginal_nb_rqr(y_te, mu, alpha_eff, tau2, seed=42 + fi)
            rows.append(dict(gene=g, z=z, y=y_te, mu=mu,
                             alpha=np.full_like(mu, alpha_eff), tau2=np.full_like(mu, tau2)))

    zdict, ppc_dict = {}, {}
    for g in genes_t:
        grecs = [r for r in rows if r["gene"] == g]
        if not grecs:
            continue
        zdict[g] = np.concatenate([r["z"] for r in grecs])
        ppc_dict[g] = dict(
            y=np.concatenate([r["y"] for r in grecs]), mu=np.concatenate([r["mu"] for r in grecs]),
            alpha=np.concatenate([r["alpha"] for r in grecs]), tau2=np.concatenate([r["tau2"] for r in grecs]),
        )
    return zdict, ppc_dict, fold_stats


EMPTY_METRICS = dict(obs_zero_frac=np.nan, pred_zero_frac=np.nan, zero_diff=np.nan, pearson_chi2=np.nan,
                    obs_mean=np.nan, pred_mean=np.nan, mean_rel_err=np.nan,
                    obs_var=np.nan, pred_var=np.nan, var_rel_err=np.nan,
                    ll_sum=np.nan, ll_mean=np.nan)


def ppc_metrics(ppc):
    y, mu, alpha, tau2 = ppc["y"], ppc["mu"], ppc["alpha"], ppc["tau2"]
    if len(y) < 8:
        return dict(EMPTY_METRICS)
    obs_zero_frac = float(np.mean(y == 0))
    pred_zero_frac = float(np.mean(nb_marginal_pmf0(mu, alpha, tau2)))
    mu_marg, var_marg = nb_marginal_mean_var(mu, alpha, tau2)
    pearson_chi2 = float(np.mean((y - mu_marg) ** 2 / np.maximum(var_marg, 1e-8)))
    obs_mean, pred_mean = float(y.mean()), float(mu_marg.mean())
    obs_var, pred_var = float(y.var()), float(var_marg.mean())
    ll = marginal_nb_loglik(y, mu, alpha, tau2)
    return dict(obs_zero_frac=obs_zero_frac, pred_zero_frac=pred_zero_frac,
               zero_diff=pred_zero_frac - obs_zero_frac, pearson_chi2=pearson_chi2,
               obs_mean=obs_mean, pred_mean=pred_mean,
               mean_rel_err=(pred_mean - obs_mean) / max(obs_mean, 1e-8),
               obs_var=obs_var, pred_var=pred_var,
               var_rel_err=(pred_var - obs_var) / max(obs_var, 1e-8),
               ll_sum=float(np.sum(ll)), ll_mean=float(np.mean(ll)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-genes", type=int, default=None,
                    help="smoke test: only run CV on the first N genes per route")
    ap.add_argument("--engine-dir", type=Path, default=config.ENGINE_MIXED_DIR,
                    help="directory to load the trained engine from (and re-save with cv_* fields into)")
    ap.add_argument("--out-dir", type=Path, default=config.CV_MIXED_DIR,
                    help="directory to write fold_stats/cv_stats/cv_zscores/cv_ppc into")
    args = ap.parse_args()

    disp_prior_path = args.engine_dir / "disp_prior.json"
    if not disp_prior_path.exists():
        disp_prior_path = None

    engine_dir = args.engine_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(engine_dir / "training_summary.csv", index_col="gene")
    summary = summary[summary["ok"]]
    model_genes = summary.index[summary["route"] == "model"].tolist()[:args.limit_genes]
    pool_genes = summary.index[summary["route"] == "pool"].tolist()[:args.limit_genes]
    stage_of = summary["stage"].to_dict()

    e2 = NormativeModelEngineMixed()
    e2.load_hc_data()
    e2.alpha_fn = load_trend(engine_dir / "dispersion_trend.json")
    n_hc = e2.X_hc_scaled.shape[0]
    folds = list(StratifiedKFold(MP["n_splits"], shuffle=True, random_state=42).split(np.zeros(n_hc), e2.batch))

    tmp = "/tmp/cv_glmm_v2"
    Path(tmp).mkdir(exist_ok=True)

    print(f"CV: {len(model_genes)} model-route genes, {len(pool_genes)} pool-route genes")
    zdict_m, ppc_dict_m, fold_stat_rows = cv_model_route(e2, model_genes, stage_of, folds, tmp, disp_prior_path)
    zdict_p, ppc_dict_p, pool_fold_stats = cv_pool_route(e2, pool_genes, folds, tmp)
    zdict = {**zdict_m, **zdict_p}
    ppc_dict = {**ppc_dict_m, **ppc_dict_p}

    fold_stats = pd.DataFrame(fold_stat_rows)
    fold_stats.to_csv(out_dir / "fold_stats.csv", index=False)
    if len(fold_stats):
        print(f"model-route fold success rate: {fold_stats['ok'].mean():.3f} ({int(fold_stats['ok'].sum())}/{len(fold_stats)})")
    pd.DataFrame(pool_fold_stats).to_csv(out_dir / "pool_fold_stats.csv", index=False)
    if pool_fold_stats:
        n_pool_ok = sum(1 for r in pool_fold_stats if r["ok"])
        print(f"pool-route fold success: {n_pool_ok}/{len(pool_fold_stats)}")

    engine = NormativeModelEngineMixed.load(engine_dir)

    stats = []
    for g, z in zdict.items():
        v = z[np.isfinite(z)]
        if len(v) < 8:
            continue
        nz = int((e2.Y_hc[:, e2._gene_col[g]] > 0).sum())
        z_sub = v if len(v) <= SHASH_MAX_N else np.random.default_rng(42).choice(v, SHASH_MAX_N, replace=False)
        calib = gene_shash_calibration(z_sub)

        ppc = ppc_metrics(ppc_dict.get(g, dict(y=np.array([]), mu=np.array([]), alpha=np.array([]), tau2=np.array([]))))

        cv_fields = dict(
            cv_shash_ok=calib["shash_ok"], cv_shash_xi=calib["shash_xi"],
            cv_shash_eta=calib["shash_eta"], cv_shash_eps=calib["shash_eps"],
            cv_shash_delta=calib["shash_delta"], cv_shash_z_lo=calib["z_lo"], cv_shash_z_hi=calib["z_hi"],
            cv_raw_skew=calib["raw_skew"], cv_raw_kurtosis=calib["raw_kurtosis"],
            cv_corrected_skew=calib["corrected_skew"], cv_corrected_kurtosis=calib["corrected_kurtosis"],
            cv_naive_exceed=calib["naive_exceed"], cv_shash_exceed=calib["shash_exceed"],
            cv_naive_fdr_reject_rate=calib["naive_fdr_reject_rate"], cv_corr_fdr_reject_rate=calib["corr_fdr_reject_rate"],
            cv_obs_zero_frac=ppc["obs_zero_frac"], cv_pred_zero_frac=ppc["pred_zero_frac"],
            cv_zero_diff=ppc["zero_diff"], cv_pearson_chi2=ppc["pearson_chi2"],
            cv_obs_mean=ppc["obs_mean"], cv_pred_mean=ppc["pred_mean"], cv_mean_rel_err=ppc["mean_rel_err"],
            cv_obs_var=ppc["obs_var"], cv_pred_var=ppc["pred_var"], cv_var_rel_err=ppc["var_rel_err"],
            cv_ll_sum=ppc["ll_sum"], cv_ll_mean=ppc["ll_mean"],
        )

        rec = engine.genes.get(g)
        if rec is not None:
            for k, val in cv_fields.items():
                setattr(rec, k, val)

        route = "model" if g in zdict_m else "pool"
        stats.append(dict(gene=g, route=route, stage=stage_of[g], nz=nz,
                          mean_z=float(v.mean()), std_z=float(v.std()), n_valid=len(v), **cv_fields))

    df = pd.DataFrame(stats)
    df.to_csv(out_dir / "cv_stats.csv", index=False)
    with open(out_dir / "cv_zscores.pkl", "wb") as f:
        pickle.dump(zdict, f)
    with open(out_dir / "cv_ppc.pkl", "wb") as f:
        pickle.dump(ppc_dict, f)
    engine.save(engine_dir)
    print(df.groupby(["route", "stage"])[["mean_z", "std_z"]].median().to_string())
    print(f"Saved -> {out_dir}/fold_stats.csv, pool_fold_stats.csv, cv_stats.csv, cv_zscores.pkl, cv_ppc.pkl, updated {engine_dir}/genes.pkl")


if __name__ == "__main__":
    main()
