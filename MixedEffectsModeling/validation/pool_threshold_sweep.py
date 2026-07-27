import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.calibration import gene_shash_calibration
from MixedEffectsModeling.core.dispersion_trend import load_trend
from MixedEffectsModeling.core.marginal_rqr import _poisson_rqr, marginal_nb_rqr
from MixedEffectsModeling.core.model_engine_mixed import NormativeModelEngineMixed
from MixedEffectsModeling.validation.cv_engine import cv_model_route

MP = config.SPIKE_PARAMS
THRESHOLDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SHASH_MAX_N = 3000


def cv_pool_route(e2, genes_t, folds, tmp):
    rows, fold_stats = [], []
    for fi, (tr, te) in enumerate(folds):
        pd.DataFrame(e2.X_hc_scaled[tr], columns=config.BIAS_COLUMNS).to_csv(f"{tmp}/Xp_{fi}.csv.gz")
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
            # glmm_fit_pool.R's fit_pooled_glmm() does not expose a reason string on
            # failure (glmmTMB error path just returns ok=FALSE) -- this is the most
            # specific info available without touching the R fitting code.
            fold_stats.append(dict(fold=fi, ok=False, family=None, fail_reason="pooled_glmm_fit_error"))
            continue
        fold_stats.append(dict(fold=fi, ok=True, family=fit["family"], fail_reason=""))

        beta = np.asarray(fit["beta"])
        tau2 = float(fit["tau2"]) if fit.get("tau2") is not None else 0.0
        mean_hc = dict(zip(fit["gene"], fit["mean_hc"]))
        eps = fit["eps"]
        X_te = e2.X_hc_scaled[te]
        mult = np.exp(X_te @ beta[1:])
        if fit.get("mult_lo") is not None:
            mult = np.clip(mult, fit["mult_lo"], fit["mult_hi"])

        for g in genes_t:
            mu = np.clip((mean_hc[g] + eps) * np.exp(beta[0]) * mult, 1e-6, 1e8)
            y_te = e2.Y_hc[te, e2._gene_col[g]].astype(np.float64)
            if fit["family"] == "poisson":
                z = _poisson_rqr(y_te, mu, seed=42 + fi)
            else:
                z = marginal_nb_rqr(y_te, mu, fit["alpha"], tau2, seed=42 + fi)
            rows.append(dict(gene=g, z=z))

    zdict = {}
    for g in genes_t:
        parts = [r["z"] for r in rows if r["gene"] == g]
        if parts:
            zdict[g] = np.concatenate(parts)
    return zdict, fold_stats


def naive_calib(z):
    v = np.asarray(z)
    v = v[np.isfinite(v)]
    if len(v) < 8:
        return dict(naive_exceed=np.nan, naive_fdr_reject_rate=np.nan, raw_skew=np.nan, raw_kurtosis=np.nan)
    v = v if len(v) <= SHASH_MAX_N else np.random.default_rng(42).choice(v, SHASH_MAX_N, replace=False)
    calib = gene_shash_calibration(v)
    return dict(naive_exceed=calib["naive_exceed"], naive_fdr_reject_rate=calib["naive_fdr_reject_rate"],
               raw_skew=calib["raw_skew"], raw_kurtosis=calib["raw_kurtosis"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-genes", type=int, default=None,
                    help="smoke test: only use the first N genes of the nz<max(THRESHOLDS) superset")
    ap.add_argument("--thresholds", type=int, nargs="+", default=None,
                    help="smoke test: override THRESHOLDS with a shorter list")
    args = ap.parse_args()
    thresholds = args.thresholds or THRESHOLDS

    out_dir = config.THRESHOLD_SWEEP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(config.ENGINE_MIXED_DIR / "training_summary.csv", index_col="gene")
    summary = summary[summary["ok"]]
    superset = summary.index[summary["nz"] < max(thresholds)].tolist()[:args.limit_genes]
    nz_of = summary["nz"].to_dict()
    stage_of = summary["stage"].to_dict()
    print(f"superset (nz < {max(thresholds)}): {len(superset)} genes")

    e2 = NormativeModelEngineMixed()
    e2.load_hc_data()
    e2.alpha_fn = load_trend()
    n_hc = e2.X_hc_scaled.shape[0]
    folds = list(StratifiedKFold(MP["n_splits"], shuffle=True, random_state=42).split(np.zeros(n_hc), e2.batch))

    tmp = "/tmp/pool_threshold_sweep"
    Path(tmp).mkdir(exist_ok=True)

    # individual (nbi-fixed-stage) CV is run exactly once, on the full nz<max(thresholds)
    ind_cache_path = out_dir / "individual_cv_cache.pkl"
    zdict_ind, ind_fold_info = {}, {}
    if ind_cache_path.exists():
        with open(ind_cache_path, "rb") as f:
            cached_superset, zdict_ind, ind_fold_info = pickle.load(f)
        if set(superset) <= set(cached_superset):
            print(f"individual (model-route) CV baseline: loaded cache -> {ind_cache_path}")
        else:
            print("individual (model-route) CV cache stale for current superset -- recomputing")
            zdict_ind, ind_fold_info = {}, {}

    if not (set(superset) <= set(ind_fold_info)):
        print(f"individual (model-route) CV baseline on {len(superset)} genes...")
        zdict_ind, _, ind_fold_stat_rows = cv_model_route(e2, superset, stage_of, folds, tmp)
        ind_fold_info = {g: dict(n_folds_ok=0, fail_reasons=[]) for g in superset}
        for r in ind_fold_stat_rows:
            info = ind_fold_info[r["gene"]]
            if r["ok"]:
                info["n_folds_ok"] += 1
            else:
                info["fail_reasons"].append(f"fold{r['fold']}:{r['fail_reason'] or 'unknown'}")
        with open(ind_cache_path, "wb") as f:
            pickle.dump((superset, zdict_ind, ind_fold_info), f)

    # pooled CV is threshold-specific and cached per threshold, so re-running the sweep
    # (e.g. after adding a new threshold) only computes the missing ones.
    per_t_dfs = []
    for t in thresholds:
        t_path = out_dir / f"pool_vs_individual_naive_z_t{t}.csv"
        if t_path.exists():
            print(f"threshold={t}: cached -> {t_path}")
            per_t_dfs.append(pd.read_csv(t_path))
            continue

        genes_t = [g for g in superset if nz_of[g] < t]
        if not genes_t:
            continue
        print(f"threshold={t}: pooled CV on {len(genes_t)} genes...")
        zdict_pool, pool_fold_stats = cv_pool_route(e2, genes_t, folds, tmp)
        pool_n_folds_ok = sum(1 for r in pool_fold_stats if r["ok"])
        pool_fail_reasons = ";".join(f"fold{r['fold']}:{r['fail_reason']}" for r in pool_fold_stats if not r["ok"])

        t_rows = []
        for g in genes_t:
            m_ind = naive_calib(zdict_ind.get(g, np.array([])))
            m_pool = naive_calib(zdict_pool.get(g, np.array([])))
            t_rows.append(dict(
                threshold=t, gene=g, nz=nz_of[g], ind_stage=stage_of[g],
                ind_n_folds_ok=ind_fold_info[g]["n_folds_ok"],
                ind_fail_reasons=";".join(ind_fold_info[g]["fail_reasons"]),
                **{f"ind_{k}": v for k, v in m_ind.items()},
                pool_n_folds_ok=pool_n_folds_ok, pool_fail_reasons=pool_fail_reasons,
                **{f"pool_{k}": v for k, v in m_pool.items()},
            ))
        t_df = pd.DataFrame(t_rows)
        t_df.to_csv(t_path, index=False)
        print(f"Saved -> {t_path} ({len(t_df)} rows)")
        per_t_dfs.append(t_df)

    df = pd.concat(per_t_dfs, ignore_index=True)
    df.to_csv(out_dir / "pool_vs_individual_naive_z.csv", index=False)
    print(f"Saved combined -> {out_dir}/pool_vs_individual_naive_z.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()
