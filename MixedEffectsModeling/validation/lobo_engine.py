import argparse
import json
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.dispersion_trend import load_trend
from MixedEffectsModeling.core.eb_shrinkage import squeeze_log_theta
from MixedEffectsModeling.core.marginal_rqr import _poisson_rqr, marginal_nb_rqr
from MixedEffectsModeling.validation.cv_engine import squeeze_fold

MP = config.SPIKE_PARAMS


def load_full_data(h5ad_path=config.H5AD_PATH):
    """Same QC filters as NormativeModelEngineMixed.load_hc_data, but keeps every
    phenotype (not just HC) so held-out disease samples can be scored too."""
    adata = sc.read_h5ad(h5ad_path)
    adata = adata[adata.obs["QC_Passed"] == True]
    adata = adata[adata.obs["Phenotype_Processed"].notna()]
    adata = adata[adata.obs["Phenotype_Processed"] != "Unknown"]
    adata = adata[adata.obs["broad_protocol_category"] != "Exome-based (EB)"]

    is_hc = (adata.obs["Phenotype_Processed"].astype(str) == "Healthy Control").values
    batch = adata.obs[config.STRATIFY_COL].astype(str).values
    bsize = pd.Series(batch[is_hc]).value_counts()
    small_hc_batches = set(bsize.loc[lambda v: v < config.MIN_HC_BATCH_SIZE].index)

    X_raw = adata.obs[config.BIAS_COLUMNS].values.astype(np.float64)
    Y_raw = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
    Y = np.round(Y_raw).astype(np.float64)
    names = adata.obs_names.astype(str).values
    is_pc = (adata.var["GeneType"] == "protein_coding").values
    pc_gene_names = adata.var_names[is_pc].tolist()
    pc_idx = np.where(is_pc)[0]
    gene_col = {g: pc_idx[i] for i, g in enumerate(pc_gene_names)}
    return dict(X_raw=X_raw, Y=Y, names=names, batch=batch, is_hc=is_hc,
               small_hc_batches=small_hc_batches, gene_col=gene_col)


def usable_batches(data):
    """Batch_IDs with >=1 HC sample surviving the small-HC-batch drop. tier A =
    disease samples also present in that batch, tier B = HC-only (noise-floor only)."""
    is_hc, batch = data["is_hc"], data["batch"]
    hc_batches = sorted(set(batch[is_hc]) - data["small_hc_batches"])
    dis_counts = pd.Series(batch[~is_hc]).value_counts()
    rows = [dict(batch=b, n_dis=int(dis_counts.get(b, 0)),
                tier="A" if dis_counts.get(b, 0) > 0 else "B") for b in hc_batches]
    return pd.DataFrame(rows)


def _refit_model_genes(tr_idx, data, genes, stage_of, scaler, tmp, disp_prior_path):
    Xs_tr = scaler.transform(data["X_raw"][tr_idx])
    Y_tr = data["Y"][tr_idx][:, [data["gene_col"][g] for g in genes]]
    pd.DataFrame(Xs_tr, columns=config.BIAS_COLUMNS).to_csv(f"{tmp}/X.csv.gz")
    pd.DataFrame(Y_tr, columns=genes).to_csv(f"{tmp}/Y.csv.gz")
    pd.DataFrame({"Batch_ID": data["batch"][tr_idx]}).to_csv(f"{tmp}/batch.csv.gz")
    pd.DataFrame({"gene": genes, "stage": [stage_of[g] for g in genes]}).to_csv(f"{tmp}/genes.csv", index=False)
    fit_params = Path(tmp) / "fit_params.json"
    fit_params.write_text(json.dumps(config.FIT_PARAMS))
    cmd = ["Rscript", str(config.GLMM_FIT_R), "--x", f"{tmp}/X.csv.gz", "--y", f"{tmp}/Y.csv.gz",
          "--batch", f"{tmp}/batch.csv.gz", "--genes", f"{tmp}/genes.csv",
          "--trend", str(config.DISPERSION_TREND_PATH), "--fit-params", str(fit_params),
          "--mode", "fixed_stage", "--out", f"{tmp}/res.csv"]
    if disp_prior_path is not None:
        cmd += ["--disp-prior", str(disp_prior_path)]
    subprocess.run(cmd, check=True, cwd=str(config.GLMM_FIT_R.parent))
    return squeeze_fold(pd.read_csv(f"{tmp}/res.csv").set_index("gene"))


def _score_model_genes(fits, genes, te_idx, data, scaler, alpha_fn, seed=42):
    Xa_te = np.column_stack([np.ones(len(te_idx)), scaler.transform(data["X_raw"][te_idx])])
    zdict, rows = {}, []
    for g in genes:
        if g not in fits.index:
            rows.append(dict(gene=g, route="model", ok=False, fail_reason="fold_output_missing"))
            continue
        row = fits.loc[g]
        ok = bool(row["ok"])
        rows.append(dict(gene=g, route="model", stage=row["stage"], ok=ok,
                         singular=bool(row["singular"]) if not pd.isna(row["singular"]) else None,
                         tau2=float(row["tau2"]) if not pd.isna(row["tau2"]) else np.nan,
                         fail_reason=row["fail_reason"] if not pd.isna(row["fail_reason"]) else ""))
        if not ok:
            continue
        mu_coef = row[[c for c in fits.columns if c.startswith("mu_coef_")]].values.astype(float)
        disp_coef = row[[c for c in fits.columns if c.startswith("disp_coef_")]].values.astype(float)
        mu = np.clip(np.exp(Xa_te @ np.nan_to_num(mu_coef, nan=0.0)), 1e-6, 1e8)
        if not np.all(np.isnan(disp_coef)):
            alpha = np.exp(-Xa_te @ np.nan_to_num(disp_coef, nan=0.0))
        elif "trend_alpha" in row.index and not pd.isna(row["trend_alpha"]):
            alpha = np.full(len(te_idx), float(row["trend_alpha"]))
        else:
            alpha = np.full(len(te_idx), alpha_fn(float(mu.mean())))
        y_te = data["Y"][te_idx, data["gene_col"][g]]
        z = marginal_nb_rqr(y_te, mu, alpha, float(row["tau2"]), seed=seed)
        zdict[g] = z.astype(np.float32)
    return zdict, rows


def _refit_and_score_pool(tr_idx, te_idx, data, genes, scaler, tmp):
    Xs_tr = scaler.transform(data["X_raw"][tr_idx])
    Y_tr = data["Y"][tr_idx][:, [data["gene_col"][g] for g in genes]]
    pd.DataFrame(Xs_tr, columns=config.BIAS_COLUMNS).to_csv(f"{tmp}/Xp.csv.gz")
    pd.DataFrame(Y_tr, columns=genes).to_csv(f"{tmp}/Yp.csv.gz")
    pd.DataFrame({"Batch_ID": data["batch"][tr_idx]}).to_csv(f"{tmp}/batchp.csv.gz")
    pd.DataFrame({"gene": genes}).to_csv(f"{tmp}/genesp.csv", index=False)
    subprocess.run([
        "Rscript", str(config.GLMM_FIT_POOL_R), "--x", f"{tmp}/Xp.csv.gz", "--y", f"{tmp}/Yp.csv.gz",
        "--batch", f"{tmp}/batchp.csv.gz", "--genes", f"{tmp}/genesp.csv",
        "--rare-overdisp-thr", str(MP["rare_overdisp_thr"]), "--out", f"{tmp}/pool_res.json",
    ], check=True, cwd=str(config.GLMM_FIT_POOL_R.parent))

    with open(f"{tmp}/pool_res.json") as f:
        fit = json.load(f)
    if not fit["ok"]:
        return {}, [dict(gene=g, route="pool", ok=False, fail_reason="pooled_glmm_fit_error") for g in genes]

    beta = np.asarray(fit["beta"])
    tau2 = float(fit["tau2"]) if fit.get("tau2") is not None else 0.0
    alpha_eff = fit["alpha"] if fit["family"] == "negbin" else 1e-8
    mean_hc = dict(zip(fit["gene"], fit["mean_hc"]))
    eps = fit["eps"]
    Xs_te = scaler.transform(data["X_raw"][te_idx])
    mult = np.exp(Xs_te @ beta[1:])
    if fit.get("mult_lo") is not None:
        mult = np.clip(mult, fit["mult_lo"], fit["mult_hi"])

    zdict, rows = {}, []
    for g in genes:
        mu = np.clip((mean_hc[g] + eps) * np.exp(beta[0]) * mult, 1e-6, 1e8)
        y_te = data["Y"][te_idx, data["gene_col"][g]]
        z = _poisson_rqr(y_te, mu, seed=42) if fit["family"] == "poisson" else marginal_nb_rqr(y_te, mu, alpha_eff, tau2, seed=42)
        zdict[g] = z.astype(np.float32)
        rows.append(dict(gene=g, route="pool", ok=True, family=fit["family"]))
    return zdict, rows


def run_one_batch(batch_id, data, summary, alpha_fn, disp_prior_path, tmp, limit_genes=None):
    is_hc, batch = data["is_hc"], data["batch"]
    held = batch == batch_id
    small = np.array([b in data["small_hc_batches"] for b in batch])
    tr_idx = np.where(is_hc & ~held & ~small)[0]
    te_idx = np.where(held)[0]

    genes = summary if limit_genes is None else summary.iloc[:limit_genes]
    model_genes = genes.index[genes["route"] == "model"].tolist()
    pool_genes = genes.index[genes["route"] == "pool"].tolist()
    stage_of = genes["stage"].to_dict()
    scaler = StandardScaler().fit(data["X_raw"][tr_idx])

    t0 = time.perf_counter()
    zdict, fold_rows = {}, []
    if model_genes:
        fits = _refit_model_genes(tr_idx, data, model_genes, stage_of, scaler, tmp, disp_prior_path)
        z_m, rows_m = _score_model_genes(fits, model_genes, te_idx, data, scaler, alpha_fn)
        zdict.update(z_m)
        fold_rows += rows_m
    if pool_genes:
        z_p, rows_p = _refit_and_score_pool(tr_idx, te_idx, data, pool_genes, scaler, tmp)
        zdict.update(z_p)
        fold_rows += rows_p

    Z = np.column_stack([zdict[g] for g in zdict]) if zdict else np.empty((len(te_idx), 0))
    return dict(
        batch_id=batch_id, n_hc_train=len(tr_idx), n_test=len(te_idx),
        test_names=data["names"][te_idx].tolist(), test_is_hc=is_hc[te_idx].tolist(),
        gene_names=list(zdict.keys()), Z=Z, fold_info=pd.DataFrame(fold_rows),
        elapsed_s=time.perf_counter() - t0,
    )


def save_batch_result(res, tier, n_dis, out_dir):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", res["batch_id"])
    bdir = out_dir / safe
    bdir.mkdir(parents=True, exist_ok=True)
    np.save(bdir / "Z_test.npy", res["Z"])
    with open(bdir / "gene_names.pkl", "wb") as f:
        pickle.dump(res["gene_names"], f)
    meta = dict(batch_id=res["batch_id"], tier=tier, n_dis=n_dis,
               n_hc_train=res["n_hc_train"], n_test=res["n_test"],
               test_names=res["test_names"], test_is_hc=res["test_is_hc"],
               n_genes_scored=len(res["gene_names"]), elapsed_s=res["elapsed_s"])
    with open(bdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    res["fold_info"].to_csv(bdir / "fold_info.csv", index=False)
    return bdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=str, default=None, help="run a single Batch_ID only (default: all usable batches)")
    ap.add_argument("--limit-genes", type=int, default=None, help="smoke test")
    ap.add_argument("--engine-dir", type=Path, default=config.ENGINE_MIXED_DIR)
    ap.add_argument("--out-dir", type=Path, default=config.LOBO_MIXED_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    disp_prior_path = args.engine_dir / "disp_prior.json"
    if not disp_prior_path.exists():
        disp_prior_path = None

    print("Loading data...")
    data = load_full_data()
    summary = pd.read_csv(args.engine_dir / "training_summary.csv", index_col="gene")
    summary = summary[summary["ok"] & summary["route"].isin(["model", "pool"])]
    alpha_fn = load_trend(args.engine_dir / "dispersion_trend.json")

    batches_df = usable_batches(data)
    batches_df.to_csv(args.out_dir / "batch_tier_assignment.csv", index=False)
    print(batches_df.to_string())

    todo = batches_df if args.batch is None else batches_df[batches_df["batch"] == args.batch]
    if todo.empty:
        raise SystemExit(f"batch {args.batch!r} not found in usable_batches()")

    tmp = "/tmp/lobo_glmm"
    Path(tmp).mkdir(exist_ok=True)
    for _, brow in todo.iterrows():
        b, tier, n_dis = brow["batch"], brow["tier"], int(brow["n_dis"])
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", b)
        if (args.out_dir / safe / "meta.json").exists():
            print(f"[skip, already done] {b}")
            continue
        print(f"\n=== LOBO batch: {b}  (tier={tier}, n_dis={n_dis}) ===", flush=True)
        res = run_one_batch(b, data, summary, alpha_fn, disp_prior_path, tmp, limit_genes=args.limit_genes)
        out_dir = save_batch_result(res, tier, n_dis, args.out_dir)
        print(f"  saved -> {out_dir}  ({res['elapsed_s']:.0f}s, "
             f"{len(res['gene_names'])} genes, n_test={res['n_test']})", flush=True)


if __name__ == "__main__":
    main()
