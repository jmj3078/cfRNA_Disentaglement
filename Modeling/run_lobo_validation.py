#!/usr/bin/env python
"""Leave-one-batch-out (LOBO) validation of the trained NormativeModelEngine.

For each usable Batch_ID (HC present, i.e. a genuine "new batch" test is
possible), refit every gene's ALREADY-DECIDED stage (route/stage taken as
fixed from engine_state/training_summary.csv, same convention as
cv_model_engine.py) on all HC samples OUTSIDE that batch, then score BOTH the
held-out HC samples and any disease samples from that same batch under the
resulting fit. This directly tests whether the batch's HC samples look normal
under a model that never saw that batch (the "new-batch noise floor"), and
whether co-located disease samples deviate further than that floor under the
exact same batch-exposure condition -- see memory
project_lobo_validation_design.md for the full rationale (replaces the
discontinued discrimination_control classifier approach).

Each batch's full result set (z-scores, fold info, gene list actually scored,
the exact HC/disease sample names + row order, and the config snapshot used)
is written to its own directory under config.MODELING_DIR / "LOBO_Results" /
<batch_id> so any batch's run can be reloaded and reproduced independently of
the others and independently of re-running this script.

Usage:
    python run_lobo_validation.py                    # all usable batches
    python run_lobo_validation.py --batch "Chang et al._Batch_1"
    python run_lobo_validation.py --limit-genes 200   # smoke test
"""

import argparse
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from pipeline import data_prep
from cv_model_engine import cv_intercept, cv_nb_fixed, cv_nbi, cv_pool, z_stats, fold_summary
from dispersion_trend import load_trend

MP = config.MODELING_PARAMS
LOBO_DIR = config.MODELING_DIR / "LOBO_Results"


def usable_batches(adata, dd):
    """Batch_IDs with >=1 HC sample AND (after OOD+min_samples filtering) either
    disease samples in the same batch (Tier A) or none (Tier B noise-floor-only).
    Batches with zero HC are dropped -- there is nothing to hold out as "new
    batch HC" for them."""
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    is_hc = (obs["Phenotype_Processed"].astype(str) == "Healthy Control").values
    batch_col = MP["stratify_col"]
    hc_batches = set(obs.loc[is_hc, batch_col].astype(str))
    dis_names = set(str(n) for n in dd.dis_names)
    dis_batch = obs.loc[obs.index.isin(dis_names), batch_col].astype(str)
    dis_counts = dis_batch.value_counts()
    rows = []
    for b in sorted(hc_batches):
        rows.append(dict(batch=b, n_dis_post_filter=int(dis_counts.get(b, 0)),
                         tier="A" if dis_counts.get(b, 0) > 0 else "B"))
    return pd.DataFrame(rows)


def load_full(adata):
    """HC + disease covariates/counts/names in one aligned array, HC scaler
    fit on HC only (mirrors model_engine training convention)."""
    obs = adata.obs
    is_hc = (obs["Phenotype_Processed"].astype(str) == "Healthy Control").values
    X_raw = data_prep.bias_matrix(adata)
    Y = data_prep.count_matrix(adata)
    scaler = StandardScaler()
    scaler.fit(X_raw[is_hc])
    Xs = scaler.transform(X_raw)
    names = np.array(adata.obs_names.astype(str))
    batch = obs[MP["stratify_col"]].astype(str).values
    var_names = np.array(adata.var_names.tolist())
    return Xs, Y, var_names, names, batch, is_hc


def run_one_batch(batch_id, Xs, Y, var_names, names, batch, is_hc, summary,
                  r_fit_fn, alpha_fn, rare_glm_full, engine_cfg, limit_genes=None):
    name2col = {g: i for i, g in enumerate(var_names)}
    held_out = (batch == batch_id)
    tr_mask = is_hc & ~held_out           # training HC = all HC minus this batch
    te_mask = held_out                    # test = every sample (HC+disease) in this batch
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]
    n_tr, n_te = len(tr_idx), len(te_idx)
    te_is_hc = is_hc[te_idx]

    genes = summary if limit_genes is None else summary.iloc[:limit_genes]
    zdict, fold_rows = {}, []
    t0 = time.perf_counter()
    for i, (gene, row) in enumerate(genes.iterrows()):
        j = name2col.get(gene)
        if j is None:
            continue
        y_tr, y_te = Y[tr_idx, j], Y[te_idx, j]
        Xs_tr, Xs_te = Xs[tr_idx], Xs[te_idx]
        route, stage = row["route"], row.get("stage", "")
        y_full = np.concatenate([y_tr, y_te])
        Xs_full = np.concatenate([Xs_tr, Xs_te])
        fold = [(np.arange(n_tr), np.arange(n_tr, n_tr + n_te))]
        if route == "pool":
            z, finfo, mu_all, sigma_all, _ = cv_pool(
                y_full, Xs_full, fold, y_tr.mean(), rare_glm_full, 42)
        elif route == "model" and stage == "intercept":
            z, finfo, mu_all, sigma_all = cv_intercept(y_full, alpha_fn, fold, 42)
        elif route == "model" and stage == "nb_fixed":
            z, finfo, mu_all, sigma_all = cv_nb_fixed(
                y_full, Xs_full, fold, alpha_fn, engine_cfg["outlier_z"],
                engine_cfg["max_outlier_iter"], engine_cfg["max_remove_frac"], 42,
                beta_explode_thr=engine_cfg["beta_explode_thr"], gaic_k=engine_cfg["gaic_k"])
        elif route == "model" and stage == "nbi":
            z, finfo, mu_all, sigma_all = cv_nbi(
                y_full, Xs_full, fold, r_fit_fn, config.BIAS_COLUMNS,
                engine_cfg["outlier_z"], engine_cfg["max_outlier_iter"],
                engine_cfg["max_remove_frac"], engine_cfg["ridge_lambda_sigma"], 42)
        else:
            continue
        zdict[gene] = z[n_tr:].astype(np.float32)   # keep only the held-out (test) half
        fold_rows.append(dict(gene=gene, route=route, stage=stage, **fold_summary(finfo, 1)))
        if (i + 1) % 2000 == 0:
            el = time.perf_counter() - t0
            print(f"    [{batch_id}] {i+1}/{len(genes)} genes  elapsed={el:.0f}s", flush=True)

    Z = np.column_stack([zdict[g] for g in zdict]) if zdict else np.empty((n_te, 0))
    scored_genes = list(zdict.keys())
    return dict(
        batch_id=batch_id, n_hc_train=n_tr, n_test=n_te,
        test_names=names[te_idx].tolist(), test_is_hc=te_is_hc.tolist(),
        gene_names=scored_genes, Z=Z, fold_info=pd.DataFrame(fold_rows),
        elapsed_s=time.perf_counter() - t0,
    )


def save_batch_result(res, tier, n_dis_post_filter, engine_cfg):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", res["batch_id"])
    out_dir = LOBO_DIR / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "Z_test.npy", res["Z"])
    with open(out_dir / "gene_names.pkl", "wb") as f:
        pickle.dump(res["gene_names"], f)
    meta = dict(
        batch_id=res["batch_id"], tier=tier, n_dis_post_filter=n_dis_post_filter,
        n_hc_train=res["n_hc_train"], n_test=res["n_test"],
        test_names=res["test_names"], test_is_hc=res["test_is_hc"],
        elapsed_s=res["elapsed_s"], engine_cfg=engine_cfg,
        n_genes_scored=len(res["gene_names"]),
    )
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    res["fold_info"].to_csv(out_dir / "fold_info.csv", index=False)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=str, default=None,
                    help="run a single Batch_ID only (default: all usable batches)")
    ap.add_argument("--limit-genes", type=int, default=None, help="smoke test")
    args = ap.parse_args()

    LOBO_DIR.mkdir(parents=True, exist_ok=True)
    (LOBO_DIR / "_ALL_DONE.marker").unlink(missing_ok=True)

    print("Loading data...")
    adata = data_prep.load_adata()
    dd = data_prep.load_disease_filtered(adata=adata)
    Xs, Y, var_names, names, batch, is_hc = load_full(adata)

    summary = pd.read_csv(config.ENGINE_DIR / "training_summary.csv", index_col="gene")
    summary = summary[summary["attempted"] & (summary["route"] != "excluded")]

    with open(config.ENGINE_DIR / "config.pkl", "rb") as f:
        engine_cfg = pickle.load(f)
    with open(config.ENGINE_DIR / "rare_glm.pkl", "rb") as f:
        rare_glm_full = pickle.load(f)
    alpha_fn = load_trend()

    print("Initialising R / gamlss...")
    ro.r(f'source("{config.R_HELPER}")')
    r_fit_fn = ro.globalenv["fit_gamlss_gene"]

    batches_df = usable_batches(adata, dd)
    batches_df.to_csv(LOBO_DIR / "batch_tier_assignment.csv", index=False)
    print(batches_df.to_string())

    todo = batches_df if args.batch is None else batches_df[batches_df["batch"] == args.batch]
    if todo.empty:
        raise SystemExit(f"batch {args.batch!r} not found in usable_batches()")

    for _, brow in todo.iterrows():
        b, tier, n_dis = brow["batch"], brow["tier"], brow["n_dis_post_filter"]
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", b)
        if (LOBO_DIR / safe / "meta.json").exists():
            print(f"[skip, already done] {b}")
            continue
        print(f"\n=== LOBO batch: {b}  (tier={tier}, n_dis_post_filter={n_dis}) ===", flush=True)
        res = run_one_batch(b, Xs, Y, var_names, names, batch, is_hc, summary,
                            r_fit_fn, alpha_fn, rare_glm_full, engine_cfg,
                            limit_genes=args.limit_genes)
        out_dir = save_batch_result(res, tier, int(n_dis), engine_cfg)
        print(f"  saved -> {out_dir}  ({res['elapsed_s']:.0f}s, "
             f"{len(res['gene_names'])} genes, n_test={res['n_test']})", flush=True)

    if args.batch is None:
        (LOBO_DIR / "_ALL_DONE.marker").write_text(
            f"all {len(batches_df)} batches finished\n")
        print("\nALL BATCHES DONE.")


if __name__ == "__main__":
    main()
