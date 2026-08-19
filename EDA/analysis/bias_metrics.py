import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import median_abs_deviation
from statsmodels.nonparametric.smoothers_lowess import lowess

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import PARAMS

DEFAULT_METRICS = [
    "gc_bias_score",
    "len_bias_score",
    "platelet_score",
    "log1p_total_counts",
]


def _get_n80_sparse(matrix):
    if not sp.isspmatrix_csr(matrix):
        matrix = sp.csr_matrix(matrix)

    n80 = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        row_data = matrix.data[matrix.indptr[i]:matrix.indptr[i + 1]]
        row_data = row_data[row_data > 0]
        if len(row_data) == 0:
            continue
        cumsum = np.cumsum(np.sort(row_data)[::-1])
        if cumsum[-1] <= 0:
            continue
        n80[i] = np.argmax(cumsum >= cumsum[-1] * 0.8) + 1
    return n80


def _compute_score_sparse(X_csr, feat_vals, n_bins=None, loess_frac=None, outlier_pct=None):
    n_bins = n_bins or PARAMS["n_bins"]
    loess_frac = loess_frac or PARAMS["loess_frac"]
    outlier_pct = outlier_pct or PARAMS["outlier_pct"]

    feat_vals = np.array(feat_vals, dtype=float)
    scores = []

    for i in range(X_csr.shape[0]):
        row_data = X_csr.data[X_csr.indptr[i]:X_csr.indptr[i + 1]]
        row_indices = X_csr.indices[X_csr.indptr[i]:X_csr.indptr[i + 1]]
        pos_mask = row_data > 0
        valid_expr = row_data[pos_mask]
        valid_indices = row_indices[pos_mask]

        if len(valid_expr) < PARAMS["min_expressed"]:
            scores.append(0.0)
            continue

        q_thresh = np.percentile(valid_expr, outlier_pct)
        keep = valid_expr <= q_thresh
        final_expr = valid_expr[keep]
        final_feat = feat_vals[valid_indices[keep]]

        df_tmp = pd.DataFrame({"expr": final_expr, "feat": final_feat})
        try:
            df_tmp["bin"] = pd.qcut(df_tmp["feat"], q=n_bins, duplicates="drop")
        except ValueError:
            scores.append(0.0)
            continue

        bin_stats = (
            df_tmp.groupby("bin", observed=True)
            .agg({"expr": "median", "feat": "mean"})
            .dropna()
        )
        if len(bin_stats) < 2:
            scores.append(0.0)
            continue

        smoothed = lowess(bin_stats["expr"], bin_stats["feat"], frac=loess_frac, it=0)
        curve_disp = median_abs_deviation(smoothed[:, 1], scale="normal")
        total_disp = median_abs_deviation(final_expr, scale="normal")
        scores.append(curve_disp / total_disp if total_disp > 0 else 0.0)

    return np.array(scores)


def calculate_diversity_ratio(adata, gene_type_col="GeneType", coding_label="protein_coding"):
    X = adata.X
    print("Calculating NG80 (sparse)...")
    ng80 = _get_n80_sparse(X)

    print(f"Calculating NP80 (subset: {coding_label})...")
    if gene_type_col in adata.var.columns:
        is_coding = (adata.var[gene_type_col] == coding_label).values
        if np.sum(is_coding) == 0:
            print(f"  [Warning] No genes with type '{coding_label}'. NP80 set to 0.")
            np80 = np.zeros(adata.n_obs)
        else:
            np80 = _get_n80_sparse(X[:, is_coding])
    else:
        print(f"  [Warning] Column '{gene_type_col}' not found. NP80 set to NaN.")
        np80 = np.full(adata.n_obs, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np80 / ng80
        ratio[ng80 == 0] = 0

    return pd.DataFrame(
        {"NG80": ng80, "NP80": np80, "NP80_NG80_ratio": ratio},
        index=adata.obs_names,
    )


def calculate_bias_metrics(
    adata,
    layer=None,
    gene_type_col="GeneType",
    target_type="protein_coding",
    gc_col="GC_Percent",
    len_col="log10_Length",
    platelet_col="is_platelet",
):
    print(f"--- Calculating Bfias Metrics (layer: {layer or 'X'}) ---")

    X_data = adata.layers[layer] if (layer and layer in adata.layers) else adata.X
    if not sp.issparse(X_data):
        X_data = sp.csr_matrix(X_data)
    elif not sp.isspmatrix_csr(X_data):
        X_data = X_data.tocsr()

    metrics_df = pd.DataFrame(index=adata.obs_names)

    if gene_type_col in adata.var.columns:
        coding_mask = (adata.var[gene_type_col] == target_type).values
        if not np.any(coding_mask):
            print(f"  [Warning] No genes for type '{target_type}'. Using all genes.")
            coding_mask = np.ones(adata.n_vars, dtype=bool)
    else:
        coding_mask = np.ones(adata.n_vars, dtype=bool)

    subset_X = X_data[:, coding_mask]
    subset_var = adata.var.iloc[coding_mask]

    if gc_col in subset_var.columns:
        print("  > GC bias (LOESS)...")
        metrics_df["gc_bias_score"] = _compute_score_sparse(subset_X, subset_var[gc_col])
    else:
        print(f"  [Skip] GC column '{gc_col}' not found.")

    if len_col in subset_var.columns:
        print("  > Length bias (LOESS)...")
        metrics_df["len_bias_score"] = _compute_score_sparse(subset_X, subset_var[len_col])
    else:
        print(f"  [Skip] Length column '{len_col}' not found.")

    if platelet_col in adata.var.columns:
        platelet_genes = adata.var_names[adata.var[platelet_col]].tolist()
        if platelet_genes:
            print(f"  > Platelet score ({len(platelet_genes)} genes)...")
            tmp = sc.AnnData(X=X_data, obs=adata.obs, var=adata.var)
            sc.tl.score_genes(tmp, gene_list=platelet_genes, score_name="platelet_score")
            metrics_df["platelet_score"] = tmp.obs["platelet_score"].values

    print("Done.\n")
    return metrics_df
