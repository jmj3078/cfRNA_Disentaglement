import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.pipeline import DataAnalysisPipeline


def compute_gene_wise_bias_rda(
    adata,
    bias_metrics,
    layer="CPM_log1p",
    phenotype_col="Phenotype_Processed",
    target_labels="Healthy Control",
    group_name="HC",
    min_expressed_frac=0.1,
):
    print(f"\n--- [{group_name}] Vectorized Gene-wise Partial RDA ---")

    if target_labels is None:
        sample_mask = np.ones(adata.n_obs, dtype=bool)
    elif isinstance(target_labels, str):
        sample_mask = adata.obs[phenotype_col] == target_labels
    else:
        sample_mask = adata.obs[phenotype_col].isin(target_labels)

    adata_sub = adata[sample_mask].copy()
    obs_cols = [m for m in bias_metrics if m in adata_sub.obs.columns]
    valid_obs_mask = adata_sub.obs[obs_cols].notna().all(axis=1)
    adata_sub = adata_sub[valid_obs_mask].copy()

    n_samples = adata_sub.n_obs
    print(f"[{group_name}] Valid samples : {n_samples:,}")

    X_expr = adata_sub.layers[layer]
    if sp.issparse(X_expr):
        X_expr = X_expr.toarray()
    X_expr = X_expr.astype(np.float32)

    expressed_frac = (X_expr > 0).mean(axis=0)
    keep = expressed_frac >= min_expressed_frac
    Y = X_expr[:, keep]
    gene_names = adata_sub.var_names[keep].tolist()
    n_genes = len(gene_names)
    print(f"[{group_name}] Genes analyzed (≥{min_expressed_frac*100:.0f}% expressed) : {n_genes:,}")

    Y_c = Y - Y.mean(axis=0)
    sst = np.sum(Y_c ** 2, axis=0)
    valid_gene_mask = sst > 1e-12

    categorical_vars = {v for v in obs_cols if not pd.api.types.is_numeric_dtype(adata_sub.obs[v])}

    def _get_design_matrix(vars_list):
        return DataAnalysisPipeline._encode_design_matrix(adata_sub.obs, vars_list, categorical_vars)[0]

    def _vectorized_adj_r2(design_X):
        n, p_plus_1 = design_X.shape
        p = p_plus_1 - 1
        if n <= p + 1:
            return np.zeros(n_genes)

        Q, _ = np.linalg.qr(design_X, mode='reduced')
        ss_reg = np.sum((Q.T @ Y_c) ** 2, axis=0)

        r2 = np.zeros(n_genes)
        r2[valid_gene_mask] = ss_reg[valid_gene_mask] / sst[valid_gene_mask]

        r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
        return np.clip(r2_adj, 0.0, 1.0)

    print(f"[{group_name}] Computing multivariate R² via orthogonal projection...")

    X_all = _get_design_matrix(obs_cols)
    r2_all = _vectorized_adj_r2(X_all)

    gene_records = {"Gene": gene_names, "Joint_R2_All_Biases": r2_all}
    sum_unique = np.zeros(n_genes)

    for bias in obs_cols:
        remaining = [v for v in obs_cols if v != bias]
        if remaining:
            X_minus = _get_design_matrix(remaining)
            r2_minus = _vectorized_adj_r2(X_minus)
        else:
            r2_minus = np.zeros(n_genes)

        unique_r2 = np.clip(r2_all - r2_minus, 0.0, 1.0)
        gene_records[f"Unique_{bias}"] = unique_r2
        sum_unique += unique_r2

    gene_records["Shared_Biases"] = np.clip(r2_all - sum_unique, 0.0, 1.0)
    gene_records["Unexplained"] = np.clip(1.0 - r2_all, 0.0, 1.0)

    df_detail = pd.DataFrame(gene_records)
    summary_data = []

    n_joint_contaminated = np.sum(r2_all > 0.10)
    summary_data.append({
        "Variance_Component": "ALL_BIASES_COMBINED (Joint R²)",
        "Max_R2": round(np.max(r2_all), 4),
        "Mean_R2": round(np.mean(r2_all), 4),
        "Genes_Highly_Biased": int(n_joint_contaminated),
        "Threshold": "> 10%"
    })

    for bias in obs_cols:
        unique_col = f"Unique_{bias}"
        vals = df_detail[unique_col].values
        n_contaminated = np.sum(vals > 0.05)

        summary_data.append({
            "Variance_Component": f"Unique: {bias}",
            "Max_R2": round(np.max(vals), 4),
            "Mean_R2": round(np.mean(vals), 4),
            "Genes_Highly_Biased": int(n_contaminated),
            "Threshold": "> 5%"
        })

    df_summary = pd.DataFrame(summary_data)

    print("=" * 75)
    print(" [Summary] Gene-level Contamination by Confounders")
    print("=" * 75)
    print(df_summary.to_string(index=False))
    print("=" * 75)

    return df_detail, df_summary
