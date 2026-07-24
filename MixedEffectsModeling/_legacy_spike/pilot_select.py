import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MixedEffectsModeling.config as config


def load_hc_slice():
    adata = sc.read_h5ad(config.H5AD_PATH)
    adata = adata[adata.obs["QC_Passed"] == True]
    adata = adata[adata.obs["Phenotype_Processed"].notna()]
    adata = adata[adata.obs["Phenotype_Processed"] != "Unknown"]
    adata = adata[adata.obs["broad_protocol_category"] != "Exome-based (EB)"]
    is_hc = (adata.obs["Phenotype_Processed"].astype(str) == "Healthy Control").values
    adata_hc = adata[is_hc].copy()

    X_raw = adata_hc.obs[config.BIAS_COLUMNS].values.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    batch = adata_hc.obs[config.STRATIFY_COL].astype(str).values

    Y_raw = adata_hc.X.toarray() if issparse(adata_hc.X) else np.asarray(adata_hc.X)
    Y = np.round(Y_raw).astype(np.float64)

    is_pc = (adata_hc.var["GeneType"] == "protein_coding").values
    gene_names = adata_hc.var_names[is_pc].tolist()
    Y_pc = Y[:, is_pc]

    sample_ids = adata_hc.obs_names.tolist()
    return sample_ids, X_scaled, batch, Y_pc, gene_names


def select_pilot_genes(Y_pc, gene_names, n_pilot, seed):
    nz = (Y_pc > 0).sum(axis=0)
    mean_hc = Y_pc.mean(axis=0)
    df = pd.DataFrame({"gene": gene_names, "nz": nz, "mean_hc": mean_hc})
    df = df[df["nz"] >= config.SPIKE_PARAMS["outlier_z"]]  # trivial floor, avoids all-zero genes

    # Stratify across the NZ range in 4 bins, weighted toward low-NZ (low-expression,
    # highest sigma-explosion risk per the design spec) -- 40% of the pilot from the
    # lowest-NZ quartile, remaining 60% spread evenly across the other three.
    rng = np.random.default_rng(seed)
    df = df.sort_values("nz").reset_index(drop=True)
    quartile_edges = np.quantile(df.index, [0, 0.25, 0.5, 0.75, 1.0]).astype(int)
    weights = [0.4, 0.2, 0.2, 0.2]
    picks = []
    for i in range(4):
        lo, hi = quartile_edges[i], quartile_edges[i + 1]
        bin_idx = df.index[lo:hi] if i < 3 else df.index[lo:hi + 1]
        n_take = max(1, round(n_pilot * weights[i]))
        n_take = min(n_take, len(bin_idx))
        picks.extend(rng.choice(bin_idx, size=n_take, replace=False).tolist())
    picked = df.loc[sorted(set(picks))].reset_index(drop=True)
    return picked


def main():
    config.SPIKE_DIR.mkdir(parents=True, exist_ok=True)
    sample_ids, X_scaled, batch, Y_pc, gene_names = load_hc_slice()
    print(f"HC samples: {len(sample_ids)}  protein-coding genes: {len(gene_names)}  "
          f"unique batches: {len(set(batch))}")

    picked = select_pilot_genes(Y_pc, gene_names, config.SPIKE_PARAMS["n_pilot_genes"],
                                config.SPIKE_PARAMS["seed"])
    gene_col = {g: i for i, g in enumerate(gene_names)}
    pilot_idx = [gene_col[g] for g in picked["gene"]]

    X_df = pd.DataFrame(X_scaled, index=sample_ids, columns=config.BIAS_COLUMNS)
    batch_df = pd.DataFrame({"Batch_ID": batch}, index=sample_ids)
    Y_df = pd.DataFrame(Y_pc[:, pilot_idx], index=sample_ids, columns=picked["gene"].tolist())

    X_df.to_csv(config.SPIKE_DIR / "pilot_X.csv.gz")
    batch_df.to_csv(config.SPIKE_DIR / "pilot_batch.csv.gz")
    Y_df.to_csv(config.SPIKE_DIR / "pilot_Y.csv.gz")
    picked.to_csv(config.SPIKE_DIR / "pilot_genes.csv", index=False)

    assert X_df.shape[0] == Y_df.shape[0] == batch_df.shape[0], "row-count mismatch across exports"
    assert Y_df.shape[1] == len(picked), "pilot gene count mismatch"
    assert batch_df["Batch_ID"].nunique() >= 2, "expected multiple batches in HC pilot slice"
    print(f"PASS: wrote pilot data for {Y_df.shape[1]} genes x {X_df.shape[0]} HC samples, "
          f"{batch_df['Batch_ID'].nunique()} batches")
    print(picked)


if __name__ == "__main__":
    main()
