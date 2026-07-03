"""Sweeps the pooling-route nz cutoff and checks 5-fold held-out calibration at
each cutoff, to see how quickly the pooled-GLM shared-beta assumption breaks
down as more (higher-nz, more heterogeneous) genes are folded into the pool.
Mirrors NormativeModelEngineV2.train_rare's pooled-GLM logic (statsmodels
Poisson/NegBinomial with offset=log(mean_hc+eps), shared beta across genes),
scored via cv_pool-style held-out RQR. No R/GAMLSS involved, so this runs fast.

Usage (run from EDA_Modeling/, cwd assumption per project convention):
    python pooling_nz_sweep.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.api as sm
from scipy.sparse import issparse
from scipy.stats import kurtosis, skew
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import NegativeBinomial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from model_engine_v2 import _nb_rqr, _poisson_rqr, _w1_normal
from viz_style import apply_style

apply_style()

MP2 = config.MODELING_PARAMS_V2
OUT_DIR = Path(__file__).resolve().parent / "Analysis_Results"
FIG_DIR = OUT_DIR / "Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NZ_THRESHOLDS = [3, 5, 7, 10, 15, 20, 25, 30, 40, 50]
SEED = 42


def load_hc():
    adata = sc.read_h5ad(config.H5AD_PATH)
    m = ((adata.obs["QC_Passed"] == True) & (adata.obs["Phenotype_Processed"].notna()) &
         (adata.obs["Phenotype_Processed"] != "Unknown") &
         (adata.obs["broad_protocol_category"] != "Exome-based (EB)"))
    a = adata[m]
    is_hc = (a.obs["Phenotype_Processed"].astype(str) == "Healthy Control").values
    is_pc = (a.var["GeneType"] == "protein_coding").values
    X = a.obs[config.BIAS_COLUMNS].values.astype(np.float64)[is_hc]
    Xs = StandardScaler().fit_transform(X)
    Y = a.X.toarray() if issparse(a.X) else np.asarray(a.X)
    Y = np.round(Y[is_hc][:, is_pc]).astype(np.float64)
    names = np.array(a.var_names[is_pc].tolist())
    strata = a.obs[MP2["stratify_col"]].astype(object).fillna("NA").astype(str).values[is_hc]
    return Xs, Y, names, strata


def fit_pooled_glm(y_tr, Xs_tr, gene_cols, rare_overdisp_thr):
    """One shared beta across all genes in gene_cols, offset=log(mean_hc+eps).
    Exact mirror of NormativeModelEngineV2.train_rare, but on a training fold."""
    n_tr = Xs_tr.shape[0]
    n_g = len(gene_cols)
    eps = 1.0 / (2 * n_tr)
    Y_g = y_tr[:, gene_cols]
    mean_hc = Y_g.mean(axis=0)
    sample_idx = np.repeat(np.arange(n_tr), n_g)
    gene_idx = np.tile(np.arange(n_g), n_tr)
    Xc = np.column_stack([np.ones(n_tr * n_g), Xs_tr[sample_idx]])
    y = Y_g[sample_idx, gene_idx]
    offset = np.log(mean_hc[gene_idx] + eps)
    pois = sm.GLM(y, Xc, family=sm.families.Poisson(), offset=offset).fit()
    ratio = float(pois.deviance / pois.df_resid)
    if ratio <= rare_overdisp_thr:
        family, beta, alpha = "poisson", np.asarray(pois.params), None
    else:
        nb = NegativeBinomial(y, Xc, offset=offset).fit(disp=False)
        family, beta, alpha = "negbin", np.asarray(nb.params[:-1]), float(nb.params[-1])
    return dict(family=family, beta=beta, alpha=alpha, eps=eps, mean_hc=mean_hc, overdisp_ratio=ratio)


def cv_pool_sweep(Y, Xs, gene_cols, folds, rare_overdisp_thr, seed):
    """Held-out z for every gene in gene_cols, refitting the pooled GLM fresh
    per fold (unlike cv_pool in cv_model_engine_v2.py, which reuses the
    full-data pooled beta -- here we want an honest held-out check of the
    pooling assumption itself, so the shared beta must not see test folds)."""
    n = Xs.shape[0]
    n_g = len(gene_cols)
    z = np.full((n, n_g), np.nan)
    for fi, (tr, te) in enumerate(folds):
        fit = fit_pooled_glm(Y[tr], Xs[tr], gene_cols, rare_overdisp_thr)
        Xc_te = np.column_stack([np.ones(len(te)), Xs[te]])
        mu = np.clip((fit["mean_hc"][None, :] + fit["eps"]) *
                     np.exp(Xc_te @ fit["beta"])[:, None], 1e-12, 1e8)
        for gi in range(n_g):
            y_te = Y[te, gene_cols[gi]]
            if fit["family"] == "poisson":
                z[te, gi] = _poisson_rqr(y_te, mu[:, gi], seed + fi)
            else:
                z[te, gi] = _nb_rqr(y_te, mu[:, gi], fit["alpha"], seed + fi)
    return z


def main():
    print("Loading HC data...")
    Xs, Y, names, strata = load_hc()
    nz = (Y > 0).sum(0).astype(int)
    n_hc = Xs.shape[0]
    folds = list(StratifiedKFold(MP2["n_splits"], shuffle=True, random_state=SEED)
                .split(np.zeros(n_hc), strata))
    print(f"HC={n_hc}  protein-coding genes={len(names)}")

    rows = []
    per_gene_rows = []
    for T in NZ_THRESHOLDS:
        gene_cols = np.where(nz < T)[0]
        if len(gene_cols) < 5:
            print(f"nz<{T}: only {len(gene_cols)} genes, skipping")
            continue
        print(f"nz<{T}: pooling {len(gene_cols)} genes...")
        z = cv_pool_sweep(Y, Xs, gene_cols, folds, MP2["rare_overdisp_thr"], SEED)

        gene_w1, gene_skew, gene_kurt, gene_std = [], [], [], []
        for gi, col in enumerate(gene_cols):
            v = z[:, gi]
            v = v[np.isfinite(v)]
            if len(v) < 8:
                continue
            gene_w1.append(_w1_normal(v))
            gene_skew.append(float(skew(v)))
            gene_kurt.append(float(kurtosis(v)))
            gene_std.append(float(v.std()))
            per_gene_rows.append(dict(nz_threshold=T, gene=names[col], nz=int(nz[col]),
                                      w1=gene_w1[-1], skew_z=gene_skew[-1],
                                      kurt_z=gene_kurt[-1], std_z=gene_std[-1]))

        rows.append(dict(
            nz_threshold=T, n_genes=len(gene_cols),
            w1_median=np.median(gene_w1), w1_p90=np.percentile(gene_w1, 90),
            skew_median=np.median(gene_skew), kurt_median=np.median(gene_kurt),
            std_median=np.median(gene_std),
        ))

    summary = pd.DataFrame(rows)
    per_gene = pd.DataFrame(per_gene_rows)
    summary.to_csv(OUT_DIR / "pooling_nz_sweep_summary.csv", index=False)
    per_gene.to_csv(OUT_DIR / "pooling_nz_sweep_per_gene.csv", index=False)
    print("\n" + summary.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    nz_a = MP2["nz_a_max"]

    ax = axes[0]
    ax.plot(summary["nz_threshold"], summary["w1_median"], "-o", color="#7570b3", label="median W1")
    ax.plot(summary["nz_threshold"], summary["w1_p90"], "-o", color="#d95f02", label="90th pct W1")
    ax.axvline(nz_a, color="k", ls="--", lw=1, label=f"nz_a_max={nz_a} (current)")
    ax.set(xlabel="nz threshold (pool = genes with nz < threshold)", ylabel="W1 (held-out CV)",
          title="Pooling calibration (W1) vs nz threshold")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(summary["nz_threshold"], summary["skew_median"], "-o", color="#1b9e77", label="median skew_z")
    ax.plot(summary["nz_threshold"], summary["kurt_median"], "-o", color="#e7298a", label="median kurt_z")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axvline(nz_a, color="k", ls="--", lw=1)
    ax.set(xlabel="nz threshold", ylabel="skew_z / kurt_z (held-out)",
          title="Higher moments vs nz threshold")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(summary["nz_threshold"], summary["std_median"], "-o", color="#66a61e")
    ax.axhline(1.0, color="k", ls=":", lw=1, label="calibrated (std=1)")
    ax.axvline(nz_a, color="k", ls="--", lw=1, label=f"nz_a_max={nz_a}")
    ax.set(xlabel="nz threshold", ylabel="median std(z) (held-out)",
          title="RQR spread vs nz threshold")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "pooling_nz_sweep.png", dpi=150)
    plt.show()
    print(f"\nSaved -> {OUT_DIR}/pooling_nz_sweep_summary.csv, pooling_nz_sweep_per_gene.csv")
    print(f"Saved -> {FIG_DIR}/pooling_nz_sweep.png")


if __name__ == "__main__":
    main()
