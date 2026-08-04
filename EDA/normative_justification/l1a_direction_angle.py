import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

ROOT = Path("/project/cfRNA_NormativeModeling")
sys.path.insert(0, str(ROOT))
from config import PATHS

BIAS = [
    "log(Total Reads)", "Spliced Reads (%)", "gDNA Contamination (Intron/Exon)",
    "rRNA Fraction", "RNA Degradation (3' Bias)", "Platelet Score",
    "GC Bias", "Gene Length Bias", "NG80", "(NP80/NG80)",
]
LAYER = "CPM_log1p"
HC = "Healthy Control"
EXCLUDE_AUTHOR = ["Ibarra et al."]
MIN_HC, MIN_DIS = 25, 20
N_PERM = 500
SEED = 0
OUT = Path(__file__).parent / "results" / "l1a_direction_angle.csv"


def read_rows(f, layer, rows, n_genes):
    g = f["layers"][layer]
    indptr = g["indptr"][:]
    out = np.zeros((len(rows), n_genes), dtype=np.float32)
    for i, r in enumerate(rows):
        s, e = indptr[r], indptr[r + 1]
        out[i, g["indices"][s:e]] = g["data"][s:e]
    return out


def proj_frac(Q, d):
    nd = float(d @ d)
    return float(np.sum((Q.T @ d) ** 2) / nd) if nd > 1e-12 else np.nan


adata = ad.read_h5ad(PATHS["merged_qc"], backed="r")
obs = adata.obs
obs = obs[(obs["QC_Passed"] == True) & (~obs["Author"].isin(EXCLUDE_AUTHOR))]
n_genes_all = adata.shape[1]
name_to_row = {n: i for i, n in enumerate(adata.obs.index)}

rng = np.random.default_rng(SEED)
f = h5py.File(PATHS["merged_qc"], "r")
records = []

for batch, g in obs.groupby("Batch_ID", observed=True):
    g = g[g[BIAS].notna().all(axis=1)]
    hc = g[g["Phenotype_Processed"] == HC]
    dis = g[(g["Phenotype_Processed"] != HC) & g["Phenotype_Processed"].notna()]
    if len(hc) < MIN_HC or len(dis) < MIN_DIS:
        continue
    print(f"[run] {batch}: n_hc={len(hc)} n_dis={len(dis)}", flush=True)

    names = list(hc.index) + list(dis.index)
    rows = np.array([name_to_row[n] for n in names])
    order = np.argsort(rows)
    Y = read_rows(f, LAYER, rows[order], n_genes_all)[np.argsort(order)]

    keep = (Y > 0).mean(axis=0) >= 0.10
    Y = Y[:, keep]
    is_hc = np.array([True] * len(hc) + [False] * len(dis))

    sd = Y[is_hc].std(axis=0)
    ok = sd > 1e-8
    Y, sd = Y[:, ok], sd[ok]
    Y = (Y - Y[is_hc].mean(axis=0)) / sd

    # technical subspace spanned by covariate coefficients fitted on HC only
    Xc = hc[BIAS].values.astype(float)
    Xc = (Xc - Xc.mean(axis=0)) / Xc.std(axis=0)
    Xc = np.column_stack([np.ones(len(Xc)), Xc])
    B = np.linalg.pinv(Xc) @ Y[is_hc]
    Q, _ = np.linalg.qr(B[1:].T, mode="reduced")

    d = Y[~is_hc].mean(axis=0) - Y[is_hc].mean(axis=0)
    obs_frac = proj_frac(Q, d)

    null = np.empty(N_PERM)
    n_hc = int(is_hc.sum())
    for i in range(N_PERM):
        perm = rng.permutation(len(is_hc))
        m = np.zeros(len(is_hc), dtype=bool)
        m[perm[:n_hc]] = True
        null[i] = proj_frac(Q, Y[~m].mean(axis=0) - Y[m].mean(axis=0))

    records.append({
        "Batch": batch, "n_hc": len(hc), "n_dis": len(dis), "n_genes": int(ok.sum()),
        "proj_frac": obs_frac,
        "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
        "p_perm": float((np.sum(null >= obs_frac) + 1) / (N_PERM + 1)),
        "effect_norm": float(np.linalg.norm(d)),
    })

df = pd.DataFrame(records)
df.to_csv(OUT, index=False)
print("\n=== disease direction inside HC technical subspace (10 covariates) ===")
print(df.round(4).to_string(index=False))
print(f"\nrandom-direction expectation ~ 10/n_genes = {10 / df['n_genes'].mean():.5f}")
