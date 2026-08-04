import json
import pickle
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

ROOT = Path("/project/cfRNA_NormativeModeling")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "EDA"))
from analysis_helper import DataAnalysisPipeline
from config import PATHS

BIAS = [
    "log(Total Reads)", "Spliced Reads (%)", "gDNA Contamination (Intron/Exon)",
    "rRNA Fraction", "RNA Degradation (3' Bias)", "Platelet Score",
    "GC Bias", "Gene Length Bias", "NG80", "(NP80/NG80)",
]
LAYERS = ["CPM_log1p", "RUVg_Platelet_k1", "RUVg_Platelet_k2", "RUVg_Platelet_k3"]
LOBO = ROOT / "MixedEffectsModeling" / "LOBO_Results_mixed"
MIN_HC = 25
OUT = Path(__file__).parent / "results" / "b0_residual_bias.csv"

enc = DataAnalysisPipeline._encode_design_matrix
adj = DataAnalysisPipeline._adj_r2


def read_rows(f, layer, rows, colmap, n_out):
    g = f["layers"][layer]
    indptr = g["indptr"][:]
    out = np.zeros((len(rows), n_out), dtype=np.float32)
    for i, r in enumerate(rows):
        s, e = indptr[r], indptr[r + 1]
        idx = g["indices"][s:e]
        dat = g["data"][s:e]
        m = colmap[idx] >= 0
        out[i, colmap[idx[m]]] = dat[m]
    return out


def unique_r2(Y, obs_sub, standardize=False):
    Y = Y - Y.mean(axis=0)
    keep = Y.std(axis=0) > 1e-8
    Y = Y[:, keep]
    if standardize:
        Y = Y / Y.std(axis=0)
    r2_all, _ = adj(Y, enc(obs_sub, BIAS, set())[0])
    rec = {"Joint": r2_all, "n_genes": int(keep.sum())}
    for v in BIAS:
        r2_minus, _ = adj(Y, enc(obs_sub, [c for c in BIAS if c != v], set())[0])
        rec[v] = max(0.0, r2_all - r2_minus)
    return rec


adata = ad.read_h5ad(PATHS["merged_qc"], backed="r")
obs = adata.obs
var_names = np.asarray(adata.var_names)
name_to_row = {n: i for i, n in enumerate(obs.index)}
gene_to_col = {g: i for i, g in enumerate(var_names)}

rows_out = []
f = h5py.File(PATHS["merged_qc"], "r")

for d in sorted(p for p in LOBO.iterdir() if (p / "meta.json").is_file()):
    meta = json.load(open(d / "meta.json"))
    names = np.asarray(meta["test_names"])
    is_hc = np.asarray(meta["test_is_hc"], dtype=bool)
    Z = np.load(d / "Z_test.npy")
    genes = pickle.load(open(d / "gene_names.pkl", "rb"))

    ok_obs = obs.loc[names[is_hc], BIAS].notna().all(axis=1).values
    hc_names = names[is_hc][ok_obs]
    if len(hc_names) < MIN_HC:
        print(f"[skip] {d.name}: n_hc_test={len(hc_names)}")
        continue

    Zh = Z[is_hc][ok_obs]
    gene_ok = np.isfinite(Zh).all(axis=0) & np.isin(genes, var_names)
    genes_ok = np.asarray(genes)[gene_ok]
    Zh = Zh[:, gene_ok]

    rows = np.array([name_to_row[n] for n in hc_names])
    order = np.argsort(rows)
    inv = np.argsort(order)
    cols = np.array([gene_to_col[g] for g in genes_ok])
    colmap = -np.ones(len(var_names), dtype=np.int64)
    colmap[cols] = np.arange(len(cols))

    obs_sub = obs.loc[hc_names, BIAS]
    print(f"[run] {d.name}: n_hc={len(hc_names)} genes={len(cols)}", flush=True)

    mats = {"NormativeZ": Zh}
    for layer in LAYERS:
        mats[layer] = read_rows(f, layer, rows[order], colmap, len(cols))[inv]

    for name, Y in mats.items():
        for std in (False, True):
            rec = unique_r2(Y, obs_sub, standardize=std)
            rec.update({"Batch": d.name, "Feature": name, "n_hc": len(hc_names),
                        "Weighting": "gene_std" if std else "raw"})
            rows_out.append(rec)

df = pd.DataFrame(rows_out)
df.to_csv(OUT, index=False)

order_f = ["CPM_log1p", "RUVg_Platelet_k1", "RUVg_Platelet_k2", "RUVg_Platelet_k3", "NormativeZ"]
for w in ("raw", "gene_std"):
    sub = df[df["Weighting"] == w]
    piv = sub.groupby("Feature")[["Joint"] + BIAS].mean().reindex(order_f)
    print(f"\n=== mean adj R2 across batches, weighting={w} (held-out HC only) ===")
    print(piv.T.round(4).to_string())
    print("\nper-batch Joint:")
    print(sub.pivot(index="Batch", columns="Feature", values="Joint")[order_f].round(3).to_string())
print("\nbatches:", df["Batch"].nunique(), "| n_hc total:", df.groupby("Batch")["n_hc"].first().sum())
