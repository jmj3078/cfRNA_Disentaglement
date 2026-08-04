"""Control-composition sensitivity of group-wise biomarker selection (Moore et al. Batch_1).

Case group fixed; HC split into tertiles of a technical-bias axis. Each stratum yields its
own marker list, and the agreement between lists is compared against a size-matched random
split of the same HC pool -- the null that separates "control composition" from "small n".
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.ood_filter import MahalanobisFilter, RangeFilter

BATCH = "Moore et al._Batch_1"
DISEASES = ["Pancreatic Cancer", "Pancreatitis"]
LAYERS = ["CPM_log1p", "TPM_log2", "TMM_log2",
          "RUVg_Platelet_k1", "RUVg_Platelet_k2", "RUVg_Platelet_k3",
          "Proposed_Full_k1", "Proposed_Full_k2", "Proposed_Full_k3"]
K_GRID = [25, 50, 100, 200, 500]
N_NULL = 500
MIN_COUNT_SUM = 10
SEED = 42
CACHE = config.CTRL_COMP_DIR / "moore_b1_cache.pkl"
RESULT_CSV = config.CTRL_COMP_DIR / "jaccard_results.csv"
NULL_CSV = config.CTRL_COMP_DIR / "null_distribution.csv"


def build_cache(h5ad_path=config.H5AD_PATH):
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    config.CTRL_COMP_DIR.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(h5ad_path)
    adata = adata[adata.obs["QC_Passed"] == True]
    adata = adata[adata.obs["Phenotype_Processed"].notna()]
    adata = adata[adata.obs["Phenotype_Processed"] != "Unknown"]
    adata = adata[adata.obs["broad_protocol_category"] != "Exome-based (EB)"]

    pheno = adata.obs["Phenotype_Processed"].astype(str).values
    batch = adata.obs[config.STRATIFY_COL].astype(str).values
    is_hc = pheno == "Healthy Control"
    bsize = pd.Series(batch[is_hc]).value_counts()
    small = set(bsize.loc[lambda v: v < config.MIN_HC_BATCH_SIZE].index)
    train_hc = is_hc & ~np.isin(batch, list(small))

    X_all = adata.obs[config.BIAS_COLUMNS].values.astype(float)
    ood = MahalanobisFilter(percentile=config.MODELING_PARAMS["ood_percentile"]
                            if hasattr(config, "MODELING_PARAMS") else 95).fit(X_all[train_hc])
    rng_f = RangeFilter(n_out_thr=2).fit(X_all[train_hc])
    inlier = ood.mask(X_all) & rng_f.mask(X_all)

    keep = (batch == BATCH) & inlier & np.isin(pheno, ["Healthy Control"] + DISEASES)
    sub = adata[keep]
    raw = sub.layers["Raw"]
    raw = raw.toarray() if issparse(raw) else np.asarray(raw)
    gene_ok = (sub.var["GeneType"] == "protein_coding").values & (raw.sum(axis=0) >= MIN_COUNT_SUM)

    layers = {}
    for name in LAYERS:
        L = sub.layers[name]
        L = L.toarray() if issparse(L) else np.asarray(L)
        layers[name] = np.asarray(L[:, gene_ok], dtype=np.float32)

    obs = pd.DataFrame({"sample": sub.obs_names.astype(str).values,
                        "phenotype": pheno[keep]})
    obs[config.BIAS_COLUMNS] = X_all[keep]
    data = dict(obs=obs, layers=layers,
                genes=sub.var_names[gene_ok].astype(str).values,
                gene_names=sub.var["GeneName"].astype(str).values[gene_ok])
    with open(CACHE, "wb") as f:
        pickle.dump(data, f)
    return data


def bias_axes(obs):
    """10 raw bias columns plus PC1 of the standardized set (composite technical axis)."""
    X = obs[config.BIAS_COLUMNS].values
    Z = (X - X.mean(0)) / X.std(0)
    u, s, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pc1 = u[:, 0] * s[0]
    if np.corrcoef(pc1, Z[:, 0])[0, 1] < 0:
        pc1 = -pc1
    axes = {c: X[:, i] for i, c in enumerate(config.BIAS_COLUMNS)}
    axes["PC1(bias)"] = pc1
    return axes, vt[0]


def tertiles(values):
    r = rankdata(values, method="ordinal") - 1
    n = len(values)
    edges = [0, n // 3, 2 * n // 3, n]
    return [np.where((r >= edges[i]) & (r < edges[i + 1]))[0] for i in range(3)]


def welch_t(A, B):
    ma, mb = A.mean(0), B.mean(0)
    va, vb = A.var(0, ddof=1), B.var(0, ddof=1)
    se = np.sqrt(va / A.shape[0] + vb / B.shape[0])
    return np.divide(ma - mb, se, out=np.zeros_like(ma), where=se > 0)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def _pair_metrics(ti, tj, k_grid):
    top_i = {k: set(np.argsort(-np.abs(ti))[:k]) for k in k_grid}
    top_j = {k: set(np.argsort(-np.abs(tj))[:k]) for k in k_grid}
    out = {}
    for k in k_grid:
        a, b = top_i[k], top_j[k]
        inter = a & b
        out[f"jaccard_k{k}"] = len(inter) / len(a | b)
        idx = np.array(sorted(inter), dtype=int)
        out[f"signflip_k{k}"] = float((np.sign(ti[idx]) != np.sign(tj[idx])).mean()) if len(idx) else np.nan
    r = rankdata(ti), rankdata(tj)
    out["spearman"] = float(np.corrcoef(r[0], r[1])[0, 1])
    return out


def strata_metrics(L, case_idx, strata, k_grid=K_GRID):
    """Mean over the 3 pairwise comparisons of the marker lists from each control stratum."""
    stats = [welch_t(L[case_idx], L[s]) for s in strata]
    rows = []
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            m = _pair_metrics(stats[i], stats[j], k_grid)
            m.update(pair=f"{i}-{j}")
            rows.append(m)
    df = pd.DataFrame(rows)
    return df.drop(columns="pair").mean().to_dict(), df


def run_all(force=False):
    if RESULT_CSV.exists() and NULL_CSV.exists() and not force:
        return pd.read_csv(RESULT_CSV), pd.read_csv(NULL_CSV)
    data = build_cache()
    obs = data["obs"]
    axes, _ = bias_axes(obs)
    hc_idx = np.where(obs["phenotype"].values == "Healthy Control")[0]
    rng = np.random.default_rng(SEED)

    rows, null_rows = [], []
    for disease in DISEASES:
        case_idx = np.where(obs["phenotype"].values == disease)[0]
        for layer in LAYERS:
            L = data["layers"][layer]
            for axis, values in axes.items():
                strata = [hc_idx[s] for s in tertiles(values[hc_idx])]
                m, _ = strata_metrics(L, case_idx, strata)
                d = [cohens_d(values[case_idx], values[s]) for s in strata]
                m.update(disease=disease, layer=layer, axis=axis, split="tertile",
                         n_case=len(case_idx), n_ctrl=int(np.mean([len(s) for s in strata])),
                         delta_d=float(max(d) - min(d)))
                rows.append(m)

            sizes = [len(s) for s in tertiles(values[hc_idx])]
            for b in range(N_NULL):
                perm = rng.permutation(hc_idx)
                strata = [perm[:sizes[0]], perm[sizes[0]:sizes[0] + sizes[1]], perm[sizes[0] + sizes[1]:]]
                m, _ = strata_metrics(L, case_idx, strata)
                m.update(disease=disease, layer=layer, draw=b, split="random",
                         n_case=len(case_idx), n_ctrl=int(np.mean(sizes)))
                null_rows.append(m)
            break_axes = True
        if break_axes:
            pass
    res, null = pd.DataFrame(rows), pd.DataFrame(null_rows)
    config.CTRL_COMP_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULT_CSV, index=False)
    null.to_csv(NULL_CSV, index=False)
    return res, null


def null_pvalue(res, null, metric="jaccard_k100"):
    """One-sided empirical p: how often a size-matched random split agrees no better
    than the bias-stratified split."""
    out = []
    for (dis, lay), g in null.groupby(["disease", "layer"]):
        ref = g[metric].values
        for _, r in res[(res.disease == dis) & (res.layer == lay)].iterrows():
            out.append(dict(disease=dis, layer=lay, axis=r["axis"], obs=r[metric],
                            null_mean=ref.mean(), null_sd=ref.std(),
                            p_lower=float((ref <= r[metric]).mean())))
    return pd.DataFrame(out)


if __name__ == "__main__":
    res, null = run_all(force="--force" in sys.argv)
    print(res.groupby(["disease", "layer"])["jaccard_k100"].mean().to_string())
    print()
    print(null.groupby(["disease", "layer"])["jaccard_k100"].agg(["mean", "std"]).to_string())
