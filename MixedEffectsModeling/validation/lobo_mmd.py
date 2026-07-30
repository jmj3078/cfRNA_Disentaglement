import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.shash import shash_transform_to_z

LOBO_DIR = config.LOBO_MIXED_DIR
MMD_SUMMARY_CSV = LOBO_DIR / "mmd_summary.csv"
MMD_SUMMARY_SHASH_CSV = LOBO_DIR / "mmd_summary_shash.csv"

# Batches below this held-out-HC count give an unstable noise floor -- see
# memory project_lobo_validation_design.md (same convention as the v1 engine).
MIN_N_HC = 25


def load_shash_params(cv_stats_path=None):
    """Per-gene SHASH params fit on in-fold CV Z (core/calibration.py
    gene_shash_calibration) -- the transform that corrects residual per-gene
    skew/kurtosis before treating a Z-score as N(0,1). Neither the LOBO
    Z_test.npy nor cv_zscores.pkl carry this correction by default; it must be
    applied explicitly (see mmd_summary(shash=True))."""
    path = cv_stats_path or (config.CV_MIXED_DIR / "cv_stats.csv")
    return pd.read_csv(path).set_index("gene")[
        ["cv_shash_ok", "cv_shash_xi", "cv_shash_eta", "cv_shash_eps", "cv_shash_delta"]]


def _shash_correct_col(z, row):
    if not bool(row["cv_shash_ok"]):
        return z
    return shash_transform_to_z(z, row["cv_shash_xi"], row["cv_shash_eta"], row["cv_shash_eps"], row["cv_shash_delta"])


def _shash_correct_matrix(Z, gene_names, params):
    Zc = Z.copy()
    for j, g in enumerate(gene_names):
        if g in params.index:
            Zc[:, j] = _shash_correct_col(Z[:, j], params.loc[g])
    return Zc


def load_batch(batch_dir):
    meta = json.load(open(batch_dir / "meta.json"))
    Z = np.load(batch_dir / "Z_test.npy")
    is_hc = np.array(meta["test_is_hc"])
    return meta, Z, is_hc


def tier_a_batches(min_n_hc=MIN_N_HC):
    df = pd.read_csv(LOBO_DIR / "batch_tier_assignment.csv")
    tier_a = df[df["tier"] == "A"]["batch"].tolist()
    if min_n_hc is None:
        return tier_a
    kept = []
    for b in tier_a:
        safe = b.replace(" ", "_")
        meta_path = LOBO_DIR / safe / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        if sum(meta["test_is_hc"]) >= min_n_hc:
            kept.append(b)
    return kept


def load_hc_reference_centroid(cv_zscores_path=None, shash_params=None):
    """Per-gene mean Z from the standard 5-fold engine CV (NOT the LOBO run) --
    the 'model trained/evaluated under normal conditions' reference point both
    held-out-HC and disease are compared against. Per-gene arrays in
    cv_zscores.pkl are ragged (a gene that failed some CV folds has fewer
    entries than one that passed all 5), so this only ever takes a per-gene
    mean -- never stacks them into a shared per-sample matrix.
    shash_params, if given, SHASH-corrects each gene's Z before averaging."""
    path = cv_zscores_path or (config.CV_MIXED_DIR / "cv_zscores.pkl")
    d = pickle.load(open(path, "rb"))
    genes = list(d.keys())
    centroid = np.empty(len(genes))
    for i, g in enumerate(genes):
        z = np.nan_to_num(d[g], nan=np.nan, posinf=10.0, neginf=-10.0)
        if shash_params is not None and g in shash_params.index:
            z = _shash_correct_col(z, shash_params.loc[g])
        centroid[i] = np.nanmean(z)
    return genes, centroid


def _mmd2_from_kernel(K, n):
    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    m = K.shape[0] - n
    sx = Kxx.sum() - np.trace(Kxx)
    sy = Kyy.sum() - np.trace(Kyy)
    return sx / (n * (n - 1)) + sy / (m * (m - 1)) - 2 * Kxy.mean()


def _mmd_permutation_test(X, Y, n_perm=1000, seed=42):
    from scipy.spatial.distance import pdist, squareform
    pooled = np.vstack([X, Y])
    n = len(X)
    d2 = squareform(pdist(pooled, metric="sqeuclidean"))
    med = np.median(d2[d2 > 0])
    gamma = 1.0 / med if med > 0 else 1.0
    K = np.exp(-gamma * d2)

    obs = _mmd2_from_kernel(K, n)
    rng = np.random.default_rng(seed)
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(pooled))
        perm_stats[i] = _mmd2_from_kernel(K[np.ix_(idx, idx)], n)
    p = (1 + (perm_stats >= obs).sum()) / (1 + n_perm)
    return obs, p


def _ref_centroid_direction(gene_names, Z, is_hc, hc_genes, hc_centroid):
    gene_index = {g: i for i, g in enumerate(hc_genes)}
    keep_cols = [i for i, g in enumerate(gene_names) if g in gene_index]
    cols = [gene_index[gene_names[i]] for i in keep_cols]
    order = np.argsort(cols)
    Zsub = np.nan_to_num(Z[:, keep_cols][:, order], nan=0.0, posinf=10.0, neginf=-10.0)
    ref = hc_centroid[np.sort(cols)]
    dist = np.sqrt(((Zsub - ref) ** 2).sum(axis=1))
    return dist[is_hc].mean(), dist[~is_hc].mean()


def mmd_summary(min_n_hc=MIN_N_HC, n_perm=1000, max_n_per_group=150, seed=42, shash=False):
    """For each Tier-A batch with a stably-estimable held-out-HC noise floor,
    tests whether held-out-HC and held-out-disease Z-vectors come from
    different distributions (MMD^2, RBF kernel over all genes jointly,
    permutation p-value), then reports direction (disease farther than
    held-out-HC from the in-fold HC reference centroid, or not).
    shash=True applies the per-gene SHASH correction (core/calibration.py,
    fit on in-fold CV Z) to both the LOBO Z and the CV reference centroid
    before comparing -- neither carries that correction otherwise."""
    shash_params = load_shash_params() if shash else None
    hc_genes, hc_centroid = load_hc_reference_centroid(shash_params=shash_params)

    rows = []
    for b in tier_a_batches(min_n_hc=min_n_hc):
        safe = b.replace(" ", "_")
        bdir = LOBO_DIR / safe
        meta, Z, is_hc = load_batch(bdir)
        Z = np.nan_to_num(Z, nan=0.0, posinf=10.0, neginf=-10.0)
        gene_names = pickle.load(open(bdir / "gene_names.pkl", "rb"))
        if shash:
            Z = _shash_correct_matrix(Z, gene_names, shash_params)
        hc_Zb, dis_Zb = Z[is_hc], Z[~is_hc]
        if len(hc_Zb) < 5 or len(dis_Zb) < 5:
            continue

        rng = np.random.default_rng(seed)
        hc_s = hc_Zb if len(hc_Zb) <= max_n_per_group else hc_Zb[rng.choice(len(hc_Zb), max_n_per_group, replace=False)]
        dis_s = dis_Zb if len(dis_Zb) <= max_n_per_group else dis_Zb[rng.choice(len(dis_Zb), max_n_per_group, replace=False)]
        mmd2, p = _mmd_permutation_test(hc_s, dis_s, n_perm=n_perm, seed=seed)

        hc_dist, dis_dist = _ref_centroid_direction(gene_names, Z, is_hc, hc_genes, hc_centroid)
        rows.append(dict(batch=b, n_hc=len(hc_Zb), n_dis=len(dis_Zb), mmd2=mmd2, perm_p=p,
                         hc_ref_dist=hc_dist, dis_ref_dist=dis_dist, disease_farther=dis_dist > hc_dist))
    return pd.DataFrame(rows).sort_values("n_dis", ascending=False)


def mmd_summary_cached(force=False, csv_path=None, shash=False, **kwargs):
    if csv_path is None:
        csv_path = MMD_SUMMARY_SHASH_CSV if shash else MMD_SUMMARY_CSV
    if not force and os.path.isfile(csv_path):
        return pd.read_csv(csv_path)
    df = mmd_summary(shash=shash, **kwargs)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
