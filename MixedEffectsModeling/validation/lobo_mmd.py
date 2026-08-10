import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.shash import load_shash_params, shash_correct_col, shash_correct_matrix

LOBO_DIR = config.LOBO_MIXED_DIR

# Batches below this held-out-HC count give an unstable noise floor -- see
# memory project_lobo_validation_design.md (same convention as the v1 engine).
MIN_N_HC = 25


def _cache_path(shash, ood_filter):
    suffix = ("_shash" if shash else "") + ("_ood" if ood_filter else "")
    return LOBO_DIR / f"mmd_summary{suffix}.csv"


def load_batch(batch_dir, ood_filter=False):
    """ood_filter=True drops held-out samples flagged by compute_ood()
    (lobo_engine.py) as covariate-extrapolating relative to that batch's own
    train-fold HC -- otherwise a large |Z| can reflect the model being asked
    to extrapolate, not disease biology."""
    meta = json.load(open(batch_dir / "meta.json"))
    Z = np.load(batch_dir / "Z_test.npy")
    is_hc = np.array(meta["test_is_hc"])
    if ood_filter:
        mask_path = batch_dir / "ood_mask.npy"
        if not mask_path.exists():
            raise FileNotFoundError(f"{mask_path} missing -- run lobo_engine.compute_ood() first")
        keep = np.load(mask_path)
        Z, is_hc = Z[keep], is_hc[keep]
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


def _ref_centroid_direction(Z, is_hc):
    """Reference centroid built ONLY from this batch's own LOBO-scored held-out HC
    (a model that never trained on this batch) -- not the earlier CV-based centroid,
    which leaked information because CV's StratifiedKFold splits within every batch
    rather than excluding one, so this batch's own samples entered 4/5 of its
    training folds. Each HC sample is compared to a leave-one-out centroid of the
    OTHER held-out HC (excluding itself), or it would be pulled toward its own
    position and its distance would be spuriously shrunk; disease samples never
    contribute to the centroid, so they compare against the full HC mean."""
    hc_Z = Z[is_hc]
    n_hc = len(hc_Z)
    hc_sum = hc_Z.sum(axis=0)
    loo_centroid = (hc_sum[None, :] - hc_Z) / (n_hc - 1)
    d_hc = np.linalg.norm(hc_Z - loo_centroid, axis=1)
    full_centroid = hc_sum / n_hc
    d_dis = np.linalg.norm(Z[~is_hc] - full_centroid[None, :], axis=1)
    _, p_direction = mannwhitneyu(d_dis, d_hc, alternative="greater")
    return d_hc.mean(), d_dis.mean(), float(p_direction)


def mmd_summary(min_n_hc=MIN_N_HC, n_perm=1000, max_n_per_group=150, seed=42, shash=False, ood_filter=False):
    """For each Tier-A batch with a stably-estimable held-out-HC noise floor,
    tests whether held-out-HC and held-out-disease Z-vectors come from
    different distributions (MMD^2, RBF kernel over all genes jointly,
    permutation p-value), then reports direction (disease farther than
    held-out-HC from a reference centroid, or not). The reference centroid is
    built ONLY from this batch's own held-out HC (see _ref_centroid_direction)
    -- an earlier version used a centroid pooled from the standard 5-fold CV,
    which leaked this batch's own samples back in (CV splits within every
    batch rather than excluding one) and diluted the noise-floor estimate.
    shash=True applies the per-gene SHASH correction (core/calibration.py,
    fit on in-fold CV Z) to the LOBO Z before comparing.
    ood_filter=True drops held-out samples flagged as covariate-extrapolating
    (see load_batch); requires lobo_engine.compute_ood() to have been run."""
    shash_params = load_shash_params(config.CV_MIXED_DIR / "cv_stats.csv") if shash else None

    rows = []
    for b in tier_a_batches(min_n_hc=min_n_hc):
        safe = b.replace(" ", "_")
        bdir = LOBO_DIR / safe
        meta, Z, is_hc = load_batch(bdir, ood_filter=ood_filter)
        Z = np.nan_to_num(Z, nan=0.0, posinf=10.0, neginf=-10.0)
        gene_names = pickle.load(open(bdir / "gene_names.pkl", "rb"))
        if shash:
            Z = shash_correct_matrix(Z, gene_names, shash_params)
        hc_Zb, dis_Zb = Z[is_hc], Z[~is_hc]
        if len(hc_Zb) < 5 or len(dis_Zb) < 5:
            continue

        rng = np.random.default_rng(seed)
        hc_s = hc_Zb if len(hc_Zb) <= max_n_per_group else hc_Zb[rng.choice(len(hc_Zb), max_n_per_group, replace=False)]
        dis_s = dis_Zb if len(dis_Zb) <= max_n_per_group else dis_Zb[rng.choice(len(dis_Zb), max_n_per_group, replace=False)]
        mmd2, p = _mmd_permutation_test(hc_s, dis_s, n_perm=n_perm, seed=seed)

        hc_dist, dis_dist, p_direction = _ref_centroid_direction(Z, is_hc)
        rows.append(dict(batch=b, n_hc=len(hc_Zb), n_dis=len(dis_Zb), mmd2=mmd2, perm_p=p,
                         hc_ref_dist=hc_dist, dis_ref_dist=dis_dist, disease_farther=dis_dist > hc_dist,
                         p_direction=p_direction))
    return pd.DataFrame(rows).sort_values("n_dis", ascending=False)


def mmd_summary_cached(force=False, csv_path=None, shash=False, ood_filter=False, **kwargs):
    csv_path = csv_path or _cache_path(shash, ood_filter)
    if not force and os.path.isfile(csv_path):
        return pd.read_csv(csv_path)
    df = mmd_summary(shash=shash, ood_filter=ood_filter, **kwargs)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
