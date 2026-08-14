import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import MixedEffectsModeling.config as config

LOBO_DIR = config.LOBO_MIXED_DIR

# Batches below this held-out-HC count give an unstable noise floor -- see
# memory project_lobo_validation_design.md (same convention as the v1 engine).
MIN_N_HC = 25


def _cache_path(shash, ood_filter):
    suffix = ("_shash" if shash else "") + ("_ood" if ood_filter else "")
    return LOBO_DIR / f"mmd_summary{suffix}.csv"


def _raw_cache_path(shash, ood_filter):
    suffix = ("_shash" if shash else "") + ("_ood" if ood_filter else "")
    return LOBO_DIR / f"mmd_raw{suffix}.pkl"


def load_batch(batch_dir, ood_filter=False, shash=False):
    """ood_filter=True drops held-out samples flagged by compute_ood()
    (lobo_engine.py) as covariate-extrapolating relative to that batch's own
    train-fold HC -- otherwise a large |Z| can reflect the model being asked
    to extrapolate, not disease biology. shash=True loads Z_test_shash.npy --
    SHASH already fit (lobo_engine.run_one_batch) on this batch's own
    train-fold in-sample Z and applied to this batch's held-out Z, never fit
    on the held-out Z itself."""
    meta = json.load(open(batch_dir / "meta.json"))
    Z = np.load(batch_dir / ("Z_test_shash.npy" if shash else "Z_test.npy"))
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


def _median_heuristic_gamma(pooled):
    d2 = squareform(pdist(pooled, metric="sqeuclidean"))
    med = np.median(d2[d2 > 0])
    return (1.0 / med if med > 0 else 1.0), d2


def _mmd2_from_kernel(K, n):
    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    m = K.shape[0] - n
    sx = Kxx.sum() - np.trace(Kxx)
    sy = Kyy.sum() - np.trace(Kyy)
    return sx / (n * (n - 1)) + sy / (m * (m - 1)) - 2 * Kxy.mean()


def _mmd_permutation_test(X, Y, n_perm=1000, seed=42):
    pooled = np.vstack([X, Y])
    n = len(X)
    gamma, d2 = _median_heuristic_gamma(pooled)
    K = np.exp(-gamma * d2)

    obs = _mmd2_from_kernel(K, n)
    rng = np.random.default_rng(seed)
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(pooled))
        perm_stats[i] = _mmd2_from_kernel(K[np.ix_(idx, idx)], n)
    p = (1 + (perm_stats >= obs).sum()) / (1 + n_perm)
    return obs, p, perm_stats


def _ref_centroid_direction(Z, is_hc):
    """Kernel-embedding distance to the held-out-HC mean embedding, in the SAME
    RKHS (same RBF kernel, same per-batch median-heuristic gamma) as the MMD
    test above -- not a raw Euclidean distance in gene space, which collapses
    19,804 genes with equal, uncorrelated weight and is only a Euclidean cousin
    of the mean|Z|/chi2 statistics already rejected for exactly that reason.
    The reference is built ONLY from this batch's own LOBO-scored held-out HC
    (a model that never trained on this batch), never from CV (CV's
    StratifiedKFold splits within every batch rather than excluding one, so it
    would leak this batch's own samples back into the reference). Each HC
    sample compares to the mean EMBEDDING of the OTHER held-out HC (leave-one-out,
    or it would be pulled toward its own position and its distance spuriously
    shrunk); disease samples never contribute to the reference, so they compare
    against the full HC mean embedding. Squared distance to a kernel mean
    embedding has a closed form in terms of the Gram matrix alone --
    ||phi(x) - mean_i phi(h_i)||^2 = k(x,x) - (2/n)sum_i k(x,h_i) + (1/n^2)sum_ij k(h_i,h_j)."""
    hc_Z, dis_Z = Z[is_hc], Z[~is_hc]
    n_hc = len(hc_Z)
    gamma, _ = _median_heuristic_gamma(np.vstack([hc_Z, dis_Z]))
    Khh = np.exp(-gamma * squareform(pdist(hc_Z, metric="sqeuclidean")))
    np.fill_diagonal(Khh, 1.0)
    Kdh = np.exp(-gamma * cdist(dis_Z, hc_Z, metric="sqeuclidean"))

    hc_row_sum = Khh.sum(axis=1) - 1.0  # sum_{k!=i} k(h_i,h_k), k(x,x)=1 excluded
    hc_total_full = Khh.sum()           # sum_{k,l} k(h_k,h_l) over ALL HC, diagonal included
    n = n_hc - 1
    # sum_{k!=i,l!=i} k(h_k,h_l): full sum minus row i, col i (each hc_row_sum+diag), plus the
    # doubly-removed (i,i)=1 term back once -- see derivation in the review that caught this.
    rest_total = hc_total_full - 2 * hc_row_sum - 1.0
    d2_hc = np.clip(1.0 - (2.0 / n) * hc_row_sum + rest_total / n**2, 0, None)
    d2_dis = np.clip(1.0 - (2.0 / n_hc) * Kdh.sum(axis=1) + hc_total_full / n_hc**2, 0, None)

    d_hc, d_dis = np.sqrt(d2_hc), np.sqrt(d2_dis)
    _, p_direction = mannwhitneyu(d_dis, d_hc, alternative="greater")
    return d_hc, d_dis, float(p_direction)


def mmd_summary(min_n_hc=MIN_N_HC, n_perm=1000, max_n_per_group=150, seed=42, shash=False, ood_filter=False,
                return_raw=False):
    """For each Tier-A batch with a stably-estimable held-out-HC noise floor,
    tests whether held-out-HC and held-out-disease Z-vectors come from
    different distributions (MMD^2, RBF kernel over all genes jointly,
    permutation p-value), then reports direction (disease farther than
    held-out-HC from a reference centroid, or not). The reference centroid is
    built ONLY from this batch's own held-out HC (see _ref_centroid_direction)
    -- an earlier version used a centroid pooled from the standard 5-fold CV,
    which leaked this batch's own samples back in (CV splits within every
    batch rather than excluding one) and diluted the noise-floor estimate.
    shash=True loads Z_test_shash.npy -- SHASH already fit (lobo_engine.
    run_one_batch) on THIS batch's own train-fold (HC minus this batch)
    in-sample Z and applied to this batch's held-out Z, never fit on the
    held-out Z itself. ood_filter=True drops held-out samples flagged as
    covariate-extrapolating (see load_batch); requires
    lobo_engine.compute_ood() to have been run."""
    rows, raw = [], {}
    for b in tier_a_batches(min_n_hc=min_n_hc):
        safe = b.replace(" ", "_")
        bdir = LOBO_DIR / safe
        meta, Z, is_hc = load_batch(bdir, ood_filter=ood_filter, shash=shash)
        Z = np.nan_to_num(Z, nan=0.0, posinf=10.0, neginf=-10.0)
        hc_Zb, dis_Zb = Z[is_hc], Z[~is_hc]
        if len(hc_Zb) < 5 or len(dis_Zb) < 5:
            continue

        rng = np.random.default_rng(seed)
        hc_s = hc_Zb if len(hc_Zb) <= max_n_per_group else hc_Zb[rng.choice(len(hc_Zb), max_n_per_group, replace=False)]
        dis_s = dis_Zb if len(dis_Zb) <= max_n_per_group else dis_Zb[rng.choice(len(dis_Zb), max_n_per_group, replace=False)]
        mmd2, p, mmd2_null = _mmd_permutation_test(hc_s, dis_s, n_perm=n_perm, seed=seed)

        d_hc, d_dis, p_direction = _ref_centroid_direction(Z, is_hc)
        hc_dist, dis_dist = float(d_hc.mean()), float(d_dis.mean())
        rows.append(dict(batch=b, n_hc=len(hc_Zb), n_dis=len(dis_Zb), mmd2=mmd2, perm_p=p,
                         hc_ref_dist=hc_dist, dis_ref_dist=dis_dist, disease_farther=dis_dist > hc_dist,
                         p_direction=p_direction))
        if return_raw:
            raw[b] = dict(d_hc=d_hc, d_dis=d_dis, mmd2_obs=mmd2, mmd2_null=mmd2_null)

    df = pd.DataFrame(rows).sort_values("n_dis", ascending=False)
    return (df, raw) if return_raw else df


def mmd_summary_cached(force=False, csv_path=None, shash=False, ood_filter=False, **kwargs):
    csv_path = csv_path or _cache_path(shash, ood_filter)
    if not force and os.path.isfile(csv_path):
        return pd.read_csv(csv_path)
    df = mmd_summary(shash=shash, ood_filter=ood_filter, **kwargs)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def mmd_raw_cached(force=False, pkl_path=None, shash=False, ood_filter=False, **kwargs):
    """Per-batch raw distributions behind mmd_summary's means: d_hc/d_dis (kernel-embedding
    distance per sample) and mmd2_null (the permutation draws behind perm_p) -- for plotting
    the actual distributions instead of just the summary points."""
    pkl_path = pkl_path or _raw_cache_path(shash, ood_filter)
    if not force and os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    _, raw = mmd_summary(shash=shash, ood_filter=ood_filter, return_raw=True, **kwargs)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(raw, f)
    return raw
