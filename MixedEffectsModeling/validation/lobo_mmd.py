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
from MixedEffectsModeling.core.calibration import gene_shash_calibration
from MixedEffectsModeling.core.shash import shash_correct_matrix

LOBO_DIR = config.LOBO_MIXED_DIR
SHASH_LOO_DIR = LOBO_DIR / "shash_loo"
SHASH_MAX_N = 3000
SHASH_MIN_POOL = 200

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


def _load_all_batches_hc():
    """Load held-out HC Z (+ gene panel) for every Tier-A batch that has a LOBO
    run on disk, keyed by batch name -- used to pool "every OTHER batch's HC"
    for the leave-one-batch-out SHASH fit below. Never OOD-filtered: the SHASH
    reference should be the same held-out-HC null the MMD test itself uses."""
    out = {}
    for b in tier_a_batches(min_n_hc=None):
        safe = b.replace(" ", "_")
        bdir = LOBO_DIR / safe
        if not (bdir / "meta.json").exists():
            continue
        _, Z, is_hc = load_batch(bdir, ood_filter=False)
        Z = np.nan_to_num(Z, nan=0.0, posinf=10.0, neginf=-10.0)
        gene_names = pickle.load(open(bdir / "gene_names.pkl", "rb"))
        out[b] = (Z[is_hc], gene_names)
    return out


def _pool_other_batches_hc(loaded, exclude_batch, gene_names):
    """Stack held-out-HC Z rows from every batch except exclude_batch, aligned
    to gene_names by gene symbol (batches share the same panel in practice,
    but this is robust to reordering)."""
    parts = []
    for b, (hc_Z, gnames) in loaded.items():
        if b == exclude_batch or len(hc_Z) == 0:
            continue
        if list(gnames) == list(gene_names):
            parts.append(hc_Z)
        else:
            col = {g: j for j, g in enumerate(gnames)}
            idx = [col.get(g, -1) for g in gene_names]
            sub = np.full((len(hc_Z), len(gene_names)), np.nan)
            present = [k for k, c in enumerate(idx) if c >= 0]
            sub[:, present] = hc_Z[:, [idx[k] for k in present]]
            parts.append(sub)
    return np.vstack(parts) if parts else np.empty((0, len(gene_names)))


def loo_shash_params(batch, loaded, gene_names, cache_dir=SHASH_LOO_DIR, force=False, seed=42):
    """Per-gene SHASH params for `batch`'s LOBO Z, fit ONLY on held-out HC from
    every OTHER Tier-A batch -- the LOBO analogue of core.shash.load_shash_params,
    which pools CV Z across ALL batches (including this one) and so leaks this
    batch's own samples into its own correction. Genes with fewer than
    SHASH_MIN_POOL pooled HC observations are left uncorrected (shash_ok=False)."""
    safe = batch.replace(" ", "_")
    cache_path = cache_dir / f"{safe}.csv"
    if not force and cache_path.exists():
        return pd.read_csv(cache_path).set_index("gene")

    pooled = _pool_other_batches_hc(loaded, batch, gene_names)
    rng = np.random.default_rng(seed)
    rows = []
    for j, g in enumerate(gene_names):
        z = pooled[:, j]
        z = z[np.isfinite(z)]
        if len(z) < SHASH_MIN_POOL:
            rows.append(dict(gene=g, cv_shash_ok=False, cv_shash_xi=0.0, cv_shash_eta=1.0,
                             cv_shash_eps=0.0, cv_shash_delta=1.0))
            continue
        if len(z) > SHASH_MAX_N:
            z = rng.choice(z, SHASH_MAX_N, replace=False)
        calib = gene_shash_calibration(z)
        rows.append(dict(gene=g, cv_shash_ok=calib["shash_ok"], cv_shash_xi=calib["shash_xi"],
                         cv_shash_eta=calib["shash_eta"], cv_shash_eps=calib["shash_eps"],
                         cv_shash_delta=calib["shash_delta"]))
    df = pd.DataFrame(rows).set_index("gene")
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    return df


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
    return obs, p


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
    return d_hc.mean(), d_dis.mean(), float(p_direction)


def mmd_summary(min_n_hc=MIN_N_HC, n_perm=1000, max_n_per_group=150, seed=42, shash=False, ood_filter=False,
                force_shash=False):
    """For each Tier-A batch with a stably-estimable held-out-HC noise floor,
    tests whether held-out-HC and held-out-disease Z-vectors come from
    different distributions (MMD^2, RBF kernel over all genes jointly,
    permutation p-value), then reports direction (disease farther than
    held-out-HC from a reference centroid, or not). The reference centroid is
    built ONLY from this batch's own held-out HC (see _ref_centroid_direction)
    -- an earlier version used a centroid pooled from the standard 5-fold CV,
    which leaked this batch's own samples back in (CV splits within every
    batch rather than excluding one) and diluted the noise-floor estimate.
    shash=True applies a per-gene SHASH correction (leave-one-batch-out: fit
    on held-out HC from every OTHER Tier-A batch, see loo_shash_params) to the
    LOBO Z before comparing -- NOT the CV-pooled params, which would let this
    batch's own within-batch CV folds leak into its own correction.
    ood_filter=True drops held-out samples flagged as covariate-extrapolating
    (see load_batch); requires lobo_engine.compute_ood() to have been run."""
    loaded = _load_all_batches_hc() if shash else None

    rows = []
    for b in tier_a_batches(min_n_hc=min_n_hc):
        safe = b.replace(" ", "_")
        bdir = LOBO_DIR / safe
        meta, Z, is_hc = load_batch(bdir, ood_filter=ood_filter)
        Z = np.nan_to_num(Z, nan=0.0, posinf=10.0, neginf=-10.0)
        gene_names = pickle.load(open(bdir / "gene_names.pkl", "rb"))
        if shash:
            shash_params = loo_shash_params(b, loaded, gene_names, force=force_shash)
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
