import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
from viz_style import apply_style

apply_style()

LOBO_DIR = config.MODELING_DIR / "LOBO_Results"
MMD_SUMMARY_CSV = LOBO_DIR / "mmd_summary.csv"

# Batches below this held-out-HC count give an unstable noise floor (SEM of
# the HC deviation estimate does not stabilize below ~n_hc=25 in this
# dataset -- see project_lobo_validation_design.md) and are excluded from the
# reported comparison rather than reported with a misleadingly precise p.
MIN_N_HC = 25


def load_batch(batch_dir, ood_filter=True):
    """ood_filter=True drops held-out samples whose covariates fall outside the
    train-fold HC distribution (see compute_lobo_ood.py) -- otherwise a sample
    scored via covariate extrapolation can show a large |Z| that reflects the
    model being asked to extrapolate, not disease biology."""
    meta = json.load(open(batch_dir / "meta.json"))
    Z = np.load(batch_dir / "Z_test.npy")
    is_hc = np.array(meta["test_is_hc"])
    if ood_filter:
        ood_path = batch_dir / "ood_mask.npy"
        if not ood_path.exists():
            raise FileNotFoundError(f"{ood_path} missing -- run compute_lobo_ood.py first")
        keep = np.load(ood_path)
        Z, is_hc = Z[keep], is_hc[keep]
    return meta, Z, is_hc


def tier_a_batches(min_n_hc=MIN_N_HC):
    """Tier-A (HC+disease co-located) batches with a held-out-HC noise floor
    large enough to be estimated stably. See project_lobo_validation_design.md
    for why batches below this were dropped rather than reported: their
    reversed disease-vs-HC direction traced to noise-floor instability (small
    n_hc), not to a genuine absence of disease signal -- e.g. Chang Batch_3/4
    (n_hc=9-14) -- whereas some large-n_hc batches (Moore Batch_1, n_hc=67)
    that still show no separation were confirmed to reflect a real weak
    signal even after splitting by phenotype, not a noise-floor artifact."""
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
        n_hc = sum(meta["test_is_hc"])
        if n_hc >= min_n_hc:
            kept.append(b)
    return kept


def load_hc_reference_z(cv_zscores_path=None):
    """In-fold HC Z reference: per-gene held-out-fold Z vectors from the
    standard 5-fold engine CV (cv_model_engine.py), NOT the LOBO run -- this is
    the 'model trained and evaluated under normal conditions' distribution
    that LOBO held-out-HC and disease are both compared against. Returns
    (gene_names, Z of shape (n_hc, n_genes))."""
    path = cv_zscores_path or (config.CV_RESULTS_DIR / "cv_zscores.pkl")
    d = pickle.load(open(path, "rb"))
    genes = list(d.keys())
    Z = np.column_stack([d[g] for g in genes])
    return genes, np.nan_to_num(Z, nan=0.0, posinf=10.0, neginf=-10.0)


def _mmd2_from_kernel(K, n):
    """Unbiased MMD^2 given a precomputed full (n+m, n+m) RBF kernel matrix
    with X in rows/cols [0:n] and Y in [n:]."""
    Kxx, Kyy, Kxy = K[:n, :n], K[n:, n:], K[:n, n:]
    m = K.shape[0] - n
    sx = Kxx.sum() - np.trace(Kxx)
    sy = Kyy.sum() - np.trace(Kyy)
    return sx / (n * (n - 1)) + sy / (m * (m - 1)) - 2 * Kxy.mean()


def _mmd_permutation_test(X, Y, n_perm=1000, seed=42):
    """Permutation p-value for MMD^2(X, Y) > 0 (X, Y drawn from different
    distributions), using ALL genes as the RBF feature space directly (no
    covariance inversion, so it stays well-defined even with p >> n). The
    (n+m, n+m) pairwise-distance / kernel matrix is built ONCE from the pooled
    sample; each permutation only re-slices that fixed matrix by shuffled
    label order, instead of recomputing an O(n*m*n_genes) kernel per
    permutation (the naive version cost ~12min/batch; this costs ~1.5s)."""
    from scipy.spatial.distance import squareform, pdist
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
    """Mean Euclidean distance of held-out-HC vs disease Z-vectors from the
    in-fold HC reference centroid, for reading MMD's 'the two groups differ'
    result as a direction (disease farther from normative than the held-out
    HC noise floor, or not)."""
    gene_index = {g: i for i, g in enumerate(hc_genes)}
    keep_cols = [i for i, g in enumerate(gene_names) if g in gene_index]
    cols = [gene_index[gene_names[i]] for i in keep_cols]
    order = np.argsort(cols)
    Zsub = np.nan_to_num(Z[:, keep_cols][:, order], nan=0.0, posinf=10.0, neginf=-10.0)
    ref = hc_centroid[np.sort(cols)]
    dist = np.sqrt(((Zsub - ref) ** 2).sum(axis=1))
    return dist[is_hc].mean(), dist[~is_hc].mean()


def mmd_summary(min_n_hc=MIN_N_HC, ood_filter=True, n_perm=1000, max_n_per_group=150, seed=42):
    """For each Tier-A batch with a stably-estimable held-out-HC noise floor
    (n_hc >= min_n_hc), tests whether held-out-HC and held-out-disease
    Z-vectors come from different distributions (MMD^2 with an RBF kernel over
    all genes jointly, permutation p-value), then reports the direction (is
    disease farther than held-out-HC from the in-fold HC reference centroid).
    This is the only retained deviation metric -- mean|Z|, chi-square sum(z^2)
    and BH-FDR extreme-fraction were all tried and dropped because collapsing
    each sample's ~20k-gene Z-vector to a single averaged/summed scalar
    dilutes a disease signal that is concentrated in a minority of genes
    (see project_lobo_validation_design.md)."""
    hc_genes, hc_Z = load_hc_reference_z()
    hc_centroid = hc_Z.mean(axis=0)

    rows = []
    for b in tier_a_batches(min_n_hc=min_n_hc):
        safe = b.replace(" ", "_")
        bdir = LOBO_DIR / safe
        meta, Z, is_hc = load_batch(bdir, ood_filter=ood_filter)
        Z = np.nan_to_num(Z, nan=0.0, posinf=10.0, neginf=-10.0)
        gene_names = pickle.load(open(bdir / "gene_names.pkl", "rb"))
        hc_Zb, dis_Zb = Z[is_hc], Z[~is_hc]
        if len(hc_Zb) < 5 or len(dis_Zb) < 5:
            continue

        rng = np.random.default_rng(seed)
        hc_s = hc_Zb if len(hc_Zb) <= max_n_per_group else hc_Zb[
            rng.choice(len(hc_Zb), max_n_per_group, replace=False)]
        dis_s = dis_Zb if len(dis_Zb) <= max_n_per_group else dis_Zb[
            rng.choice(len(dis_Zb), max_n_per_group, replace=False)]
        mmd2, p = _mmd_permutation_test(hc_s, dis_s, n_perm=n_perm, seed=seed)

        hc_dist, dis_dist = _ref_centroid_direction(gene_names, Z, is_hc, hc_genes, hc_centroid)
        rows.append(dict(
            batch=b, n_hc=len(hc_Zb), n_dis=len(dis_Zb),
            mmd2=mmd2, perm_p=p,
            hc_ref_dist=hc_dist, dis_ref_dist=dis_dist,
            disease_farther=dis_dist > hc_dist,
        ))
    return pd.DataFrame(rows).sort_values("n_dis", ascending=False)


def mmd_summary_cached(force=False, csv_path=None, **kwargs):
    """Cache-first wrapper around mmd_summary(): loads csv_path if it exists,
    otherwise recomputes and saves. force=True always recomputes (e.g. after
    changing MIN_N_HC or re-running LOBO)."""
    csv_path = csv_path or MMD_SUMMARY_CSV
    if not force and os.path.isfile(csv_path):
        return pd.read_csv(csv_path)
    df = mmd_summary(**kwargs)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def _p_to_asterisk(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _short_batch_label(batch):
    """'Ward Z et al._Batch_1' -> 'Ward Z_1' -- drop the 'et al.' filler and
    the repeated 'Batch_' prefix so axis labels stay legible."""
    name, _, num = batch.replace(" et al.", "").partition("_Batch_")
    return f"{name}_{num}" if num else name


def plot_mmd_bar(df, fig_dir=None, save=True):
    """Bar chart of MMD^2 per batch. Asterisks mark permutation-test
    significance: * p<0.05, ** p<0.01, *** p<0.001."""
    fig_dir = fig_dir or LOBO_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = df.sort_values("mmd2", ascending=False).reset_index(drop=True)
    labels = [_short_batch_label(b) for b in d["batch"]]

    fig, ax = plt.subplots(figsize=(4, 6))
    bars = ax.bar(range(len(d)), d["mmd2"], color="#00C78B")
    y_span = d["mmd2"].max() - d["mmd2"].min()
    for bar, p in zip(bars, d["perm_p"]):
        label = _p_to_asterisk(p)
        if not label:
            continue
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01 * y_span, label,
               ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("MMD$^2$ (held-out HC vs disease)")
    ax.set_xlabel("LOBO batch")
    plt.tight_layout()
    if save:
        fig.savefig(fig_dir / "mmd_bar.png", bbox_inches="tight")
    return fig


def plot_mmd_direction(df, fig_dir=None, save=True):
    """Paired hc_ref_dist-vs-dis_ref_dist dot plot showing the direction
    (disease farther from the in-fold HC reference than the held-out-HC
    noise floor, or not) for every reported batch."""
    fig_dir = fig_dir or LOBO_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = df.sort_values("mmd2", ascending=False).reset_index(drop=True)
    labels = [_short_batch_label(b) for b in d["batch"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    y = np.arange(len(d))
    ax.hlines(y, d["hc_ref_dist"], d["dis_ref_dist"],
             color=np.where(d["disease_farther"], "#00C78B", "#999999"), linewidth=3)
    ax.scatter(d["hc_ref_dist"], y, color="#A4AFB8", label="held-out HC", zorder=3, s=90, edgecolors="black", linewidth=0.5)
    ax.scatter(d["dis_ref_dist"], y, color="#00C78B", label="disease", zorder=3, s=90, edgecolors="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # top = highest MMD^2, matching plot_mmd_bar's left-to-right order
    ax.set_xlabel("Distance from HC reference")
    ax.set_ylabel("LOBO batch")

    ax.legend(frameon=False)

    plt.tight_layout()
    if save:
        fig.savefig(fig_dir / "mmd_direction.png", bbox_inches="tight")
    return fig


def plot_mmd_summary(df, fig_dir=None, save=True):
    """Returns (fig_bar, fig_direction) -- see plot_mmd_bar / plot_mmd_direction."""
    return plot_mmd_bar(df, fig_dir=fig_dir, save=save), plot_mmd_direction(df, fig_dir=fig_dir, save=save)


def run_all(force=False, fig_dir=None):
    """Thin-runner entry point: load/compute the cached MMD summary table,
    save the summary figures, print a short pass/fail readout. Notebook usage:
    `from pipeline import lobo_validation as lv; lv.run_all()`."""
    df = mmd_summary_cached(force=force)
    print(df.to_string(index=False))
    fig_bar, fig_direction = plot_mmd_summary(df, fig_dir=fig_dir)
    n_sig_and_farther = ((df["perm_p"] < 0.05) & df["disease_farther"]).sum()
    print(f"\n{n_sig_and_farther}/{len(df)} batches: MMD significant (p<0.05) "
         f"AND disease farther from the in-fold HC reference than the "
         f"held-out-HC noise floor.")
    return df, fig_bar, fig_direction
