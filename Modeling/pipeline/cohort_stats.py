import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def storey_qvalue(pvals, lam=None):
    """Storey & Tibshirani (2003) q-value: estimates pi0 (true-null fraction) from the
    p-value histogram instead of assuming the worst case pi0=1 like BH, so it's less
    conservative when most tests are truly non-null. Prefer BH when comparing against a
    BH-based reference (e.g. DESeq2 padj); storey is an option for exploratory/single-method
    use where power matters more than cross-method comparability."""
    pvals = np.asarray(pvals)
    m = len(pvals)
    if lam is None:
        lam = np.arange(0.05, 0.96, 0.05)
    pi0s = np.array([np.mean(pvals >= l) / (1 - l) for l in lam])
    coeffs = np.polyfit(lam, pi0s, 3)
    pi0 = np.clip(np.polyval(coeffs, lam.max()), 0, 1)
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    ranks = np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate((pi0 * m * p_sorted / ranks)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q = np.empty(m)
    q[order] = q_sorted
    return q


def adjust_pvalues(pvals, method='fdr_bh'):
    """method='storey' for Storey's q-value; anything else is passed to
    statsmodels.multipletests (e.g. 'fdr_bh', 'fdr_by', 'bonferroni')."""
    if method == 'storey':
        return storey_qvalue(pvals)
    return multipletests(pvals, method=method)[1]


def _raw_stats(Z, gene_names, ref=None):
    if ref is None:
        stat, pval = stats.ttest_1samp(Z, 0.0, axis=0)
        mean_diff = Z.mean(axis=0)
    else:
        stat, pval = stats.ttest_ind(Z, ref, axis=0, equal_var=False)
        mean_diff = Z.mean(axis=0) - ref.mean(axis=0)
    return pd.DataFrame({'gene': gene_names, 'mean_diff': mean_diff, 'stat': stat, 'pval': pval})


def _fdr(df, fdr_method):
    df = df.copy()
    df['padj'] = adjust_pvalues(df['pval'].values, method=fdr_method)
    return df


def _test(Z, gene_names, ref=None, route=None, fdr_method='fdr_bh'):
    """Shared implementation for test_vs_hc / test_cohort_vs_cohort.

    route : optional array aligned to gene_names (e.g. scoring.gene_stage() mapped to
    'pool'/'model'). When given, BH is applied SEPARATELY within each route stratum, since
    pool-route genes share a single fitted beta across genes (not independent per-gene
    hypotheses) and are dominated by RQR jitter at near-zero counts -- pooling them into one
    correction with model-route genes both violates BH's independence/PRDS assumption and
    wastes power (see Bourgon et al. 2010 PNAS on independent filtering).
    """
    df = _raw_stats(Z, gene_names, ref=ref)
    if route is None:
        df = _fdr(df, fdr_method)
    else:
        df['route'] = np.asarray(route)
        strata = np.where(df['route'].values == 'pool', 'pool', 'model')
        df = pd.concat([_fdr(g, fdr_method) for _, g in df.groupby(strata, sort=False)])
    return df.sort_values('pval').reset_index(drop=True)


def test_vs_hc(Z, gene_names, route=None, fdr_method='fdr_bh'):
    """One-sample t-test per gene (H0: mean_Z == 0), BH-adjusted padj.

    Z : (n_samples, n_genes) cohort Z-score matrix.
    Returns DataFrame [gene, (route), mean_diff, stat, pval, padj], sorted by pval.
    """
    return _test(Z, gene_names, route=route, fdr_method=fdr_method)


def test_cohort_vs_cohort(Z_a, Z_b, gene_names, route=None, fdr_method='fdr_bh'):
    """Two-sample Welch's t-test per gene between two disease cohorts, BH-adjusted.

    Z_a, Z_b : (n_samples, n_genes) Z-score matrices for cohort A and cohort B.
    Returns DataFrame [gene, (route), mean_diff, stat, pval, padj], sorted by pval.
    """
    return _test(Z_a, gene_names, ref=Z_b, route=route, fdr_method=fdr_method)


def sig_gene_set(Z, gene_names, ref=None, route=None, padj_thr=0.05, fdr_method='fdr_bh',
                  include_pool=False):
    """Set of gene_names with padj < padj_thr from test_vs_hc or test_cohort_vs_cohort.

    ref : optional second Z matrix -> cohort_vs_cohort; otherwise vs_hc.
    route/include_pool : when route is given, 'pool'-route genes are excluded from the
    returned set by default (include_pool=True to opt in) -- see _test docstring.
    """
    df = _test(Z, gene_names, ref=ref, route=route, fdr_method=fdr_method)
    if route is not None and not include_pool:
        df = df[df['route'] != 'pool']
    return set(df.loc[df['padj'] < padj_thr, 'gene'])
