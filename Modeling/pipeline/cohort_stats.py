import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import config

MP = config.MODELING_PARAMS


def gene_symbol_map(gene_names, gene_syms):
    """Series of gene_syms aligned to gene_names, NaN falls back to the gene id itself."""
    sym = pd.Series(gene_syms, index=gene_names)
    return sym.fillna(pd.Series(gene_names, index=gene_names))


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


def adjust_pvalues(pvals, method=MP['fdr_method']):
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


def _test(Z, gene_names, ref=None, route=None, fdr_method=MP['fdr_method']):
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


def test_vs_hc(Z, gene_names, route=None, fdr_method=MP['fdr_method']):
    """One-sample t-test per gene (H0: mean_Z == 0), BH-adjusted padj.

    Z : (n_samples, n_genes) cohort Z-score matrix.
    Returns DataFrame [gene, (route), mean_diff, stat, pval, padj], sorted by pval.
    """
    return _test(Z, gene_names, route=route, fdr_method=fdr_method)


def test_cohort_vs_cohort(Z_a, Z_b, gene_names, route=None, fdr_method=MP['fdr_method']):
    """Two-sample Welch's t-test per gene between two disease cohorts, BH-adjusted.

    Z_a, Z_b : (n_samples, n_genes) Z-score matrices for cohort A and cohort B.
    Returns DataFrame [gene, (route), mean_diff, stat, pval, padj], sorted by pval.
    """
    return _test(Z_a, gene_names, ref=Z_b, route=route, fdr_method=fdr_method)
