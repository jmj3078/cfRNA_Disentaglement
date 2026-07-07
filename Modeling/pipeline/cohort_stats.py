import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def test_vs_hc(Z, gene_names):
    """One-sample t-test per gene (H0: mean_Z == 0) with BH-adjusted padj.

    Z : (n_samples, n_genes) cohort Z-score matrix.
    Returns DataFrame [gene, stat, pval, padj], sorted by pval.
    """
    stat, pval = stats.ttest_1samp(Z, 0.0, axis=0)
    padj = multipletests(pval, method='fdr_bh')[1]
    return pd.DataFrame({'gene': gene_names, 'stat': stat, 'pval': pval, 'padj': padj}) \
        .sort_values('pval').reset_index(drop=True)


def test_cohort_vs_cohort(Z_a, Z_b, gene_names):
    """Two-sample Welch's t-test per gene between two disease cohorts, BH-adjusted.

    Z_a, Z_b : (n_samples, n_genes) Z-score matrices for cohort A and cohort B.
    Returns DataFrame [gene, stat, pval, padj], sorted by pval.
    """
    stat, pval = stats.ttest_ind(Z_a, Z_b, axis=0, equal_var=False)
    padj = multipletests(pval, method='fdr_bh')[1]
    return pd.DataFrame({'gene': gene_names, 'stat': stat, 'pval': pval, 'padj': padj}) \
        .sort_values('pval').reset_index(drop=True)


def sig_gene_set(Z, gene_names, padj_thr=0.05, ref=None):
    """Set of gene_names with padj < padj_thr from test_vs_hc or test_cohort_vs_cohort.

    ref : optional second Z matrix -> cohort_vs_cohort; otherwise vs_hc.
    """
    df = test_vs_hc(Z, gene_names) if ref is None else test_cohort_vs_cohort(Z, ref, gene_names)
    return set(df.loc[df['padj'] < padj_thr, 'gene'])
