import numpy as np
import pandas as pd
import gseapy as gp

import config
from pipeline.cohort_stats import adjust_pvalues, test_cohort_vs_cohort, test_vs_hc

MP = config.MODELING_PARAMS

COMPARISONS = [
    ('CAD_HF+_vs_CAD_HF-', 'CAD_HF+ (Ward)', 'CAD_HF- (Ward)'),
    ('Pancreatitis_vs_PDAC', 'Pancreatitis (Moore)', 'Pancreatic Cancer (Moore)'),
    ('ICI_Cancer_vs_ICIm', 'ICI-treated Cancer (Raissadati)', 'ICI-m (Raissadati)'),
]


def compare_path(name):
    return config.COHORT_COMPARE_DIR / f'deg_{name}.csv'


def run_comparison(dd, name, pheno_a, pheno_b, route=None, fdr_method='fdr_bh',
                   min_hc_dev=None, save=True):
    path = compare_path(name)
    if save and path.exists():
        return pd.read_csv(path)
    Za = dd.Z_dis[dd.dis_pheno == pheno_a]
    Zb = dd.Z_dis[dd.dis_pheno == pheno_b]
    df = test_cohort_vs_cohort(Za, Zb, dd.gene_names, route=route, fdr_method=fdr_method)
    if min_hc_dev is not None:
        df_a = test_vs_hc(Za, dd.gene_names)
        df_b = test_vs_hc(Zb, dd.gene_names)
        dev = pd.concat([df_a.set_index('gene')['mean_diff'].abs(),
                         df_b.set_index('gene')['mean_diff'].abs()], axis=1).max(axis=1)
        df['hc_dev_max'] = df['gene'].map(dev)
        model_mask = (df['route'] != 'pool').values if 'route' in df.columns else np.ones(len(df), bool)
        keep = model_mask & (df['hc_dev_max'].values > min_hc_dev)
        df.loc[model_mask, 'padj'] = np.nan
        df.loc[keep, 'padj'] = adjust_pvalues(df.loc[keep, 'pval'].values, method=fdr_method)
    sym = dict(zip(dd.gene_names, dd.gene_syms))
    df['gene_sym'] = df['gene'].map(sym).fillna(df['gene'])
    df.insert(0, 'pheno_b', pheno_b)
    df.insert(0, 'pheno_a', pheno_a)
    df.insert(0, 'comparison', name)
    if save:
        config.COHORT_COMPARE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def run_all(dd, route=None, fdr_method='fdr_bh', min_hc_dev=None, save=True):
    return {name: run_comparison(dd, name, a, b, route=route, fdr_method=fdr_method,
                                 min_hc_dev=min_hc_dev, save=save)
            for name, a, b in COMPARISONS}


def gsea_run_dir(name):
    return config.COHORT_COMPARE_GSEA_DIR / name


def run_gsea(df, name, min_size=10, max_size=500, fdr_thr=0.25, save=True):
    outdir = gsea_run_dir(name)
    out_path = outdir / f'gsea_result_{name}.csv'
    if save and out_path.exists():
        return pd.read_csv(out_path)
    sub = df[df['route'] != 'pool'] if 'route' in df.columns else df
    rng = np.random.default_rng(MP['gsea_seed'])
    jitter = rng.normal(0, 1e-7, len(sub))
    rnk_df = (pd.DataFrame({'gene': sub['gene_sym'].values, 'score': sub['stat'].values + jitter})
             .sort_values('score', ascending=False).reset_index(drop=True))
    res = gp.prerank(rnk=rnk_df, gene_sets=MP['gsea_gene_sets'], outdir=None,
                     min_size=min_size, max_size=max_size,
                     permutation_num=MP['gsea_perm'], seed=MP['gsea_seed'], verbose=False)
    out = res.res2d[res.res2d['FDR q-val'] < fdr_thr].copy()
    if save:
        outdir.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
    return out
