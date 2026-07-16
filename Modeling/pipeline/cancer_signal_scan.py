import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import fisher_exact

import config
from pipeline.cohort_stats import adjust_pvalues, gene_symbol_map, test_vs_hc

MP = config.MODELING_PARAMS

# Cancer phenotypes with a valid Open Targets disease mapping (see build_disease_reference.py
# PHENO_QUERY). ICI-treated Cancer has no single OT disease ID (heterogeneous cohort) and is
# excluded entirely, per project decision.
CANCER_PHENOTYPES = [
    'Colorectal Cancer (Chen)',
    'Esophagus Cancer (Chen)',
    'Liver Cancer (Chen)',
    'Liver Cancer (Roskams-Hieter)',
    'Lung Cancer (Chen)',
    'MGUS (Roskams-Hieter)',
    'MM (Roskams-Hieter)',
    'Pancreatic Cancer (Moore)',
    'Stomach Cancer (Chen)',
    'Other Cancer (Moore)',
]


def vs_hc_path(pheno):
    stem = pheno.replace('/', '_')
    return config.CANCER_SCAN_DIR / f'vs_hc_{stem}.csv'


def vs_hc_table(dd, pheno, route=None, fdr_method=MP['fdr_method'], save=True):
    """One-sample vs-HC Welch/t-test DEG table for a single cancer cohort (route-stratified
    BH -- pool-route genes excluded from the FDR family, see cohort_stats.py). Cache-first."""
    path = vs_hc_path(pheno)
    if save and path.exists():
        return pd.read_csv(path)
    Z = dd.Z_dis[dd.dis_pheno == pheno]
    df = test_vs_hc(Z, dd.gene_names, route=route, fdr_method=fdr_method)
    df['gene_sym'] = gene_symbol_map(dd.gene_names, dd.gene_syms).loc[df['gene']].values
    df.insert(0, 'phenotype', pheno)
    if save:
        config.CANCER_SCAN_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return df


def load_ot_reference(pheno):
    """{gene_symbol: score} from the Open Targets disease_reference JSON, or None if this
    phenotype has no OT mapping (excluded from analysis)."""
    stem = pheno.replace('/', '_')
    path = config.BENCHMARK_DIR / 'disease_reference' / f'{stem}.json'
    if not path.exists():
        return None
    rec = json.load(open(path))
    if not rec.get('efo') or not rec.get('genes'):
        return None
    return dict(rec['genes'])


def enrichment_test(sig_syms, ot_syms, universe_syms):
    """Fisher's exact test (one-sided, 'greater') for overlap between a significant-gene set
    and the OT reference set, within the tested gene universe. Returns (odds_ratio, pval,
    n_overlap, n_sig, n_ot_in_universe, n_universe)."""
    sig_syms, ot_syms, universe_syms = set(sig_syms), set(ot_syms), set(universe_syms)
    ot_u = ot_syms & universe_syms
    a = len(sig_syms & ot_u)
    b = len(sig_syms) - a
    c = len(ot_u) - a
    d = len(universe_syms) - a - b - c
    odds, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
    return odds, pval, a, len(sig_syms), len(ot_u), len(universe_syms)


def per_sample_path():
    return config.CANCER_SCAN_DIR / 'per_sample_summary.csv'


def per_sample_scan(dd, route=None, fdr_method=MP['fdr_method'], padj_thr=MP['padj_thr'], save=True):
    """Per-INDIVIDUAL-SAMPLE significant-gene identification against Open Targets.

    For each single sample (not cohort-aggregate), BH-adjusts that sample's own per-gene
    p-values (p = 2*(1-Phi(|z|)), same construction as plots.py's plot_sample cutoff) across
    model-route genes only, takes the padj<padj_thr flagged gene set, and tests it against
    the phenotype's OT reference gene set with Fisher's exact test. This answers "does this
    one patient's own flagged-gene profile look like the known disease gene set" -- a
    genuinely per-sample question, unlike run_all()'s cohort-level one-sample t-test.
    Returns one row per sample, sorted by enrichment p-value (best individual cases first).
    """
    path = per_sample_path()
    if save and path.exists():
        return pd.read_csv(path)
    gene_syms = gene_symbol_map(dd.gene_names, dd.gene_syms).values
    model_mask = (np.asarray(route) != 'pool') if route is not None else np.ones(len(dd.gene_names), bool)
    universe = set(gene_syms[model_mask])
    rows = []
    for pheno in CANCER_PHENOTYPES:
        ot = load_ot_reference(pheno)
        if ot is None:
            continue
        ph_mask = dd.dis_pheno == pheno
        names = np.array(dd.dis_names)[ph_mask]
        Z = dd.Z_dis[ph_mask][:, model_mask]
        syms = gene_syms[model_mask]
        pval = 2 * stats.norm.sf(np.abs(Z))
        for i in range(Z.shape[0]):
            padj = adjust_pvalues(pval[i], method=fdr_method)
            sig = set(syms[padj < padj_thr])
            odds, p, n_ov, n_sig, n_ot_u, n_u = enrichment_test(sig, ot.keys(), universe)
            rows.append({'phenotype': pheno, 'sample': names[i], 'n_sig_genes': n_sig,
                         'n_overlap': n_ov, 'odds_ratio': odds, 'enrichment_pval': p})
    summary = pd.DataFrame(rows).sort_values('enrichment_pval').reset_index(drop=True)
    if save:
        config.CANCER_SCAN_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(path, index=False)
    return summary


def run_all(dd, route=None, fdr_method=MP['fdr_method'], padj_thr=MP['padj_thr'], save=True):
    """Per-cancer-phenotype vs-HC DEG + OT-reference enrichment scan. Returns a summary
    DataFrame sorted by enrichment p-value (most validated signal first). Route-stratified:
    only model-route genes (route != 'pool') are counted toward padj significance and the
    enrichment universe -- see cohort_stats.py."""
    rows = []
    for pheno in CANCER_PHENOTYPES:
        ot = load_ot_reference(pheno)
        if ot is None:
            continue
        df = vs_hc_table(dd, pheno, route=route, fdr_method=fdr_method, save=save)
        d = df[df['route'] != 'pool'] if 'route' in df.columns else df
        sig = set(d.loc[d['padj'] < padj_thr, 'gene_sym'])
        universe = set(d['gene_sym'])
        odds, pval, n_ov, n_sig, n_ot_u, n_u = enrichment_test(sig, ot.keys(), universe)
        n = (dd.dis_pheno == pheno).sum()
        rows.append({'phenotype': pheno, 'n_samples': n, 'n_sig_genes': n_sig,
                     'n_ot_genes_in_universe': n_ot_u, 'n_overlap': n_ov,
                     'odds_ratio': odds, 'enrichment_pval': pval})
    summary = pd.DataFrame(rows).sort_values('enrichment_pval').reset_index(drop=True)
    if save:
        config.CANCER_SCAN_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(config.CANCER_SCAN_DIR / 'summary.csv', index=False)
    return summary


def attach_ood_distance(summary, save=True):
    """Adds mahal_dist + ood_threshold columns to a per_sample_scan summary (Mahalanobis
    distance of each sample's covariates from the HC-fit distribution, config.MODELING_PARAMS
    ood_percentile threshold) -- lets downstream plots/filters distinguish a genuine
    per-sample disease signal from a broad covariate outlier riding near the OOD cutoff."""
    from pipeline import data_prep
    from sample_filter import MahalanobisFilter
    adata = data_prep.load_adata()
    is_hc, _, _ = data_prep.make_phenotypes(adata)
    X_raw = data_prep.bias_matrix(adata)
    sample_ids = np.array(adata.obs_names)
    ood = MahalanobisFilter(percentile=config.MODELING_PARAMS['ood_percentile'])
    ood.fit(X_raw[is_hc])
    dist_map = dict(zip(sample_ids, ood.distances(X_raw)))
    summary = summary.copy()
    summary['mahal_dist'] = summary['sample'].map(dist_map)
    summary['ood_threshold'] = ood.threshold_
    if save:
        summary.to_csv(per_sample_path(), index=False)
    return summary
