"""Symmetric group-vs-normative DB-hit comparison (gene-level + pathway-level).

Every method is scored by the same rule -- own significant genes/pathways intersected
with an Open Targets reference -- so counts (recall) and rates (precision) are directly
comparable. See MixedEffectsModeling/CLAUDE.md and PathwayConvergence for the underlying
normative sample-level results this reuses.
"""
import pickle
import re

import numpy as np
import pandas as pd
import scanpy as sc

import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.calibration import bh_fdr_reject
from MixedEffectsModeling.core.pathway_convergence import (
    collapse_to_symbols, load_pathway_library, load_symbol_vocab, slugify,
)

HERE = config.ROOT / "MixedEffectsModeling" / "Benchmark"
DESEQ_DIR = HERE / "DESeq2"
REF_DIR = HERE / "disease_reference"
PC_DIR = config.PATHWAY_CONV_DIR
NULL_DIR = HERE / "DESeq2_pathway_null"
ZDIR = config.ROOT / "MixedEffectsModeling" / "Z_scores_mixed"
DESIGN_LABELS = {"no_covariate": "deseq2_no_cov", "covariate": "deseq2_cov",
                 "ruvg_k1": "deseq2_ruvg_k1", "ruvg_k2": "deseq2_ruvg_k2", "ruvg_k3": "deseq2_ruvg_k3"}
# same reference definition as 3_disease_scoring.ipynb cell 10 (marker panel): top-N by OT score,
# floored -- keeps reference size comparable across phenotypes instead of scaling with literature
# volume (an unfloored/unranked reference balloons to ~80% of the genome for well-studied cancers)
MARKER_TOPN = 300
MARKER_SCORE_FLOOR = 0.05


def load_reference(topn=MARKER_TOPN, score_floor=MARKER_SCORE_FLOOR):
    """{phenotype: set(symbols)} from Open Targets, top-N by score above a floor."""
    out = {}
    for f in REF_DIR.glob('*.json'):
        import json
        r = json.load(open(f))
        if r['phenotype'] == 'MGUS':  # OT association pool too thin (22 genes) for any panel size
            continue
        ranked = sorted(r['genes'], key=lambda gs: -gs[1])
        floored = [g for g, s in ranked if s >= score_floor]
        out[r['phenotype']] = set(floored[:topn]) | set(r.get('supplement', {}).get('genes', []))
    return out


def ensg_to_symbol():
    return sc.read_h5ad(config.H5AD_PATH, backed='r').var['GeneName']


def deseq2_tag(study, pheno):
    return f"{study.replace(' ', '_')}__{pheno.replace('/', '-').replace(' ', '_')}"


def deseq2_study_results(design):
    """{(study, phenotype): results_df} for one design, indexed by base ENSG (no version)."""
    summary = pd.read_csv(DESEQ_DIR / 'summary.csv')
    summary = summary[summary.design == design]
    out = {}
    for _, row in summary.iterrows():
        path = DESEQ_DIR / design / f"{deseq2_tag(row.study, row.phenotype)}.csv"
        if not path.exists():
            continue
        res = pd.read_csv(path, index_col=0)
        res.index = res.index.str.split('.').str[0]
        out[(row.study, row.phenotype)] = res
    return out


def deseq2_gene_sets(design, sym_of):
    """{phenotype: set(symbols)} pooling padj<0.05 genes across studies (union)."""
    sym_of = sym_of.copy()
    sym_of.index = sym_of.index.str.split('.').str[0]
    out = {}
    for (study, pheno), res in deseq2_study_results(design).items():
        sig = res.index[res['padj'] < 0.05]
        syms = set(sym_of.reindex(sig).dropna())
        out.setdefault(pheno, set()).update(syms)
    return out


def pc_dirs_by_phenotype(sample_meta):
    """{phenotype: [pdir, ...]} -- resolves PathwayConvergence subdirs (some phenotypes are
    split per-study) back to the canonical phenotype via the samples actually inside sig.pkl,
    not by parsing directory names."""
    ph_of = sample_meta.set_index('sample')['phenotype']
    out = {}
    for pdir in sorted(d for d in PC_DIR.iterdir() if d.is_dir() and (d / 'sig.pkl').exists()):
        d = pickle.load(open(pdir / 'sig.pkl', 'rb'))
        phenos = ph_of.reindex(d['names_c']).dropna().unique()
        if len(phenos) != 1:
            raise ValueError(f"{pdir} mixes phenotypes: {phenos}")
        out.setdefault(phenos[0], []).append(pdir)
    return out


def model_route_mask(gene_names):
    """Bool array over gene_names, True for individually-fitted ('model' route) genes --
    False excludes the ~2060 pooled-GLM ('rare') genes, for the no-rare-pooling comparison."""
    ts = pd.read_csv(config.ENGINE_MIXED_DIR / 'training_summary.csv').set_index('gene')['route']
    return (ts.reindex(gene_names) == 'model').values


def normative_gene_sets(sample_meta, gene_names, sym_of, exclude_pool=False):
    """{phenotype: (union_symbol_set, per_patient_rate_gene_symbols_list)}.
    per_patient list holds each patient's own significant-symbol set for rate distributions.
    exclude_pool=True restricts to individually-fitted genes (drops the pooled-GLM rare route)."""
    sym_arr = sym_of.reindex(gene_names).values
    col_mask = model_route_mask(gene_names) if exclude_pool else None
    out_union, out_per_patient = {}, {}
    for pheno, pdirs in pc_dirs_by_phenotype(sample_meta).items():
        union = set()
        per_patient = []
        for pdir in pdirs:
            d = pickle.load(open(pdir / 'sig.pkl', 'rb'))
            for row in d['gene_sig']:
                if col_mask is not None:
                    row = row & col_mask
                syms = {s for s in sym_arr[row] if pd.notna(s)}
                per_patient.append(syms)
                union |= syms
        out_union[pheno] = union
        out_per_patient[pheno] = per_patient
    return out_union, out_per_patient


def gene_venn_sets():
    """{phenotype: (deseq2_no_cov, deseq2_cov, normative_union)} symbol sets, phenotypes
    present in all three methods only."""
    sym_of = ensg_to_symbol()
    sm = pd.read_csv(ZDIR / 'sample_meta.csv')
    gene_names = pickle.load(open(ZDIR / 'gene_names.pkl', 'rb'))
    nocov = deseq2_gene_sets('no_covariate', sym_of)
    cov = deseq2_gene_sets('covariate', sym_of)
    norm_union, _ = normative_gene_sets(sm, gene_names, sym_of)
    return {ph: (nocov[ph], cov[ph], norm_union[ph])
            for ph in sorted(set(nocov) & set(cov) & set(norm_union))}


def plot_gene_venn(ncols=3, save=True):
    from matplotlib_venn import venn3
    import matplotlib.pyplot as plt
    sets = gene_venn_sets()
    n = len(sets)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (pheno, (a, b, c)) in zip(axes, sets.items()):
        venn3([a, b, c], set_labels=('DESeq2 no-cov', 'DESeq2 w/cov', 'Normative'), ax=ax)
        ax.set_title(pheno)
    for ax in axes[len(sets):]:
        ax.axis('off')
    plt.tight_layout()
    if save:
        (HERE / 'Figures').mkdir(exist_ok=True)
        fig.savefig(HERE / 'Figures' / 'venn_gene_benchmarks.png', bbox_inches='tight')
    return fig


def db_hit_row(phenotype, method, sig_set, ref):
    dref = ref.get(phenotype, set())
    n_sig, n_db = len(sig_set), len(sig_set & dref)
    return dict(phenotype=phenotype, method=method, n_sig=n_sig, n_db=n_db,
                db_hit_rate=round(n_db / n_sig, 3) if n_sig else np.nan,
                has_ot_ref=len(dref) > 0)


def plot_db_hit_bars(rates, title, fname, ncols=3, save=True):
    """One panel per phenotype (equal weight -- no pooled/volume-weighted average, which lets a
    method with near-zero detections in most phenotypes look artificially strong off one lucky
    win). Within a panel, one bar per method: an unfilled outline at height n_sig (total
    detections) with a solid fill at height n_db (DB-hit subset) -- count and rate in one glyph."""
    import matplotlib.pyplot as plt
    sub = rates[rates.has_ot_ref].copy()
    phenos = sorted(sub.phenotype.unique())
    methods = list(dict.fromkeys(sub.method))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    mcolor = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    nrows = -(-len(phenos) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    x = np.arange(len(methods))
    for ax, pheno in zip(axes, phenos):
        row = sub[sub.phenotype == pheno].set_index('method')
        n_sig = row['n_sig'].reindex(methods).fillna(0)
        n_db = row['n_db'].reindex(methods).fillna(0)
        for xi, m in enumerate(methods):
            c = mcolor[m]
            ax.bar(xi, n_sig[m], width=0.7, facecolor='none', edgecolor=c, linewidth=1.3)
            ax.bar(xi, n_db[m], width=0.7, facecolor=c, edgecolor='none')
            if n_sig[m] > 0:
                ax.text(xi, n_sig[m], f"{n_db[m]:.0f}/{n_sig[m]:.0f}", ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=60, ha='right', fontsize=8)
        ax.set_title(pheno, fontsize=10)
    for ax in axes[len(phenos):]:
        ax.axis('off')

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=mcolor[m], edgecolor=mcolor[m]) for m in methods]
    fig.legend(handles, methods, loc='lower center', ncol=min(len(methods), 4), bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"{title}\n(outline = total significant, fill = DB-hit subset)")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if save:
        (HERE / 'Figures').mkdir(exist_ok=True)
        fig.savefig(HERE / 'Figures' / fname, bbox_inches='tight')
    return fig


def gene_level_db_hits(save=True, designs=('no_covariate', 'covariate', 'ruvg_k1', 'ruvg_k2', 'ruvg_k3')):
    """Symmetric gene-level DB-hit table: every DESeq2 design in `designs` + normative_union
    (with and without rare-gene pooling) + per-patient normative rate distribution."""
    ref = load_reference()
    sym_of = ensg_to_symbol()
    sm = pd.read_csv(ZDIR / 'sample_meta.csv')
    gene_names = pickle.load(open(ZDIR / 'gene_names.pkl', 'rb'))

    design_sets = {d: deseq2_gene_sets(d, sym_of) for d in designs}
    norm_union, norm_pp = normative_gene_sets(sm, gene_names, sym_of)
    norm_union_nopool, _ = normative_gene_sets(sm, gene_names, sym_of, exclude_pool=True)

    all_phenos = set(norm_union)
    for ds in design_sets.values():
        all_phenos |= set(ds)

    rows = []
    for pheno in sorted(all_phenos):
        for design, ds in design_sets.items():
            if pheno in ds:
                rows.append(db_hit_row(pheno, DESIGN_LABELS[design], ds[pheno], ref))
        if pheno in norm_union:
            rows.append(db_hit_row(pheno, 'normative_union', norm_union[pheno], ref))
        if pheno in norm_union_nopool:
            rows.append(db_hit_row(pheno, 'normative_union_no_pool', norm_union_nopool[pheno], ref))

    rates = pd.DataFrame(rows)

    pp_rows = []
    for pheno, patients in norm_pp.items():
        for syms in patients:
            r = db_hit_row(pheno, 'normative_persample', syms, ref)
            pp_rows.append(r)
    persample = pd.DataFrame(pp_rows)
    persample_summary = (persample[persample.has_ot_ref]
                          .groupby('phenotype')['db_hit_rate']
                          .agg(['median', 'count']).reset_index())

    if save:
        rates.to_csv(HERE / 'gene_db_hit_rates.csv', index=False)
        persample.to_csv(HERE / 'gene_db_hit_rates_persample.csv', index=False)
    return rates, persample_summary


# --- pathway-level: same permutation-null scorer as pathway_convergence.run_phenotype,
# fed the DESeq2 Wald stat as a single "sample" instead of a per-patient Z row.

def deseq2_pathway_sig(stat_series, gene_names_base, universe_syms, sym2idx, col2sym, terms, M,
                       cache_key=None, n_perm=800, seed=42):
    # col2sym is indexed in gene_names order (the modeled 19858 genes), not the full H5AD var table
    full = pd.Series(np.nan, index=gene_names_base)
    common = stat_series.index.intersection(full.index)
    full.loc[common] = stat_series.loc[common]
    Z = full.values[None, :]

    N = len(universe_syms)
    Zu, Fm = collapse_to_symbols(Z, col2sym, N)
    Mf = M.astype(float)
    num, den = (Zu * Fm) @ Mf.T, Fm @ Mf.T
    T = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    # the permutation null re-shuffles this exact stat vector's own values -> deterministic given
    # (stat_series, seed), so it's cacheable per (design, study, phenotype) like pathway_convergence's null.npz
    null_path = (NULL_DIR / f"{cache_key}.npz") if cache_key else None
    if null_path and null_path.exists():
        d = np.load(null_path)
        null_mean, null_sd = d['null_mean'], d['null_sd']
    else:
        rng = np.random.default_rng(seed)
        null_sum, null_sumsq = np.zeros_like(T), np.zeros_like(T)
        for _ in range(n_perm):
            perm = rng.permutation(N)
            Zp, Fp = Zu[:, perm], Fm[:, perm]
            nu, de = (Zp * Fp) @ Mf.T, Fp @ Mf.T
            Tn = np.divide(nu, de, out=np.zeros_like(nu), where=de > 0)
            null_sum += Tn
            null_sumsq += Tn ** 2
        null_mean = null_sum / n_perm
        null_sd = np.sqrt(np.clip(null_sumsq / n_perm - null_mean ** 2, 1e-12, None))
        if null_path:
            NULL_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(null_path, null_mean=null_mean, null_sd=null_sd)

    Tz = (T - null_mean) / null_sd
    from scipy.stats import norm
    p = 2 * norm.sf(np.abs(Tz))
    reject = bh_fdr_reject(p[0], q=config.PATHWAY_CONV_PARAMS['fdr_q'])
    return {t for t, r in zip(terms, reject) if r}


def reference_pathways(ref_syms, sym2idx, terms, M, q=0.05):
    """Pathways where reference genes are significantly OVER-represented (hypergeometric ORA,
    BH-FDR), not just any-overlap -- 'any member gene' saturates at 80-90% of the library once
    a phenotype has >500 reference genes, since large gene sets share members with most pathways."""
    from scipy.stats import hypergeom
    idx = [sym2idx[s] for s in ref_syms if s in sym2idx]
    if not idx:
        return set()
    N = M.shape[1]
    K = len(idx)
    n = M.sum(axis=1)
    x = M[:, idx].sum(axis=1)
    p = hypergeom.sf(x - 1, N, K, n)
    reject = bh_fdr_reject(p, q=q)
    return {t for t, r in zip(terms, reject) if r}


def pathway_level_db_hits(save=True, designs=('no_covariate', 'covariate', 'ruvg_k1', 'ruvg_k2', 'ruvg_k3')):
    ref = load_reference()
    universe_syms, sym2idx, col2sym = load_symbol_vocab(None)
    terms, M = load_pathway_library()
    ref_path = {ph: reference_pathways(syms, sym2idx, terms, M) for ph, syms in ref.items()}

    gene_names = pickle.load(open(ZDIR / 'gene_names.pkl', 'rb'))
    gene_names_base = pd.Index(gene_names).str.split('.').str[0]

    sm = pd.read_csv(ZDIR / 'sample_meta.csv')
    pc_map = pc_dirs_by_phenotype(sm)
    summary = pd.read_csv(DESEQ_DIR / 'summary.csv')

    rows = []
    for pheno, pdirs in pc_map.items():
        norm_union = set()
        for pdir in pdirs:
            d = pickle.load(open(pdir / 'sig.pkl', 'rb'))
            for row in d['path_sig']:
                norm_union |= {t for t, s in zip(d['terms'], row) if s}
        rows.append(db_hit_row(pheno, 'normative_union', norm_union, ref_path))

        for design in designs:
            label = DESIGN_LABELS[design]
            studies = summary[(summary.design == design) & (summary.phenotype == pheno)]
            union = set()
            for _, r in studies.iterrows():
                path = DESEQ_DIR / design / f"{deseq2_tag(r.study, pheno)}.csv"
                if not path.exists():
                    continue
                res = pd.read_csv(path, index_col=0)
                res.index = res.index.str.split('.').str[0]
                cache_key = f"{design}__{deseq2_tag(r.study, pheno)}"
                sig = deseq2_pathway_sig(res['stat'].dropna(), gene_names_base, universe_syms,
                                         sym2idx, col2sym, terms, M, cache_key=cache_key)
                union |= sig
            if studies.shape[0]:
                rows.append(db_hit_row(pheno, label, union, ref_path))

    rates = pd.DataFrame(rows)
    if save:
        rates.to_csv(HERE / 'pathway_db_hit_rates.csv', index=False)
    return rates
