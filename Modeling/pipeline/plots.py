import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

import config
from pipeline import signatures as sig
from pipeline.cohort_stats import adjust_pvalues
from viz_style import apply_style

apply_style()

MP = config.MODELING_PARAMS

UP, DN = '#d62728', '#1f77b4'
# Engine stage/route palette (single pipeline: demotion-chain stages + pool route).
STAGE_COLOR = {'nbi': '#E41A1C', 'nb_fixed': '#377EB8', 'intercept': '#4DAF4A', 'pool': '#984EA3'}
STAGE_ORDER = ['nbi', 'nb_fixed', 'intercept', 'pool']
THEME_TINTS = ['#355475', '#AB5D10', '#3A7134', '#7C5471', '#9F3C3C', '#4F807C', '#A68D29', '#6D5141']
LEGEND_W = 0.15


# ── GSEA dotplots (gene_enrichment) ─────────────────────────────────────────
def _prep_plot_df(df_subset):
    if df_subset.empty:
        return df_subset
    d = df_subset.copy()
    d['Term_clean'] = d['Term'].str.split('__').str[1].fillna(d['Term'])
    d['tag_n'] = d['Tag %'].str.split('/').str[0].astype(float)
    d['FDR q-val'] = pd.to_numeric(d['FDR q-val'], errors='coerce').fillna(1.0)
    d['neg_log_q'] = -np.log10(d['FDR q-val'].clip(lower=1e-3))
    return d


def plot_gsea_dotplots(gsea_results, fdr_thr=None, top_n=None, fig_dir=None, sample_sizes=None):
    """Save per-phenotype bar/up/dn dotplots (gsea_bar/up/dn_{ph}.png)."""
    fdr_thr = MP['gsea_fdr_thr'] if fdr_thr is None else fdr_thr
    top_n = MP['gsea_top_n'] if top_n is None else top_n
    fig_dir = fig_dir or (config.GSEA_DIR / 'Figures' / 'gsea_dotplot')
    fig_dir.mkdir(parents=True, exist_ok=True)
    sample_sizes = sample_sizes or {}

    for ph, df in gsea_results.items():
        df = df.copy()
        df['NES'] = pd.to_numeric(df['NES'], errors='coerce')
        n_sig_pos = len(df[df['NES'] > 0])
        n_sig_neg = len(df[df['NES'] < 0])
        fname = ph.replace(' ', '_').replace('/', '_')
        n_samp = sample_sizes.get(ph, None)
        samp_str = f" (n={n_samp})" if n_samp else ""

        fig_bar, ax_bar = plt.subplots(figsize=(6, 2.5))
        bars = ax_bar.barh([0, 1], [n_sig_neg, n_sig_pos], color=[DN, UP], height=0.5, alpha=0.85)
        ax_bar.set_yticks([0, 1])
        ax_bar.set_yticklabels(['NES < 0 (Down)', 'NES > 0 (Up)'])
        ax_bar.set_xlabel('Total Significant Terms Count')
        ax_bar.set_title(f'GSEA Summary — {ph}{samp_str}\n(FDR < {fdr_thr})')
        max_count = max(n_sig_pos, n_sig_neg)
        for bar in bars:
            width = bar.get_width()
            if width > 0:
                ax_bar.text(width + (max_count * 0.02), bar.get_y() + bar.get_height() / 2,
                            f' {int(width)}', va='center', ha='left')
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        if max_count > 0:
            ax_bar.set_xlim(0, max_count * 1.2)
        plt.tight_layout()
        plt.savefig(fig_dir / f'gsea_bar_{fname}.png', bbox_inches='tight')
        plt.close(fig_bar)

        pos_df = _prep_plot_df(df[df['NES'] > 0].nlargest(top_n, 'NES').copy())
        neg_df = _prep_plot_df(df[df['NES'] < 0].nsmallest(top_n, 'NES').copy())
        if not pos_df.empty:
            pos_df = pos_df.sort_values('NES', ascending=True)
        if not neg_df.empty:
            neg_df = neg_df.sort_values('NES', ascending=False)

        for sub, cmap, sign_lbl, color, xlim_fn, fn in [
            (pos_df, 'Reds', 'Upregulated Pathways (NES > 0)', UP,
             lambda d: (-0.2, d['NES'].max() + 0.5), f'gsea_up_{fname}.png'),
            (neg_df, 'Blues', 'Downregulated Pathways (NES < 0)', DN,
             lambda d: (d['NES'].min() - 0.5, 0.2), f'gsea_dn_{fname}.png'),
        ]:
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, max(6, len(sub) * 0.45)))
            ax.set_position([0.45, 0.18, 0.35, 0.75])
            scat = ax.scatter(sub['NES'], range(len(sub)), s=sub['tag_n'] * 15,
                              c=sub['neg_log_q'], cmap=cmap, vmin=0, alpha=0.85,
                              edgecolors='black', linewidths=1., zorder=3)
            ax.axvline(0, color='black', lw=1, ls='--', alpha=0.5)
            ax.set_yticks(range(len(sub)))
            ax.set_yticklabels(sub['Term_clean'])
            ax.set_xlim(*xlim_fn(sub))
            ax.set_title(f'{sign_lbl} — {ph}{samp_str}', color=color, pad=20)
            ax.set_xlabel('NES')
            ax.grid(alpha=0.3, axis='y')
            cax = fig.add_axes([0.83, 0.55, 0.02, 0.25])
            cb = fig.colorbar(scat, cax=cax, orientation='vertical')
            cb.ax.set_title('-log10\n(FDR)', pad=10, loc='left')
            tag_min, tag_max = int(sub['tag_n'].min()), int(sub['tag_n'].max())
            if tag_min == tag_max:
                size_vals = [tag_min]
            else:
                step = (tag_max - tag_min) / 3
                size_vals = sorted(set(int(tag_min + i * step) for i in range(4)))
            size_ex = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333',
                              alpha=0.8, markersize=np.sqrt(n * 15), label=str(n)) for n in size_vals]
            ax.legend(handles=size_ex, title='Count', loc='upper left',
                      bbox_to_anchor=(1.05, 0.48), frameon=False, labelspacing=1.8)
            plt.savefig(fig_dir / fn, bbox_inches='tight')
            plt.close(fig)
        print(f'[{ph}] GSEA Plots generated successfully.')


# ── Heuristic signature figure (gsea_heuristic_signatures) ──────────────────
def plot_signature(ph, ctx, gsea_dir=None, fig_dir=None, themes=None, save=True):
    """Plot theme-shaded lollipop (left) and lead-gene specificity strip (right)."""
    fig_dir = fig_dir or (config.GSEA_DIR / 'Figures' / 'Heuristic_Signatures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows, bands, lead_pool, yc = sig.theme_rows(ph, ctx, themes=themes, gsea_dir=gsea_dir)
    sym_to_idx, meanZ, phenos_u, samp_n = ctx.sym_to_idx, ctx.meanZ, ctx.phenos_u, ctx.samp_n

    rdf = pd.DataFrame(rows)
    H = max(8, yc * 0.35)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(20, H), gridspec_kw={'width_ratios': [1.55, 1]})

    for b in bands:
        axL.axhspan(b['y0'], b['y1'], color=THEME_TINTS[b['ti'] % len(THEME_TINTS)], alpha=0.13, zorder=0)
    cols = [UP if v > 0 else DN for v in rdf['nes']]
    axL.hlines(rdf['y'], 0, rdf['nes'], color=cols, lw=1.2, alpha=0.55, zorder=2)
    axL.scatter(rdf['nes'], rdf['y'], s=np.clip(rdf['tag'] * 7, 25, 300),
                c=cols, edgecolors='black', linewidths=0.7, zorder=3)
    axL.axvline(0, color='black', lw=1, ls='--', alpha=0.5)
    axL.set_yticks(rdf['y'])
    axL.set_yticklabels(rdf['term'])
    axL.tick_params(axis='y', length=0)
    axL.set_ylim(-1, yc - 0.4)
    axL.invert_yaxis()
    for tk, th in zip(axL.get_yticklabels(), rdf['theme']):
        tk.set_color(THEME_TINTS[int(th) % len(THEME_TINTS)])
    axL.set_xlabel('NES')
    axL.set_title(f'Signature pathways by theme — {ph} (n={samp_n[ph]})', pad=10)
    axL.grid(axis='x', alpha=0.3)
    axL.margins(x=0.12)

    seen = []
    for gg in lead_pool:
        if gg in sym_to_idx and gg not in seen:
            seen.append(gg)
    tgt0 = np.array([meanZ[ph][sym_to_idx[gg]] for gg in seen])
    genes = [seen[i] for i in np.argsort(-np.abs(tgt0))[:16]]
    allmat = np.array([[meanZ[p][sym_to_idx[gg]] for p in phenos_u] for gg in genes])
    tgt = np.array([meanZ[ph][sym_to_idx[gg]] for gg in genes])
    order = np.argsort(tgt)
    genes = [genes[i] for i in order]
    allmat = allmat[order]
    tgt = tgt[order]
    yy = np.arange(len(genes))
    for i in range(len(genes)):
        axR.scatter(allmat[i], [yy[i]] * allmat.shape[1], s=12, c='#bbbbbb', alpha=0.6, zorder=1)
    axR.scatter(tgt, yy, s=80, c=[UP if v > 0 else DN for v in tgt],
                edgecolors='black', linewidths=0.8, zorder=3)
    axR.axvline(0, color='black', lw=1, ls='--', alpha=0.5)
    axR.set_yticks(yy)
    axR.set_yticklabels(genes)
    axR.set_ylim(-1, len(genes))
    axR.set_xlabel('mean Z-score')
    axR.set_title('Lead-gene specificity')
    axR.grid(axis='x', alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1 - LEGEND_W, 1])
    handles = [Patch(facecolor=THEME_TINTS[b['ti'] % len(THEME_TINTS)], alpha=0.5, label=b['label'])
               for b in bands]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1 - LEGEND_W + 0.005, 0.5),
               frameon=False, handlelength=1.2, title='Theme  (n = FDR<0.05 Pathways)',
               alignment='left')
    if save:
        plt.savefig(fig_dir / f"signature_{ph.replace(' ', '_').replace('/', '_')}.png",
                    bbox_inches='tight')
    plt.show()


# ── gene_selection plots ────────────────────────────────────────────────────
def _fdr_sig_count_per_sample(Z, padj_thr, fdr_method):
    """Per-sample count of genes with adjusted padj/qval < padj_thr, treating each gene's z
    as a per-sample z-test (p = 2*(1-Phi(|z|))). fdr_method: 'fdr_bh' (default), 'fdr_by',
    'storey', or anything cohort_stats.adjust_pvalues accepts. Correction is applied
    per-sample row (each sample is its own multiple-testing family)."""
    pval = 2 * stats.norm.sf(np.abs(Z))
    counts = np.empty(Z.shape[0], dtype=int)
    for i in range(Z.shape[0]):
        padj = adjust_pvalues(pval[i], method=fdr_method)
        counts[i] = (padj < padj_thr).sum()
    return counts


def plot_zscore_outlier_hist(Z_dis, dis_pheno, route=None, padj_thr=0.05, fdr_method='fdr_bh',
                             fig_dir=None, save=True):
    """Per-phenotype distribution of the per-sample count of FDR-significant genes
    (padj/qval < padj_thr), replacing the old raw |z|>thresh count. fdr_method: 'fdr_bh'
    (default, comparable to DESeq2 padj), 'fdr_by' (valid under arbitrary dependence,
    more conservative), or 'storey' (estimates the true-null fraction from the data instead
    of assuming the worst case -- less conservative, but not directly comparable to a
    BH-based reference like DESeq2 padj). route, if given (aligned to Z_dis columns, e.g.
    scoring.gene_stage(dd.gene_names)), excludes pool-route ("rare") genes from the count --
    they share a single fitted beta across genes and are not independent per-gene
    hypotheses, so mixing them into this FDR count would be invalid (see cohort_stats.py for
    the full rationale). Without route, all genes are counted."""
    fig_dir = fig_dir or config.CV_FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    Z = Z_dis if route is None else Z_dis[:, np.asarray(route) != 'pool']
    pheno_list = np.unique(dis_pheno)
    counts = {ph: _fdr_sig_count_per_sample(Z[dis_pheno == ph], padj_thr, fdr_method)
             for ph in pheno_list}
    order = sorted(pheno_list, key=lambda ph: -np.median(counts[ph]))
    n_pheno = len(order); n_cols = 4; n_rows = (n_pheno + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.array(axes).flatten()
    for ax, ph in zip(axes, order):
        c = counts[ph]
        ax.hist(c, bins=25, color='gray', edgecolor='white', linewidth=0.5)
        ax.axvline(np.median(c), color='tomato', linestyle='--', linewidth=1.0,
                   label=f'med={np.median(c):.0f}')
        ax.set_title(f'{ph} (n={len(c)})')
        ax.set_xlabel(f'# genes {fdr_method}<{padj_thr}'); ax.set_ylabel('samples')
        ax.legend(frameon=False)
    for ax in axes[n_pheno:]:
        ax.set_visible(False)
    fig.suptitle(f'Per-phenotype distribution of FDR-significant ({fdr_method}<{padj_thr}) gene count', y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig(fig_dir / 'zscore_outlier_gene_dist.png', bbox_inches='tight')
    plt.show()


# ── cohort-vs-cohort DEG (cohort_compare) ───────────────────────────────────
def plot_volcano(df, name, padj_thr=0.05, top_n_labels=15, fig_dir=None, save=True):
    """Volcano plot for a cohort_compare.run_comparison() DEG table.

    x = mean_diff: the difference in mean Z between the two cohorts. This is a standardized
    deviation-scale effect size, NOT a log2 fold-change -- Z is already covariate-adjusted,
    so magnitudes are comparable across samples for the same gene but are not on the same
    scale as DESeq2 log2FC across genes with different baseline dispersion; read this plot
    as "how many SD apart", not "how many fold different".
    Pool-route genes (df['route']=='pool'), if present, are excluded -- they share a single
    fitted beta across genes and aren't independent per-gene hypotheses (see cohort_stats.py).
    Genes with NaN padj (independent-filtering excluded, cohort_compare's min_hc_dev) are
    also dropped from the plot rather than shown as ns.
    """
    d = df[df['route'] != 'pool'].copy() if 'route' in df.columns else df.copy()
    d = d.dropna(subset=['padj'])
    d['neglog10_padj'] = -np.log10(d['padj'].clip(lower=1e-300))
    sig = d['padj'] < padj_thr
    up = sig & (d['mean_diff'] > 0)
    down = sig & (d['mean_diff'] < 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(d.loc[~sig, 'mean_diff'], d.loc[~sig, 'neglog10_padj'], s=4, color='lightgrey',
              alpha=0.5, label=f'ns (n={(~sig).sum():,})', rasterized=True)
    ax.scatter(d.loc[up, 'mean_diff'], d.loc[up, 'neglog10_padj'], s=8, color=UP, alpha=0.8,
              label=f'up (n={up.sum():,})')
    ax.scatter(d.loc[down, 'mean_diff'], d.loc[down, 'neglog10_padj'], s=8, color=DN, alpha=0.8,
              label=f'down (n={down.sum():,})')
    top = d[sig].sort_values('padj').head(top_n_labels)
    for _, row in top.iterrows():
        ax.annotate(row['gene_sym'], (row['mean_diff'], row['neglog10_padj']), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    ax.axhline(-np.log10(padj_thr), color='grey', lw=0.8, ls='--')
    ax.axvline(0, color='grey', lw=0.6)
    ax.set_xlabel('Mean Z difference (cohort A - cohort B)')
    ax.set_ylabel('-log10(padj)')
    ax.set_title(name, fontweight='bold')
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    if save:
        fig_dir = fig_dir or config.COHORT_COMPARE_FIG_DIR
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_dir / f'volcano_{name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def _roc_shadow(ax, roc_d, color, label, auc, lw=1.5, alpha=0.18):
    if not roc_d or not roc_d.get('fprs'):
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, color='grey')
        return
    tprs = [np.interp(np.linspace(0, 1, 101), f, t) for f, t in zip(roc_d['fprs'], roc_d['tprs'])]
    mean_t = np.mean(tprs, axis=0); std_t = np.std(tprs, axis=0)
    auc_str = f'{auc:.2f}' if not np.isnan(auc) else '—'
    base = np.linspace(0, 1, 101)
    ax.plot(base, mean_t, color=color, lw=lw, label=f'{label} AUC={auc_str}')
    ax.fill_between(base, np.clip(mean_t - std_t, 0, 1), np.clip(mean_t + std_t, 0, 1),
                    alpha=alpha, color=color)


def _pr_shadow(ax, pr_d, color, label, ap, prevalence=None, lw=1.5, alpha=0.18):
    if not pr_d or not pr_d.get('recs'):
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes, color='grey')
        return
    base = np.linspace(0, 1, 101)
    precs = []
    for rec, prec in zip(pr_d['recs'], pr_d['precs']):
        order = np.argsort(rec)
        precs.append(np.interp(base, rec[order], prec[order]))
    mean_p = np.mean(precs, axis=0); std_p = np.std(precs, axis=0)
    ap_str = f'{ap:.2f}' if not np.isnan(ap) else '—'
    ax.plot(base, mean_p, color=color, lw=lw, label=f'{label} AP={ap_str}')
    ax.fill_between(base, np.clip(mean_p - std_p, 0, 1), np.clip(mean_p + std_p, 0, 1),
                    alpha=alpha, color=color)
    if prevalence is not None:
        ax.axhline(prevalence, color='k', lw=0.7, ls='--', alpha=0.35)


def _curve_grid(curves, value_of, kind, color, label, suptitle, fname, fig_dir, save):
    """Familiar per-phenotype shadow-curve grid (ROC or PR), mirroring plot_roc_curves."""
    phenos = sorted(curves, key=lambda p: -(curves[p]['n'] if curves[p] else 0))
    ncols = 5; nrows = (len(phenos) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axf = axes.flatten()
    for ax, ph in zip(axf, phenos):
        cv = curves[ph]
        if cv is None:
            ax.text(0.5, 0.5, '(excluded)', ha='center', va='center',
                    transform=ax.transAxes, color='grey')
            ax.set_title(f'{ph}'); ax.axis('off'); continue
        val = value_of.get(ph, np.nan)
        if kind == 'roc':
            _roc_shadow(ax, cv['roc'], color, label, val)
            ax.plot([0, 1], [0, 1], 'k--', lw=0.7, alpha=0.35)
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        else:
            _pr_shadow(ax, cv['pr'], color, label, val, prevalence=cv['prevalence'])
            ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.set_title(f'{ph}  (n={cv["n"]})'); ax.tick_params(labelsize=6)
        ax.legend(frameon=False, loc='lower right' if kind == 'roc' else 'upper right',
                  fontsize=7)
    for ax in axf[len(phenos):]:
        ax.axis('off')
    fig.suptitle(suptitle, y=1.005)
    plt.tight_layout()
    if save:
        plt.savefig(fig_dir / fname, bbox_inches='tight', dpi=150)
    plt.show()

# ── disease_scoring per-sample Manhattan (disease_scoring) ──────────────────
# Fine-grained stage/route colouring. The flagged table's score_type
# (nbi_z / nb_fixed_z / intercept_z / rare_glm) maps to the engine stage/route.
_SCORE_STAGE_COLOR = STAGE_COLOR
_SCORE_STAGE_LABEL = {'nbi': 'full NBI', 'nb_fixed': 'fixed-disp NB',
                      'intercept': 'intercept NB', 'pool': 'pooled GLM (rare)'}
_SCORE_TYPE_TO_STAGE = {'nbi_z': 'nbi', 'nb_fixed_z': 'nb_fixed',
                        'intercept_z': 'intercept', 'rare_glm': 'pool'}


def _stage_of(df):
    """Map the flagged table's score_type to the engine stage/route."""
    return df['score_type'].map(_SCORE_TYPE_TO_STAGE).fillna('nbi')


MODEL_STAGES = ('nbi', 'nb_fixed', 'intercept')


def _model_padj_cutoff(model_df, padj_thr, fdr_method='fdr_bh'):
    """Smallest |z| among model-route genes with adjusted padj/qval < padj_thr, treating each
    gene's own score as a per-sample z-test (p = 2*(1-Phi(|z|))). None if nothing passes.
    fdr_method: 'fdr_bh' (default), 'fdr_by', 'storey', or anything cohort_stats.adjust_pvalues
    accepts. Only meaningful for model-route genes -- pool-route genes share a single fitted
    beta across genes, so their |z| is not an independent per-gene test statistic (see
    cohort_stats.py); they are never given a significance cutoff, only shown descriptively."""
    if len(model_df) == 0:
        return None
    pval = 2 * stats.norm.sf(model_df['abs_score'].values)
    padj = adjust_pvalues(pval, method=fdr_method)
    passing = model_df['abs_score'].values[padj < padj_thr]
    return passing.min() if len(passing) else None


def _top_flagged_panel(ax, top, title, cutoff=None):
    if len(top) == 0:
        ax.text(0.5, 0.5, 'No flagged genes', ha='center', va='center',
                transform=ax.transAxes, color='grey')
        ax.axis('off')
        return
    colors = [_SCORE_STAGE_COLOR.get(s, 'grey') for s in top['stage']]
    ax.barh(range(len(top)), top['abs_score'].values, color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{g}  ({c:.0f}ct)" for g, c in
                        zip(top['label'], top['raw_count'].fillna(0))], fontsize=7)
    ax.invert_yaxis()
    if cutoff is not None:
        ax.axvline(cutoff, color='red', lw=1, ls='--', alpha=0.6)
    ax.set_xlabel('|Score|')
    ax.set_title(title, fontweight='bold')
    seen = [s for s in MODEL_STAGES if s in set(top['stage'])]
    legend_els = [Patch(facecolor=_SCORE_STAGE_COLOR[s], label=_SCORE_STAGE_LABEL[s]) for s in seen]
    ax.legend(handles=legend_els, fontsize=7, loc='lower right')


def _rare_heatmap_panel(ax, rare_df, max_labels=40):
    """Heatmap of ALL pool-route ("rare") genes by |z| -- descriptive only, no significance
    cutoff, since these genes don't carry an independent per-gene test statistic (shared
    fitted beta across the pool). See cohort_stats.py for the rationale. With many genes,
    only max_labels evenly-spaced rows get a gene-name ytick to stay legible."""
    top = rare_df.sort_values('abs_score', ascending=False).reset_index(drop=True)
    n = len(top)
    if n == 0:
        ax.text(0.5, 0.5, 'No pool-route genes', ha='center', va='center',
                transform=ax.transAxes, color='grey')
        ax.axis('off')
        return
    vmax = max(top['abs_score'].max(), 1e-6)
    im = ax.imshow(top['score'].values.reshape(-1, 1), cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   aspect='auto')
    ax.set_xticks([])
    ticks = range(n) if n <= max_labels else np.linspace(0, n - 1, max_labels).round().astype(int)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{top['label'].iloc[i]}  ({top['raw_count'].fillna(0).iloc[i]:.0f}ct)"
                        for i in ticks], fontsize=5)
    ax.set_title(f'All {n} Pool-route Genes\n(rare, descriptive)', fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.08, pad=0.15)
    cbar.set_label('z (rare_glm)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def plot_sample(df, sample_id, phenotype='', top_n=20, padj_thr=0.05, fdr_method='fdr_bh'):
    """Model-route genes are flagged by an adjusted padj<padj_thr cutoff on their own
    per-sample z (nbi-based cutoff, computed only across nbi/nb_fixed/intercept genes).
    fdr_method: 'fdr_bh' (default), 'fdr_by', 'storey', or anything
    cohort_stats.adjust_pvalues accepts. Pool-route ("rare") genes are shown separately as a
    small descriptive heatmap with no significance cutoff, since they share a single fitted
    beta across genes rather than each having its own regression -- see cohort_stats.py for
    the full rationale."""
    df = df.dropna(subset=['score']).copy()
    df['abs_score'] = df['score'].abs()
    df['stage'] = _stage_of(df)
    if 'gene_sym' in df.columns:
        same = df['gene_sym'].isna() | (df['gene_sym'] == df['gene'])
        df['label'] = np.where(same, df['gene'], df['gene_sym'] + ' (' + df['gene'] + ')')
    else:
        df['label'] = df['gene']
    df_sorted = df.sort_values('score', ascending=False).reset_index(drop=True)
    model_mask = df_sorted['stage'].isin(MODEL_STAGES)
    model_df = df_sorted[model_mask]
    rare_df = df_sorted[df_sorted['stage'] == 'pool']
    z_cut = _model_padj_cutoff(model_df, padj_thr, fdr_method=fdr_method)
    if z_cut is None:
        z_cut = MP['z_flag']

    fig_h = max(5, min(0.08 * len(rare_df), 30))
    fig, axes = plt.subplots(1, 3, figsize=(22, fig_h), gridspec_kw={'width_ratios': [3, 1, 1]})

    ax = axes[0]
    for stage in ['nbi', 'nb_fixed', 'intercept', 'pool']:
        sub = df_sorted[df_sorted['stage'] == stage]
        if len(sub) == 0:
            continue
        ax.scatter(sub.index, sub['score'], s=0.2, alpha=0.25, color=_SCORE_STAGE_COLOR[stage],
                   label=f'{_SCORE_STAGE_LABEL[stage]} (n={len(sub):,})', rasterized=True)
    flag_sub = model_df[model_df['abs_score'] >= z_cut]
    ax.scatter(flag_sub.index, flag_sub['score'], s=5, color='black', zorder=5, alpha=0.8)
    for _, row in flag_sub.head(5).iterrows():
        ax.annotate(row['label'], (row.name, row['score']), xytext=(5, 3),
                    textcoords='offset points', fontsize=7, alpha=0.85)
    ax.axhline(z_cut, color='red', lw=1, ls='--', alpha=0.6,
              label=f'{fdr_method}<{padj_thr} (nbi-based)')
    ax.axhline(-z_cut, color='red', lw=1, ls='--', alpha=0.6)
    ax.axhline(0, color='grey', lw=0.8, ls='-', alpha=0.4)
    ax.set_xlabel('Genes (sorted by score)')
    ax.set_ylabel('Anomaly Score (z / rare_score)')
    ax.set_title(f'{sample_id}\n{phenotype}', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', markerscale=8)

    _top_flagged_panel(axes[1], flag_sub.sort_values('abs_score', ascending=False).head(top_n),
                       f'Top {top_n} Flagged Genes\n(model route)', cutoff=z_cut)
    _rare_heatmap_panel(axes[2], rare_df)
    plt.tight_layout()
    return fig


def plot_stage_score_diagnostics(flagged, z_flag=None, fig_dir=None, save=True):
    """Check whether any stage over-values z (extreme |z| driven by tiny counts, or an
    inconsistent per-stage z ceiling). Reads the flagged parquet (all disease samples).

    3 panels: (1) per-stage |z| distribution + max-|z| ceiling annotation (RQR epsilon clip
    differs by stage: nbi ~5.6, others ~7.0); (2) |z| vs raw_count hexbin per stage to spot
    high-z-from-low-count overvaluation; (3) fraction of |z|>=6 flags coming from raw_count
    <= 3 per stage (overvaluation index)."""
    z_flag = MP['z_flag'] if z_flag is None else z_flag
    fig_dir = fig_dir or config.CV_FIG_DIR
    df = flagged.dropna(subset=['score']).copy()
    df['stage'] = _stage_of(df)
    df['absz'] = df['score'].abs()
    present = [s for s in STAGE_ORDER if s in set(df['stage'])]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for s in present:
        a = df.loc[df['stage'] == s, 'absz']
        ax.hist(a, bins=np.linspace(z_flag, a.max() + 0.2, 40), histtype='step', lw=1.6,
                color=_SCORE_STAGE_COLOR[s], density=True,
                label=f'{_SCORE_STAGE_LABEL[s]} (max |z|={a.max():.2f})')
    ax.set_xlabel('|z| (flagged genes only)')
    ax.set_ylabel('density')
    ax.set_title('Per-stage |z| distribution + ceiling', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    ax = axes[1]
    for s in present:
        sub = df[df['stage'] == s]
        ax.scatter(sub['raw_count'].clip(lower=0.5), sub['absz'], s=3, alpha=0.15,
                   color=_SCORE_STAGE_COLOR[s], label=_SCORE_STAGE_LABEL[s], rasterized=True)
    ax.set_xscale('log')
    ax.axhline(6, color='black', lw=1, ls='--', alpha=0.6)
    ax.set_xlabel('raw count (observed, log)')
    ax.set_ylabel('|z|')
    ax.set_title('|z| vs raw count (low-count high-z = overvaluation)', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right', markerscale=4)

    ax = axes[2]
    idx = []
    for s in present:
        hi = df[(df['stage'] == s) & (df['absz'] >= 6)]
        frac = (hi['raw_count'] <= 3).mean() * 100 if len(hi) else 0.0
        idx.append((s, frac, len(hi)))
    ax.bar(range(len(idx)), [f for _, f, _ in idx],
           color=[_SCORE_STAGE_COLOR[s] for s, _, _ in idx], alpha=0.85)
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([f'{_SCORE_STAGE_LABEL[s]}\n(n={n:,})' for s, _, n in idx], fontsize=8)
    ax.set_ylabel('% of |z|>=6 flags with raw_count <= 3')
    ax.set_title('Overvaluation index (extreme flags from tiny counts)', fontweight='bold')
    for i, (_, f, _) in enumerate(idx):
        ax.annotate(f'{f:.1f}%', (i, f), ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save:
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_dir / 'stage_score_diagnostics.png', bbox_inches='tight', dpi=150)
    return fig



def plot_venn_benchmarks(dd, padj_thr=0.05, ncols=3, fig_dir=None, save=True):
    """Per-phenotype 3-way Venn: DESeq2 (no-cov) | DESeq2 (w/cov) | Normative Model.
    Phenotypes without any DESeq2 result file (e.g. no matched HC) are skipped.
    """
    import warnings
    from matplotlib_venn import venn3
    from pipeline.benchmark import build_venn_sets, deseq2_path

    fig_dir = fig_dir or (config.BENCHMARK_DIR / 'Figures')
    fig_dir.mkdir(parents=True, exist_ok=True)

    phenos = [ph for ph in sorted(np.unique(dd.dis_pheno))
              if deseq2_path(ph, cov=False).exists() or deseq2_path(ph, cov=True).exists()]

    nrows = (len(phenos) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.0))
    axf = np.array(axes).flatten()

    SET_COLORS = ('#4C72B0', '#C44E52', '#55A868')
    SET_LABELS = ('DESeq2\n(no-cov)', 'DESeq2\n(w/cov)', 'Normative\nModel')

    for ax, ph in zip(axf, phenos):
        sets = build_venn_sets(dd, ph, padj_thr=padj_thr)
        s_a, s_b, s_c = sets[SET_LABELS[0]], sets[SET_LABELS[1]], sets[SET_LABELS[2]]
        only_a = len(s_a - s_b - s_c)
        only_b = len(s_b - s_a - s_c)
        only_c = len(s_c - s_a - s_b)
        ab = len((s_a & s_b) - s_c)
        ac = len((s_a & s_c) - s_b)
        bc = len((s_b & s_c) - s_a)
        abc = len(s_a & s_b & s_c)
        raw = (only_a, only_b, ab, only_c, ac, bc, abc)
        scaled = tuple(int(np.sqrt(r) * 10) if r > 0 else 0 for r in raw)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            v = venn3(subsets=scaled, set_labels=SET_LABELS, ax=ax,
                      set_colors=SET_COLORS, alpha=0.55)
        if v is not None:
            for text in (v.set_labels or []):
                if text:
                    text.set_fontsize(8)
            if v.subset_labels:
                for lbl, actual in zip(v.subset_labels, raw):
                    if lbl is not None:
                        lbl.set_text(str(actual))
                        lbl.set_fontsize(9)
        n_a, n_b, n_c = len(s_a), len(s_b), len(s_c)
        ax.set_title(f'{ph}\n(no-cov:{n_a}  w/cov:{n_b}  NM:{n_c})', fontsize=8, pad=4)

    for ax in axf[len(phenos):]:
        ax.axis('off')

    fig.suptitle(
        f'Gene-level hit overlap per phenotype  (DESeq2 padj<{padj_thr}, Normative |z|≥{MP["z_flag"]} any sample)',
        y=1.01, fontsize=10)
    plt.tight_layout()
    if save:
        plt.savefig(fig_dir / 'venn_benchmarks.png', bbox_inches='tight', dpi=200)
    plt.show()
    return fig


def plot_venn_gsea_pathways(wr, dq, dq_cov, ncols=3, fig_dir=None, save=True):
    """Per-phenotype 3-way pathway Venn: DESeq2(no-cov) | DESeq2(w/cov) | NM(with_rare).
    Only phenotypes present in at least one DESeq2 GSEA result are shown.
    Circle areas are sqrt-scaled; labels show true counts.
    Palette matches the gene-level Venn (plot_venn_benchmarks).
    """
    import warnings
    from matplotlib_venn import venn3

    fig_dir = fig_dir or (config.BENCHMARK_DIR / 'Figures')
    fig_dir.mkdir(parents=True, exist_ok=True)

    phenos = sorted((set(dq) | set(dq_cov)) & set(wr))
    nrows = (len(phenos) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.0))
    axf = np.array(axes).flatten()

    SET_COLORS = ('#4C72B0', '#C44E52', '#55A868')
    SET_LABELS = ('DESeq2\n(no-cov)', 'DESeq2\n(w/cov)', 'NM\n(with_rare)')

    def _terms(d, ph):
        return set(d.get(ph, pd.DataFrame()).get('Term', pd.Series()).dropna())

    for ax, ph in zip(axf, phenos):
        s_a = _terms(dq, ph)
        s_b = _terms(dq_cov, ph)
        s_c = _terms(wr, ph)
        only_a = len(s_a - s_b - s_c)
        only_b = len(s_b - s_a - s_c)
        only_c = len(s_c - s_a - s_b)
        ab = len((s_a & s_b) - s_c)
        ac = len((s_a & s_c) - s_b)
        bc = len((s_b & s_c) - s_a)
        abc = len(s_a & s_b & s_c)
        raw = (only_a, only_b, ab, only_c, ac, bc, abc)
        scaled = tuple(int(np.sqrt(r) * 10) if r > 0 else 0 for r in raw)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            v = venn3(subsets=scaled, set_labels=SET_LABELS, ax=ax,
                      set_colors=SET_COLORS, alpha=0.55)
        if v is not None:
            for text in (v.set_labels or []):
                if text:
                    text.set_fontsize(8)
            if v.subset_labels:
                for lbl, actual in zip(v.subset_labels, raw):
                    if lbl is not None:
                        lbl.set_text(str(actual))
                        lbl.set_fontsize(9)
        ax.set_title(f'{ph}\n(no-cov:{len(s_a)}  w/cov:{len(s_b)}  NM:{len(s_c)})',
                     fontsize=8, pad=4)

    for ax in axf[len(phenos):]:
        ax.axis('off')

    fig.suptitle('GSEA pathway overlap per phenotype  (FDR < 0.05)',
                 y=1.01, fontsize=10)
    plt.tight_layout()
    if save:
        plt.savefig(fig_dir / 'venn_gsea_pathways.png', bbox_inches='tight', dpi=200)
    plt.show()
    return fig


DB_METHOD_STYLE = {'deseq2': ('#DD8452', 'DESeq2'),
                   'deseq2_cov': ('#C44E52', 'DESeq2 + covariates'),
                   'only_nbi': ('#4C72B0', 'Normative (no rare pooling)'),
                   'with_rare': ('#55A868', 'Normative (with rare pooling)')}


def plot_db_hit_rates(rates, summary, fig_dir=None, save=True):
    """Symmetric DB-support comparison, counts first. Left: per-phenotype DB-supported term
    counts (n_db) for each method. Right: pooled DB-supported (filled) inside all-significant
    (outline) with the pooled DB-hit rate annotated, so absolute coverage and precision are
    both visible. Only phenotypes with an Open Targets reference are shown."""
    fig_dir = fig_dir or (config.BENCHMARK_DIR / 'Figures')
    fig_dir.mkdir(parents=True, exist_ok=True)
    sub = rates[rates['has_ot_ref']].copy()
    methods = [m for m in DB_METHOD_STYLE if m in set(sub['method'])]
    order = sub.groupby('phenotype')['n_db'].sum().sort_values().index.tolist()
    piv = sub.pivot(index='phenotype', columns='method', values='n_db').reindex(order)
    y = np.arange(len(order))
    h = 0.8 / len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), gridspec_kw={'width_ratios': [2.4, 1]})
    ax = axes[0]
    for i, m in enumerate(methods):
        c, lab = DB_METHOD_STYLE[m]
        ax.barh(y + (i - (len(methods) - 1) / 2) * h, piv[m].values, height=h, color=c, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel('DB-supported significant terms (count)')
    ax.set_title('Per-phenotype DB-hit counts')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax2 = axes[1]
    sm = summary.set_index('method').reindex(methods)
    xb = np.arange(len(methods))
    cols = [DB_METHOD_STYLE[m][0] for m in methods]
    ax2.bar(xb, sm['total_sig'].values, color='none', edgecolor='grey', lw=1.0)
    ax2.bar(xb, sm['total_db'].values, color=cols)
    for k, m in enumerate(methods):
        ax2.text(k, sm.loc[m, 'total_db'],
                 f"{int(sm.loc[m, 'total_db'])}\n({sm.loc[m, 'pooled_db_hit_rate']:.2f})",
                 ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(xb)
    ax2.set_xticklabels([DB_METHOD_STYLE[m][1] for m in methods], rotation=30, ha='right', fontsize=8)
    ax2.set_ylabel('pooled term count')
    ax2.set_title('DB-supported (fill) vs all significant (outline)\nlabel: n_db (pooled DB-hit rate)')
    fig.tight_layout()
    if save:
        plt.savefig(fig_dir / 'db_hit_rates.png', bbox_inches='tight', dpi=200)
    plt.show()
    return fig
