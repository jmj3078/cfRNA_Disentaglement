"""CV calibration + posterior-predictive-check diagnostics for the normative engine.

Reads the CV outputs (config.CV_RESULTS_DIR): cv_stats.csv (per-gene held-out RQR
calibration: w1/mean_z/std_z/skew_z/kurt_z + route/stage/nz), cv_zscores.pkl (per-gene
held-out z vectors), cv_ppc.pkl (per-held-out-point y/mu/sigma/family/stage). Every plot
follows the plots.py convention (fig_dir=None -> config.CV_FIG_DIR, apply_style, save +
return), and the PPC summary is cached to cv_diagnostics/ppc_summary_stats.csv.

Notebook usage is a thin runner: `from pipeline import cv_diagnostics as cvd; cvd.run_all()`.
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

import config
from viz_style import apply_style

apply_style()

STAGE_COLOR = {'nbi': '#E41A1C', 'nb_fixed': '#377EB8', 'intercept': '#4DAF4A', 'pool': '#984EA3'}
STAGE_ORDER = ['nbi', 'nb_fixed', 'intercept', 'pool']


def _fig_dir(fig_dir):
    fig_dir = fig_dir or config.CV_FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def load_cv(cv_dir=None):
    """Load cv_stats.csv (+ stage_key) and cv_zscores.pkl."""
    cv_dir = cv_dir or config.CV_RESULTS_DIR
    df = pd.read_csv(cv_dir / 'cv_stats.csv')
    df['stage_key'] = np.where(df['route'] == 'pool', 'pool', df['stage'])
    with open(cv_dir / 'cv_zscores.pkl', 'rb') as f:
        zdict = pickle.load(f)
    return df, zdict


def stages_present(df):
    return [s for s in STAGE_ORDER if s in df['stage_key'].unique()]


def summary_table(df, save=True, cv_dir=None):
    """Median calibration metrics + gene counts per stage."""
    cv_dir = cv_dir or config.CV_RESULTS_DIR
    tab = df.groupby('stage_key')[['w1', 'mean_z', 'std_z', 'skew_z', 'kurt_z', 'n_valid']].median()
    tab['n_genes'] = df['stage_key'].value_counts()
    tab = tab.reindex(stages_present(df))
    if save:
        tab.to_csv(cv_dir / 'cv_summary_by_stage.csv')
    return tab


def plot_metric_hists(df, fig_dir=None, save=True):
    fig_dir = _fig_dir(fig_dir)
    metrics = [('w1', (0, 0.3)), ('mean_z', (-0.5, 0.5)), ('std_z', (0.5, 1.5)),
               ('skew_z', (-7, 7)), ('kurt_z', (-10, 10))]
    present = stages_present(df)
    fig, axes = plt.subplots(len(present), len(metrics),
                             figsize=(4 * len(metrics), 3 * len(present)), squeeze=False)
    for r, stage in enumerate(present):
        sub = df[df['stage_key'] == stage]
        for c, (col, xlim) in enumerate(metrics):
            ax = axes[r, c]
            ax.hist(sub[col].clip(*xlim), bins=60, color=STAGE_COLOR[stage], alpha=0.85)
            target = None if col == 'w1' else (1.0 if col == 'std_z' else 0.0)
            if target is not None:
                ax.axvline(target, color='black', lw=1, ls='--')
            ax.set_xlim(xlim)
            if r == 0:
                ax.set_title(col)
            if c == 0:
                ax.set_ylabel(f'{stage}\n(n={len(sub):,})')
    fig.tight_layout()
    if save:
        fig.savefig(fig_dir / 'cv_metric_hists.png', bbox_inches='tight', dpi=150)
    return fig


def plot_calibration_vs_nz(df, fig_dir=None, save=True):
    """Bin genes by log-spaced NZ; plot bin mean +/- SEM per calibration metric, per stage."""
    fig_dir = _fig_dir(fig_dir)
    cal = [('mean_z', 0.0, (-0.5, 0.5)), ('std_z', 1.0, (0.5, 1.5)),
           ('skew_z', 0.0, (-7, 7)), ('kurt_z', 0.0, (-10, 10))]
    edges = np.geomspace(df['nz'].clip(lower=1).min(), df['nz'].max(), 16)
    df = df.copy()
    df['nz_bin'] = pd.cut(df['nz'].clip(lower=1), bins=edges, include_lowest=True)
    bin_mid = df.groupby('nz_bin', observed=True)['nz'].median()
    present = stages_present(df)
    fig, axes = plt.subplots(1, len(cal), figsize=(4.5 * len(cal), 4.5), squeeze=False)
    for ax, (metric, target, ylim) in zip(axes[0], cal):
        for stage in present:
            grp = df[df['stage_key'] == stage].dropna(subset=[metric]).groupby('nz_bin', observed=True)[metric]
            mean, sem = grp.mean(), grp.sem()
            valid = grp.count() >= 3
            x = bin_mid.reindex(mean.index)
            ax.errorbar(x[valid], mean[valid], yerr=sem[valid], fmt='-o', ms=4, lw=1.3,
                        capsize=2, color=STAGE_COLOR[stage], label=stage)
        if target is not None:
            ax.axhline(target, color='black', lw=1, ls='--')
        ax.set_xscale('log')
        ax.set_ylim(*ylim)
        ax.set_xlabel('HC Non-zero counts')
        ax.set_ylabel(f'{metric} (bin mean +/- SE)')
    axes[0][0].legend(frameon=False, loc='lower left')
    fig.tight_layout()
    if save:
        fig.savefig(fig_dir / 'cv_calibration_vs_nz.png', bbox_inches='tight', dpi=150)
    return fig


def plot_pooled_z_hist(df, zdict, fig_dir=None, save=True):
    fig_dir = _fig_dir(fig_dir)
    present = stages_present(df)
    fig, axes = plt.subplots(1, len(present), figsize=(4.5 * len(present), 4), squeeze=False)
    xs = np.linspace(-5, 5, 200)
    for c, stage in enumerate(present):
        genes = df.loc[df['stage_key'] == stage, 'gene']
        z_all = np.concatenate([zdict[g][np.isfinite(zdict[g])] for g in genes])
        ax = axes[0, c]
        ax.hist(z_all, bins=100, density=True, color=STAGE_COLOR[stage], alpha=0.8)
        ax.plot(xs, norm.pdf(xs), color='black', lw=1.2)
        ax.set_ylabel(f'{stage} (mean={z_all.mean():.3f} std={z_all.std():.3f})')
        ax.set_xlim(-5, 5)
    fig.tight_layout()
    if save:
        fig.savefig(fig_dir / 'cv_pooled_z_hist.png', bbox_inches='tight', dpi=150)
    return fig


def plot_qq_best_worst(df, zdict, fig_dir=None, save=True):
    fig_dir = _fig_dir(fig_dir)
    present = stages_present(df)
    fig, axes = plt.subplots(2, len(present), figsize=(4 * len(present), 8), squeeze=False)
    for c, stage in enumerate(present):
        sub = df[df['stage_key'] == stage].dropna(subset=['w1'])
        pairs = [(sub.loc[sub['w1'].idxmin(), 'gene'], 'best'),
                 (sub.loc[sub['w1'].idxmax(), 'gene'], 'worst')]
        for r, (gene, label) in enumerate(pairs):
            ax = axes[r, c]
            z = zdict[gene]
            z = np.sort(z[np.isfinite(z)])
            n = len(z)
            ref = norm.ppf(np.linspace(1 / (2 * n), 1 - 1 / (2 * n), n))
            ax.scatter(ref, z, s=6, alpha=0.5, color=STAGE_COLOR[stage])
            ax.plot([ref.min(), ref.max()], [ref.min(), ref.max()], color='black', lw=1, ls='--')
            ax.set_ylabel(f'{stage} / {label}')
            if r == 1:
                ax.set_xlabel('theoretical N(0,1) quantile')
    fig.tight_layout()
    if save:
        fig.savefig(fig_dir / 'cv_qq_best_worst.png', bbox_inches='tight', dpi=150)
    return fig


def _simulate_once(mu, sigma, family, seed):
    """One held-out replicate y_sim ~ NB2(mu, sigma), Poisson limit when 1/sigma huge."""
    rng = np.random.default_rng(seed)
    mu = np.clip(mu, 1e-8, 1e6)
    sigma = np.clip(sigma, 1e-8, None)
    r = 1.0 / sigma
    near_poisson = (family == 'poisson') | (r > 1e6)
    y = np.where(near_poisson, rng.poisson(mu), 0)
    nb = ~near_poisson
    if nb.any():
        p = r[nb] / (r[nb] + mu[nb])
        y[nb] = rng.negative_binomial(r[nb], p)
    return y


def build_ppc_df(df, cv_dir=None):
    """Real vs simulated mean/std/detection-rate per gene (single held-out replicate)."""
    cv_dir = cv_dir or config.CV_RESULTS_DIR
    with open(cv_dir / 'cv_ppc.pkl', 'rb') as f:
        ppc = pickle.load(f)
    stage_map = df.set_index('gene')['stage_key']
    records, seed = [], 42
    for gene, d in ppc.items():
        valid = np.isfinite(d['y']) & np.isfinite(d['mu'])
        y, mu, sigma = d['y'][valid], d['mu'][valid], d['sigma'][valid]
        if len(y) < 10:
            continue
        y_sim = _simulate_once(mu, sigma, d['family'], seed)
        records.append({
            'gene': gene, 'stage_key': stage_map.get(gene),
            'real_mean': float(y.mean()), 'sim_mean': float(y_sim.mean()),
            'real_std': float(y.std()), 'sim_std': float(y_sim.std()),
            'real_det': float((y > 0).mean()), 'sim_det': float((y_sim > 0).mean()),
        })
        seed += 1
    return pd.DataFrame(records)


def plot_ppc(df, ppc_df=None, fig_dir=None, save=True, cv_dir=None):
    """Real-vs-simulated scatter per statistic (Mean / Std / Detection rate), color by
    stage. Reports Spearman rho + RMSE + Bias (log2 for the two count stats, raw for
    detection rate; discreteness-robust) and raw-scale MAE. Saves ppc_summary_stats.csv.

    Cache-first: reloads ppc_summary_stats.csv if present (the scatter still needs ppc_df,
    which is rebuilt only when not supplied)."""
    fig_dir = _fig_dir(fig_dir)
    cv_dir = cv_dir or config.CV_RESULTS_DIR
    if ppc_df is None:
        ppc_df = build_ppc_df(df, cv_dir)
    present = stages_present(df)
    stats = [('real_mean', 'sim_mean', 'Mean Count', 'log'),
             ('real_std', 'sim_std', 'Standard Deviation', 'log'),
             ('real_det', 'sim_det', 'Detection Rate', 'linear')]
    fig, axes = plt.subplots(1, len(stats), figsize=(6 * len(stats), 5.5))
    rows = []
    for ax, (rx, sy, title, scale) in zip(axes, stats):
        for stage in present:
            sub = ppc_df[ppc_df['stage_key'] == stage].dropna(subset=[rx, sy])
            x = sub[rx].clip(lower=1e-3)
            y = sub[sy].clip(lower=1e-3)
            ax.scatter(x, y, alpha=0.15, s=4, color=STAGE_COLOR[stage],
                       label=f'{stage} (n={len(sub):,})', rasterized=True)
            xs, ys = (np.log2(x + 1), np.log2(y + 1)) if scale == 'log' else (x, y)
            rows.append((stage, title, spearmanr(xs, ys)[0], np.abs(x - y).mean(),
                         (ys - xs).mean(), np.sqrt(((ys - xs) ** 2).mean())))
        xf = ppc_df[rx].dropna().clip(lower=1e-3)
        yf = ppc_df[sy].dropna().clip(lower=1e-3)
        xfs, yfs = (np.log2(xf + 1), np.log2(yf + 1)) if scale == 'log' else (xf, yf)
        rho, mae = spearmanr(xfs, yfs)[0], np.abs(xf - yf).mean()
        bias, rmse = (yfs - xfs).mean(), np.sqrt(((yfs - xfs) ** 2).mean())
        rows.append(('ALL', title, rho, mae, bias, rmse))
        tag = 'log2-scale' if scale == 'log' else 'raw-scale'
        ax.text(0.03, 0.97, f'Spearman rho ({tag}) = {rho:.4f}\nRMSE ({tag}) = {rmse:.3f}\n'
                f'Bias ({tag}) = {bias:+.3f}\nMAE (raw-scale) = {mae:.3f}',
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
        lo = min(xf.quantile(0.005), yf.quantile(0.005))
        hi = max(xf.quantile(0.995), yf.quantile(0.995))
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='y = x')
        if scale == 'log':
            ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(f'Real (HC) -- {title}')
        ax.set_ylabel(f'Simulated -- {title}')
        ax.legend(frameon=False, markerscale=3, loc='lower right')
    fig.tight_layout()
    stats_df = pd.DataFrame(rows, columns=['stage', 'statistic', 'spearman_rho', 'mae', 'bias', 'rmse'])
    if save:
        fig.savefig(fig_dir / 'ppc_real_vs_sim.png', bbox_inches='tight', dpi=150)
        stats_df.to_csv(cv_dir / 'ppc_summary_stats.csv', index=False)
    return fig, stats_df


def run_all(cv_dir=None, fig_dir=None):
    """Thin-runner entry point: load CV outputs, emit every figure + summary CSV."""
    df, zdict = load_cv(cv_dir)
    tab = summary_table(df, cv_dir=cv_dir)
    plot_metric_hists(df, fig_dir)
    plot_calibration_vs_nz(df, fig_dir)
    plot_pooled_z_hist(df, zdict, fig_dir)
    plot_qq_best_worst(df, zdict, fig_dir)
    ppc_df = build_ppc_df(df, cv_dir)
    _, stats_df = plot_ppc(df, ppc_df, fig_dir, cv_dir=cv_dir)
    return dict(summary=tab, ppc_stats=stats_df)
