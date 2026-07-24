import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.marginal_rqr import _poisson_rqr, marginal_nb_rqr
from MixedEffectsModeling.pool_vs_individual_sweep import load_hc
from MixedEffectsModeling.ppc_mixed import simulate_many
from MixedEffectsModeling.spike_in_power_test import reconstruct_individual_full
from viz_style import apply_style

OUT = config.THRESHOLD_SWEEP_DIR
THRESHOLDS = [50, 75, 150]
N_REPS = 300
MIN_VALID = 100
CSV_PATH = OUT / "gene_level_diagnostics.csv"
CSV_PATH_ALIGNED = OUT / "gene_level_diagnostics_aligned.csv"


def _row(gene, T, fit_name, d, seed):
    y, mu, alpha, tau2 = d["y"], d["mu"], d["alpha"], d["tau2"]
    family = d.get("family", "negbin")
    y_rep = simulate_many(mu, alpha, tau2, family, N_REPS, seed)
    if family == "poisson":
        z = _poisson_rqr(y, mu, seed=seed)
    else:
        z = marginal_nb_rqr(y, mu, alpha, tau2, seed=seed)
    return dict(
        nz_threshold=T, gene=gene, fit=fit_name,
        obs_mean=float(y.mean()), pred_mean=float(y_rep.mean(axis=1).mean()),
        obs_std=float(y.std()), pred_std=float(y_rep.std(axis=1).mean()),
        obs_nonzero=float((y > 0).mean()), pred_nonzero=float((y_rep > 0).mean(axis=1).mean()),
        mean_z=float(np.mean(z)), std_z=float(np.std(z)))


def build_rows_aligned(limit_genes):
    Xs, Y, batch, gene_names = load_hc()
    with open(OUT / "pool_vs_individual_ppc_pooled.pkl", "rb") as f:
        ppc_pooled_by_threshold = pickle.load(f)
    superset = sorted(set().union(*[set(ppc_pooled_by_threshold[T].keys()) for T in ppc_pooled_by_threshold]))
    ppc_individual = reconstruct_individual_full(Xs, batch, gene_names, superset)

    rows = []
    for T in THRESHOLDS:
        ppc_pooled = ppc_pooled_by_threshold[T]
        genes = [g for g in ppc_pooled if g in ppc_individual]
        if limit_genes is not None:
            genes = genes[:limit_genes]
        used = 0
        for gi, g in enumerate(genes):
            mu_ind = ppc_individual[g]["mu"]; alpha_ind = ppc_individual[g]["alpha"]; tau2_ind = ppc_individual[g]["tau2"]
            mu_pool = ppc_pooled[g]["mu"]; alpha_pool = ppc_pooled[g]["alpha"]; tau2_pool = ppc_pooled[g]["tau2"]
            y = ppc_pooled[g]["y"]
            valid = np.isfinite(mu_ind) & np.isfinite(alpha_ind) & np.isfinite(mu_pool)
            if valid.sum() < MIN_VALID:
                continue
            used += 1
            d_ind = dict(y=y[valid], mu=mu_ind[valid], alpha=alpha_ind[valid], tau2=tau2_ind[valid], family="negbin")
            d_pool = dict(y=y[valid], mu=mu_pool[valid], alpha=alpha_pool[valid], tau2=tau2_pool[valid], family="negbin")
            rows.append(_row(g, T, "individual", d_ind, seed=1000 + gi))
            rows.append(_row(g, T, "pooled", d_pool, seed=2000 + gi))
        print(f"nz<{T}: {used}/{len(genes)} genes retained (aligned, >= {MIN_VALID} valid samples)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", action="store_true")
    ap.add_argument("--limit-genes", type=int, default=None)
    args = ap.parse_args()

    csv_path = CSV_PATH_ALIGNED if args.aligned else CSV_PATH
    fig_name = "gene_level_diagnostics_aligned.png" if args.aligned else "gene_level_diagnostics.png"

    if csv_path.exists() and args.limit_genes is None:
        print(f"Loading cached -> {csv_path}")
        df = pd.read_csv(csv_path)
    elif args.aligned:
        rows = build_rows_aligned(args.limit_genes)
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Saved -> {csv_path}")
    else:
        with open(OUT / "pool_vs_individual_ppc_individual.pkl", "rb") as f:
            ppc_individual = pickle.load(f)
        with open(OUT / "pool_vs_individual_ppc_pooled.pkl", "rb") as f:
            ppc_pooled_by_threshold = pickle.load(f)

        rows = []
        for T in THRESHOLDS:
            ppc_pooled = ppc_pooled_by_threshold[T]
            genes = [g for g in ppc_pooled if g in ppc_individual]
            if args.limit_genes is not None:
                genes = genes[:args.limit_genes]
            print(f"nz<{T}: {len(genes)} genes")
            for gi, g in enumerate(genes):
                rows.append(_row(g, T, "individual", ppc_individual[g], seed=1000 + gi))
                rows.append(_row(g, T, "pooled", ppc_pooled[g], seed=2000 + gi))
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"Saved -> {csv_path}")

    apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(THRESHOLDS), 4, figsize=(16, 4 * len(THRESHOLDS)))
    colors = {"individual": "tab:blue", "pooled": "tab:orange"}
    for ri, T in enumerate(THRESHOLDS):
        sub = df[df.nz_threshold == T]
        ax = axes[ri]
        for fit_name, c in colors.items():
            s = sub[sub.fit == fit_name]
            ax[0].scatter(s.obs_mean, s.pred_mean, s=4, alpha=0.3, color=c, label=fit_name)
            ax[1].scatter(s.obs_std, s.pred_std, s=4, alpha=0.3, color=c, label=fit_name)
            ax[2].scatter(s.obs_nonzero, s.pred_nonzero, s=4, alpha=0.3, color=c, label=fit_name)
            ax[3].scatter(s.mean_z, s.std_z, s=4, alpha=0.3, color=c, label=fit_name)

        for j, (xl, yl, log) in enumerate([
            ("observed mean(y)", "predicted mean(y)", True),
            ("observed std(y)", "predicted std(y)", True),
            ("observed nonzero frac", "predicted nonzero frac", False),
        ]):
            a = ax[j]
            if log:
                a.set_xscale("log"); a.set_yscale("log")
                lo, hi = a.get_xlim()
                a.plot([lo, hi], [lo, hi], ls=":", color="gray", zorder=0)
            else:
                a.plot([0, 1], [0, 1], ls=":", color="gray", zorder=0)
            a.set(xlabel=xl, ylabel=yl, title=f"nz<{T}")

        ax[3].axvline(0, ls=":", color="gray"); ax[3].axhline(1, ls=":", color="gray")
        ax[3].set(xlabel="mean(z) per gene (bias)", ylabel="std(z) per gene (dispersion)", title=f"nz<{T} Z-calibration")
        ax[3].set_xlim(-2, 2); ax[3].set_ylim(0, 2.5)
    axes[0][0].legend(markerscale=3)
    fig.tight_layout()
    fig_path = config.THRESHOLD_SWEEP_FIG_DIR / fig_name
    fig.savefig(fig_path, dpi=150)
    print(f"Saved -> {fig_path}")


if __name__ == "__main__":
    main()
