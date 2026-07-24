import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.marginal_rqr import marginal_nb_loglik
from MixedEffectsModeling.pool_vs_individual_sweep import load_hc
from MixedEffectsModeling.spike_in_power_test import reconstruct_individual_full
from viz_style import apply_style

OUT = config.THRESHOLD_SWEEP_DIR
MIN_VALID = 100


def main():
    print("Loading HC data for nz map...")
    Xs, Y, batch, gene_names = load_hc()
    nz_map = dict(zip(gene_names, (Y > 0).sum(axis=0)))

    print("Loading cached pooled PPC pickle...")
    with open(OUT / "pool_vs_individual_ppc_pooled.pkl", "rb") as f:
        ppc_pooled_by_threshold = pickle.load(f)

    superset = sorted(set().union(*[set(ppc_pooled_by_threshold[T].keys()) for T in ppc_pooled_by_threshold]))
    print(f"Reconstructing full-length individual per-sample params ({len(superset)} genes)...")
    ppc_individual = reconstruct_individual_full(Xs, batch, gene_names, superset)

    thresholds = sorted(ppc_pooled_by_threshold.keys())
    gene_rows, summary_rows = [], []
    for T in thresholds:
        ppc_pool = ppc_pooled_by_threshold[T]
        genes = [g for g in ppc_pool if g in ppc_individual]
        diffs = []
        for g in genes:
            mu_ind = ppc_individual[g]["mu"]
            alpha_ind = ppc_individual[g]["alpha"]
            tau2_ind = ppc_individual[g]["tau2"]
            mu_pool = ppc_pool[g]["mu"]
            alpha_pool = ppc_pool[g]["alpha"]
            tau2_pool = ppc_pool[g]["tau2"]
            y = ppc_pool[g]["y"]

            valid = np.isfinite(mu_ind) & np.isfinite(alpha_ind) & np.isfinite(mu_pool)
            n_valid = int(valid.sum())
            if n_valid < MIN_VALID:
                continue

            ll_ind = float(marginal_nb_loglik(y[valid], mu_ind[valid], alpha_ind[valid], tau2_ind[valid]).mean())
            ll_pool = float(marginal_nb_loglik(y[valid], mu_pool[valid], alpha_pool[valid], tau2_pool[valid]).mean())
            ll_diff = ll_ind - ll_pool
            diffs.append(ll_diff)
            gene_rows.append(dict(nz_threshold=T, gene=g, nz=int(nz_map[g]), n_valid=n_valid,
                                  ll_individual=ll_ind, ll_pooled=ll_pool, ll_diff=ll_diff))

        if not diffs:
            continue
        diffs = np.array(diffs)
        frac_individual_wins = float((diffs > 0).mean())
        summary_rows.append(dict(nz_threshold=T, n_genes=len(diffs), median_ll_diff=float(np.median(diffs)),
                                 mean_ll_diff=float(diffs.mean()), frac_individual_wins=frac_individual_wins))
        print(f"nz<{T}: n_genes={len(diffs)} median_ll_diff={np.median(diffs):.4f} "
              f"frac_individual_wins={frac_individual_wins:.3f}")

    gene_df = pd.DataFrame(gene_rows)
    summary_df = pd.DataFrame(summary_rows)
    gene_df.to_csv(OUT / "pool_vs_individual_gene_level_aligned.csv", index=False)
    summary_df.to_csv(OUT / "pool_vs_individual_summary_aligned.csv", index=False)

    apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(summary_df["nz_threshold"], summary_df["median_ll_diff"], "-o")
    axes[0].axhline(0, ls=":", color="gray")
    axes[0].set(xlabel="Pooling threshold (HC nonzero count)",
               ylabel="Median held-out LL(individual) - LL(pooled)\n(>0: individual model wins)")
    axes[1].plot(summary_df["nz_threshold"], summary_df["frac_individual_wins"], "-o", color="tab:orange")
    axes[1].axhline(0.5, ls=":", color="gray")
    axes[1].set(xlabel="Pooling threshold (HC nonzero count)", ylabel="Fraction of genes where individual wins")
    fig.tight_layout()
    fig.savefig(config.THRESHOLD_SWEEP_FIG_DIR / "pool_vs_individual_aligned.png", dpi=150)

    print(f"Saved -> {OUT}/pool_vs_individual_gene_level_aligned.csv, "
          f"{OUT}/pool_vs_individual_summary_aligned.csv, "
          f"{config.THRESHOLD_SWEEP_FIG_DIR}/pool_vs_individual_aligned.png")


if __name__ == "__main__":
    main()
