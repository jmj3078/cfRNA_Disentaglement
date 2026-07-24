import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.marginal_rqr import marginal_nb_rqr
from MixedEffectsModeling.pool_vs_individual_sweep import load_hc
from viz_style import apply_style

OUT = config.THRESHOLD_SWEEP_DIR
FIG_OUT = config.THRESHOLD_SWEEP_FIG_DIR
OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)

SHIFT_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
N_REP = 50
FLAG = 3.0  # project z_flag convention
IND_TMP = "/tmp/pool_vs_individual/individual"
MIN_VALID = 100  # need enough non-nan draws given N_REP=50


def reconstruct_individual_full(Xs, batch, gene_names, superset):
    cache = OUT / "ppc_individual_full.pkl"
    if cache.is_file():
        with open(cache, "rb") as f:
            return pickle.load(f)

    res_paths = [Path(IND_TMP) / f"res_{fi}.csv" for fi in range(config.SPIKE_PARAMS["n_splits"])]
    missing = [p for p in res_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing individual fold coefficient files: {missing}")

    n_hc = Xs.shape[0]
    folds = list(StratifiedKFold(config.SPIKE_PARAMS["n_splits"], shuffle=True, random_state=42)
                 .split(np.zeros(n_hc), batch))

    parts = {g: {"mu": [], "alpha": [], "tau2": []} for g in superset}
    for fi, (tr, te) in enumerate(folds):
        fold_fits = pd.read_csv(res_paths[fi]).set_index("gene")
        Xa_te = np.column_stack([np.ones(len(te)), Xs[te]])
        for g in superset:
            if g not in fold_fits.index or not bool(fold_fits.loc[g, "ok"]):
                parts[g]["mu"].append(np.full(len(te), np.nan))
                parts[g]["alpha"].append(np.full(len(te), np.nan))
                parts[g]["tau2"].append(np.full(len(te), np.nan))
                continue
            row = fold_fits.loc[g]
            mu_coef = row[[c for c in fold_fits.columns if c.startswith("mu_coef_")]].values.astype(float)
            disp_coef = row[[c for c in fold_fits.columns if c.startswith("disp_coef_")]].values.astype(float)
            mu = np.clip(np.exp(Xa_te @ np.nan_to_num(mu_coef, nan=0.0)), 1e-6, 1e8)
            if not np.all(np.isnan(disp_coef)):
                alpha = np.exp(-Xa_te @ np.nan_to_num(disp_coef, nan=0.0))
            elif "fixed_alpha" in row.index and not pd.isna(row["fixed_alpha"]):
                alpha = np.full(len(te), float(row["fixed_alpha"]))
            else:
                alpha = np.full(len(te), np.nan)
            parts[g]["mu"].append(mu)
            parts[g]["alpha"].append(alpha)
            parts[g]["tau2"].append(np.full(len(te), float(row["tau2"])))

    ppc_full = {}
    for g in superset:
        ppc_full[g] = dict(mu=np.concatenate(parts[g]["mu"]).astype(np.float64),
                            alpha=np.concatenate(parts[g]["alpha"]).astype(np.float64),
                            tau2=np.concatenate(parts[g]["tau2"]).astype(np.float64))
    with open(cache, "wb") as f:
        pickle.dump(ppc_full, f)
    return ppc_full


def simulate_and_score(params_gen, params_ind, params_pool, shift, seed):
    mu_gen, alpha_gen, tau2_gen = params_gen
    n_s = len(mu_gen)
    rng = np.random.default_rng(seed)

    b = rng.normal(0.0, np.sqrt(np.maximum(tau2_gen, 0.0)), size=(N_REP, n_s))
    mean_syn = mu_gen[None, :] * shift * np.exp(b)
    alpha_tiled = np.tile(alpha_gen, N_REP)
    n_nb = 1.0 / alpha_tiled
    p_nb = n_nb / (n_nb + mean_syn.ravel())
    y_syn = rng.negative_binomial(n_nb, p_nb).astype(np.float64)

    mu_ind, alpha_ind, tau2_ind = params_ind
    mu_pool, alpha_pool, tau2_pool = params_pool
    mu_ind_t = np.tile(mu_ind, N_REP)
    alpha_ind_t = np.tile(alpha_ind, N_REP)
    tau2_ind_t = np.tile(tau2_ind, N_REP)
    mu_pool_t = np.tile(mu_pool, N_REP)
    alpha_pool_t = np.tile(alpha_pool, N_REP)
    tau2_pool_t = np.tile(tau2_pool, N_REP)

    z_ind = marginal_nb_rqr(y_syn, mu_ind_t, alpha_ind_t, tau2_ind_t, seed)
    z_pool = marginal_nb_rqr(y_syn, mu_pool_t, alpha_pool_t, tau2_pool_t, seed)
    return z_ind, z_pool


def gene_params(ppc_gene):
    return (ppc_gene["mu"].astype(np.float64), ppc_gene["alpha"].astype(np.float64),
            ppc_gene["tau2"].astype(np.float64))


NZ_BINS = [0, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150]


def individual_fit_success_diagnostic(nz_map):
    n_splits = config.SPIKE_PARAMS["n_splits"]
    ok = None
    for fi in range(n_splits):
        fold = pd.read_csv(Path(IND_TMP) / f"res_{fi}.csv").set_index("gene")["ok"]
        ok = fold.to_frame(fi) if ok is None else ok.join(fold.to_frame(fi), how="outer")
    fold_ok_rate = ok.mean(axis=1)
    all5_ok = (ok.fillna(False).sum(axis=1) == n_splits)

    summary_train = pd.read_csv(config.ENGINE_MIXED_DIR / "training_summary.csv", index_col="gene")["stage"]

    diag = pd.DataFrame({"gene": ok.index})
    diag["nz"] = diag["gene"].map(nz_map)
    diag["fold_ok_rate"] = diag["gene"].map(fold_ok_rate)
    diag["all5_ok"] = diag["gene"].map(all5_ok)
    diag["stage"] = diag["gene"].map(summary_train)
    diag = diag.dropna(subset=["nz"])
    diag.to_csv(OUT / "individual_fit_success.csv", index=False)

    diag["nz_bin"] = pd.cut(diag["nz"], bins=NZ_BINS + [max(diag["nz"].max() + 1, NZ_BINS[-1] + 1)], right=False)
    binned = diag.groupby("nz_bin", observed=True).agg(
        mean_fold_ok_rate=("fold_ok_rate", "mean"), frac_all5_ok=("all5_ok", "mean"),
        n_genes=("gene", "size")).reset_index()
    print(binned.to_string(index=False))

    apply_style()
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(binned))
    ax1.plot(x, binned["mean_fold_ok_rate"], "-o", label="mean fold_ok_rate")
    ax1.plot(x, binned["frac_all5_ok"], "-o", label="frac all5_ok")
    ax1.axhline(0.95, ls=":", color="gray")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(b) for b in binned["nz_bin"]], rotation=45, ha="right", fontsize=6)
    ax1.set(xlabel="nz bin", ylabel="fraction")
    ax1.legend(fontsize=7)

    stage_frac = diag.groupby("nz_bin", observed=True)["stage"].value_counts(normalize=True).unstack(fill_value=0)
    stage_frac = stage_frac.reindex(binned["nz_bin"])
    for stage in stage_frac.columns:
        ax2.plot(x, stage_frac[stage].values, "-o", label=stage)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(b) for b in binned["nz_bin"]], rotation=45, ha="right", fontsize=6)
    ax2.set(xlabel="nz bin", ylabel="fraction of genes at stage")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "individual_fit_success.png", dpi=150)
    print(f"Saved -> {OUT}/individual_fit_success.csv, {FIG_OUT}/individual_fit_success.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-genes", type=int, default=None)
    ap.add_argument("--thresholds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    print("Loading HC data for nz map...")
    Xs, Y, batch, gene_names = load_hc()
    nz = (Y > 0).sum(axis=0)
    nz_map = dict(zip(gene_names, nz))

    print("Individual fit success diagnostic...")
    individual_fit_success_diagnostic(nz_map)

    print("Loading cached pooled PPC pickle...")
    with open(OUT / "pool_vs_individual_ppc_pooled.pkl", "rb") as f:
        ppc_pooled_by_threshold = pickle.load(f)

    superset = sorted(set().union(*[set(ppc_pooled_by_threshold[T].keys()) for T in ppc_pooled_by_threshold]))
    print(f"Reconstructing full-length individual per-sample params ({len(superset)} genes)...")
    ppc_individual = reconstruct_individual_full(Xs, batch, gene_names, superset)

    thresholds = args.thresholds if args.thresholds is not None else sorted(ppc_pooled_by_threshold.keys())

    base_seed = config.SPIKE_PARAMS["seed"]
    rows = []
    for T in thresholds:
        if T not in ppc_pooled_by_threshold:
            print(f"Skipping T={T}: not in cached pooled pkl")
            continue
        ppc_pool = ppc_pooled_by_threshold[T]
        genes = [g for g in ppc_pool if g in ppc_individual]
        if args.limit_genes is not None:
            genes = genes[:args.limit_genes]

        used_genes = []
        for gi, g in enumerate(genes):
            mu_ind, alpha_ind, tau2_ind = gene_params(ppc_individual[g])
            mu_pool, alpha_pool, tau2_pool = gene_params(ppc_pool[g])
            valid = np.isfinite(mu_ind) & np.isfinite(alpha_ind) & np.isfinite(mu_pool)
            n_valid = int(valid.sum())
            if n_valid < MIN_VALID:
                continue
            used_genes.append(g)
            params_ind = (mu_ind[valid], alpha_ind[valid], tau2_ind[valid])
            params_pool = (mu_pool[valid], alpha_pool[valid], tau2_pool[valid])
            for generator, params_gen in (("ind", params_ind), ("pool", params_pool)):
                for si, shift in enumerate(SHIFT_GRID):
                    seed = base_seed + T * 100000 + gi * 1000 + (0 if generator == "ind" else 500) + si
                    z_ind, z_pool = simulate_and_score(params_gen, params_ind, params_pool, shift, seed)
                    rows.append(dict(
                        nz_threshold=T, gene=g, nz=int(nz_map[g]), generator=generator, shift=shift,
                        flag_ind=float((np.abs(z_ind) > FLAG).mean()),
                        flag_pool=float((np.abs(z_pool) > FLAG).mean()),
                        meanabs_ind=float(np.abs(z_ind).mean()),
                        meanabs_pool=float(np.abs(z_pool).mean()),
                        divergence=float(np.abs(z_ind - z_pool).mean()),
                        n_samples=n_valid, n_valid_samples=n_valid,
                    ))
        print(f"T={T}: {len(used_genes)}/{len(genes)} genes retained (>= {MIN_VALID} valid samples)")

    gene_df = pd.DataFrame(rows)
    gene_df.to_csv(OUT / "spike_in_gene_level.csv", index=False)
    print(f"spike_in_gene_level.csv: {gene_df.shape}")
    print(gene_df.head())

    summary = gene_df.groupby(["nz_threshold", "generator", "shift"]).agg(
        mean_flag_ind=("flag_ind", "mean"), median_flag_ind=("flag_ind", "median"),
        mean_flag_pool=("flag_pool", "mean"), median_flag_pool=("flag_pool", "median"),
        mean_meanabs_ind=("meanabs_ind", "mean"), median_meanabs_ind=("meanabs_ind", "median"),
        mean_meanabs_pool=("meanabs_pool", "mean"), median_meanabs_pool=("meanabs_pool", "median"),
        mean_divergence=("divergence", "mean"), median_divergence=("divergence", "median"),
        n_genes=("gene", "nunique"),
    ).reset_index()
    summary.to_csv(OUT / "spike_in_summary.csv", index=False)
    print(f"spike_in_summary.csv: {summary.shape}")
    print(summary.head())

    apply_style()
    import matplotlib.pyplot as plt

    rep_ts = sorted(set(thresholds) & set(summary["nz_threshold"].unique()))
    if len(rep_ts) > 4:
        idx = np.linspace(0, len(rep_ts) - 1, 4).astype(int)
        rep_ts = [rep_ts[i] for i in idx]

    fig, axes = plt.subplots(1, len(rep_ts), figsize=(4.2 * len(rep_ts), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, T in zip(axes, rep_ts):
        sub = summary[summary["nz_threshold"] == T]
        for generator, color in (("ind", "tab:blue"), ("pool", "tab:orange")):
            for score_col, ls, label in (("mean_flag_ind", "-", f"{generator} gen, ind score"),
                                          ("mean_flag_pool", "--", f"{generator} gen, pool score")):
                s = sub[sub["generator"] == generator]
                ax.plot(s["shift"], s[score_col], ls, color=color, marker="o", label=label)
        ax.axvline(1.0, ls=":", color="gray")
        ax.set_xscale("log")
        ax.set(xlabel="Multiplicative shift", title=f"nz_threshold={T}")
    axes[0].set_ylabel(f"Flag rate (|z|>{FLAG})")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_OUT / "spike_in_flag_curves.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    for shift in SHIFT_GRID:
        s = summary[summary["shift"] == shift].groupby("nz_threshold")["mean_divergence"].mean().reset_index()
        ax2.plot(s["nz_threshold"], s["mean_divergence"], "-o", label=f"shift={shift}")
    ax2.set(xlabel="Pooling threshold (HC nonzero count)", ylabel="Mean |z_ind - z_pool|")
    ax2.legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig(FIG_OUT / "spike_in_divergence.png", dpi=150)

    print(f"Saved -> {OUT}/spike_in_gene_level.csv, {OUT}/spike_in_summary.csv, "
          f"{FIG_OUT}/spike_in_flag_curves.png, {FIG_OUT}/spike_in_divergence.png")


if __name__ == "__main__":
    main()
