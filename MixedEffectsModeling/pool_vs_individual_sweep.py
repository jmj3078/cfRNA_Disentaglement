import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.marginal_rqr import marginal_nb_loglik
from viz_style import apply_style

MP = config.SPIKE_PARAMS
OUT = config.THRESHOLD_SWEEP_DIR
OUT.mkdir(parents=True, exist_ok=True)
config.THRESHOLD_SWEEP_FIG_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [3, 5, 7, 10, 15, 20, 25, 30, 50, 75, 100, 150]
TMP = "/tmp/pool_vs_individual"


def load_hc():
    adata = sc.read_h5ad(config.H5AD_PATH)
    m = ((adata.obs["QC_Passed"] == True) & (adata.obs["Phenotype_Processed"].notna()) &
         (adata.obs["Phenotype_Processed"] != "Unknown") &
         (adata.obs["broad_protocol_category"] != "Exome-based (EB)"))
    a = adata[m]
    is_hc = (a.obs["Phenotype_Processed"].astype(str) == "Healthy Control").values
    is_pc = (a.var["GeneType"] == "protein_coding").values
    gene_names = a.var_names[is_pc].values
    X = a.obs[config.BIAS_COLUMNS].values.astype(np.float64)[is_hc]
    Xs = StandardScaler().fit_transform(X)
    Y = a.X.toarray() if issparse(a.X) else np.asarray(a.X)
    Y = np.round(Y[is_hc][:, is_pc]).astype(np.float64)
    batch = a.obs["Batch_ID"].astype(str).values[is_hc]
    return Xs, Y, batch, gene_names


def individual_cascade_ppc(Xs, Y, batch, gene_names, genes, stage_of, folds, tmp):
    """Refit each gene at its ALREADY-KNOWN stage (engine_state_mixed/training_summary.csv,
    from the Task 6a full-data cascade) via glmm_fit.R --mode fixed_stage -- same
    convention already validated in cv_glmm_engine.py's cv_model_route(). Re-searching
    the demotion chain from scratch per fold (--mode cascade) would be redundant: the
    full-data fit already tells us which stage each gene converges to, and fixed_stage
    is the cheaper, already-proven way to refit it per fold. Runs ONCE on the superset
    of genes, independent of any pooling threshold, and is reused for every threshold.

    Returns dict[gene] = dict(y, mu, alpha, tau2, family, stage) -- same per-sample
    schema as cv_glmm_engine.py's cv_ppc.pkl, so downstream PPC/log-likelihood
    re-diagnosis never needs to rerun the R fits."""
    Path(tmp).mkdir(parents=True, exist_ok=True)
    name2col = {g: i for i, g in enumerate(gene_names)}
    parts = {g: [] for g in genes}
    for fi, (tr, te) in enumerate(folds):
        pd.DataFrame(Xs[tr], columns=config.BIAS_COLUMNS).to_csv(f"{tmp}/X_{fi}.csv.gz")
        Y_tr = Y[tr][:, [name2col[g] for g in genes]]
        pd.DataFrame(Y_tr, columns=genes).to_csv(f"{tmp}/Y_{fi}.csv.gz")
        pd.DataFrame({"Batch_ID": batch[tr]}).to_csv(f"{tmp}/batch_{fi}.csv.gz")
        pd.DataFrame({"gene": genes, "stage": [stage_of[g] for g in genes]}).to_csv(
            f"{tmp}/genes_{fi}.csv", index=False)
        subprocess.run([
            "Rscript", str(config.GLMM_FIT_R), "--x", f"{tmp}/X_{fi}.csv.gz", "--y", f"{tmp}/Y_{fi}.csv.gz",
            "--batch", f"{tmp}/batch_{fi}.csv.gz", "--genes", f"{tmp}/genes_{fi}.csv",
            "--trend", str(config.DISPERSION_TREND_PATH), "--mode", "fixed_stage", "--out", f"{tmp}/res_{fi}.csv",
        ], check=True, cwd=str(config.GLMM_FIT_R.parent))

        fold_fits = pd.read_csv(f"{tmp}/res_{fi}.csv").set_index("gene")
        Xa_te = np.column_stack([np.ones(len(te)), Xs[te]])
        for g in genes:
            if g not in fold_fits.index or not bool(fold_fits.loc[g, "ok"]):
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
                continue  # should not happen: cascade always resolves fixed_alpha by "intercept"
            y_te = Y[te, name2col[g]]
            parts[g].append(dict(y=y_te.astype(np.float32), mu=mu.astype(np.float32),
                                 alpha=np.asarray(alpha, dtype=np.float32),
                                 tau2=np.full(len(te), float(row["tau2"]), dtype=np.float32)))
    ppc = {}
    for g, ps in parts.items():
        if not ps:
            continue
        ppc[g] = dict(y=np.concatenate([p["y"] for p in ps]), mu=np.concatenate([p["mu"] for p in ps]),
                      alpha=np.concatenate([p["alpha"] for p in ps]), tau2=np.concatenate([p["tau2"] for p in ps]),
                      family="negbin", stage=stage_of[g])
    return ppc


def pooled_ppc_at_threshold(Xs, Y, batch, gene_names, genes, folds, tmp):
    """Returns dict[gene] = dict(y, mu, alpha, tau2, family, stage='pool'), same
    per-sample schema as individual_cascade_ppc / cv_glmm_engine.py's cv_ppc.pkl."""
    Path(tmp).mkdir(parents=True, exist_ok=True)
    name2col = {g: i for i, g in enumerate(gene_names)}
    parts = {g: [] for g in genes}
    for fi, (tr, te) in enumerate(folds):
        pd.DataFrame(Xs[tr], columns=config.BIAS_COLUMNS).to_csv(f"{tmp}/Xp_{fi}.csv.gz")
        Y_tr = Y[tr][:, [name2col[g] for g in genes]]
        pd.DataFrame(Y_tr, columns=genes).to_csv(f"{tmp}/Yp_{fi}.csv.gz")
        pd.DataFrame({"Batch_ID": batch[tr]}).to_csv(f"{tmp}/batchp_{fi}.csv.gz")
        pd.DataFrame({"gene": genes}).to_csv(f"{tmp}/genesp_{fi}.csv", index=False)
        subprocess.run([
            "Rscript", str(config.GLMM_FIT_POOL_R), "--x", f"{tmp}/Xp_{fi}.csv.gz", "--y", f"{tmp}/Yp_{fi}.csv.gz",
            "--batch", f"{tmp}/batchp_{fi}.csv.gz", "--genes", f"{tmp}/genesp_{fi}.csv",
            "--rare-overdisp-thr", str(MP["rare_overdisp_thr"]), "--out", f"{tmp}/resp_{fi}.json",
        ], check=True, cwd=str(config.GLMM_FIT_POOL_R.parent))
        with open(f"{tmp}/resp_{fi}.json") as f:
            fit = json.load(f)
        if not fit["ok"]:
            continue

        beta = np.asarray(fit["beta"])
        tau2 = float(fit["tau2"]) if fit.get("tau2") is not None else 0.0
        eps = 1.0 / (2 * len(tr))
        mean_hc = dict(zip(fit["gene"], fit["mean_hc"]))
        Xte = Xs[te]
        mult = np.exp(Xte @ beta[1:])
        if fit.get("mult_lo") is not None:
            mult = np.clip(mult, fit["mult_lo"], fit["mult_hi"])
        alpha_val = 1e-6 if fit["family"] == "poisson" else fit["alpha"]  # ~Poisson limit
        for g in genes:
            mu = np.clip((mean_hc[g] + eps) * np.exp(beta[0]) * mult, 1e-6, 1e8)
            y_te = Y[te, name2col[g]]
            parts[g].append(dict(y=y_te.astype(np.float32), mu=mu.astype(np.float32),
                                 alpha=np.full(len(te), alpha_val, dtype=np.float32),
                                 tau2=np.full(len(te), tau2, dtype=np.float32)))
    ppc = {}
    for g, ps in parts.items():
        if not ps:
            continue
        ppc[g] = dict(y=np.concatenate([p["y"] for p in ps]), mu=np.concatenate([p["mu"] for p in ps]),
                      alpha=np.concatenate([p["alpha"] for p in ps]), tau2=np.concatenate([p["tau2"] for p in ps]),
                      family="negbin", stage="pool")
    return ppc


def _mean_ll(ppc_gene):
    return float(marginal_nb_loglik(ppc_gene["y"], ppc_gene["mu"], ppc_gene["alpha"], ppc_gene["tau2"]).mean())


def main():
    print("Loading HC data...")
    Xs, Y, batch, gene_names = load_hc()
    nz = (Y > 0).sum(axis=0)
    max_t = max(THRESHOLDS)
    superset = gene_names[nz < max_t].tolist()
    print(f"Superset (nz<{max_t}): {len(superset)} genes")

    summary = pd.read_csv(config.ENGINE_MIXED_DIR / "training_summary.csv", index_col="gene")
    stage_of = summary["stage"].to_dict()
    superset = [g for g in superset if g in stage_of]  # need a known stage to refit fixed_stage

    n_hc = Xs.shape[0]
    folds = list(StratifiedKFold(MP["n_splits"], shuffle=True, random_state=42).split(np.zeros(n_hc), batch))

    print("Fitting individual per-gene cascade at its known stage (once, reused across all thresholds)...")
    ppc_individual = individual_cascade_ppc(Xs, Y, batch, gene_names, superset, stage_of, folds, f"{TMP}/individual")
    print(f"  {len(ppc_individual)}/{len(superset)} genes converged individually (some resolve to intercept)")
    ll_individual = {g: _mean_ll(d) for g, d in ppc_individual.items()}

    rows, gene_rows = [], []
    ppc_pooled_by_threshold = {}
    nz_map = dict(zip(gene_names, nz))
    for T in THRESHOLDS:
        genes_t = [g for g in superset if nz_map[g] < T]
        if len(genes_t) < 5:
            continue
        print(f"Pooled fit for nz<{T} ({len(genes_t)} genes)...")
        ppc_pooled = pooled_ppc_at_threshold(Xs, Y, batch, gene_names, genes_t, folds, f"{TMP}/pooled_{T}")
        ppc_pooled_by_threshold[T] = ppc_pooled
        ll_pooled = {g: _mean_ll(d) for g, d in ppc_pooled.items()}

        diffs = []
        for g in genes_t:
            if g not in ll_individual or g not in ll_pooled:
                continue
            d = ll_individual[g] - ll_pooled[g]
            diffs.append(d)
            gene_rows.append(dict(nz_threshold=T, gene=g, ll_individual=ll_individual[g],
                                  ll_pooled=ll_pooled[g], ll_diff=d))
        diffs = np.array(diffs)
        frac_individual_wins = float((diffs > 0).mean())
        rows.append(dict(nz_threshold=T, n_genes=len(diffs), median_ll_diff=float(np.median(diffs)),
                         mean_ll_diff=float(diffs.mean()), frac_individual_wins=frac_individual_wins))
        print(f"  nz<{T}: median(LL_individual-LL_pooled)={np.median(diffs):.4f}  "
             f"frac_individual_wins={frac_individual_wins:.3f}")

    df = pd.DataFrame(rows)
    gene_df = pd.DataFrame(gene_rows)
    df.to_csv(OUT / "pool_vs_individual_summary.csv", index=False)
    gene_df.to_csv(OUT / "pool_vs_individual_gene_level.csv", index=False)

    # Raw per-sample y/mu/alpha/tau2, same schema as cv_glmm_engine.py's cv_ppc.pkl --
    # lets later PPC/log-likelihood re-diagnosis (e.g. via ppc_mixed.py's simulate_mixed)
    # run against cached data instead of rerunning the R fits.
    with open(OUT / "pool_vs_individual_ppc_individual.pkl", "wb") as f:
        pickle.dump(ppc_individual, f)
    with open(OUT / "pool_vs_individual_ppc_pooled.pkl", "wb") as f:
        pickle.dump(ppc_pooled_by_threshold, f)

    apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(df["nz_threshold"], df["median_ll_diff"], "-o")
    axes[0].axhline(0, ls=":", color="gray")
    axes[0].set(xlabel="Pooling threshold (HC nonzero count)",
               ylabel="Median held-out LL(individual) - LL(pooled)\n(>0: individual model wins)")
    axes[1].plot(df["nz_threshold"], df["frac_individual_wins"], "-o", color="tab:orange")
    axes[1].axhline(0.5, ls=":", color="gray")
    axes[1].set(xlabel="Pooling threshold (HC nonzero count)", ylabel="Fraction of genes where individual wins")
    fig.tight_layout()
    fig.savefig(config.THRESHOLD_SWEEP_FIG_DIR / "pool_vs_individual.png", dpi=150)
    print(f"Saved -> {OUT}/pool_vs_individual_summary.csv, {OUT}/pool_vs_individual_gene_level.csv, "
         f"{OUT}/pool_vs_individual_ppc_individual.pkl, {OUT}/pool_vs_individual_ppc_pooled.pkl, "
         f"{config.THRESHOLD_SWEEP_FIG_DIR}/pool_vs_individual.png")


if __name__ == "__main__":
    main()
