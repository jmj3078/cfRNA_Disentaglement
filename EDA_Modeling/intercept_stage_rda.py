"""Is the 'intercept' stage (final demotion-chain fallback -- zero covariate
conditioning) landing on genes that are genuinely covariate-insensitive
biologically, or on genes that COULD support covariate modeling but the
engine's fit machinery just failed to find it?

Route/stage is already fully decided by the trained engine
(engine_state_v2/training_summary.csv), so this only needs the existing
gene-wise vectorized Partial RDA (EDA/analysis_helper.compute_gene_wise_bias_rda)
run on the 282 intercept-stage genes, using the exact same covariates
(config.BIAS_COLUMNS) the model was fit on. A low Joint R^2 for these genes
(comparable to what nbi-stage genes near the same expression level show)
would support "biologically insensitive"; a Joint R^2 comparable to
successfully-fit nbi genes would instead point to a modeling-limitation
(likely low-nz identifiability) explanation, consistent with the
intercept-stage conservative-interpretation caveat already flagged for
downstream disease scoring.

Usage (run from EDA_Modeling/, cwd assumption per project convention):
    python intercept_stage_rda.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "EDA"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Modeling"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from analysis_helper import compute_gene_wise_bias_rda
from viz_style import apply_style

apply_style()

OUT_DIR = Path(__file__).resolve().parent / "Analysis_Results"
FIG_DIR = OUT_DIR / "Figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def main():
    summary = pd.read_csv(config.ENGINE_DIR_V2 / "training_summary.csv")

    intercept_genes = summary.loc[summary["stage"] == "intercept", "gene"].tolist()
    print(f"intercept-stage genes: {len(intercept_genes)}")
    nz_lo, nz_hi = summary.loc[summary["stage"] == "intercept", "nz"].agg(["min", "max"])
    control_genes = summary.loc[(summary["stage"] == "nbi") &
                                (summary["nz"] >= nz_lo) & (summary["nz"] <= nz_hi), "gene"].tolist()
    print(f"nz-matched nbi control genes ({nz_lo}<=nz<={nz_hi}): {len(control_genes)}")

    adata = sc.read_h5ad(config.H5AD_PATH)
    m = ((adata.obs["QC_Passed"] == True) & (adata.obs["Phenotype_Processed"].notna()) &
         (adata.obs["Phenotype_Processed"] != "Unknown") &
         (adata.obs["broad_protocol_category"] != "Exome-based (EB)"))
    adata = adata[m]

    all_genes = intercept_genes + control_genes
    adata_sub = adata[:, adata.var_names.isin(all_genes)].copy()
    print(f"Genes found in adata: {adata_sub.n_vars} / {len(all_genes)}")

    df_detail, _ = compute_gene_wise_bias_rda(
        adata_sub, bias_metrics=config.BIAS_COLUMNS, layer="CPM_log1p",
        phenotype_col="Phenotype_Processed", target_labels="Healthy Control",
        group_name="intercept_vs_nbi_control", min_expressed_frac=0.0,
    )
    df_detail["group"] = np.where(df_detail["Gene"].isin(intercept_genes), "intercept", "nbi_control")
    nz_map = summary.set_index("gene")["nz"]
    df_detail["nz"] = df_detail["Gene"].map(nz_map)
    df_detail.to_csv(OUT_DIR / "intercept_stage_rda.csv", index=False)

    print("\nJoint R^2 (all biases combined) by group:")
    print(df_detail.groupby("group")["Joint_R2_All_Biases"].describe().to_string())
    
    R2_EXCEPTION_THR = 0.15
    df_detail["is_exception"] = ((df_detail["group"] == "intercept") &
                                 (df_detail["Joint_R2_All_Biases"] > R2_EXCEPTION_THR))
    df_detail.to_csv(OUT_DIR / "intercept_stage_rda.csv", index=False)

    exceptions = df_detail[df_detail["is_exception"]].merge(
        summary[["gene", "fail_reason", "nbi_explode"]], left_on="Gene", right_on="gene", how="left"
    ).drop(columns="gene").sort_values("Joint_R2_All_Biases", ascending=False)
    exceptions.to_csv(OUT_DIR / "intercept_stage_exceptions.csv", index=False)
    print(f"\n{len(exceptions)} intercept-stage EXCEPTION genes (Joint R2 > {R2_EXCEPTION_THR}, "
         "fit failure rather than covariate-insensitivity -- recommend excluding from downstream scoring):")
    print(exceptions[["Gene", "nz", "Joint_R2_All_Biases", "fail_reason", "nbi_explode"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for grp, color in [("intercept", "#1b9e77"), ("nbi_control", "#c3c3c3")]:
        vals = df_detail.loc[df_detail["group"] == grp, "Joint_R2_All_Biases"]
        ax.hist(vals, bins=25, alpha=0.6, color=color, label=f"{grp} (n={len(vals)})", density=True)
    ax.axvline(R2_EXCEPTION_THR, color="#d95f02", ls="--", lw=1.5,
              label=f"exception threshold ({R2_EXCEPTION_THR})")
    ax.set(xlabel="Joint R2 (all 10 covariates, HC)", ylabel="density",
          title="Covariate-explained variance:\nintercept-stage vs nz-matched nbi-stage")
    ax.legend(fontsize=8)

    ax = axes[1]
    colors = np.where(df_detail["is_exception"], "#d95f02",
                      np.where(df_detail["group"] == "intercept", "#1b9e77", "#c3c3c3"))
    ax.scatter(df_detail["nz"], df_detail["Joint_R2_All_Biases"], c=colors, s=14, alpha=0.7)
    for _, r in exceptions.iterrows():
        ax.annotate(r["Gene"].split(".")[0], (r["nz"], r["Joint_R2_All_Biases"]),
                   fontsize=7, xytext=(4, 4), textcoords="offset points", color="#d95f02")

    int_nz = df_detail.loc[df_detail["group"] == "intercept", "nz"]
    ax.plot([int_nz.min(), int_nz.max()], [R2_EXCEPTION_THR] * 2,
           color="#d95f02", ls="--", lw=1.5, label="exception threshold (intercept-stage only)")
    ax.set(xlabel="nz (HC nonzero count)", ylabel="Joint R2")
    ax.title.set_fontsize(11)
    ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "intercept_stage_rda.png", dpi=150)
    plt.show()
    print(f"\nSaved -> {OUT_DIR}/intercept_stage_rda.csv (with is_exception column)")
    print(f"Saved -> {OUT_DIR}/intercept_stage_exceptions.csv")
    print(f"Saved -> {FIG_DIR}/intercept_stage_rda.png")

    # Raw HC count distribution (per-study) for the excluded genes -- do these
    # high-Joint-R2 exceptions actually look like the study-clustered,
    # non-smooth patterns already seen in the sigma-explode nz>200 cases, or is
    # this a different failure mode?
    adata_exc = adata[adata.obs["Phenotype_Processed"].astype(str) == "Healthy Control",
                      adata.var_names.isin(exceptions["Gene"])]
    X_exc = adata_exc.X.toarray() if hasattr(adata_exc.X, "toarray") else np.asarray(adata_exc.X)
    X_exc = np.round(X_exc).astype(np.float64)
    study = adata_exc.obs["Batch_ID"].astype(str).str.split("_Batch").str[0].values
    studies = sorted(set(study))
    cmap = plt.get_cmap("tab10")
    scolor = {s: cmap(i % 10) for i, s in enumerate(studies)}
    gsym = {g: adata.var.loc[g, "GeneName"] for g in exceptions["Gene"]}

    n_g = len(exceptions)
    ncols = 4
    nrows = -(-n_g // ncols)
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes2 = np.atleast_1d(axes2).flatten()
    for ax, (_, r) in zip(axes2, exceptions.iterrows()):
        g = r["Gene"]
        j = list(adata_exc.var_names).index(g)
        y = X_exc[:, j]
        for s in studies:
            ys = np.log1p(y[(study == s) & (y > 0)])
            if len(ys) < 2:
                continue
            ax.hist(ys, bins=20, alpha=0.55, color=scolor[s], label=s)
        ax.set_title(f"{gsym[g]}\n({g})  nz={int(r['nz'])}  R2={r['Joint_R2_All_Biases']:.2f}", fontsize=9)
        ax.set_xlabel("log1p(raw count) | count>0")
        ax.set_ylabel("n samples")
    for ax in axes2[n_g:]:
        ax.axis("off")
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="upper center", ncol=len(studies), bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "intercept_stage_exceptions_raw_hist.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {FIG_DIR}/intercept_stage_exceptions_raw_hist.png")


if __name__ == "__main__":
    main()
