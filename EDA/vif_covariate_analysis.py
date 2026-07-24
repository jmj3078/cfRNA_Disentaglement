import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import BIAS_COLUMNS, H5AD_PATH, ROOT

parent_dir = str(_ROOT)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from viz_style import apply_style

RESULTS_DIR = ROOT / "EDA" / "Analysis_Results"
OUT_CSV = RESULTS_DIR / "vif_covariate_candidates.csv"
OUT_FIG = RESULTS_DIR / "vif_covariate_candidates.png"
OUT_FIG_CURRENT = RESULTS_DIR / "vif_current_10.png"

# QC/bias metrics not currently used as model covariates (BIAS_COLUMNS).
# umi_tools-* excluded: only 13.5% coverage (tool not run for most batches).
EXTRA_CANDIDATES = [
    "NP80",
    "picard_mark_duplicates-PERCENT_DUPLICATION",
    "samtools_stats-error_rate",
    "samtools_stats-reads_mapped_percent",
    "samtools_stats-reads_properly_paired_percent",
    "samtools_stats-reads_MQ0_percent",
    "samtools_stats-insert_size_average",
    "salmon-percent_mapped",
    "star-mapped_percent",
    "star-uniquely_mapped_percent",
    "fastp-pct_duplication",
    "fastp-after_filtering_q30_rate",
    "fastp-after_filtering_gc_content",
    "fastp-pct_surviving",
    "fastp-pct_adapter",
    "cds_exons_tag_pct",
    "5_utr_exons_tag_pct",
    "3_utr_exons_tag_pct",
    "introns_tag_pct",
    "tss_up_1kb_tag_pct",
    "tes_down_1kb_tag_pct",
    "other_intergenic_tag_pct",
    "Total Centrifugation Force (g)",
    "Sample Volume (mL)",
]


def compute_vif(df):
    X = df.assign(const=1.0).values
    vifs = [variance_inflation_factor(X, i) for i in range(df.shape[1])]
    return pd.Series(vifs, index=df.columns).sort_values(ascending=False)


def plot_current_vif(result):
    apply_style()
    import matplotlib.pyplot as plt

    vif = result.loc[BIAS_COLUMNS, "VIF_current_10_only"].sort_values()

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#c0392b" if v >= 5 else "#2c7bb6" for v in vif.values]
    ax.barh(vif.index, vif.values, color=colors)
    ax.axvline(5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("VIF")
    ax.set_title("VIF among current 10 covariates")
    fig.tight_layout()
    fig.savefig(OUT_FIG_CURRENT)
    print(f"Saved: {OUT_FIG_CURRENT}")


def main():
    if OUT_CSV.is_file():
        print(f"Cache found: {OUT_CSV}, loading instead of recomputing.")
        result = pd.read_csv(OUT_CSV, index_col=0)
        print(result)
        plot_current_vif(result)
        return result

    adata = ad.read_h5ad(H5AD_PATH, backed="r")
    obs = adata.obs[adata.obs["QC_Passed"] == True]

    cols = BIAS_COLUMNS + EXTRA_CANDIDATES
    df = obs[cols].apply(pd.to_numeric, errors="coerce").dropna()
    print(f"Complete-case n = {len(df)} / {len(obs)} ({100 * len(df) / len(obs):.1f}%)")

    constant_cols = df.columns[df.std() == 0].tolist()
    if constant_cols:
        print(f"Dropping constant columns (zero variance, undefined VIF): {constant_cols}")
        df = df.drop(columns=constant_cols)

    vif_current = compute_vif(df[BIAS_COLUMNS]).rename("VIF_current_10_only")
    vif_extended = compute_vif(df).rename("VIF_with_extra_candidates")

    result = pd.concat([vif_current, vif_extended], axis=1)
    result["group"] = ["current_10" if c in BIAS_COLUMNS else "extra_candidate" for c in result.index]
    result = result.sort_values("VIF_with_extra_candidates", ascending=False)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV)
    print(result)

    apply_style()
    import matplotlib.pyplot as plt

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"Covariate candidate correlation matrix (n={len(df)})")
    fig.tight_layout()
    fig.savefig(OUT_FIG)
    print(f"Saved: {OUT_FIG}")

    plot_current_vif(result)

    return result


if __name__ == "__main__":
    main()
