from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "OpenAccess_nfcore"

PATHS = {
    "merged_raw":    DATA_DIR / "Merged_Processed_AnnData.h5ad",
    "merged_biases": DATA_DIR / "Merged_Processed_AnnData_with_Batch_Biases.h5ad",
    "merged_qc":     DATA_DIR / "Merged_Processed_AnnData_with_Batch_Biases_QC_Status.h5ad",
}

PARAMS = {
    "min_study_samples": 10,
    "n_top_genes":       2000,
    "n_pcs":             50,
    "loess_frac":        0.7,
    "n_bins":            20,
    "outlier_pct":       99,
    "min_expressed":     50,
}

H5AD_PATH = PATHS["merged_qc"]   # normative modeling

CTRL_COMP_DIR     = ROOT / "EDA" / "Analysis_Results" / "Control_Composition"
CTRL_COMP_W_DIR   = CTRL_COMP_DIR / "ruvg_W"
CTRL_COMP_EXPR_DIR = CTRL_COMP_DIR / "expr"
CTRL_COMP_STAT_DIR = CTRL_COMP_DIR / "tstats"
CTRL_COMP_FIG_DIR = CTRL_COMP_DIR / "Figures"
CTRL_COMP_DESEQ2_DIR = CTRL_COMP_DIR / "deseq2_stats"

BIAS_COLUMNS = [
    "log(Total Reads)",
    "Spliced Reads (%)",
    "gDNA Contamination (Intron/Exon)",
    "rRNA Fraction",
    "RNA Degradation (3' Bias)",
    "Platelet Score",
    "GC Bias",
    "Gene Length Bias",
    "NG80",
    "(NP80/NG80)",
]

# Used by EDA/control_composition/run_control_composition.py's MahalanobisFilter.
MODELING_PARAMS = {"ood_percentile": 95}
