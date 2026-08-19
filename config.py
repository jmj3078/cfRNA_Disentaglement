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

EDA_RESULTS_DIR   = ROOT / "EDA" / "Analysis_Results"
EDA_OVERVIEW_DIR  = EDA_RESULTS_DIR / "Overview"
EDA_QC_DIR        = EDA_RESULTS_DIR / "QC"
EDA_PCA_NOHVG_DIR = EDA_RESULTS_DIR / "PCA" / "NoHVG"
EDA_PCA_HVG_DIR   = EDA_RESULTS_DIR / "PCA" / "HVG"
EDA_RDA_DIR       = EDA_RESULTS_DIR / "RDA"
EDA_RDA_NOHVG_DIR = EDA_RDA_DIR / "NoHVG"
EDA_RDA_HVG_DIR   = EDA_RDA_DIR / "HVG"
EDA_RDA_HC_DIR    = EDA_RDA_DIR / "HC"
EDA_BIAS_BATCH_DIR = EDA_RESULTS_DIR / "BiasBatch"
EDA_BIAS_PHENOTYPE_DIR = EDA_RESULTS_DIR / "BiasPhenotype"
EDA_GENE_BIAS_HC_DIR = EDA_RESULTS_DIR / "GeneBiasHC"
EDA_VIF_DIR       = EDA_RESULTS_DIR / "VIF"

CTRL_COMP_DIR     = EDA_RESULTS_DIR / "Control_Composition"
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
