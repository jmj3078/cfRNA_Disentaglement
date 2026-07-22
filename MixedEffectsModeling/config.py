from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
H5AD_PATH = ROOT / "OpenAccess_nfcore" / "Merged_Processed_AnnData_with_Batch_Biases_QC_Status.h5ad"
GAMLSS_R_HELPER = ROOT / "Modeling" / "gamlss.r"

SPIKE_DIR = Path(__file__).resolve().parent / "Spike_Results"

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

STRATIFY_COL = "Batch_ID"

SPIKE_PARAMS = {
    "outlier_z": 5.0,
    "max_outlier_iter": 3,
    "max_remove_frac": 0.05,
    "ridge_lambda_sigma": 0.05,
    "beta_explode_thr": 3.0,
    "gaic_k": 2.0,
    "n_pilot_genes": 40,
    "seed": 42,
}
