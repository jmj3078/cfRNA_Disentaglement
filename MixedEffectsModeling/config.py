from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
H5AD_PATH = ROOT / "OpenAccess_nfcore" / "Merged_Processed_AnnData_with_Batch_Biases_QC_Status.h5ad"
GAMLSS_R_HELPER = ROOT / "Modeling" / "gamlss.r"

SPIKE_DIR = Path(__file__).resolve().parent / "Spike_Results"

# Production engine paths -- fully independent of root config.py/Modeling/ by
# design (see docs/superpowers/specs/2026-07-22-mixed-effects-production-engine-design.md).
_HERE = Path(__file__).resolve().parent
ENGINE_MIXED_DIR        = _HERE / "engine_state_mixed"
CV_MIXED_DIR            = _HERE / "CV_Results_mixed"
CV_MIXED_FIG_DIR        = CV_MIXED_DIR / "Figures"
THRESHOLD_SWEEP_DIR     = _HERE / "Threshold_Sweep"
THRESHOLD_SWEEP_FIG_DIR = THRESHOLD_SWEEP_DIR / "Figures"
GLMM_HELPERS_R = _HERE / "core" / "glmm_helpers.R"
GLMM_FIT_R     = _HERE / "core" / "glmm_fit.R"
GLMM_FIT_POOL_R = _HERE / "core" / "glmm_fit_pool.R"
POOL_SWEEP_R   = _HERE / "pool_threshold_sweep.R"
DISPERSION_TREND_PATH = ENGINE_MIXED_DIR / "dispersion_trend.json"

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
    "trend_min_nz": 30,
    "rare_overdisp_thr": 2.0,
    "alpha_floor": 1e-2,
    "alpha_cap": 50.0,
    "n_splits": 5,
}
