from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
H5AD_PATH = ROOT / "OpenAccess_nfcore" / "Merged_Processed_AnnData_with_Batch_Biases_QC_Status.h5ad"
GAMLSS_R_HELPER = ROOT / "Modeling" / "gamlss.r"

_HERE = Path(__file__).resolve().parent

ENGINE_MIXED_DIR        = _HERE / "engine_state_mixed"
CV_MIXED_DIR            = _HERE / "CV_Results_mixed"
CV_MIXED_FIG_DIR        = CV_MIXED_DIR / "Figures"
THRESHOLD_SWEEP_DIR     = _HERE / "Threshold_Sweep"
THRESHOLD_SWEEP_FIG_DIR = THRESHOLD_SWEEP_DIR / "Figures"
PCIS_CAL_DIR            = _HERE / "PCIS_Calibration"
PCIS_CAL_FIG_DIR        = PCIS_CAL_DIR / "Figures"
LOG_DIR                 = _HERE / "Logs"
GLMM_HELPERS_R = _HERE / "core" / "glmm_helpers.R"
PCIS_NULL_R    = _HERE / "core" / "pcis_null.R"
GLMM_FIT_R     = _HERE / "core" / "glmm_fit.R"
GLMM_FIT_POOL_R = _HERE / "core" / "glmm_fit_pool.R"
POOL_SWEEP_R   = _HERE / "pool_threshold_sweep.R"
DISPERSION_TREND_PATH = ENGINE_MIXED_DIR / "dispersion_trend.json"
DISP_PRIOR_PATH = ENGINE_MIXED_DIR / "disp_prior.json"

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

NZ_A_MAX = 40
MIN_HC_BATCH_SIZE = 5

SPIKE_PARAMS = {
    "beta_explode_thr": 3.0,
    "seed": 42,
    "rare_overdisp_thr": 2.0,
    "alpha_floor": 1e-2,
    "alpha_cap": 50.0,
    "n_splits": 5,
    "trend_min_nz": 30,
}

# Empirical-Bayes dispersion shrinkage + PCIS (Prior-Conditioned Impact Score)
# outlier removal
EB_PARAMS = {
    "calib_n_genes": 2000,
    "calib_n_strata": 10,
    "tau_floor": 1e-3,
}

FIT_PARAMS = {
    "beta_explode_thr": SPIKE_PARAMS["beta_explode_thr"],
    "tau2_max": SPIKE_PARAMS["beta_explode_thr"] ** 2,
    "disp_intercept_max": 10.0,
    "pcis_cut": 2.25,
    "max_outlier_frac": 0.05,
    "chunk_size": 200,
    "cores": 12,
}
