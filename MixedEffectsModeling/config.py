from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
H5AD_PATH = ROOT / "OpenAccess_nfcore" / "Merged_Processed_AnnData_with_Batch_Biases_QC_Status.h5ad"
GAMLSS_R_HELPER = ROOT / "Modeling" / "gamlss.r"

_HERE = Path(__file__).resolve().parent

ENGINE_MIXED_DIR        = _HERE / "engine_state_mixed"
CV_MIXED_DIR            = _HERE / "CV_Results_mixed"
CV_MIXED_FIG_DIR        = CV_MIXED_DIR / "Figures"
LOBO_MIXED_DIR          = _HERE / "LOBO_Results_mixed"
ZSCORES_MIXED_DIR       = _HERE / "Z_scores_mixed"
THRESHOLD_SWEEP_DIR     = _HERE / "Threshold_Sweep"
THRESHOLD_SWEEP_FIG_DIR = THRESHOLD_SWEEP_DIR / "Figures"
PCIS_CAL_DIR            = _HERE / "PCIS_Calibration"
PCIS_CAL_FIG_DIR        = PCIS_CAL_DIR / "Figures"
PATHWAY_CONV_DIR        = _HERE / "PathwayConvergence"
PATHWAY_CONV_FIG_DIR    = PATHWAY_CONV_DIR / "Figures"
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

# Pooling cutoff. Set from the nz_a_max=0 run (every gene through the individual cascade).
# Raised 25 -> 31 (2026-07-29): CV fold-level convergence (all 5 folds) drops to ~0.6 in the
# nz 25-30 bin, below the bar the nz_a_max choice is meant to hold.
NZ_A_MAX = 31
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
    "tau2_max": 3.0,
    "disp_intercept_max": 10.0,
    "pcis_cut": 2.25,
    "max_outlier_frac": 0.05,
    "chunk_size": 200,
    "cores": 12,
}

# Gene- vs pathway-level deviation convergence (4_gene_enrichment.ipynb): patient-level BH-sig
# genes are heterogeneous, but does the same deviation converge onto shared pathways? Mirrors
# Wolfers 2018 JAMA Psych / Segal 2023 Nat Neurosci deviation-overlap design (see
# EDA/normative_modeling_literature.md).
PATHWAY_CONV_PARAMS = {
    "gene_sets": ["KEGG_2021_Human", "Reactome_2022"],
    "min_pathway_size": 5,
    "n_null_perm": 200,
    "fdr_q": 0.05,
    "seed": 42,
    # Name-based keyword match misses pathways that are translation-dominated by gene COMPOSITION but
    # unrelated by NAME (Influenza Infection, SLIT/ROBO signaling, Cellular Response To Starvation all
    # came out >45% ribosomal-protein genes empirically) -- so exclusion is composition-based: any
    # pathway sharing > ribo_frac_max of its genes with the reference KEGG "Ribosome" set is dropped.
    # Keyword list stays as a fast belt-and-suspenders for OXPHOS/neurodegeneration, which the
    # ribosome-composition check does not catch (feedback_gsea_interpretation).
    "ribo_reference_term": "Ribosome",
    "ribo_frac_max": 0.15,
    "max_pathway_size_select": 300,
    "top_k_pathways": 6,
    "redundancy_jaccard_max": 0.5,
    "exclude_keywords": [
        "oxidative phosphorylation", "electron transport", "respiratory chain",
        "alzheimer", "parkinson", "huntington", "prion disease", "amyotrophic lateral sclerosis",
    ],
}
