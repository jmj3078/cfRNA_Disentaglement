from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "OpenAccess_nfcore"
PIPELINE_DIR = ROOT / "Saved_Pipeline"

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

# ---------------------------------------------------------------------------
# Modeling pipeline (single normative engine: NZ-gated demotion chain)
# ---------------------------------------------------------------------------
MODELING_DIR   = ROOT / "Modeling"
ENGINE_DIR     = MODELING_DIR / "engine_state"
CV_RESULTS_DIR = MODELING_DIR / "CV_Results"
CV_FIG_DIR     = CV_RESULTS_DIR / "Figures"
Z_SCORES_DIR   = MODELING_DIR / "Z_scores"
GSEA_DIR       = MODELING_DIR / "GSEA"
GSEA_FIG_DIR   = GSEA_DIR / "Figures"
RARE_REF       = Z_SCORES_DIR / "rare_event_ref.pkl"
R_HELPER       = MODELING_DIR / "gamlss.r"
DISPERSION_TREND_PATH = ENGINE_DIR / "dispersion_trend.json"

H5AD_PATH = PATHS["merged_qc"]   # normative modeling

# scoring 산출 Z-score 행렬 (disease_scoring/scoring.py) → Z_scores/ 디렉토리
Z_DISEASE      = Z_SCORES_DIR / "Z_disease.npy"
Z_SAMPLE_NAMES = Z_SCORES_DIR / "Z_sample_names.npy"
Z_GENE_NAMES   = Z_SCORES_DIR / "Z_gene_names.npy"
Z_HC           = Z_SCORES_DIR / "Z_hc.npy"
Z_HC_NAMES     = Z_SCORES_DIR / "Z_hc_names.npy"
Z_RARE_DISEASE    = Z_SCORES_DIR / "Z_rare_disease.npy"
Z_RARE_HC         = Z_SCORES_DIR / "Z_rare_hc.npy"
Z_RARE_GENE_NAMES = Z_SCORES_DIR / "Z_rare_gene_names.npy"
RARE_GLM       = ENGINE_DIR / "rare_glm.pkl"

COHORT_COMPARE_DIR     = MODELING_DIR / "Cohort_Compare"
COHORT_COMPARE_GSEA_DIR = COHORT_COMPARE_DIR / "GSEA"
COHORT_COMPARE_FIG_DIR  = COHORT_COMPARE_DIR / "Figures"

BENCHMARK_DIR      = MODELING_DIR / "Benchmark"
DESEQ2_RESULTS_DIR = BENCHMARK_DIR / "deseq2_results"
DESEQ2_GSEA_DIR    = BENCHMARK_DIR / "deseq2_gsea"
DESEQ2_COV_RESULTS_DIR = BENCHMARK_DIR / "deseq2_covariate_results"
DESEQ2_COV_GSEA_DIR    = BENCHMARK_DIR / "deseq2_covariate_gsea"

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

# Genes excluded from DOWNSTREAM analysis only (gene selection / GSEA / signatures).
# Scoring still scores and saves all genes; data_prep.load_disease_filtered drops these
# columns. Seeded with the intercept-stage exception genes (EDA_Modeling: irls_diverged +
# nbi sigma-explode, study-driven mean-shift artifacts rather than covariate-insensitive).
EXCLUDED_GENES = {
    "ENSG00000262526.2", "ENSG00000255073.8", "ENSG00000284779.3", "ENSG00000271723.5",
    "ENSG00000280148.1", "ENSG00000146385.2", "ENSG00000214107.10",
}

MODELING_PARAMS = {
    # analysis / downstream (shared by pipeline modules)
    "ood_percentile":  95,
    "min_samples":     5,
    "z_flag":          3.0,
    "stratify_col":    "Batch_ID",
    "n_splits":        5,
    "gsea_gene_sets":  ["KEGG_2021_Human", "GO_Biological_Process_2023", "Reactome_2022"],
    "gsea_fdr_thr":    0.05,
    "gsea_top_n":      30,
    "gsea_perm":       100,
    "gsea_seed":       42,
    "sig_cap_per_theme": 8,
    "emap_sim_thr":    0.50,
    # engine (NZ-gated demotion chain)
    "nz_a_max":          7,
    "trend_min_nz":      30,     # min HC nonzero samples for a gene to enter dispersion-trend fitting
    "alpha_floor":       1e-2,
    "alpha_cap":         50.0,
    "ridge_lambda_sigma": 0.05,  # L2 penalty on stage nbi's sigma submodel only (gamlss ridgeVec); mu is never penalized
    "outlier_z":         5.0,
    "max_outlier_iter":  3,
    "max_remove_frac":   0.05,
    "beta_explode_thr":  3.0,    # |slope coef| threshold flagging non-identifiable mean fit
    "gaic_k":            2.0,    # GAIC penalty weight (k=2 == AIC convention, gamlss default)
    "rare_overdisp_thr": 2.0,
}
