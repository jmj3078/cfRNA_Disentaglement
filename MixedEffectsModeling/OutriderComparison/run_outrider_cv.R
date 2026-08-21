suppressPackageStartupMessages({
  library(OUTRIDER)
  library(data.table)
  library(jsonlite)
})

dir <- "/project/cfRNA_NormativeModeling/MixedEffectsModeling/OutriderComparison"

hc_counts <- fread(file.path(dir, "hc_counts.csv"))
hc_genes <- hc_counts[[1]]
hc_mat <- as.matrix(hc_counts[, -1]); rownames(hc_mat) <- hc_genes

folds <- fromJSON(file.path(dir, "cv_folds.json"))
q <- 20

for (fi in names(folds)) {
  cat(sprintf("=== fold %s ===\n", fi))
  train_names <- folds[[fi]]$train
  test_names <- folds[[fi]]$test
  mat <- hc_mat[, c(train_names, test_names), drop = FALSE]

  ods <- OutriderDataSet(countData = mat)
  ods <- ods[rowSums(counts(ods)) > 0 & rowSums(counts(ods)) >= ncol(ods)/100, ]
  ods <- estimateSizeFactors(ods)
  set.seed(42)
  ods <- controlForConfounders(ods, q = q, iterations = 5)
  ods <- fit(ods)
  ods <- computeZscores(ods)
  z <- t(as.matrix(assay(ods, "zScore")))

  z_test <- z[test_names, , drop = FALSE]
  saveRDS(z_test, file.path(dir, paste0("cv_z_test_fold", fi, ".rds")))
}
cat("ALL DONE\n")
