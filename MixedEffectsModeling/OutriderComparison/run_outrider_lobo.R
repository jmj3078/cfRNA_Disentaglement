suppressPackageStartupMessages({
  library(OUTRIDER)
  library(data.table)
  library(jsonlite)
})

dir <- "/project/cfRNA_NormativeModeling/MixedEffectsModeling/OutriderComparison"

hc_counts <- fread(file.path(dir, "hc_counts.csv"))
hc_genes <- hc_counts[[1]]
hc_mat <- as.matrix(hc_counts[, -1]); rownames(hc_mat) <- hc_genes
hc_meta <- fread(file.path(dir, "hc_meta.csv"))
stopifnot(identical(colnames(hc_mat), hc_meta$sample))

dis_counts <- fread(file.path(dir, "disease_counts.csv"))
dis_genes <- dis_counts[[1]]
dis_mat <- as.matrix(dis_counts[, -1]); rownames(dis_mat) <- dis_genes
stopifnot(identical(hc_genes, dis_genes))

test_meta <- fromJSON(file.path(dir, "lobo_test_meta.json"))
batches <- names(test_meta)

q <- 20
out_rows <- list()

for (b in batches) {
  safe <- gsub(" ", "_", b)
  cat(sprintf("=== %s ===\n", b))
  test_names <- test_meta[[b]]$test_names
  test_is_hc <- test_meta[[b]]$test_is_hc

  train_hc_names <- hc_meta$sample[hc_meta$batch != b]
  train_mat <- hc_mat[, train_hc_names, drop = FALSE]

  test_from_hc <- intersect(test_names, colnames(hc_mat))
  test_from_dis <- intersect(test_names, colnames(dis_mat))
  test_mat <- cbind(hc_mat[, test_from_hc, drop = FALSE], dis_mat[, test_from_dis, drop = FALSE])
  test_mat <- test_mat[, test_names, drop = FALSE]  # restore original order

  mat <- cbind(train_mat, test_mat)
  ods <- OutriderDataSet(countData = mat)
  ods <- ods[rowSums(counts(ods)) > 0 & rowSums(counts(ods)) >= ncol(ods)/100, ]
  ods <- estimateSizeFactors(ods)
  set.seed(42)
  ods <- controlForConfounders(ods, q = q, iterations = 5)
  ods <- fit(ods)
  ods <- computeZscores(ods)
  z <- t(as.matrix(assay(ods, "zScore")))  # samples x genes

  z_test <- z[test_names, , drop = FALSE]
  saveRDS(z_test, file.path(dir, paste0("z_test_", safe, ".rds")))
  out_rows[[b]] <- data.table(batch = b, sample = test_names, is_hc = test_is_hc)
}

fwrite(rbindlist(out_rows), file.path(dir, "outrider_lobo_membership.csv"))
cat("ALL DONE\n")
