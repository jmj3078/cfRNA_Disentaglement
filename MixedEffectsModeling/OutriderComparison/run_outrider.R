suppressPackageStartupMessages({
  library(OUTRIDER)
  library(data.table)
})

dir <- "/project/cfRNA_NormativeModeling/MixedEffectsModeling/OutriderComparison"
counts <- fread(file.path(dir, "hc_counts.csv"))
genes <- counts[[1]]
mat <- as.matrix(counts[, -1])
rownames(mat) <- genes
mode(mat) <- "integer"

meta <- fread(file.path(dir, "hc_meta.csv"))
stopifnot(identical(colnames(mat), meta$sample))

gene_len <- fread(file.path(dir, "gene_lengths.csv"))
setnames(gene_len, c("gene", "length"))
gene_len <- gene_len[match(genes, gene_len$gene)]
stopifnot(identical(gene_len$gene, genes))

ods <- OutriderDataSet(countData = mat)
mcols(ods)$basepairs <- gene_len$length
ods <- filterExpression(ods, filterGenes = TRUE, fpkmCutoff = 1)
ods <- estimateSizeFactors(ods)

set.seed(42)
q <- 20
bp <- MulticoreParam(workers = 4, stop.on.error = FALSE)
ods <- controlForConfounders(ods, q = q, iterations = 5, BPPARAM = bp)
ods <- fit(ods, BPPARAM = bp)
ods <- computePvalues(ods, alternative = "two.sided")
ods <- computeZscores(ods)

pv <- as.matrix(assay(ods, "pValue"))
padj <- as.matrix(assay(ods, "padjust"))
z <- as.matrix(assay(ods, "zScore"))

aberrant <- padj < 0.05
per_sample_aberrant <- colSums(aberrant, na.rm = TRUE)
per_sample_mean_absz <- colMeans(abs(z), na.rm = TRUE)
per_sample_pos_frac <- colMeans(z > 0, na.rm = TRUE)

out <- data.table(sample = colnames(mat), batch = meta$batch,
                   n_aberrant = per_sample_aberrant,
                   mean_absz = per_sample_mean_absz,
                   pos_frac = per_sample_pos_frac)
fwrite(out, file.path(dir, "outrider_per_sample.csv"))
saveRDS(ods, file.path(dir, "ods_fit.rds"))
cat("DONE q=", q, "\n")
