suppressMessages({library(RUVSeq); library(edgeR)})

args <- commandArgs(trailingOnly = TRUE)
counts_path <- args[1]
controls_path <- args[2]
k <- as.integer(args[3])
out_path <- args[4]

counts <- as.matrix(read.csv(counts_path, row.names = 1, check.names = FALSE))
control_genes <- intersect(readLines(controls_path), rownames(counts))
stopifnot(length(control_genes) >= 5)

dge <- calcNormFactors(DGEList(counts = counts), method = "TMM")
tmm_log2 <- cpm(dge, normalized.lib.sizes = TRUE, log = TRUE, prior.count = 1)
tmm_log2[!is.finite(tmm_log2)] <- 0

W <- RUVg(tmm_log2, control_genes, k = k, isLog = TRUE, center = TRUE)$W
colnames(W) <- paste0("W_", seq_len(k))
rownames(W) <- colnames(counts)
write.csv(W, out_path)
