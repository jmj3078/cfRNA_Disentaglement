suppressMessages({library(RUVSeq)})

args <- commandArgs(trailingOnly = TRUE)
tmm_path <- args[1]
controls_path <- args[2]
subsets_path <- args[3]
k <- as.integer(args[4])
out_dir <- args[5]

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
tmm_log2 <- as.matrix(read.csv(gzfile(tmm_path), row.names = 1, check.names = FALSE))
control_genes <- intersect(readLines(controls_path), rownames(tmm_log2))
stopifnot(length(control_genes) >= 5)
subsets <- read.csv(subsets_path, colClasses = "character")

ids <- unique(subsets$subset_id)
for (i in seq_along(ids)) {
    out_path <- file.path(out_dir, paste0(ids[i], ".csv"))
    if (file.exists(out_path)) next
    samples <- subsets$sample[subsets$subset_id == ids[i]]
    set_ruvg <- RUVg(tmm_log2[, samples, drop = FALSE], control_genes, k = k,
                     isLog = TRUE, center = TRUE)
    W <- set_ruvg$W
    colnames(W) <- paste0("W_", seq_len(k))
    write.csv(data.frame(sample = samples, W, check.names = FALSE), out_path, row.names = FALSE)
    if (i == 1) {
        write.csv(set_ruvg$normalizedCounts,
                  gzfile(file.path(out_dir, "_check_normalizedCounts.csv.gz")))
    }
    if (i %% 50 == 0) message(sprintf("[ruvg] %d/%d", i, length(ids)))
}
message(sprintf("[ruvg] done %d subsets", length(ids)))
