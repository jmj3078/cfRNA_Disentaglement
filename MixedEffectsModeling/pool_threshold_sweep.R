suppressPackageStartupMessages(library(optparse))

# sys.frame(1)$ofile only works inside source()'d files, not top-level Rscript
# execution -- parse --file= from commandArgs instead, robust to invocation cwd
# (matches glmm_fit.R).
.args <- commandArgs(trailingOnly = FALSE)
.script_path <- sub("--file=", "", grep("--file=", .args, value = TRUE))
source(file.path(dirname(normalizePath(.script_path)), "glmm_helpers.R"))

opt <- parse_args(OptionParser(option_list = list(
  make_option("--x", type = "character"), make_option("--y", type = "character"),
  make_option("--batch", type = "character"), make_option("--folds", type = "character"),
  make_option("--thresholds", type = "character", default = "3,5,7,10,15,20,25,30,40,50"),
  make_option("--out", type = "character")
)))

X <- as.matrix(read.csv(opt$x, row.names = 1))
Y <- as.matrix(read.csv(opt$y, row.names = 1))
batch <- read.csv(opt$batch, row.names = 1)[[1]]
folds <- read.csv(opt$folds)  # columns: sample_idx (0-based), fold
nz <- colSums(Y > 0)
thresholds <- as.numeric(strsplit(opt$thresholds, ",")[[1]])
n_hc <- nrow(X)
eps <- 1 / (2 * n_hc)

rows <- list()
for (T in thresholds) {
  cols <- which(nz < T)
  if (length(cols) < 5) next
  w1s <- c()
  for (fi in unique(folds$fold)) {
    tr <- folds$sample_idx[folds$fold != fi] + 1
    te <- folds$sample_idx[folds$fold == fi] + 1
    mean_hc <- colMeans(Y[tr, cols, drop = FALSE])
    fit <- fit_pooled_glmm(Y[tr, cols, drop = FALSE], X[tr, , drop = FALSE], batch[tr], mean_hc, eps, 2.0)
    if (!isTRUE(fit$ok)) next
    Xc_te <- cbind(1, X[te, , drop = FALSE])
    mu <- (mean_hc[rep(seq_along(cols), each = length(te))] + eps) *
      exp(Xc_te[rep(seq_along(te), length(cols)), , drop = FALSE] %*% fit$beta)
    y_te <- as.vector(Y[te, cols, drop = FALSE])
    theta <- if (fit$family == "poisson") NA else 1 / fit$alpha
    p <- if (is.na(theta)) NA else theta / (theta + mu)
    u <- if (fit$family == "poisson") ppois(y_te, mu) else pnbinom(y_te, theta, p)
    z <- qnorm(pmin(pmax(u, 1e-8), 1 - 1e-8))
    w1s <- c(w1s, mean(abs(sort(z) - qnorm(ppoints(length(z))))))
  }
  if (length(w1s) == 0) next
  rows[[length(rows) + 1]] <- list(nz_threshold = T, n_genes = length(cols),
    w1_median = median(w1s), w1_p90 = quantile(w1s, 0.9, names = FALSE))
  cat(sprintf("nz<%d: n_genes=%d w1_median=%.3f\n", T, length(cols), median(w1s)))
}
write.csv(do.call(rbind, lapply(rows, as.data.frame)), opt$out, row.names = FALSE)
cat("DONE\n")
