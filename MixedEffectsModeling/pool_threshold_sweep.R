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
  make_option("--thresholds", type = "character", default = "3,5,7,10,15,20,25,30"),
  make_option("--out", type = "character"),
  make_option("--gene-out", type = "character", default = NULL)
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
gene_rows <- list()
for (T in thresholds) {
  cols <- which(nz < T)
  if (length(cols) < 5) next
  gene_names <- colnames(Y)[cols]
  gene_z <- setNames(vector("list", length(cols)), gene_names)
  fold_diag <- list()

  for (fi in unique(folds$fold)) {
    tr <- folds$sample_idx[folds$fold != fi] + 1
    te <- folds$sample_idx[folds$fold == fi] + 1
    mean_hc <- colMeans(Y[tr, cols, drop = FALSE])
    fit <- fit_pooled_glmm(Y[tr, cols, drop = FALSE], X[tr, , drop = FALSE], batch[tr], mean_hc, eps, 2.0)
    fold_diag[[length(fold_diag) + 1]] <- list(fold = fi, ok = isTRUE(fit$ok),
      family = if (isTRUE(fit$ok)) fit$family else NA,
      tau2 = if (isTRUE(fit$ok)) fit$tau2 else NA,
      overdisp_ratio = if (isTRUE(fit$ok)) fit$overdisp_ratio else NA)
    if (!isTRUE(fit$ok)) next

    Xc_te <- cbind(1, X[te, , drop = FALSE])
    mu <- (mean_hc[rep(seq_along(cols), each = length(te))] + eps) *
      exp(Xc_te[rep(seq_along(te), length(cols)), , drop = FALSE] %*% fit$beta)
    y_te <- as.vector(Y[te, cols, drop = FALSE])
    theta <- if (fit$family == "poisson") NA else 1 / fit$alpha
    p <- if (is.na(theta)) NA else theta / (theta + mu)
    # Randomized quantile residual: sample u ~ Unif(F(y-1), F(y)), not just
    # qnorm(F(y)) -- matches marginal_rqr.py's _nb_rqr/_poisson_rqr convention.
    # A plain qnorm(F(y)) is biased for discrete counts (y=0-heavy pool genes
    # get systematically pinned to the CDF's lower jump), which was inflating
    # w1 regardless of how well pooling actually fit.
    if (fit$family == "poisson") {
      lo <- ifelse(y_te > 0, ppois(y_te - 1, mu), 0)
      hi <- ppois(y_te, mu)
    } else {
      lo <- ifelse(y_te > 0, pnbinom(y_te - 1, theta, p), 0)
      hi <- pnbinom(y_te, theta, p)
    }
    lo <- pmin(pmax(lo, 1e-8), 1 - 1e-8)
    hi <- pmin(pmax(hi, 1e-8), 1 - 1e-8)
    set.seed(42 + fi)
    z <- qnorm(runif(length(y_te), pmin(lo, hi), pmax(lo, hi)))

    # Reshape the stacked (gene-major blocks of length(te)) vector into a
    # length(te) x length(cols) matrix so per-gene z can be tracked across
    # folds, not just pooled into one big vector -- needed to tell "a few bad
    # genes" apart from "uniformly poor calibration".
    z_mat <- matrix(z, nrow = length(te), ncol = length(cols))
    for (j in seq_along(cols)) gene_z[[j]] <- c(gene_z[[j]], z_mat[, j])
  }

  fd <- do.call(rbind, lapply(fold_diag, as.data.frame))
  all_z <- unlist(gene_z); all_z <- all_z[is.finite(all_z)]
  gene_w1 <- vapply(gene_z, function(v) {
    v <- v[is.finite(v)]; n <- length(v)
    if (n < 8) return(NA_real_)
    mean(abs(sort(v) - qnorm(ppoints(n))))
  }, numeric(1))
  for (j in seq_along(cols)) {
    v <- gene_z[[j]][is.finite(gene_z[[j]])]
    gene_rows[[length(gene_rows) + 1]] <- list(nz_threshold = T, gene = gene_names[j],
      w1 = gene_w1[j], n_valid = length(v))
  }

  mean_z <- mean(all_z); sd_z <- sd(all_z)
  skew_z <- mean((all_z - mean_z)^3) / sd_z^3
  kurt_z <- mean((all_z - mean_z)^4) / sd_z^4 - 3
  rows[[length(rows) + 1]] <- list(nz_threshold = T, n_genes = length(cols),
    ok_rate = mean(fd$ok), family_negbin_frac = mean(fd$family[fd$ok] == "negbin"),
    tau2_fit_median = median(fd$tau2[fd$ok], na.rm = TRUE),
    overdisp_ratio_median = median(fd$overdisp_ratio[fd$ok], na.rm = TRUE),
    w1_median = median(gene_w1, na.rm = TRUE), w1_p90 = quantile(gene_w1, 0.9, na.rm = TRUE, names = FALSE),
    w1_max = max(gene_w1, na.rm = TRUE),
    mean_z = mean_z, std_z = sd_z, skew_z = skew_z, kurt_z = kurt_z)
  cat(sprintf("nz<%d: n_genes=%d ok_rate=%.2f negbin_frac=%.2f tau2_fit=%.4f w1_median=%.3f w1_p90=%.3f mean_z=%.3f std_z=%.3f\n",
    T, length(cols), mean(fd$ok), mean(fd$family[fd$ok] == "negbin"), median(fd$tau2[fd$ok], na.rm = TRUE),
    median(gene_w1, na.rm = TRUE), quantile(gene_w1, 0.9, na.rm = TRUE, names = FALSE), mean_z, sd_z))
}
write.csv(do.call(rbind, lapply(rows, as.data.frame)), opt$out, row.names = FALSE)
if (!is.null(opt$`gene-out`)) write.csv(do.call(rbind, lapply(gene_rows, as.data.frame)), opt$`gene-out`, row.names = FALSE)
cat("DONE\n")
