suppressPackageStartupMessages({
  library(optparse); library(parallel); library(jsonlite)
})

# sys.frame(1)$ofile only works inside source()'d files, not top-level Rscript
# execution -- parse --file= from commandArgs instead, robust to invocation cwd.
.args <- commandArgs(trailingOnly = FALSE)
.script_path <- sub("--file=", "", grep("--file=", .args, value = TRUE))
source(file.path(dirname(normalizePath(.script_path)), "glmm_helpers.R"))

opt <- parse_args(OptionParser(option_list = list(
  make_option("--x", type = "character"), make_option("--y", type = "character"),
  make_option("--batch", type = "character"), make_option("--genes", type = "character"),
  make_option("--trend", type = "character"), make_option("--mode", type = "character", default = "cascade"),
  make_option("--out", type = "character"), make_option("--chunk-size", type = "integer", default = 200),
  make_option("--cores", type = "integer", default = min(parallel::detectCores() - 1, 8))
)))

X <- as.matrix(read.csv(opt$x, row.names = 1))
Y <- read.csv(opt$y, row.names = 1)
batch <- read.csv(opt$batch, row.names = 1)[[1]]
gene_meta <- read.csv(opt$genes)  # columns: gene, [stage] (stage only needed for fixed_stage mode)
# Matches Modeling/dispersion_trend.py's actual saved schema: log-scale lowess
# grid + floor/cap, not a plain mean/alpha grid.
trend <- fromJSON(opt$trend)
alpha_of <- function(mean_y) {
  lm <- log(max(mean_y, 1e-8))
  s <- exp(approx(trend$lowess_logmu, trend$lowess_logsigma, xout = lm, rule = 2)$y)
  min(max(s, trend$alpha_floor), trend$alpha_cap)
}
safe_names <- sanitize_names(colnames(X))
colnames(X) <- safe_names
BETA_EXPLODE_THR <- 3.0
TAU2_MAX <- BETA_EXPLODE_THR^2
priors_df <- data.frame(prior = "normal(0, 0.05)", class = "betad", coef = "")

done_genes <- character(0)
if (file.exists(opt$out)) done_genes <- read.csv(opt$out)$gene

fit_one_cascade <- function(g) {
  y <- as.numeric(Y[[g]])
  alpha_g <- alpha_of(mean(y))
  fixed_log_theta <- rep(-log(alpha_g), length(y))
  for (stage in c("nbi", "nbi_disp_intercept", "nb_fixed", "intercept")) {
    pr <- if (stage == "nbi") priors_df else NULL
    r <- fit_stage_gene(y, safe_names, X, batch, stage, fixed_log_theta, pr, BETA_EXPLODE_THR, TAU2_MAX)
    if (isTRUE(r$ok) || stage == "intercept") { gc(); return(c(list(gene = g), r)) }
  }
}

fit_one_fixed <- function(g) {
  stage <- gene_meta$stage[gene_meta$gene == g]
  y <- as.numeric(Y[[g]])
  alpha_g <- alpha_of(mean(y))
  fixed_log_theta <- rep(-log(alpha_g), length(y))
  pr <- if (stage == "nbi") priors_df else NULL
  r <- fit_stage_gene(y, safe_names, X, batch, stage, fixed_log_theta, pr, BETA_EXPLODE_THR, TAU2_MAX)
  gc(); c(list(gene = g), r)
}

worker <- if (opt$mode == "cascade") fit_one_cascade else fit_one_fixed
genes_todo <- setdiff(gene_meta$gene, done_genes)
chunks <- split(genes_todo, ceiling(seq_along(genes_todo) / opt$`chunk-size`))

for (i in seq_along(chunks)) {
  results <- mclapply(chunks[[i]], worker, mc.cores = opt$cores)
  rows <- lapply(results, function(r) {
    p <- 11  # 1 intercept + 10 covariates
    mu_padded <- c(r$mu_coef, rep(NA, p - length(r$mu_coef)))[1:p]
    disp_padded <- c(r$disp_coef, rep(NA, p - length(r$disp_coef)))[1:p]
    row <- c(list(gene = r$gene, stage = r$stage, ok = r$ok, singular = r$singular,
                 tau2 = r$tau2, fail_reason = r$fail_reason))
    for (j in seq_len(p)) { row[[paste0("mu_coef_", j-1)]] <- mu_padded[j]; row[[paste0("disp_coef_", j-1)]] <- disp_padded[j] }
    row
  })
  df <- do.call(rbind, lapply(rows, as.data.frame))
  write.table(df, opt$out, sep = ",", append = file.exists(opt$out), col.names = !file.exists(opt$out), row.names = FALSE)
  gc()
  cat(sprintf("chunk %d/%d done (%d genes)\n", i, length(chunks), length(chunks[[i]])))
}
cat("DONE\n")
