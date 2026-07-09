suppressPackageStartupMessages(suppressWarnings(library(gamlss)))

# ── Column name sanitization ──────────────────────────────────────
# Covariate names from Python (e.g. "log(Total Reads)") contain special
# characters that R formula parser rejects. We replace them with safe names
# and store the mapping in a global for reuse.

sanitize_names <- function(names) {
  safe <- gsub("[^A-Za-z0-9_]", "_", names)
  # R identifiers must start with a letter or dot, not underscore/digit
  bad <- grepl("^[^A-Za-z.]", safe)
  safe[bad] <- paste0("v", safe[bad])
  safe
}

# ── Formula builders ─────────────────────────────────────────────
# mu: standard linear (unpenalized)
.mu_fml <- function(safe_names) {
  as.formula(paste("y__ ~", paste(safe_names, collapse = " + ")))
}

# sigma: unpenalized formula (same structure as mu).
# Ridge penalty is applied via ridgeVec in gamlss.control (see .ridge_vec()).
.sigma_fml <- function(safe_names) {
  as.formula(paste("~", paste(safe_names, collapse = " + ")))
}

# Build ridgeVec for gamlss.control:
#   mu  coefficients (p total)   : no penalty
#   sigma intercept (1)          : no penalty   <- intentional: don't shrink baseline
#   sigma slope coefficients (p-1): lambda_sigma
# Total length = p + p = 2p, matching NBI coefficient layout.
.ridge_vec <- function(p, lambda_sigma) {
  if (lambda_sigma > 0)
    c(rep(0, p + 1L), rep(lambda_sigma, p - 1L))
  else
    NULL
}

# ── NBI RQR on new data ───────────────────────────────────────────
# gamlss::qresid() only works on fitted objects (training data).
# For test-set RQR we compute manually via pNBI CDF.

rqr_nbi <- function(y, mu, sigma, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  y <- as.integer(round(y))
  a <- ifelse(y > 0, pNBI(y - 1L, mu = mu, sigma = sigma), 0)
  b <- pNBI(y,        mu = mu, sigma = sigma)
  # Clamp to avoid Inf from qnorm
  a <- pmax(pmin(a, 1 - 1e-8), 1e-8)
  b <- pmax(pmin(b, 1 - 1e-8), 1e-8)
  # Guard against floating-point inversion
  lo <- pmin(a, b); hi <- pmax(a, b)
  u  <- runif(length(y), min = lo, max = hi)
  qnorm(u)
}

# ── Outlier selection for the iterative removal loop ───────────────
# Returns indices (into z) of the worst |z| points to drop this iteration,
# capped at the remaining removal budget over ALL iterations combined. Taking
# the worst points up to the cap (instead of refusing to remove anything once
# the raw outlier count exceeds max_remove_frac) means the loop always makes
# progress toward the budget rather than giving up entirely.
.select_outliers <- function(z, outlier_z, max_remove_frac, n_total, n_removed_so_far) {
  outlier <- is.finite(z) & (abs(z) > outlier_z)
  budget <- floor(max_remove_frac * n_total) - n_removed_so_far
  if (!any(outlier) || budget <= 0) return(integer(0))
  idx <- which(outlier)
  if (length(idx) > budget) idx <- idx[order(-abs(z[idx]))][seq_len(budget)]
  idx
}

# ── Internal: fit NBI on a data frame and return the gamlss object ──
.fit_nbi <- function(df_tr, mu_fml, sigma_fml, n_cyc, ridge_vec = NULL) {
  ctrl <- if (is.null(ridge_vec))
    gamlss.control(n.cyc = n_cyc, trace = FALSE)
  else
    gamlss.control(n.cyc = n_cyc, trace = FALSE, ridgeVec = ridge_vec)
  tryCatch(
    gamlss(
      formula       = mu_fml,
      sigma.formula = sigma_fml,
      family        = NBI(),
      data          = df_tr,
      control       = ctrl
    ),
    error = function(e) e
  )
}

# ── NBI main fitting function (with iterative outlier removal) ─────
#
# Additional args vs. original:
#   outlier_z     : |z_train| threshold to flag and remove outliers (default 5)
#   max_iter      : maximum refinement iterations (default 2)
#   lambda_sigma  : L2 ridge penalty on sigma submodel via ri() (default 0.1)
#                   Applied jointly to all covariates as a single design matrix.
#                   Set to 0 to disable.
#
# Returns a named list:
#   z, mu_test, sigma_test, success, msg
#   n_removed : total training samples removed across all iterations

fit_gamlss_gene <- function(y_train, y_test, X_train, X_test,
                             seed = NULL, n_cyc = 50,
                             outlier_z = 5.0, max_iter = 2L,
                             max_remove_frac = 0.05,
                             lambda_sigma = 0.1) {
  n_te       <- length(y_test)
  n_tr_orig  <- length(y_train)
  safe_names <- sanitize_names(colnames(X_train))

  p         <- ncol(X_train) + 1L
  mu_fml    <- .mu_fml(safe_names)
  sigma_fml <- .sigma_fml(safe_names)
  ridge_vec <- .ridge_vec(p, lambda_sigma)

  df_tr <- as.data.frame(X_train); colnames(df_tr) <- safe_names
  df_tr$y__ <- as.integer(round(y_train))
  df_te <- as.data.frame(X_test);  colnames(df_te) <- safe_names

  na_result <- list(z = rep(NA_real_, n_te), mu_test = rep(NA_real_, n_te),
                    sigma_test = rep(NA_real_, n_te),
                    success = FALSE, msg = "", n_removed = 0L)

  keep      <- rep(TRUE, n_tr_orig)
  n_removed <- 0L

  for (iter in seq_len(max_iter)) {
    fit <- .fit_nbi(df_tr[keep, , drop = FALSE], mu_fml, sigma_fml, n_cyc, ridge_vec)
    if (inherits(fit, "error")) {
      na_result$msg <- conditionMessage(fit)
      return(na_result)
    }

    pred_tr <- tryCatch(
      predictAll(fit, newdata = df_tr[keep, , drop = FALSE],
                 type = "response", data = df_tr[keep, , drop = FALSE]),
      error = function(e) e
    )
    if (inherits(pred_tr, "error")) break

    z_tr    <- rqr_nbi(df_tr$y__[keep],
                        mu    = as.numeric(pred_tr$mu),
                        sigma = as.numeric(pred_tr$sigma))
    drop_idx <- .select_outliers(z_tr, outlier_z, max_remove_frac, n_tr_orig, n_removed)
    if (length(drop_idx) == 0) break

    idx_keep  <- which(keep)
    keep[idx_keep[drop_idx]] <- FALSE
    n_removed <- n_removed + length(drop_idx)
  }

  pred_te <- tryCatch(
    predictAll(fit, newdata = df_te, type = "response",
               data = df_tr[keep, , drop = FALSE]),
    error = function(e) e
  )
  if (inherits(pred_te, "error")) {
    na_result$msg <- conditionMessage(pred_te)
    return(na_result)
  }

  mu_te    <- as.numeric(pred_te$mu)
  sigma_te <- as.numeric(pred_te$sigma)

  list(z          = rqr_nbi(y_test, mu = mu_te, sigma = sigma_te, seed = seed),
       mu_test    = mu_te,
       sigma_test = sigma_te,
       success    = TRUE,
       msg        = "",
       n_removed  = n_removed)
}

# ══════════════════════════════════════════════════════════════════
# Model Engine helper — full-data NBI training for inference
# Returns coefficient vectors so Python can score without R.
# ══════════════════════════════════════════════════════════════════

train_nbi_coeffs <- function(y_train, X_train,
                              n_cyc = 50,
                              outlier_z = 5.0, max_iter = 2L,
                              max_remove_frac = 0.05,
                              lambda_sigma = 0.1) {
  n_tr_orig  <- length(y_train)
  safe_names <- sanitize_names(colnames(X_train))

  p         <- ncol(X_train) + 1L
  mu_fml    <- .mu_fml(safe_names)
  sigma_fml <- .sigma_fml(safe_names)
  ridge_vec <- .ridge_vec(p, lambda_sigma)

  df_tr <- as.data.frame(X_train); colnames(df_tr) <- safe_names
  df_tr$y__ <- as.integer(round(y_train))

  na_result <- list(mu_coef = rep(NA_real_, p), sigma_coef = rep(NA_real_, p),
                    success = FALSE, msg = "", n_removed = 0L)

  keep <- rep(TRUE, n_tr_orig); n_removed <- 0L; fit <- NULL

  for (iter in seq_len(max_iter)) {
    fit <- .fit_nbi(df_tr[keep, , drop = FALSE], mu_fml, sigma_fml, n_cyc, ridge_vec)
    if (inherits(fit, "error")) { na_result$msg <- conditionMessage(fit); return(na_result) }
    pred_tr <- tryCatch(
      predictAll(fit, newdata = df_tr[keep, , drop = FALSE],
                 type = "response", data = df_tr[keep, , drop = FALSE]),
      error = function(e) e)
    if (inherits(pred_tr, "error")) break
    z_tr    <- rqr_nbi(df_tr$y__[keep],
                        mu = as.numeric(pred_tr$mu), sigma = as.numeric(pred_tr$sigma))
    drop_idx <- .select_outliers(z_tr, outlier_z, max_remove_frac, n_tr_orig, n_removed)
    if (length(drop_idx) == 0) break
    keep[which(keep)[drop_idx]] <- FALSE
    n_removed <- n_removed + length(drop_idx)
  }

  if (is.null(fit) || inherits(fit, "error")) return(na_result)
  mu_c <- as.numeric(fit$mu.coefficients); sigma_c <- as.numeric(fit$sigma.coefficients)
  list(mu_coef    = mu_c,
       sigma_coef = sigma_c,
       success    = TRUE, msg = "", n_removed = n_removed,
       converged  = isTRUE(fit$converged),
       mu_finite  = all(is.finite(mu_c)), sigma_finite = all(is.finite(sigma_c)),
       gaic       = GAIC(fit, k = 2), edf = fit$df.fit)
}
