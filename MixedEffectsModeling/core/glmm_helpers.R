suppressPackageStartupMessages(library(glmmTMB))

sanitize_names <- function(names) {
  safe <- gsub("[^A-Za-z0-9_]", "_", names)
  bad <- grepl("^[^A-Za-z.]", safe)
  safe[bad] <- paste0("v", safe[bad])
  safe
}

safe_max_abs <- function(x) if (length(x) == 0) 0 else max(abs(x))
is_converged <- function(fit, beta_explode_thr, tau2_max, disp_intercept_max) {
  if (inherits(fit, "try-error")) return(list(ok = FALSE, singular = NA, tau2 = NA))
  beta_max <- safe_max_abs(c(fixef(fit)$cond[-1], fixef(fit)$disp[-1]))
  disp0 <- fixef(fit)$disp[1]
  tau2 <- as.numeric(VarCorr(fit)$cond[[1]][1, 1])
  if (isTRUE(beta_max >= beta_explode_thr)) return(list(ok = FALSE, singular = NA, tau2 = tau2))
  if (length(disp0) > 0 && isTRUE(abs(disp0) >= disp_intercept_max)) return(list(ok = FALSE, singular = NA, tau2 = tau2))
  if (isTRUE(fit$sdr$pdHess)) {
    if (isTRUE(tau2 >= tau2_max)) return(list(ok = FALSE, singular = NA, tau2 = tau2))
    return(list(ok = TRUE, singular = FALSE, tau2 = tau2))
  }
  if (isTRUE(tau2 < 1e-5)) return(list(ok = TRUE, singular = TRUE, tau2 = 0.0))
  return(list(ok = FALSE, singular = NA, tau2 = tau2))
}

# Standard errors of the dispersion fixed effects. NA when sdreport is unusable
# -- the downstream EB squeeze reads NA as SE^2=inf and returns exactly the
# lowess trend value, recovering v2's hard-fixed dispersion as a limiting case.
disp_se <- function(fit, ncoef) {
  V <- tryCatch(vcov(fit, full = TRUE), error = function(e) NULL, warning = function(w) NULL)
  if (is.null(V)) return(rep(NA_real_, ncoef))
  idx <- grep("^disp", names(rownames(V)))
  if (length(idx) != ncoef) return(rep(NA_real_, ncoef))
  s <- suppressWarnings(sqrt(diag(V))[idx])
  ifelse(is.finite(s), s, NA_real_)
}

# One-step Pregibon (1981) Cook's distance for the NB2 log-link fit. The
# estimated random intercept is absorbed into mu and the hat matrix uses the
# fixed-effect design only (standard approximation). Cutoff qf(0.99, p, n-p) and
# the 5% cap follow DESeq2's outlier rule; observations are dropped rather than
# replaced by a trimmed mean, which would fabricate counts and bias dispersion
# downward. Returns indices ordered by decreasing influence.
#
# The variance uses the lowess TREND dispersion, not the gene's own fitted one.
# With a freely estimated dispersion an outlier masks itself: measured on
# synthetic near-Poisson data, three 20x outliers inflated alpha 0.004 -> 0.147
# (36x), which shrank their Pearson residuals enough that Cook's D fell from
# 4.5-9.4 to 0.6-1.1 -- below the 2.29 cutoff, so nothing was ever flagged at any
# outlier magnitude. The trend value is the same reference the EB intercept
# squeeze shrinks toward, so it is the pipeline's own definition of "typical
# dispersion at this expression level".
cook_outliers <- function(fit, Xa, y, trend_alpha, f_q, max_frac) {
  p <- ncol(Xa); n <- length(y)
  if (n - p < 4) return(integer(0))
  mu <- tryCatch(as.numeric(predict(fit, type = "response")), error = function(e) NULL)
  if (is.null(mu) || !all(is.finite(mu)) || !is.finite(trend_alpha) || trend_alpha <= 0) return(integer(0))
  alpha <- trend_alpha
  V <- mu + alpha * mu^2
  r <- (y - mu) / sqrt(V)
  Xw <- Xa * sqrt(mu / (1 + alpha * mu))
  XtX <- tryCatch(solve(crossprod(Xw)), error = function(e) NULL)
  if (is.null(XtX)) return(integer(0))
  h <- pmin(pmax(rowSums((Xw %*% XtX) * Xw), 0), 1 - 1e-8)
  D <- r^2 * h / (p * (1 - h)^2)
  over <- which(is.finite(D) & D > qf(f_q, p, n - p))
  if (length(over) == 0) return(integer(0))
  n_keep <- min(length(over), floor(max_frac * n))
  if (n_keep < 1) return(integer(0))
  over[order(-D[over])][seq_len(n_keep)]
}

# Fits ONE stage for ONE gene. Caller (glmm_fit.R) drives the demotion order.
# v3: stages are "nbi_full_eb" (dispersion regressed on covariates) and "nbi_intercept_eb"
# (dispersion intercept only). "nbi_disp_intercept" is gone -- with a
# properly-scaled slope prior it was no longer a distinct model, and "nbi_intercept_eb"
# now occupies its structural position. The dispersion INTERCEPT is left
# unpenalized here; it is squeezed toward the lowess trend analytically
# downstream (eb_shrinkage.squeeze_log_theta). tau_slope is the per-covariate EB
# prior sd for the dispersion SLOPES, estimated by a --mode pilot run.
fit_stage_gene <- function(y, safe_names, X, batch, stage, tau_slope, trend_alpha,
                           beta_explode_thr, tau2_max, disp_intercept_max,
                           cook_f_q, max_outlier_frac) {
  df <- as.data.frame(X); colnames(df) <- safe_names
  df$y__ <- as.integer(round(y))
  df$batch__ <- factor(batch)

  mu_fml <- as.formula(paste("y__ ~", paste(safe_names, collapse = " + "), "+ (1 | batch__)"))
  disp_fml <- switch(stage,
    nbi_full_eb = as.formula(paste("~", paste(safe_names, collapse = " + "))),
    nbi_intercept_eb = as.formula("~ 1"),
    stop(sprintf("fit_stage_gene: unknown stage '%s'", stage)))
  n_disp <- if (stage == "nbi_full_eb") length(safe_names) + 1L else 1L

  pr <- NULL
  if (stage == "nbi_full_eb" && !is.null(tau_slope)) {
    pr <- data.frame(prior = sprintf("normal(0, %.6g)", tau_slope),
                     class = "betad", coef = safe_names)
  }
  run <- function(d) tryCatch({
    if (is.null(pr)) glmmTMB(mu_fml, dispformula = disp_fml, family = nbinom2(), data = d)
    else glmmTMB(mu_fml, dispformula = disp_fml, family = nbinom2(), data = d, priors = pr)
  }, error = function(e) structure(conditionMessage(e), class = "try-error"))

  fit <- run(df)
  if (inherits(fit, "try-error")) {
    return(list(stage = stage, ok = FALSE, singular = NA, tau2 = NA, n_outliers = 0L,
               outlier_refit_failed = FALSE, mu_coef = numeric(0), disp_coef = numeric(0),
               disp_se = rep(NA_real_, n_disp), fail_reason = as.character(fit)))
  }

  conv <- is_converged(fit, beta_explode_thr, tau2_max, disp_intercept_max)
  n_out <- 0L; refit_failed <- FALSE
  # Cook's distance is computed even when the first fit failed the convergence
  # gate -- outliers can be the cause of the failure, so the refit gets a chance.
  drop_idx <- cook_outliers(fit, cbind(1, as.matrix(X)), df$y__, trend_alpha, cook_f_q, max_outlier_frac)
  if (length(drop_idx) > 0) {
    fit2 <- run(df[-drop_idx, , drop = FALSE])
    conv2 <- if (inherits(fit2, "try-error")) list(ok = FALSE) else
      is_converged(fit2, beta_explode_thr, tau2_max, disp_intercept_max)
    if (isTRUE(conv2$ok)) {
      fit <- fit2; conv <- conv2; n_out <- length(drop_idx)
    } else {
      refit_failed <- TRUE
    }
  }

  list(stage = stage, ok = conv$ok, singular = conv$singular, tau2 = conv$tau2,
      n_outliers = n_out, outlier_refit_failed = refit_failed,
      mu_coef = as.numeric(fixef(fit)$cond), disp_coef = as.numeric(fixef(fit)$disp),
      disp_se = disp_se(fit, n_disp),
      fail_reason = if (isTRUE(conv$ok)) "" else "not_converged_or_explosion_or_tau2_bound")
}

# Shared-beta pooled GLM (route "pool") + batch random intercept. Unused this
# round (pooling deferred until nz_a_max is picked) -- kept so the file stays
# a complete, working unit.
fit_pooled_glmm <- function(Y_block, X, batch, mean_hc, eps, rare_overdisp_thr) {
  n_hc <- nrow(X); n_g <- ncol(Y_block)
  safe_names <- sanitize_names(colnames(X))
  sample_idx <- rep(seq_len(n_hc), n_g)
  gene_idx <- rep(seq_len(n_g), each = n_hc)
  df <- as.data.frame(X[sample_idx, , drop = FALSE]); colnames(df) <- safe_names
  df$y__ <- as.integer(round(Y_block[cbind(sample_idx, gene_idx)]))
  df$batch__ <- factor(batch[sample_idx])
  df$off__ <- log(mean_hc[gene_idx] + eps)
  mu_fml <- as.formula(paste("y__ ~ offset(off__) +", paste(safe_names, collapse = " + "), "+ (1 | batch__)"))

  mult_clip <- function(beta) list(
    mult_lo = as.numeric(quantile(exp(X %*% beta[-1]), 0.001)),
    mult_hi = as.numeric(quantile(exp(X %*% beta[-1]), 0.999)))

  pool_tau2 <- function(fit) as.numeric(VarCorr(fit)$cond[[1]][1, 1])

  fit_pois <- tryCatch(glmmTMB(mu_fml, family = poisson(), data = df), error = function(e) NULL)
  if (is.null(fit_pois)) return(list(ok = FALSE))
  ratio <- sum(residuals(fit_pois, type = "pearson")^2) / df.residual(fit_pois)
  if (ratio <= rare_overdisp_thr) {
    beta <- as.numeric(fixef(fit_pois)$cond)
    mc <- mult_clip(beta)
    return(list(ok = TRUE, family = "poisson", beta = beta, alpha = NA, tau2 = pool_tau2(fit_pois),
               overdisp_ratio = ratio, mult_lo = mc$mult_lo, mult_hi = mc$mult_hi))
  }
  fit_nb <- tryCatch(glmmTMB(mu_fml, family = nbinom2(), data = df), error = function(e) NULL)
  if (is.null(fit_nb)) return(list(ok = FALSE))
  beta <- as.numeric(fixef(fit_nb)$cond)
  mc <- mult_clip(beta)
  list(ok = TRUE, family = "negbin", beta = beta,
      alpha = exp(-fixef(fit_nb)$disp[["(Intercept)"]]),
      tau2 = pool_tau2(fit_nb), overdisp_ratio = ratio, mult_lo = mc$mult_lo, mult_hi = mc$mult_hi)
}
