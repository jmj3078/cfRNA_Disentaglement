suppressPackageStartupMessages(library(glmmTMB))

sanitize_names <- function(names) {
  safe <- gsub("[^A-Za-z0-9_]", "_", names)
  bad <- grepl("^[^A-Za-z.]", safe)
  safe[bad] <- paste0("v", safe[bad])
  safe
}

safe_max_abs <- function(x) if (length(x) == 0) 0 else max(abs(x))

# Slopes only (intercept excluded, matches Modeling/model_engine.py's beta[1:]
# convention) for the MEAN submodel -- a very negative mean intercept is
# legitimate for a low-expression gene. explosion checked before pdHess
# branching, in either submodel.
#
# The DISPERSION intercept is checked separately (disp_intercept_max), not
# folded into beta_max: an exploded disp intercept collapses alpha to ~0
# (near-deterministic NB, no overdispersion) while its slopes stay ~0 --
# passes every other check (pdHess, tau2 bound) and was silently marked
# ok=TRUE, then produced catastrophic held-out log-likelihoods (~-80000/obs)
# under pool_vs_individual_sweep.py. Empirically bimodal: normal fits cluster
# in [-5, -0.6], degenerate ones jump to [17, 31] with a clean gap between --
# disp_intercept_max=10 sits in that gap.
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

# Fits ONE stage for ONE gene. Caller (glmm_fit.R) drives the demotion order.
fit_stage_gene <- function(y, safe_names, X, batch, stage, fixed_log_theta,
                           priors_df, beta_explode_thr, tau2_max, disp_intercept_max) {
  df <- as.data.frame(X); colnames(df) <- safe_names
  df$y__ <- as.integer(round(y))
  df$batch__ <- factor(batch)
  if (!is.null(fixed_log_theta)) df$fixed_log_theta <- fixed_log_theta

  mu_fml <- if (stage %in% c("intercept")) as.formula("y__ ~ 1 + (1 | batch__)") else
    as.formula(paste("y__ ~", paste(safe_names, collapse = " + "), "+ (1 | batch__)"))
  disp_fml <- switch(stage,
    nbi = as.formula(paste("~", paste(safe_names, collapse = " + "))),
    nbi_disp_intercept = as.formula("~ 1"),
    as.formula("~ 0 + offset(fixed_log_theta)"))  # nb_fixed, intercept

  fit <- tryCatch({
    if (!is.null(priors_df)) glmmTMB(mu_fml, dispformula = disp_fml, family = nbinom2(), data = df, priors = priors_df)
    else glmmTMB(mu_fml, dispformula = disp_fml, family = nbinom2(), data = df)
  }, error = function(e) structure(conditionMessage(e), class = "try-error"))

  if (inherits(fit, "try-error")) {
    return(list(stage = stage, ok = FALSE, singular = NA, tau2 = NA,
               mu_coef = numeric(0), disp_coef = numeric(0), fail_reason = as.character(fit)))
  }
  conv <- is_converged(fit, beta_explode_thr, tau2_max, disp_intercept_max)
  list(stage = stage, ok = conv$ok, singular = conv$singular, tau2 = conv$tau2,
      mu_coef = as.numeric(fixef(fit)$cond), disp_coef = as.numeric(fixef(fit)$disp),
      fail_reason = if (conv$ok) "" else "not_converged_or_explosion_or_tau2_bound")
}

# Shared-beta pooled GLM (route "pool") + batch random intercept. Mirrors
# Modeling/model_engine.py's train_rare() offset/family-selection logic, with
# (1|batch__) added on the stacked long-format design.
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

  # Covariate-multiplier clip bounds from the HC-observed range (slopes only,
  # intercept excluded); mirrors Modeling/model_engine.py's train_rare() clip
  # -- guards scoring-time extrapolation for OOD-adjacent covariates.
  mult_clip <- function(beta) list(
    mult_lo = as.numeric(quantile(exp(X %*% beta[-1]), 0.001)),
    mult_hi = as.numeric(quantile(exp(X %*% beta[-1]), 0.999)))

  # (1|batch__) tau2, extracted positionally like is_converged() -- pool route
  # has no demotion chain so it doesn't gate on tau2, but scoring/CV still needs
  # this to marginalize correctly instead of silently assuming tau2==0.
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
      alpha = exp(-fixef(fit_nb)$disp[["(Intercept)"]]),  # theta->sigma reciprocal, same convention
      tau2 = pool_tau2(fit_nb), overdisp_ratio = ratio, mult_lo = mc$mult_lo, mult_hi = mc$mult_hi)
}
