suppressPackageStartupMessages({
  library(glmmTMB)
  library(jsonlite)
})

X <- as.matrix(read.csv("Spike_Results/pilot_X.csv.gz", row.names = 1))
Y <- read.csv("Spike_Results/pilot_Y.csv.gz", row.names = 1)
batch <- read.csv("Spike_Results/pilot_batch.csv.gz", row.names = 1)$Batch_ID
genes <- colnames(Y)
safe_names <- sub("^X", "v", make.names(colnames(X), unique = TRUE))
safe_names <- gsub("[^A-Za-z0-9_]", "_", safe_names)
colnames(X) <- safe_names

caps <- fromJSON("Spike_Results/glmmtmb_capabilities.json")
use_priors <- isTRUE(caps$priors_probe_success)
cat(sprintf("use_priors = %s\n", use_priors))

fml_mu <- as.formula(paste("y__ ~", paste(safe_names, collapse = " + "), "+ (1 | batch__)"))
fml_disp <- as.formula(paste("~", paste(safe_names, collapse = " + ")))
priors_df <- if (use_priors) data.frame(prior = "normal(0, 0.05)", class = "betad", coef = "") else NULL

# Mirrors MixedEffectsModeling/config.py's SPIKE_PARAMS["beta_explode_thr"] (which in turn
# mirrors the root project's config.MODELING_PARAMS["beta_explode_thr"] convention). R and
# Python config are kept fully separate by design in this spike, so this is duplicated here.
BETA_EXPLODE_THR <- 3.0

# Reused threshold, not a separately derived one: BETA_EXPLODE_THR is calibrated as a
# regression-slope explosion check (log(mu) change per SD of a standardized covariate),
# which is not the same quantity as a random-intercept's total spread (sd(b)). Squaring
# it to get a tau2 cutoff is a convenience borrow -- picking the project's one existing
# "implausible magnitude" convention for consistency -- not a statistically derived bound.
# sd(b)=3 already implies e^{+-6}-fold batch swings, which is implausible enough on its
# own to justify treating tau2 >= 9 as non-identifiable regardless of the exact number.
TAU2_MAX <- BETA_EXPLODE_THR^2

# Principled convergence/singularity check based on TMB's own positive-definite-Hessian
# diagnostic (fit$sdr$pdHess), rather than fragile string-matching on R warning text.
#
# Explosion is checked FIRST and unconditionally, on SLOPES ONLY (both submodels'
# intercepts excluded), before branching on pdHess at all: a positive-definite Hessian
# around an exploded slope coefficient does not make that coefficient trustworthy, and a
# low-expression gene's log(mu)/log(theta) INTERCEPT is legitimately large-negative, not a
# sign of instability -- this mirrors the project's established beta_explode convention
# (Modeling/model_engine.py's beta[1:], Modeling/gamlss.r's "sigma intercept: no penalty"
# comment), which always excludes the intercept from this kind of magnitude check.
#
# tau2 is extracted positionally (VarCorr(fit)$cond[[1]]), not via the grouping variable's
# literal name, since this model design only ever has exactly one random-effect grouping
# term regardless of what it happens to be named in the formula.
#
# pdHess FALSE does not always mean a genuine optimizer failure: a batch variance
# component (tau2) estimated at/near the zero boundary makes the Hessian look
# non-positive-definite purely because the parameter sits at the edge of its domain --
# a normal, expected "boundary singular fit" outcome for genes with no real batch effect.
# We rescue that case as converged (singular = TRUE, tau2 forced to exactly 0) provided the
# slopes already passed the explosion check above.
# Conversely, pdHess TRUE is not sufficient either: a Hessian can look positive-definite
# while tau2 itself has blown up to an implausible magnitude (a handful of extreme-n
# batches -- this pilot's HC batches range from n=1 to n=116 -- can pull the optimizer to
# a huge batch-variance estimate that the Hessian still certifies as a local optimum). We
# treat tau2 >= tau2_max as non-identifiable and refuse to trust it downstream, exactly as
# we already refuse to trust an exploding fixed-effect slope.
is_converged <- function(fit, beta_explode_thr, tau2_max) {
  if (inherits(fit, "try-error")) return(list(ok = FALSE, singular = NA, tau2 = NA))

  beta_max <- max(abs(c(fixef(fit)$cond[-1], fixef(fit)$disp[-1])))
  tau2 <- as.numeric(VarCorr(fit)$cond[[1]][1, 1])

  if (isTRUE(beta_max >= beta_explode_thr)) {
    # Explosion in EITHER submodel's slopes is disqualifying regardless of pdHess.
    return(list(ok = FALSE, singular = NA, tau2 = tau2))
  }

  if (isTRUE(fit$sdr$pdHess)) {
    if (isTRUE(tau2 >= tau2_max)) {
      # Hessian is fine but the batch variance itself is implausibly large -- treat as
      # a non-identifiable fit, not a trustworthy convergence.
      return(list(ok = FALSE, singular = NA, tau2 = tau2))
    }
    return(list(ok = TRUE, singular = FALSE, tau2 = tau2))
  }

  if (isTRUE(tau2 < 1e-5)) {
    # pdHess FALSE, but slopes already confirmed not exploding above and tau2 sits at
    # the zero boundary: the benign boundary-singular case.
    return(list(ok = TRUE, singular = TRUE, tau2 = 0.0))
  }
  # pdHess FALSE, tau2 not near the boundary, slopes not exploding -- a real convergence
  # failure with no simple explanation.
  return(list(ok = FALSE, singular = NA, tau2 = tau2))
}

rows <- list()
for (g in genes) {
  df <- as.data.frame(X)
  df$y__ <- as.integer(round(Y[[g]]))
  df$batch__ <- factor(batch)

  t0 <- Sys.time()
  fit_res <- tryCatch({
    fit <- if (use_priors) {
      glmmTMB(fml_mu, dispformula = fml_disp, family = nbinom2(), data = df, priors = priors_df)
    } else {
      glmmTMB(fml_mu, dispformula = fml_disp, family = nbinom2(), data = df)
    }
    conv <- is_converged(fit, BETA_EXPLODE_THR, TAU2_MAX)
    list(converged = conv$ok, singular = conv$singular, tau2 = conv$tau2,
         mu_coef = as.numeric(fixef(fit)$cond), disp_coef = as.numeric(fixef(fit)$disp))
  }, error = function(e) list(converged = FALSE, singular = NA, tau2 = NA,
                              mu_coef = rep(NA, length(safe_names) + 1),
                              disp_coef = rep(NA, length(safe_names) + 1)))
  wall <- as.numeric(Sys.time() - t0, units = "secs")

  row <- list(gene = g, converged = isTRUE(fit_res$converged), singular = isTRUE(fit_res$singular),
             tau2 = fit_res$tau2, wall_time_sec = wall, used_priors = use_priors)
  for (i in seq_along(fit_res$mu_coef)) {
    row[[paste0("mu_coef_", i - 1)]] <- fit_res$mu_coef[i]
    row[[paste0("disp_coef_", i - 1)]] <- fit_res$disp_coef[i]
  }
  rows[[g]] <- row
  cat(sprintf("%s: converged=%s singular=%s tau2=%s time=%.2fs\n",
             g, row$converged, row$singular, format(row$tau2), wall))
  rm(fit_res); gc()
}

out <- do.call(rbind, lapply(rows, as.data.frame))
write.csv(out, "Spike_Results/random_intercept_fits.csv", row.names = FALSE)
cat("Wrote Spike_Results/random_intercept_fits.csv\n")
cat(sprintf("PASS: %d/%d genes converged, %d singular, mean wall time %.2fs\n",
           sum(out$converged), nrow(out), sum(out$singular, na.rm = TRUE), mean(out$wall_time_sec)))
