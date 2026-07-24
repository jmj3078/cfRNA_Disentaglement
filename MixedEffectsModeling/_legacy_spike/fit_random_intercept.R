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
BETA_EXPLODE_THR <- 3.0
TAU2_MAX <- BETA_EXPLODE_THR^2

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
