source("../Modeling/gamlss.r")  # not used here, kept for name-sanitizing consistency if needed later
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

rows <- list()
for (g in genes) {
  df <- as.data.frame(X)
  df$y__ <- as.integer(round(Y[[g]]))
  df$batch__ <- factor(batch)

  t0 <- Sys.time()
  fit_res <- tryCatch({
    warn_msgs <- character(0)
    fit <- withCallingHandlers(
      if (use_priors) {
        glmmTMB(fml_mu, dispformula = fml_disp, family = nbinom2(), data = df, priors = priors_df)
      } else {
        glmmTMB(fml_mu, dispformula = fml_disp, family = nbinom2(), data = df)
      },
      warning = function(w) { warn_msgs <<- c(warn_msgs, conditionMessage(w)); invokeRestart("muffleWarning") }
    )
    vc <- VarCorr(fit)$cond$batch__
    tau2 <- as.numeric(vc[1, 1])
    singular <- any(grepl("singular|convergence", warn_msgs, ignore.case = TRUE)) || isTRUE(tau2 < 1e-6)
    list(converged = TRUE, singular = singular, tau2 = tau2,
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
