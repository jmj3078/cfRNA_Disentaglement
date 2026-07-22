source("../Modeling/gamlss.r")  # reuses sanitize_names(), train_nbi_coeffs() -- read-only
suppressPackageStartupMessages(library(glmmTMB))

X <- as.matrix(read.csv("Spike_Results/pilot_X.csv.gz", row.names = 1))
Y <- read.csv("Spike_Results/pilot_Y.csv.gz", row.names = 1)
genes <- colnames(Y)
safe_names <- sanitize_names(colnames(X))
colnames(X) <- safe_names

rows <- list()
for (g in genes) {
  y <- as.integer(round(Y[[g]]))

  gam_res <- tryCatch(
    train_nbi_coeffs(y, X, n_cyc = 50, outlier_z = 5.0, max_iter = 2L,
                     max_remove_frac = 0.05, lambda_sigma = 0.0),  # unpenalized, for a clean comparison
    error = function(e) list(success = FALSE)
  )

  df <- as.data.frame(X)
  df$y__ <- y
  fml_mu <- as.formula(paste("y__ ~", paste(safe_names, collapse = " + ")))
  fml_disp <- as.formula(paste("~", paste(safe_names, collapse = " + ")))
  tmb_res <- tryCatch({
    fit <- glmmTMB(fml_mu, dispformula = fml_disp, family = nbinom2(), data = df)
    list(success = TRUE,
         mu_coef = as.numeric(fixef(fit)$cond),
         disp_coef = as.numeric(fixef(fit)$disp))
  }, error = function(e) list(success = FALSE))

  row <- list(gene = g,
             gamlss_success = isTRUE(gam_res$success),
             glmmtmb_success = isTRUE(tmb_res$success))
  p <- length(safe_names) + 1
  for (i in seq_len(p)) {
    row[[paste0("gamlss_mu_coef_", i - 1)]] <- if (isTRUE(gam_res$success)) gam_res$mu_coef[i] else NA
    row[[paste0("gamlss_sigma_coef_", i - 1)]] <- if (isTRUE(gam_res$success)) gam_res$sigma_coef[i] else NA
    row[[paste0("glmmtmb_mu_coef_", i - 1)]] <- if (isTRUE(tmb_res$success)) tmb_res$mu_coef[i] else NA
    row[[paste0("glmmtmb_disp_coef_", i - 1)]] <- if (isTRUE(tmb_res$success)) tmb_res$disp_coef[i] else NA
  }
  rows[[g]] <- row
  cat(sprintf("%s: gamlss=%s glmmTMB=%s\n", g, row$gamlss_success, row$glmmtmb_success))
}

out <- do.call(rbind, lapply(rows, as.data.frame))
write.csv(out, "Spike_Results/fixed_only_fits.csv", row.names = FALSE)
cat("Wrote Spike_Results/fixed_only_fits.csv\n")
