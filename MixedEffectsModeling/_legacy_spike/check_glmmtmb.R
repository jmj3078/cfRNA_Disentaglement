suppressWarnings(suppressPackageStartupMessages({
  ok <- requireNamespace("glmmTMB", quietly = TRUE)
}))

library(jsonlite)

result <- list(installed = ok, version = "", has_priors_arg = FALSE,
              priors_probe_success = FALSE, priors_probe_message = "")

if (ok) {
  library(glmmTMB)
  result$version <- as.character(packageVersion("glmmTMB"))
  result$has_priors_arg <- "priors" %in% names(formals(glmmTMB))

  # Minimal probe: does a priors= call actually run without error on toy data?
  set.seed(1)
  n <- 200
  toy <- data.frame(
    y = rnbinom(n, mu = 5, size = 2),
    x1 = rnorm(n),
    grp = factor(sample(letters[1:5], n, replace = TRUE))
  )
  probe <- tryCatch({
    priors_df <- data.frame(prior = "normal(0, 1)", class = "beta", coef = "")
    fit <- glmmTMB(y ~ x1 + (1 | grp), dispformula = ~x1,
                   family = nbinom2(), data = toy, priors = priors_df)
    list(success = TRUE, message = "ok")
  }, error = function(e) list(success = FALSE, message = conditionMessage(e)))
  result$priors_probe_success <- probe$success
  result$priors_probe_message <- probe$message
}

dir.create("Spike_Results", showWarnings = FALSE)
write(toJSON(result, auto_unbox = TRUE, pretty = TRUE), "Spike_Results/glmmtmb_capabilities.json")
cat("Wrote Spike_Results/glmmtmb_capabilities.json\n")
cat(toJSON(result, auto_unbox = TRUE, pretty = TRUE), "\n")
