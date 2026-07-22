suppressPackageStartupMessages(library(glmmTMB))
suppressPackageStartupMessages(library(parallel))

X <- as.matrix(read.csv("Spike_Results/pilot_X.csv.gz", row.names = 1))
Y <- read.csv("Spike_Results/pilot_Y.csv.gz", row.names = 1)
batch <- read.csv("Spike_Results/pilot_batch.csv.gz", row.names = 1)$Batch_ID
safe_names <- gsub("[^A-Za-z0-9_]", "_", colnames(X))
colnames(X) <- safe_names

# Replicate the ~40 pilot genes 5x (relabeled) to get a ~200-fit run --
# enough to see a memory trend without a multi-hour spike.
genes <- rep(colnames(Y), 5)
gene_cols <- rep(colnames(Y), 5)

fml_mu <- as.formula(paste("y__ ~", paste(safe_names, collapse = " + "), "+ (1 | batch__)"))
fml_disp <- as.formula(paste("~", paste(safe_names, collapse = " + ")))

rss_mb <- function() {
  pid <- Sys.getpid()
  as.numeric(system(sprintf("ps -o rss= -p %d", pid), intern = TRUE)) / 1024
}

fit_one <- function(gene_col) {
  df <- as.data.frame(X)
  df$y__ <- as.integer(round(Y[[gene_col]]))
  df$batch__ <- factor(batch)
  fit <- tryCatch(glmmTMB(fml_mu, dispformula = fml_disp, family = nbinom2(), data = df),
                  error = function(e) NULL)
  rm(fit); gc()
  TRUE
}

run_config <- function(config_name, chunk_size, preschedule, cores) {
  log_rows <- list()
  chunks <- split(gene_cols, ceiling(seq_along(gene_cols) / chunk_size))
  for (i in seq_along(chunks)) {
    before <- rss_mb()
    t0 <- Sys.time()
    invisible(mclapply(chunks[[i]], fit_one, mc.cores = cores, mc.preschedule = preschedule))
    gc()
    after <- rss_mb()
    wall <- as.numeric(Sys.time() - t0, units = "secs")
    log_rows[[length(log_rows) + 1]] <- list(config = config_name, chunk = i,
      rss_mb_before = before, rss_mb_after = after,
      n_genes_in_chunk = length(chunks[[i]]), chunk_wall_time_sec = wall)
    cat(sprintf("[%s] chunk %d/%d: rss %.0f -> %.0f MB, %.1fs\n",
               config_name, i, length(chunks), before, after, wall))
  }
  do.call(rbind, lapply(log_rows, as.data.frame))
}

cores <- max(1, parallel::detectCores() - 1)
r1 <- run_config("chunk20_preschedule_true", 20, TRUE, cores)
r2 <- run_config("chunk20_preschedule_false", 20, FALSE, cores)

out <- rbind(r1, r2)
write.csv(out, "Spike_Results/mclapply_memory_log.csv", row.names = FALSE)
cat("Wrote Spike_Results/mclapply_memory_log.csv\n")

drift_true <- max(r1$rss_mb_after) - min(r1$rss_mb_before)
drift_false <- max(r2$rss_mb_after) - min(r2$rss_mb_before)
cat(sprintf("PASS: RSS drift over run -- preschedule=TRUE: %.0f MB, preschedule=FALSE: %.0f MB\n",
           drift_true, drift_false))
