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
  # RSS must be sampled from *this* forked worker's own PID -- the parent's
  # RSS never sees fork-local (copy-on-write) memory, and it is fully
  # reclaimed by the OS the instant the child exits regardless of what the
  # child leaked internally while alive.
  list(pid = Sys.getpid(), rss_mb = rss_mb(), gene = gene_col)
}

run_config <- function(config_name, chunk_size, preschedule, cores) {
  chunk_log_rows <- list()
  call_log_rows <- list()
  chunks <- split(gene_cols, ceiling(seq_along(gene_cols) / chunk_size))
  for (i in seq_along(chunks)) {
    before <- rss_mb()
    t0 <- Sys.time()
    results <- mclapply(chunks[[i]], fit_one, mc.cores = cores, mc.preschedule = preschedule)
    gc()
    after <- rss_mb()
    wall <- as.numeric(Sys.time() - t0, units = "secs")
    for (call_order in seq_along(results)) {
      r <- results[[call_order]]
      call_log_rows[[length(call_log_rows) + 1]] <- list(config = config_name, chunk = i,
        call_order = call_order, pid = r$pid, rss_mb = r$rss_mb, gene = r$gene)
    }
    chunk_log_rows[[length(chunk_log_rows) + 1]] <- list(config = config_name, chunk = i,
      rss_mb_before = before, rss_mb_after = after,
      n_genes_in_chunk = length(chunks[[i]]), chunk_wall_time_sec = wall)
    cat(sprintf("[%s] chunk %d/%d: parent rss %.0f -> %.0f MB, %.1fs\n",
               config_name, i, length(chunks), before, after, wall))
  }
  list(chunk_log = do.call(rbind, lapply(chunk_log_rows, as.data.frame)),
       call_log = do.call(rbind, lapply(call_log_rows, as.data.frame)))
}

# Cap cores well below chunk_size (20): if cores >= chunk_size, every item in
# a chunk gets its own core/fork even under mc.preschedule=TRUE, so no worker
# ever handles more than one gene and the leak-accumulation scenario this
# test exists to check can never occur, no matter what the code does. A small
# core count forces mc.preschedule=TRUE to queue multiple genes per fork.
cores <- min(max(1, parallel::detectCores() - 1), 4)
res1 <- run_config("chunk20_preschedule_true", 20, TRUE, cores)
res2 <- run_config("chunk20_preschedule_false", 20, FALSE, cores)

call_log <- rbind(res1$call_log, res2$call_log)
write.csv(call_log, "Spike_Results/mclapply_memory_log.csv", row.names = FALSE)
cat("Wrote Spike_Results/mclapply_memory_log.csv (per-call, per-PID granularity)\n")

# Per-PID RSS growth: only PIDs that handled >1 gene in sequence within the
# same persistent fork can show accumulation (mc.preschedule=FALSE forks
# fresh per item, so growth there must be ~0 by construction). Group by
# (chunk, pid), not pid alone -- each chunk is a fresh mclapply() call, so a
# fork lifetime never spans a chunk boundary, but the OS could in principle
# reuse the same PID number across two different chunks' forks. Grouping by
# pid alone would silently merge those two unrelated fork lifetimes and
# difference across them, corrupting the growth number with no warning.
per_pid_growth <- function(call_log) {
  max_growth <- 0
  call_log$group_key <- paste(call_log$chunk, call_log$pid, sep = "_")
  for (key in unique(call_log$group_key)) {
    sub <- call_log[call_log$group_key == key, ]
    sub <- sub[order(sub$call_order), ]
    if (nrow(sub) > 1) {
      growth <- sub$rss_mb[nrow(sub)] - sub$rss_mb[1]
      max_growth <- max(max_growth, growth)
    }
  }
  max_growth
}

max_growth_true <- per_pid_growth(res1$call_log)
max_growth_false <- per_pid_growth(res2$call_log)
n_pid_true <- length(unique(res1$call_log$pid))
n_pid_false <- length(unique(res2$call_log$pid))
busiest_true <- max(table(res1$call_log$pid))
busiest_false <- max(table(res2$call_log$pid))

cat(sprintf("PASS: max per-PID RSS growth (within a persistent fork's sequence of fits) -- preschedule=TRUE: %.0f MB (%d PIDs, busiest handled %d genes), preschedule=FALSE: %.0f MB (%d PIDs, busiest handled %d genes)\n",
           max_growth_true, n_pid_true, busiest_true, max_growth_false, n_pid_false, busiest_false))
