# Mixed-Effects Engine v2 (3-Stage Cascade + SHASH Calibration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `MixedEffectsModeling/core/` and `MixedEffectsModeling/validation/` around a 3-stage demotion cascade (drop the `intercept` fallback stage entirely; a gene that fails `nb_fixed` is excluded) and a SHASH-based per-gene Z-score calibration module, then run the full non-pooled sweep with the new engine.

**Architecture:** Same overall shape as the current engine (R `glmmTMB` per-gene cascade orchestrated from Python, `NormativeModelEngineMixed`), but the demotion chain shortens from `nbi -> nbi_disp_intercept -> nb_fixed -> intercept` to `nbi -> nbi_disp_intercept -> nb_fixed`, with `nb_fixed` failure now propagating to `route="excluded"` (previously masked by the always-succeeding `intercept` fallback). Every stage attempt (not just the final one) now records its own reject reason, so the gene fell to stage nb_fixed because reasons at nbi/nbi_disp_intercept are auditable. A new `core/calibration.py` wraps the existing `core/shash.py` primitives into per-gene SHASH fit + naive-vs-corrected exceedance/FDR diagnostics, reusable by both this round's smoke tests and next round's gene-level report. CV fold-level logging is implemented in `validation/cv_engine.py` (per-fold success/failure recorded explicitly, not silently dropped) but **not executed** this round — only the code is delivered.

**Tech Stack:** R (`glmmTMB`, `optparse`, `jsonlite`, `parallel::mclapply`), Python (`scipy`, `numpy`, `pandas`, `scikit-learn`, `rpy2`-free — R invoked via `subprocess`).

## Global Constraints

- No test suite/linter/CI in this repo (research code) — verification is "run the script, read the printed diagnostic output," not `pytest`. Every task's last step is a real script run with expected console output, not a unit test.
- No type hints on function signatures. No comments unless explaining a non-obvious WHY. English-only comments. Alphabetical imports. No alignment whitespace. No emoji.
- `MixedEffectsModeling/` stays fully independent of `Modeling/`/root `config.py` — any code shared between them is copy-pasted, never imported cross-package (existing convention, keep it).
- All paths/params come from `MixedEffectsModeling/config.py` — never re-declare a path or magic number that already has a config entry.
- `core/`: engine logic only. `validation/`: large-scale validation/CV pipelines only. Root-level `.ipynb`: visualization/analysis only (thin runner) — this plan does not touch any notebook.
- Never run `nbconvert` / execute notebooks yourself — the user runs notebooks by hand.
- Never overwrite `engine_state_mixed/`'s current contents without it being an explicit, called-out step in this plan (Task 9 is that step).

---

## Task 1: Finish legacifying the old engine (`core/` + `validation/` → `_legacy/`)

`_legacy/core_v1/` already has committed (staged) copies of the 7 old `core/` files from a prior session. The working-tree originals in `core/` are still sitting there un-removed, and `validation/` (4 files) hasn't been backed up at all. This task finishes that move so `core/`/`validation/` start clean for the new implementation.

**Files:**
- Delete: `MixedEffectsModeling/core/cv_glmm_engine.py`, `core/dispersion_trend.py`, `core/glmm_fit.R`, `core/glmm_fit_pool.R`, `core/glmm_helpers.R`, `core/marginal_rqr.py`, `core/model_engine_mixed.py` (already duplicated into `_legacy/core_v1/`)
- Move: `MixedEffectsModeling/validation/gene_level_diagnostics.py`, `validation/ppc_mixed.py`, `validation/run_glmm_full_unconstrained.py`, `validation/spike_in_power_test.py` → `MixedEffectsModeling/_legacy/validation_v1/`
- Keep as-is: `MixedEffectsModeling/core/shash.py` (already new, not legacy)

**Interfaces:** N/A (file move only).

- [ ] **Step 1: Remove the now-duplicated `core/` originals**

```bash
cd /project/cfRNA_NormativeModeling
git rm MixedEffectsModeling/core/cv_glmm_engine.py MixedEffectsModeling/core/dispersion_trend.py \
  MixedEffectsModeling/core/glmm_fit.R MixedEffectsModeling/core/glmm_fit_pool.R \
  MixedEffectsModeling/core/glmm_helpers.R MixedEffectsModeling/core/marginal_rqr.py \
  MixedEffectsModeling/core/model_engine_mixed.py
```

Expected: 7 files staged for deletion; `git status` shows them as `D` (they remain recoverable via `_legacy/core_v1/`, already staged as `A`).

- [ ] **Step 2: Move `validation/` originals into a new `_legacy/validation_v1/`**

```bash
mkdir -p MixedEffectsModeling/_legacy/validation_v1
git mv MixedEffectsModeling/validation/gene_level_diagnostics.py MixedEffectsModeling/_legacy/validation_v1/
git mv MixedEffectsModeling/validation/ppc_mixed.py MixedEffectsModeling/_legacy/validation_v1/
git mv MixedEffectsModeling/validation/run_glmm_full_unconstrained.py MixedEffectsModeling/_legacy/validation_v1/
git mv MixedEffectsModeling/validation/spike_in_power_test.py MixedEffectsModeling/_legacy/validation_v1/
```

Expected: `MixedEffectsModeling/validation/` now empty except `__pycache__/` (gitignored); `_legacy/validation_v1/` has the 4 files.

- [ ] **Step 3: Verify layout**

```bash
ls MixedEffectsModeling/core/ MixedEffectsModeling/validation/ MixedEffectsModeling/_legacy/core_v1/ MixedEffectsModeling/_legacy/validation_v1/
```

Expected: `core/` → only `shash.py` (+ `__pycache__`); `validation/` → empty; `_legacy/core_v1/` → 7 files; `_legacy/validation_v1/` → 4 files.

- [ ] **Step 4: Commit**

```bash
git add -A MixedEffectsModeling/_legacy MixedEffectsModeling/core MixedEffectsModeling/validation
git commit -m "chore: finish legacifying core_v1/validation_v1 for mixed-effects engine v2"
```

---

## Task 2: `config.py` — fix the latent missing `trend_min_nz` key

`core/dispersion_trend.py`'s `build_trend()` and `model_engine_mixed.py`'s `build_dispersion_trend()` both read `config.SPIKE_PARAMS["trend_min_nz"]`, but `config.py` never defines that key — a `KeyError` waiting to happen the moment the trend is rebuilt from scratch (it hasn't been hit yet because `engine_state_mixed/dispersion_trend.json` already exists and gets loaded instead). `dispersion_trend.json`'s own `min_nz` field records `30` as the value actually used to build it, so that's the correct value to backfill.

**Files:**
- Modify: `MixedEffectsModeling/config.py`

**Interfaces:**
- Produces: `config.SPIKE_PARAMS["trend_min_nz"]` (int), consumed by `core/dispersion_trend.py` and `core/model_engine_mixed.py` (Task 4/5).

- [ ] **Step 1: Add the key**

```python
SPIKE_PARAMS = {
    "beta_explode_thr": 3.0,
    "seed": 42,
    "rare_overdisp_thr": 2.0,
    "alpha_floor": 1e-2,
    "alpha_cap": 50.0,
    "n_splits": 5,
    "trend_min_nz": 30,
}
```

- [ ] **Step 2: Verify**

```bash
cd /project/cfRNA_NormativeModeling && python3 -c "import MixedEffectsModeling.config as config; print(config.SPIKE_PARAMS['trend_min_nz'])"
```

Expected: `30`.

- [ ] **Step 3: Commit**

```bash
git add MixedEffectsModeling/config.py
git commit -m "fix: add missing trend_min_nz to SPIKE_PARAMS"
```

---

## Task 3: R core — 3-stage cascade + per-stage reject reasons

Recreates `core/glmm_helpers.R`, `core/glmm_fit.R`, `core/glmm_fit_pool.R` in the new `core/`. `glmm_helpers.R`'s `fit_stage_gene` drops the `intercept`-specific `mu_fml`/`disp_fml` branches (that stage no longer exists) and now fails loudly (`switch` with no default) if an unexpected stage string is ever passed, instead of silently falling through. `glmm_fit.R`'s cascade shortens to `nbi -> nbi_disp_intercept -> nb_fixed`, force-returns at `nb_fixed` regardless of `ok` (so a converged-nowhere gene now correctly comes back `ok=FALSE` instead of masquerading as a trivial intercept fit), and accumulates each stage's own fail reason into 3 new output columns. `glmm_fit_pool.R` is copied over unchanged — pooling isn't run this round, but the file stays for when `nz_a_max` is picked later.

**Files:**
- Create: `MixedEffectsModeling/core/glmm_helpers.R`
- Create: `MixedEffectsModeling/core/glmm_fit.R`
- Create: `MixedEffectsModeling/core/glmm_fit_pool.R`

**Interfaces:**
- Produces: `fit_stage_gene(y, safe_names, X, batch, stage, fixed_log_theta, priors_df, beta_explode_thr, tau2_max, disp_intercept_max)` where `stage` must be one of `"nbi"`, `"nbi_disp_intercept"`, `"nb_fixed"` — returns `list(stage, ok, singular, tau2, mu_coef, disp_coef, fail_reason)`.
- Produces: `fit_pooled_glmm(Y_block, X, batch, mean_hc, eps, rare_overdisp_thr)` — unchanged signature/return, consumed later when pooling is re-enabled.
- Produces (CLI): `glmm_fit.R --mode cascade` writes a CSV with columns `gene, stage, ok, singular, tau2, fixed_alpha, fail_reason, nbi_reject_reason, nbi_disp_intercept_reject_reason, nb_fixed_reject_reason, mu_coef_0..10, disp_coef_0..10`. Consumed by `core/model_engine_mixed.py` (Task 4).

- [ ] **Step 1: `core/glmm_helpers.R`**

```r
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

# Fits ONE stage for ONE gene. Caller (glmm_fit.R) drives the demotion order.
# stage must be one of "nbi" / "nbi_disp_intercept" / "nb_fixed" -- the
# "intercept" fallback stage was removed (v2: a gene that fails nb_fixed is
# excluded outright rather than trivially "converging" on a 1-df model).
fit_stage_gene <- function(y, safe_names, X, batch, stage, fixed_log_theta,
                           priors_df, beta_explode_thr, tau2_max, disp_intercept_max) {
  df <- as.data.frame(X); colnames(df) <- safe_names
  df$y__ <- as.integer(round(y))
  df$batch__ <- factor(batch)
  if (!is.null(fixed_log_theta)) df$fixed_log_theta <- fixed_log_theta

  mu_fml <- as.formula(paste("y__ ~", paste(safe_names, collapse = " + "), "+ (1 | batch__)"))
  disp_fml <- switch(stage,
    nbi = as.formula(paste("~", paste(safe_names, collapse = " + "))),
    nbi_disp_intercept = as.formula("~ 1"),
    nb_fixed = as.formula("~ 0 + offset(fixed_log_theta)"))

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
```

- [ ] **Step 2: `core/glmm_fit.R`**

```r
suppressPackageStartupMessages({
  library(optparse); library(parallel); library(jsonlite)
})

.args <- commandArgs(trailingOnly = FALSE)
.script_path <- sub("--file=", "", grep("--file=", .args, value = TRUE))
source(file.path(dirname(normalizePath(.script_path)), "glmm_helpers.R"))

opt <- parse_args(OptionParser(option_list = list(
  make_option("--x", type = "character"), make_option("--y", type = "character"),
  make_option("--batch", type = "character"), make_option("--genes", type = "character"),
  make_option("--trend", type = "character"), make_option("--mode", type = "character", default = "cascade"),
  make_option("--out", type = "character"), make_option("--chunk-size", type = "integer", default = 200),
  make_option("--cores", type = "integer", default = min(parallel::detectCores() - 1, 8))
)))

X <- as.matrix(read.csv(opt$x, row.names = 1))
Y <- read.csv(opt$y, row.names = 1)
batch <- read.csv(opt$batch, row.names = 1)[[1]]
gene_meta <- read.csv(opt$genes)  # columns: gene, [stage] (stage only needed for fixed_stage mode)
trend <- fromJSON(opt$trend)
alpha_of <- function(mean_y) {
  lm <- log(max(mean_y, 1e-8))
  s <- exp(approx(trend$lowess_logmu, trend$lowess_logsigma, xout = lm, rule = 2)$y)
  min(max(s, trend$alpha_floor), trend$alpha_cap)
}
safe_names <- sanitize_names(colnames(X))
colnames(X) <- safe_names
BETA_EXPLODE_THR <- 3.0
TAU2_MAX <- BETA_EXPLODE_THR^2
DISP_INTERCEPT_MAX <- 10.0
priors_df <- data.frame(prior = "normal(0, 0.05)", class = "betad", coef = "")

done_genes <- character(0)
if (file.exists(opt$out)) done_genes <- read.csv(opt$out)$gene

# v2: 3-stage cascade (nbi -> nbi_disp_intercept -> nb_fixed), the old
# "intercept" fallback is gone. force-return at nb_fixed regardless of ok --
# a gene that fails there is genuinely unmodelable and comes back ok=FALSE
# (caller routes it to "excluded"), instead of always trivially "succeeding"
# on a 1-df intercept model. Every stage's own reject reason is preserved,
# not just the final stage's, for downstream per-gene auditing.
fit_one_cascade <- function(g) {
  y <- as.numeric(Y[[g]])
  alpha_g <- alpha_of(mean(y))
  fixed_log_theta <- rep(-log(alpha_g), length(y))
  reasons <- list(nbi = "", nbi_disp_intercept = "", nb_fixed = "")
  for (stage in c("nbi", "nbi_disp_intercept", "nb_fixed")) {
    pr <- if (stage == "nbi") priors_df else NULL
    r <- fit_stage_gene(y, safe_names, X, batch, stage, fixed_log_theta, pr, BETA_EXPLODE_THR, TAU2_MAX, DISP_INTERCEPT_MAX)
    reasons[[stage]] <- if (isTRUE(r$ok)) "" else r$fail_reason
    if (isTRUE(r$ok) || stage == "nb_fixed") {
      gc()
      return(c(list(gene = g, fixed_alpha = alpha_g,
                    nbi_reject_reason = reasons$nbi,
                    nbi_disp_intercept_reject_reason = reasons$nbi_disp_intercept,
                    nb_fixed_reject_reason = reasons$nb_fixed), r))
    }
  }
}

fit_one_fixed <- function(g) {
  stage <- gene_meta$stage[gene_meta$gene == g]
  y <- as.numeric(Y[[g]])
  alpha_g <- alpha_of(mean(y))
  fixed_log_theta <- rep(-log(alpha_g), length(y))
  pr <- if (stage == "nbi") priors_df else NULL
  r <- fit_stage_gene(y, safe_names, X, batch, stage, fixed_log_theta, pr, BETA_EXPLODE_THR, TAU2_MAX, DISP_INTERCEPT_MAX)
  gc()
  c(list(gene = g, fixed_alpha = alpha_g,
        nbi_reject_reason = "", nbi_disp_intercept_reject_reason = "", nb_fixed_reject_reason = ""), r)
}

worker <- if (opt$mode == "cascade") fit_one_cascade else fit_one_fixed
genes_todo <- setdiff(gene_meta$gene, done_genes)
chunks <- split(genes_todo, ceiling(seq_along(genes_todo) / opt$`chunk-size`))

t0 <- Sys.time()
n_ok_cum <- length(done_genes)
n_total_cum <- length(done_genes)
for (i in seq_along(chunks)) {
  results <- mclapply(chunks[[i]], worker, mc.cores = opt$cores)
  rows <- lapply(results, function(r) {
    p <- 11  # 1 intercept + 10 covariates
    mu_padded <- c(r$mu_coef, rep(NA, p - length(r$mu_coef)))[1:p]
    disp_padded <- c(r$disp_coef, rep(NA, p - length(r$disp_coef)))[1:p]
    row <- c(list(gene = r$gene, stage = r$stage, ok = r$ok, singular = r$singular,
                 tau2 = r$tau2, fixed_alpha = r$fixed_alpha, fail_reason = r$fail_reason,
                 nbi_reject_reason = r$nbi_reject_reason,
                 nbi_disp_intercept_reject_reason = r$nbi_disp_intercept_reject_reason,
                 nb_fixed_reject_reason = r$nb_fixed_reject_reason))
    for (j in seq_len(p)) { row[[paste0("mu_coef_", j-1)]] <- mu_padded[j]; row[[paste0("disp_coef_", j-1)]] <- disp_padded[j] }
    row
  })
  df <- do.call(rbind, lapply(rows, as.data.frame))
  write.table(df, opt$out, sep = ",", append = file.exists(opt$out), col.names = !file.exists(opt$out), row.names = FALSE)
  gc()

  n_ok_cum <- n_ok_cum + sum(df$ok, na.rm = TRUE)
  n_total_cum <- n_total_cum + nrow(df)
  elapsed_min <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
  eta_min <- (length(chunks) - i) * (elapsed_min / i)
  stage_counts <- paste(sprintf("%s=%d", names(table(df$stage)), table(df$stage)), collapse = ",")
  cat(sprintf("[%s] chunk %d/%d done (%d genes, ok_rate=%.2f, %s) | elapsed=%.1fmin eta=%.1fmin | cum_ok_rate=%.3f (%d/%d)\n",
             format(Sys.time(), "%H:%M:%S"), i, length(chunks), nrow(df), mean(df$ok, na.rm = TRUE), stage_counts,
             elapsed_min, eta_min, n_ok_cum / n_total_cum, n_ok_cum, n_total_cum))
}
cat("DONE\n")
```

- [ ] **Step 3: `core/glmm_fit_pool.R`** (unchanged copy, unused this round)

```r
suppressPackageStartupMessages({
  library(optparse); library(jsonlite)
})

.args <- commandArgs(trailingOnly = FALSE)
.script_path <- sub("--file=", "", grep("--file=", .args, value = TRUE))
source(file.path(dirname(normalizePath(.script_path)), "glmm_helpers.R"))

opt <- parse_args(OptionParser(option_list = list(
  make_option("--x", type = "character"), make_option("--y", type = "character"),
  make_option("--batch", type = "character"), make_option("--genes", type = "character"),
  make_option("--rare-overdisp-thr", type = "double", default = 2.0),
  make_option("--out", type = "character")
)))

X <- as.matrix(read.csv(opt$x, row.names = 1))
Y <- as.matrix(read.csv(opt$y, row.names = 1))
batch <- read.csv(opt$batch, row.names = 1)[[1]]
genes <- read.csv(opt$genes)$gene
n_hc <- nrow(X)
eps <- 1 / (2 * n_hc)
mean_hc <- colMeans(Y)

fit <- fit_pooled_glmm(Y, X, batch, mean_hc, eps, opt$`rare-overdisp-thr`)

out <- list(ok = isTRUE(fit$ok), family = if (isTRUE(fit$ok)) fit$family else NA,
           beta = if (isTRUE(fit$ok)) as.numeric(fit$beta) else numeric(0),
           alpha = if (isTRUE(fit$ok) && !is.na(fit$alpha)) as.numeric(fit$alpha) else NA,
           tau2 = if (isTRUE(fit$ok)) as.numeric(fit$tau2) else NA,
           overdisp_ratio = if (isTRUE(fit$ok)) as.numeric(fit$overdisp_ratio) else NA,
           mult_lo = if (isTRUE(fit$ok)) as.numeric(fit$mult_lo) else NA,
           mult_hi = if (isTRUE(fit$ok)) as.numeric(fit$mult_hi) else NA,
           eps = eps, gene = genes, mean_hc = as.numeric(mean_hc))
write(toJSON(out, auto_unbox = TRUE, na = "null"), opt$out)
cat("DONE\n")
```

- [ ] **Step 4: Syntax-check all three R files**

```bash
cd /project/cfRNA_NormativeModeling/MixedEffectsModeling/core
Rscript -e 'invisible(parse("glmm_helpers.R")); invisible(parse("glmm_fit.R")); invisible(parse("glmm_fit_pool.R")); cat("OK\n")'
```

Expected: `OK` (no parse errors). This only checks R syntax, not runtime behavior — Task 8's smoke run is the real check.

- [ ] **Step 5: Commit**

```bash
git add MixedEffectsModeling/core/glmm_helpers.R MixedEffectsModeling/core/glmm_fit.R MixedEffectsModeling/core/glmm_fit_pool.R
git commit -m "feat: 3-stage glmm cascade (drop intercept fallback), per-stage reject reasons"
```

---

## Task 4: Python core — `dispersion_trend.py` + `marginal_rqr.py` (unchanged copies)

Phase 0 dispersion trend and marginal RQR math are untouched by the intercept-stage removal (trend is covariate-free; RQR math doesn't know about demotion stages). Recreate them verbatim in the new `core/` so `model_engine_mixed.py` (Task 5) has its dependencies.

**Files:**
- Create: `MixedEffectsModeling/core/dispersion_trend.py`
- Create: `MixedEffectsModeling/core/marginal_rqr.py`

**Interfaces:**
- Produces: `build_trend(Y_hc, min_nz=None, ...)`, `save_trend(trend, path=None)`, `load_trend(path=None) -> alpha_of(mean)`. Consumed by Task 5.
- Produces: `_poisson_rqr`, `_nb_rqr`, `marginal_nb_loglik`, `marginal_nb_rqr`. Consumed by Task 5 and Task 8 (CV).

- [ ] **Step 1: `core/dispersion_trend.py`**

```python
"""Covariate-free mean-dispersion trend for the Route B (Tier 2) engine.

Ignore covariates entirely. For every HC gene compute the closed-form NB2
method-of-moments dispersion from the raw counts:
    sigma_j = max(0, (var_j - mean_j) / mean_j^2)      (Var = mu + sigma*mu^2)
Genes with too few nonzero HC samples (nz < trend_min_nz) give a structurally
noisy MoM estimate (a single outlying observation can dominate the sample
variance) and are excluded from trend fitting -- not trimmed as outliers, since
the noise is not an outlier problem but an information-poverty problem.

The reliable genes are summarized by nonzero-weighted median in log(mu) bins,
then smoothed with lowess in log-log space (log(sigma) ~ log(mu)); a single
log-log line underfits both the low-mu plateau and the high-mu asymptote, so
lowess is the canonical trend, matching edgeR/DESeq2 trended-dispersion shape.

Route B modeling fixes each gene's dispersion at alpha_of(mean_train), so the
covariates spend all their degrees of freedom on the mean.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

import MixedEffectsModeling.config as config

MP = config.SPIKE_PARAMS


def weighted_median(x, w):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def build_trend(Y_hc, min_nz=None, n_bins=25, min_bin=20, lowess_frac=0.5):
    """Y_hc: (n_hc, n_genes) raw HC counts. Returns dict with lowess log-log curve."""
    min_nz = MP["trend_min_nz"] if min_nz is None else min_nz
    nz = (Y_hc > 0).sum(0).astype(int)
    mean_c = Y_hc.mean(0)
    var_c = Y_hc.var(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_mom = np.where(mean_c > 0, (var_c - mean_c) / mean_c ** 2, np.nan)
    sigma_mom = np.clip(sigma_mom, 0, None)

    reliable = (nz >= min_nz) & (mean_c > 0) & np.isfinite(sigma_mom)
    mu = mean_c[reliable]
    sig = sigma_mom[reliable]
    w = nz[reliable].astype(float)

    edges = np.geomspace(mu.min(), mu.max(), n_bins + 1)
    binid = np.clip(np.digitize(mu, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = binid == b
        if m.sum() < min_bin:
            continue
        rows.append({"mu_bin": weighted_median(mu[m], w[m]),
                     "sigma_wmed": weighted_median(sig[m], w[m]), "n": int(m.sum())})
    bins = pd.DataFrame(rows).sort_values("mu_bin")

    blx, bly = np.log(bins["mu_bin"].values), np.log(bins["sigma_wmed"].values)
    sm_curve = lowess(bly, blx, frac=lowess_frac, return_sorted=True)

    return {
        "a0": None, "a1": None,  # legacy parametric slot, unused (lowess is canonical)
        "alpha_floor": MP["alpha_floor"], "alpha_cap": MP["alpha_cap"],
        "min_nz": min_nz, "n_reliable": int(reliable.sum()), "n_bins_used": len(bins),
        "lowess_logmu": sm_curve[:, 0].tolist(), "lowess_logsigma": sm_curve[:, 1].tolist(),
    }


def save_trend(trend, path=None):
    path = Path(path or config.DISPERSION_TREND_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trend, indent=2))


def load_trend(path=None):
    """Returns alpha_of(mean) -> fixed NB2 dispersion for Route B scoring/training."""
    path = Path(path or config.DISPERSION_TREND_PATH)
    cf = json.loads(path.read_text())
    logmu = np.asarray(cf["lowess_logmu"])
    logsig = np.asarray(cf["lowess_logsigma"])
    floor, cap = cf["alpha_floor"], cf["alpha_cap"]

    def alpha_of(mean):
        lm = np.log(max(float(mean), 1e-8))
        s = float(np.exp(np.interp(lm, logmu, logsig, left=logsig[0], right=logsig[-1])))
        return float(np.clip(s, floor, cap))

    return alpha_of
```

- [ ] **Step 2: `core/marginal_rqr.py`**

```python
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import nbinom, norm, poisson

RQR_EPS = 1e-8


def _poisson_rqr(y, mu, seed=None):
    y = np.asarray(y)
    lo = np.where(y > 0, poisson.cdf(y - 1, mu), 0.0)
    hi = poisson.cdf(y, mu)
    lo = np.clip(lo, RQR_EPS, 1 - RQR_EPS); hi = np.clip(hi, RQR_EPS, 1 - RQR_EPS)
    rng = np.random.default_rng(seed)
    return norm.ppf(rng.uniform(np.minimum(lo, hi), np.maximum(lo, hi))).astype(np.float32)


def _nb_cdf(y, mu, alpha):
    n = 1.0 / alpha
    p = np.clip(n / (n + mu), RQR_EPS, 1 - RQR_EPS)
    return nbinom.cdf(y, n, p)


def _nb_rqr(y, mu, alpha, seed=None):
    y = np.asarray(y)
    n = 1.0 / alpha
    p = np.clip(n / (n + mu), RQR_EPS, 1 - RQR_EPS)
    lo = np.where(y > 0, nbinom.cdf(y - 1, n, p), 0.0)
    hi = nbinom.cdf(y, n, p)
    lo = np.clip(lo, RQR_EPS, 1 - RQR_EPS); hi = np.clip(hi, RQR_EPS, 1 - RQR_EPS)
    rng = np.random.default_rng(seed)
    return norm.ppf(rng.uniform(np.minimum(lo, hi), np.maximum(lo, hi))).astype(np.float32)


def _nb_logpmf(y, mu, alpha):
    n = 1.0 / alpha
    p = np.clip(n / (n + mu), RQR_EPS, 1 - RQR_EPS)
    return nbinom.logpmf(y, n, p)


def marginal_nb_loglik(y, mu, alpha, tau2, n_nodes=7):
    y = np.asarray(y)
    tau2 = np.asarray(tau2, dtype=np.float64)
    if np.all(tau2 < 1e-6):
        return _nb_logpmf(y, mu, alpha)
    nodes, weights = hermegauss(n_nodes)
    weights = weights / weights.sum()
    sd = np.sqrt(np.maximum(tau2, 0.0))
    logpmf_k = np.stack([
        _nb_logpmf(y, mu * np.exp(sd * node), alpha) + np.log(w)
        for node, w in zip(nodes, weights)
    ])
    m = logpmf_k.max(axis=0)
    return m + np.log(np.exp(logpmf_k - m).sum(axis=0))


def marginal_nb_rqr(y, mu, alpha, tau2, seed, n_nodes=7):
    y = np.asarray(y)
    tau2 = np.asarray(tau2, dtype=np.float64)
    if np.all(tau2 < 1e-6):
        return _nb_rqr(y, mu, alpha, seed)

    nodes, weights = hermegauss(n_nodes)
    weights = weights / weights.sum()
    sd = np.sqrt(np.maximum(tau2, 0.0))
    lo = np.zeros_like(y, dtype=np.float64)
    hi = np.zeros_like(y, dtype=np.float64)
    for node, w in zip(nodes, weights):
        mu_b = mu * np.exp(sd * node)
        lo += w * np.where(y > 0, _nb_cdf(y - 1, mu_b, alpha), 0.0)
        hi += w * _nb_cdf(y, mu_b, alpha)
    lo = np.clip(lo, RQR_EPS, 1 - RQR_EPS)
    hi = np.clip(hi, RQR_EPS, 1 - RQR_EPS)
    rng = np.random.default_rng(seed)
    return norm.ppf(rng.uniform(np.minimum(lo, hi), np.maximum(lo, hi))).astype(np.float32)
```

- [ ] **Step 3: Verify import + a sanity numeric check**

```bash
cd /project/cfRNA_NormativeModeling && python3 -c "
import numpy as np
from MixedEffectsModeling.core.marginal_rqr import marginal_nb_rqr
y = np.array([3.0, 0.0, 10.0]); mu = np.array([4.0, 2.0, 9.0]); alpha = np.array([0.5, 0.5, 0.5]); tau2 = np.array([0.0, 0.0, 0.0])
print(marginal_nb_rqr(y, mu, alpha, tau2, seed=1))
"
```

Expected: 3 finite float values (no exception).

- [ ] **Step 4: Commit**

```bash
git add MixedEffectsModeling/core/dispersion_trend.py MixedEffectsModeling/core/marginal_rqr.py
git commit -m "feat: restore dispersion_trend/marginal_rqr in new core/ (unchanged logic)"
```

---

## Task 5: `core/model_engine_mixed.py` — 3-stage `GeneRecordMixed` + reject-reason plumbing

`GeneRecordMixed` gains 3 new string fields for per-stage reject reasons (empty string = that stage wasn't attempted or succeeded there). `train()` reads the 3 new CSV columns `glmm_fit.R` now emits (Task 3) into those fields. Everything else (`load_hc_data`, `assign_routes`, `train_pool`, `score`, `save`/`load`) is functionally unchanged — pooling code stays present but unused since `nz_a_max` defaults to 0 (no `Threshold_Sweep/nz_a_max.txt` yet), so every gene routes to `"model"`.

**Files:**
- Create: `MixedEffectsModeling/core/model_engine_mixed.py`

**Interfaces:**
- Consumes: `build_trend/load_trend/save_trend` (Task 4), `_poisson_rqr/marginal_nb_rqr` (Task 4), `config.GLMM_FIT_R`/`config.GLMM_FIT_POOL_R`/`config.DISPERSION_TREND_PATH`/`config.ENGINE_MIXED_DIR`/`config.BIAS_COLUMNS`/`config.H5AD_PATH`.
- Produces: `NormativeModelEngineMixed` with `.genes: dict[str, GeneRecordMixed]`, `GeneRecordMixed.nbi_reject_reason/nbi_disp_intercept_reject_reason/nb_fixed_reject_reason: str`. Consumed by Task 7 (`run_engine.py`) and, next round, by the gene-report generator and `validation/cv_engine.py` (Task 8, code only).

- [ ] **Step 1: Write the file**

```python
import json
import pickle
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.dispersion_trend import build_trend, load_trend, save_trend
from MixedEffectsModeling.core.marginal_rqr import _poisson_rqr, marginal_nb_rqr

MP = config.SPIKE_PARAMS


@dataclass
class GeneRecordMixed:
    name: str
    route: str = ""
    stage: str = ""
    nz: int = 0
    ok: bool = False
    singular: bool = False
    tau2: float = 0.0
    mu_coef: np.ndarray = None
    disp_coef: np.ndarray = None
    fail_reason: str = ""
    nbi_reject_reason: str = ""
    nbi_disp_intercept_reject_reason: str = ""
    nb_fixed_reject_reason: str = ""
    mean_hc: float = None
    fixed_alpha: float = None


class NormativeModelEngineMixed:
    def __init__(self):
        self.X_hc_scaled = None
        self.Y_hc = None
        self.scaler = None
        self.batch = None
        self.pc_gene_names = []
        self._gene_col = {}
        self.genes = {}
        self.alpha_fn = None
        self.nz_a_max = None
        self.rare_glm = None

    def load_hc_data(self, h5ad_path=config.H5AD_PATH):
        adata = sc.read_h5ad(h5ad_path)
        adata = adata[adata.obs["QC_Passed"] == True]
        adata = adata[adata.obs["Phenotype_Processed"].notna()]
        adata = adata[adata.obs["Phenotype_Processed"] != "Unknown"]
        adata = adata[adata.obs["broad_protocol_category"] != "Exome-based (EB)"]
        is_hc = (adata.obs["Phenotype_Processed"].astype(str) == "Healthy Control").values
        X_raw = adata.obs[config.BIAS_COLUMNS].values.astype(np.float64)
        self.scaler = StandardScaler()
        self.X_hc_scaled = self.scaler.fit_transform(X_raw[is_hc])
        self.batch = adata.obs["Batch_ID"].astype(str).values[is_hc]
        Y_raw = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
        self.Y_hc = np.round(Y_raw[is_hc]).astype(np.float64)
        is_pc = (adata.var["GeneType"] == "protein_coding").values
        self.pc_gene_names = adata.var_names[is_pc].tolist()
        pc_indices = np.where(is_pc)[0]
        self._gene_col = {g: pc_indices[i] for i, g in enumerate(self.pc_gene_names)}

    def build_dispersion_trend(self):
        Y_pc = self.Y_hc[:, list(self._gene_col.values())]
        trend = build_trend(Y_pc, min_nz=MP["trend_min_nz"])
        save_trend(trend)
        self.alpha_fn = load_trend()

    def assign_routes(self):
        # nz_a_max is deferred -- default to 0 (no gene routed to "pool", every
        # gene attempts the model cascade) until a real threshold is chosen and
        # Threshold_Sweep/nz_a_max.txt exists.
        nz_a_max_path = config.THRESHOLD_SWEEP_DIR / "nz_a_max.txt"
        self.nz_a_max = int(nz_a_max_path.read_text().strip()) if nz_a_max_path.exists() else 0
        nz = (self.Y_hc[:, list(self._gene_col.values())] > 0).sum(axis=0)
        for i, g in enumerate(self.pc_gene_names):
            n = int(nz[i])
            route = "pool" if n < self.nz_a_max else "model"
            self.genes[g] = GeneRecordMixed(name=g, route=route, nz=n)

    def train(self, limit=None, tmp_dir="/tmp/glmm_train"):
        Path(tmp_dir).mkdir(exist_ok=True)
        model_genes = [g for g, r in self.genes.items() if r.route == "model"][:limit]
        pd.DataFrame(self.X_hc_scaled, columns=config.BIAS_COLUMNS).to_csv(f"{tmp_dir}/X.csv.gz")
        Y_model = self.Y_hc[:, [self._gene_col[g] for g in model_genes]]
        pd.DataFrame(Y_model, columns=model_genes).to_csv(f"{tmp_dir}/Y.csv.gz")
        pd.DataFrame({"Batch_ID": self.batch}).to_csv(f"{tmp_dir}/batch.csv.gz")
        pd.DataFrame({"gene": model_genes}).to_csv(f"{tmp_dir}/genes.csv", index=False)

        subprocess.run([
            "Rscript", str(config.GLMM_FIT_R), "--x", f"{tmp_dir}/X.csv.gz", "--y", f"{tmp_dir}/Y.csv.gz",
            "--batch", f"{tmp_dir}/batch.csv.gz", "--genes", f"{tmp_dir}/genes.csv",
            "--trend", str(config.DISPERSION_TREND_PATH), "--mode", "cascade", "--out", f"{tmp_dir}/results.csv",
        ], check=True, cwd=str(config.GLMM_FIT_R.parent))

        results = pd.read_csv(f"{tmp_dir}/results.csv").set_index("gene")
        for g, row in results.iterrows():
            rec = self.genes[g]
            rec.stage, rec.ok, rec.singular, rec.tau2 = row["stage"], bool(row["ok"]), bool(row["singular"]), float(row["tau2"])
            rec.fixed_alpha = float(row["fixed_alpha"]) if "fixed_alpha" in row and not pd.isna(row["fixed_alpha"]) else None
            rec.mu_coef = row[[c for c in results.columns if c.startswith("mu_coef_")]].values.astype(float)
            rec.disp_coef = row[[c for c in results.columns if c.startswith("disp_coef_")]].values.astype(float)
            rec.fail_reason = row["fail_reason"]
            rec.nbi_reject_reason = row["nbi_reject_reason"] if not pd.isna(row["nbi_reject_reason"]) else ""
            rec.nbi_disp_intercept_reject_reason = row["nbi_disp_intercept_reject_reason"] if not pd.isna(row["nbi_disp_intercept_reject_reason"]) else ""
            rec.nb_fixed_reject_reason = row["nb_fixed_reject_reason"] if not pd.isna(row["nb_fixed_reject_reason"]) else ""
            if not rec.ok:
                rec.route = "excluded"

        self.train_pool(tmp_dir=tmp_dir)

    def train_pool(self, tmp_dir="/tmp/glmm_train"):
        """Route "pool": one shared-beta pooled GLM (+ batch random intercept)
        fit jointly across all pool-route genes. Unused this round (nz_a_max
        defaults to 0, so no gene ever routes to "pool"); kept so re-enabling
        pooling later needs no changes here."""
        pool_genes = [g for g, r in self.genes.items() if r.route == "pool"]
        if not pool_genes:
            return
        Path(tmp_dir).mkdir(exist_ok=True)
        Y_pool = self.Y_hc[:, [self._gene_col[g] for g in pool_genes]]
        pd.DataFrame(Y_pool, columns=pool_genes).to_csv(f"{tmp_dir}/Y_pool.csv.gz")
        pd.DataFrame({"gene": pool_genes}).to_csv(f"{tmp_dir}/genes_pool.csv", index=False)

        subprocess.run([
            "Rscript", str(config.GLMM_FIT_POOL_R), "--x", f"{tmp_dir}/X.csv.gz", "--y", f"{tmp_dir}/Y_pool.csv.gz",
            "--batch", f"{tmp_dir}/batch.csv.gz", "--genes", f"{tmp_dir}/genes_pool.csv",
            "--rare-overdisp-thr", str(MP["rare_overdisp_thr"]), "--out", f"{tmp_dir}/results_pool.json",
        ], check=True, cwd=str(config.GLMM_FIT_POOL_R.parent))

        with open(f"{tmp_dir}/results_pool.json") as f:
            fit = json.load(f)

        if not fit["ok"]:
            for g in pool_genes:
                self.genes[g].route = "excluded"
                self.genes[g].fail_reason = "fit_pooled_glmm failed"
            return

        n_hc = self.X_hc_scaled.shape[0]
        self.rare_glm = {"family": fit["family"], "beta": np.asarray(fit["beta"]),
                         "alpha": fit["alpha"], "eps": 1.0 / (2 * n_hc),
                         "tau2": float(fit["tau2"]) if fit.get("tau2") is not None else 0.0,
                         "mult_lo": fit["mult_lo"], "mult_hi": fit["mult_hi"]}
        for g, m in zip(fit["gene"], fit["mean_hc"]):
            rec = self.genes[g]
            rec.mean_hc, rec.ok, rec.stage = float(m), True, "pool"

    def training_summary(self):
        rows = [dict(gene=r.name, route=r.route, stage=r.stage, nz=r.nz, ok=r.ok,
                    singular=r.singular, tau2=r.tau2, fail_reason=r.fail_reason,
                    nbi_reject_reason=r.nbi_reject_reason,
                    nbi_disp_intercept_reject_reason=r.nbi_disp_intercept_reject_reason,
                    nb_fixed_reject_reason=r.nb_fixed_reject_reason)
               for r in self.genes.values()]
        return pd.DataFrame(rows).set_index("gene")

    def score(self, X_test_raw, Y_test, gene_names=None, seed=42, as_dict=False):
        gene_names = gene_names or [g for g in self.genes if self.genes[g].ok]
        X_test = self.scaler.transform(X_test_raw.astype(np.float64))
        Xa = np.column_stack([np.ones(len(X_test)), X_test])
        Z = np.full((len(X_test), len(gene_names)), np.nan, dtype=np.float32)
        for j, g in enumerate(gene_names):
            rec = self.genes.get(g)
            if rec is None or not rec.ok:
                continue
            if rec.route == "pool":
                mult = np.exp(X_test @ self.rare_glm["beta"][1:])
                if "mult_lo" in self.rare_glm and self.rare_glm["mult_lo"] is not None:
                    mult = np.clip(mult, self.rare_glm["mult_lo"], self.rare_glm["mult_hi"])
                mu = np.clip((rec.mean_hc + self.rare_glm["eps"]) * np.exp(self.rare_glm["beta"][0]) * mult, 1e-6, 1e8)
                if self.rare_glm["family"] == "poisson":
                    Z[:, j] = _poisson_rqr(Y_test[:, j].astype(np.float64), mu, seed + j)
                else:
                    Z[:, j] = marginal_nb_rqr(Y_test[:, j].astype(np.float64), mu, self.rare_glm["alpha"],
                                              self.rare_glm.get("tau2", 0.0), seed + j)
                continue
            mu = np.clip(np.exp(Xa @ np.nan_to_num(rec.mu_coef, nan=0.0)), 1e-6, 1e8)
            if not np.all(np.isnan(rec.disp_coef)):
                alpha = np.exp(-Xa @ np.nan_to_num(rec.disp_coef, nan=0.0))
            elif rec.fixed_alpha is not None:
                alpha = np.full(len(X_test), rec.fixed_alpha)
            else:
                alpha = np.full(len(X_test), self.alpha_fn(float(mu.mean())))
            Z[:, j] = marginal_nb_rqr(Y_test[:, j].astype(np.float64), mu, alpha, rec.tau2, seed + j)
        return Z if not as_dict else {"combined": Z, "gene_names": list(gene_names)}

    def save(self, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "genes.pkl", "wb") as f: pickle.dump(self.genes, f)
        with open(directory / "scaler.pkl", "wb") as f: pickle.dump(self.scaler, f)
        if self.rare_glm is not None:
            with open(directory / "rare_glm.pkl", "wb") as f: pickle.dump(self.rare_glm, f)
        self.training_summary().to_csv(directory / "training_summary.csv")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        engine = cls()
        with open(directory / "genes.pkl", "rb") as f: engine.genes = pickle.load(f)
        with open(directory / "scaler.pkl", "rb") as f: engine.scaler = pickle.load(f)
        rare_glm_path = directory / "rare_glm.pkl"
        if rare_glm_path.exists():
            with open(rare_glm_path, "rb") as f: engine.rare_glm = pickle.load(f)
        engine.alpha_fn = load_trend()
        return engine
```

- [ ] **Step 2: Verify import**

```bash
cd /project/cfRNA_NormativeModeling && python3 -c "
from MixedEffectsModeling.core.model_engine_mixed import NormativeModelEngineMixed, GeneRecordMixed
r = GeneRecordMixed(name='X')
print(r.nbi_reject_reason, r.nbi_disp_intercept_reject_reason, r.nb_fixed_reject_reason)
"
```

Expected: three empty strings printed, no exception.

- [ ] **Step 3: Commit**

```bash
git add MixedEffectsModeling/core/model_engine_mixed.py
git commit -m "feat: GeneRecordMixed carries per-stage reject reasons (v2 3-stage engine)"
```

---

## Task 6: `core/calibration.py` — SHASH per-gene calibration module

Wraps `core/shash.py`'s MLE fit into per-gene diagnostics: naive vs SHASH-corrected exceedance at the 95% level, corrected skew/kurtosis, and naive-vs-SHASH-corrected BH-FDR reject rate on the gene's own held-out Z values (since held-out HC is a true null, any nonzero FDR-reject rate is exactly the false-positive inflation the SHASH correction is meant to fix). This generalizes what `demotion_dispersion_diagnostics.ipynb` Section 16 did at group level (control/problem pools of genes) down to a per-gene function, reusable by both a smoke test here and the deferred gene-report generator.

**Files:**
- Create: `MixedEffectsModeling/core/calibration.py`

**Interfaces:**
- Consumes: `core/shash.py`'s `fit_shash`, `shash_quantile`, `shash_transform_to_z`.
- Produces: `bh_fdr_reject(pvals, q=0.05) -> np.ndarray[bool]`, `gene_shash_calibration(z) -> dict` with keys `shash_ok, shash_xi, shash_eta, shash_eps, shash_delta, z_lo, z_hi, raw_skew, raw_kurtosis, corrected_skew, corrected_kurtosis, naive_exceed, shash_exceed, naive_fdr_reject_rate, corr_fdr_reject_rate`. Consumed next round by the gene-report generator.

- [ ] **Step 1: Write the file**

```python
import numpy as np
from scipy.stats import kurtosis, norm, skew

from MixedEffectsModeling.core.shash import fit_shash, shash_quantile, shash_transform_to_z


def bh_fdr_reject(pvals, q=0.05):
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    reject = np.zeros(n, dtype=bool)
    if passed.any():
        k_max = np.nonzero(passed)[0].max()
        reject[order[:k_max + 1]] = True
    return reject


# z is expected to be one gene's held-out (CV) RQR Z-scores -- held-out HC is
# a true null, so naive_fdr_reject_rate/corr_fdr_reject_rate directly measure
# false-positive inflation under the naive N(0,1) assumption vs after SHASH
# warping (Fraza et al. 2021 NeuroImage; Efron 2007 Annals of Statistics).
def gene_shash_calibration(z):
    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z)]
    xi, eta, eps, delta, ok = fit_shash(z)
    z_lo, z_hi = shash_quantile(np.array([0.025, 0.975]), xi, eta, eps, delta)
    naive_exceed = float(np.mean(np.abs(z) > 1.96))
    shash_exceed = float(np.mean((z < z_lo) | (z > z_hi)))
    z_corr = shash_transform_to_z(z, xi, eta, eps, delta) if ok else z.copy()
    p_naive = 2 * norm.sf(np.abs(z))
    p_corr = 2 * norm.sf(np.abs(z_corr))
    naive_fdr = float(bh_fdr_reject(p_naive).mean())
    corr_fdr = float(bh_fdr_reject(p_corr).mean())
    return dict(
        shash_ok=ok, shash_xi=float(xi), shash_eta=float(eta), shash_eps=float(eps), shash_delta=float(delta),
        z_lo=float(z_lo), z_hi=float(z_hi),
        raw_skew=float(skew(z)), raw_kurtosis=float(kurtosis(z)),
        corrected_skew=float(skew(z_corr)), corrected_kurtosis=float(kurtosis(z_corr)),
        naive_exceed=naive_exceed, shash_exceed=shash_exceed,
        naive_fdr_reject_rate=naive_fdr, corr_fdr_reject_rate=corr_fdr,
    )
```

- [ ] **Step 2: Verify with synthetic data (no real CV output exists yet, so this checks correctness on known inputs)**

```bash
cd /project/cfRNA_NormativeModeling && python3 -c "
import numpy as np
from MixedEffectsModeling.core.calibration import gene_shash_calibration, bh_fdr_reject

rng = np.random.default_rng(0)
z_normal = rng.normal(0, 1, 2000)
r_normal = gene_shash_calibration(z_normal)
print('normal:', {k: round(v, 3) if isinstance(v, float) else v for k, v in r_normal.items()})

z_skewed = rng.standard_exponential(2000) * 2 - 2  # heavily right-skewed, mean-shifted
r_skewed = gene_shash_calibration(z_skewed)
print('skewed:', {k: round(v, 3) if isinstance(v, float) else v for k, v in r_skewed.items()})

assert abs(r_normal['naive_exceed'] - 0.05) < 0.02
assert r_skewed['raw_skew'] > 1.0
print('OK')
"
```

Expected: two printed dicts, then `OK`. The `normal` case should have `raw_skew`/`raw_kurtosis` near 0 and `naive_exceed`/`shash_exceed` both near 0.05; the `skewed` case should have `raw_skew` well above 1.0 and `naive_fdr_reject_rate` (falsely flagged nulls under the wrong N(0,1) assumption) higher than `corr_fdr_reject_rate`.

- [ ] **Step 3: Commit**

```bash
git add MixedEffectsModeling/core/calibration.py
git commit -m "feat: per-gene SHASH calibration module (naive vs corrected exceedance/FDR)"
```

---

## Task 7: `core/run_engine.py` — entry point for the full non-pooled sweep

The engine previously had no dedicated run script (its `engine_state_mixed/` artifacts were produced ad hoc). This adds one, modeled on the deleted `validation/run_glmm_full_unconstrained.py` but driving the real `NormativeModelEngineMixed` (so it writes the canonical `genes.pkl`/`scaler.pkl`/`training_summary.csv` via `.save()`, not a bare CSV dump). Since `Threshold_Sweep/nz_a_max.txt` doesn't exist, `assign_routes()` defaults `nz_a_max=0` and every gene routes to `"model"` — this is exactly the "no pooling" sweep requested. Reuses the cached `engine_state_mixed/dispersion_trend.json` instead of rebuilding it (rebuilding needs the full raw HC matrix and is unrelated to the intercept-stage change).

**Files:**
- Create: `MixedEffectsModeling/core/run_engine.py`

**Interfaces:**
- Consumes: `NormativeModelEngineMixed` (Task 5), `config.ENGINE_MIXED_DIR`, `config.DISPERSION_TREND_PATH`.
- Produces (CLI): `python core/run_engine.py [--limit N]` writes `config.ENGINE_MIXED_DIR/{genes.pkl,scaler.pkl,training_summary.csv}`.

- [ ] **Step 1: Write the file**

```python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.model_engine_mixed import NormativeModelEngineMixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    engine = NormativeModelEngineMixed()
    engine.load_hc_data()
    if config.DISPERSION_TREND_PATH.exists():
        from MixedEffectsModeling.core.dispersion_trend import load_trend
        engine.alpha_fn = load_trend()
        print(f"Reusing cached trend -> {config.DISPERSION_TREND_PATH}")
    else:
        engine.build_dispersion_trend()
        print(f"Built trend -> {config.DISPERSION_TREND_PATH}")

    engine.assign_routes()
    n_pool = sum(1 for r in engine.genes.values() if r.route == "pool")
    n_model = sum(1 for r in engine.genes.values() if r.route == "model")
    print(f"HC={engine.X_hc_scaled.shape[0]} genes={len(engine.genes)} nz_a_max={engine.nz_a_max} pool_route={n_pool} model_route={n_model}")

    engine.train(limit=args.limit)
    engine.save(config.ENGINE_MIXED_DIR)

    summary = engine.training_summary()
    print(summary.groupby(["route", "stage"]).size().to_string())
    print(f"ok={int(summary['ok'].sum())}/{len(summary)}")
    print(f"Saved -> {config.ENGINE_MIXED_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add MixedEffectsModeling/core/run_engine.py
git commit -m "feat: add run_engine.py entry point for the v2 non-pooled full sweep"
```

---

## Task 8: `validation/cv_engine.py` — fold-level stats + success logging (code only, not run)

Reimplements `_legacy/core_v1/cv_glmm_engine.py`'s model-route CV logic, but every `(gene, fold)` pair is now written to `fold_stats.csv` regardless of convergence — the old version silently `continue`d past a fold that failed to converge (`if g not in fold_fits.index or not bool(fold_fits.loc[g, "ok"]): continue`), discarding exactly the fold-level failure information this plan's spec asks to keep. This task only writes the code; per the confirmed scope, running the actual 5-fold CV (multi-hour R refit) and building the SHASH-based per-gene report are deferred to the next plan.

**Files:**
- Create: `MixedEffectsModeling/validation/cv_engine.py`

**Interfaces:**
- Consumes: `NormativeModelEngineMixed` (Task 5), `marginal_nb_rqr`/`_poisson_rqr` (Task 4), `config.ENGINE_MIXED_DIR`, `config.CV_MIXED_DIR`, `config.GLMM_FIT_R`.
- Produces: `config.CV_MIXED_DIR/fold_stats.csv` (columns: `gene, fold, stage, ok, singular, tau2, fail_reason, n_test`), plus the existing `cv_stats.csv`/`cv_zscores.pkl`/`cv_ppc.pkl` contract (unchanged shape) restricted to genes/folds that converged. Not consumed by anything yet this round — this is the deliverable itself.

- [ ] **Step 1: Write the file**

```python
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.dispersion_trend import load_trend
from MixedEffectsModeling.core.marginal_rqr import marginal_nb_rqr
from MixedEffectsModeling.core.model_engine_mixed import NormativeModelEngineMixed

MP = config.SPIKE_PARAMS


def _w1_normal(z):
    v = z[np.isfinite(z)]
    n = len(v)
    if n < 8:
        return np.nan
    ref = norm.ppf(np.linspace(1 / (2 * n), 1 - 1 / (2 * n), n))
    return float(np.mean(np.abs(np.sort(v) - ref)))


# Every (gene, fold) pair is recorded here regardless of convergence -- the
# v1 CV script silently dropped non-converging folds, which hid exactly the
# fold-level failure information this module exists to keep.
def cv_model_route(e2, model_genes, stage_of, folds, tmp):
    if not model_genes:
        return {}, {}, []
    rows, fold_stat_rows = [], []
    for fi, (tr, te) in enumerate(folds):
        pd.DataFrame(e2.X_hc_scaled[tr], columns=config.BIAS_COLUMNS).to_csv(f"{tmp}/X_{fi}.csv.gz")
        Y_tr = e2.Y_hc[tr][:, [e2._gene_col[g] for g in model_genes]]
        pd.DataFrame(Y_tr, columns=model_genes).to_csv(f"{tmp}/Y_{fi}.csv.gz")
        pd.DataFrame({"Batch_ID": e2.batch[tr]}).to_csv(f"{tmp}/batch_{fi}.csv.gz")
        pd.DataFrame({"gene": model_genes, "stage": [stage_of[g] for g in model_genes]}).to_csv(
            f"{tmp}/genes_{fi}.csv", index=False)
        subprocess.run([
            "Rscript", str(config.GLMM_FIT_R), "--x", f"{tmp}/X_{fi}.csv.gz", "--y", f"{tmp}/Y_{fi}.csv.gz",
            "--batch", f"{tmp}/batch_{fi}.csv.gz", "--genes", f"{tmp}/genes_{fi}.csv",
            "--trend", str(config.DISPERSION_TREND_PATH), "--mode", "fixed_stage", "--out", f"{tmp}/res_{fi}.csv",
        ], check=True, cwd=str(config.GLMM_FIT_R.parent))

        fold_fits = pd.read_csv(f"{tmp}/res_{fi}.csv").set_index("gene")
        Xa_te = np.column_stack([np.ones(len(te)), e2.X_hc_scaled[te]])
        for g in model_genes:
            if g not in fold_fits.index:
                fold_stat_rows.append(dict(gene=g, fold=fi, stage=stage_of[g], ok=False,
                                           singular=None, tau2=np.nan, fail_reason="fold_output_missing", n_test=len(te)))
                continue
            row = fold_fits.loc[g]
            ok = bool(row["ok"])
            fold_stat_rows.append(dict(gene=g, fold=fi, stage=row["stage"], ok=ok,
                                       singular=bool(row["singular"]) if not pd.isna(row["singular"]) else None,
                                       tau2=float(row["tau2"]) if not pd.isna(row["tau2"]) else np.nan,
                                       fail_reason=row["fail_reason"] if not pd.isna(row["fail_reason"]) else "",
                                       n_test=len(te)))
            if not ok:
                continue
            mu_coef = row[[c for c in fold_fits.columns if c.startswith("mu_coef_")]].values.astype(float)
            disp_coef = row[[c for c in fold_fits.columns if c.startswith("disp_coef_")]].values.astype(float)
            mu = np.clip(np.exp(Xa_te @ np.nan_to_num(mu_coef, nan=0.0)), 1e-6, 1e8)
            if not np.all(np.isnan(disp_coef)):
                alpha = np.exp(-Xa_te @ np.nan_to_num(disp_coef, nan=0.0))
            elif "fixed_alpha" in row.index and not pd.isna(row["fixed_alpha"]):
                alpha = np.full(len(te), float(row["fixed_alpha"]))
            else:
                alpha = np.full(len(te), e2.alpha_fn(float(mu.mean())))
            y_te = e2.Y_hc[te, e2._gene_col[g]]
            tau2 = float(row["tau2"])
            z = marginal_nb_rqr(y_te, mu, alpha, tau2, seed=42 + fi)
            rows.append(dict(gene=g, fold=fi, y=y_te.astype(np.float32), mu=mu.astype(np.float32),
                             alpha=np.asarray(alpha, dtype=np.float32),
                             tau2=np.full(len(te), tau2, dtype=np.float32), z=z))

    zdict, ppc_dict = {}, {}
    for g in model_genes:
        grecs = [r for r in rows if r["gene"] == g]
        if not grecs:
            continue
        zdict[g] = np.concatenate([r["z"] for r in grecs])
        ppc_dict[g] = dict(
            y=np.concatenate([r["y"] for r in grecs]),
            mu=np.concatenate([r["mu"] for r in grecs]),
            alpha=np.concatenate([r["alpha"] for r in grecs]),
            tau2=np.concatenate([r["tau2"] for r in grecs]),
            family="negbin", stage=stage_of[g])
    return zdict, ppc_dict, fold_stat_rows


def main():
    out_dir = config.CV_MIXED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(config.ENGINE_MIXED_DIR / "training_summary.csv", index_col="gene")
    summary = summary[summary["ok"]]
    model_genes = summary.index[summary["route"] == "model"].tolist()
    stage_of = summary["stage"].to_dict()

    e2 = NormativeModelEngineMixed()
    e2.load_hc_data()
    e2.alpha_fn = load_trend()
    n_hc = e2.X_hc_scaled.shape[0]
    folds = list(StratifiedKFold(MP["n_splits"], shuffle=True, random_state=42).split(np.zeros(n_hc), e2.batch))

    tmp = "/tmp/cv_glmm_v2"
    Path(tmp).mkdir(exist_ok=True)

    print(f"CV: {len(model_genes)} model-route genes (pool route not run this round)")
    zdict, ppc_dict, fold_stat_rows = cv_model_route(e2, model_genes, stage_of, folds, tmp)

    fold_stats = pd.DataFrame(fold_stat_rows)
    fold_stats.to_csv(out_dir / "fold_stats.csv", index=False)
    print(f"fold success rate: {fold_stats['ok'].mean():.3f} ({int(fold_stats['ok'].sum())}/{len(fold_stats)})")

    stats = []
    for g, z in zdict.items():
        v = z[np.isfinite(z)]
        if len(v) < 8:
            continue
        nz = int((e2.Y_hc[:, e2._gene_col[g]] > 0).sum())
        stats.append(dict(gene=g, route="model", stage=stage_of[g], nz=nz,
                          w1=_w1_normal(v), mean_z=float(v.mean()), std_z=float(v.std()),
                          skew_z=float(skew(v)), kurt_z=float(kurtosis(v)), n_valid=len(v)))
    df = pd.DataFrame(stats)
    df.to_csv(out_dir / "cv_stats.csv", index=False)
    with open(out_dir / "cv_zscores.pkl", "wb") as f:
        pickle.dump(zdict, f)
    with open(out_dir / "cv_ppc.pkl", "wb") as f:
        pickle.dump(ppc_dict, f)
    print(df.groupby("stage")[["w1", "mean_z", "std_z"]].median().to_string())
    print(f"Saved -> {out_dir}/fold_stats.csv, cv_stats.csv, cv_zscores.pkl, cv_ppc.pkl")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax/import check only (do NOT run `main()` -- a full run is multi-hour R refitting, explicitly deferred to next round)**

```bash
cd /project/cfRNA_NormativeModeling && python3 -c "
import ast
ast.parse(open('MixedEffectsModeling/validation/cv_engine.py').read())
print('syntax OK')
"
cd /project/cfRNA_NormativeModeling && python3 -c "
import MixedEffectsModeling.validation.cv_engine as m
print(callable(m.cv_model_route), callable(m.main))
"
```

Expected: `syntax OK`, then `True True`.

- [ ] **Step 3: Commit**

```bash
git add MixedEffectsModeling/validation/cv_engine.py
git commit -m "feat: cv_engine.py records per-fold success/failure explicitly (not run yet)"
```

---

## Task 9: Run the full non-pooled sweep

Smoke-test with `--limit`, confirm the 3-stage cascade behaves as expected (no `intercept` stage anywhere in the output, `nb_fixed` failures show up as `ok=False`/`route=excluded`), then run the real full sweep across all ~20,097 protein-coding genes. This is the plan's stopping point.

**Files:** none (execution only).

- [ ] **Step 1: Smoke test on 40 genes**

```bash
cd /project/cfRNA_NormativeModeling && python MixedEffectsModeling/core/run_engine.py --limit 40
```

Expected: prints `HC=... genes=20097 nz_a_max=0 pool_route=0 model_route=20097` (or `model_route=40` capped by `--limit` inside `train()`), then per-`(route, stage)` counts with stages only among `{nbi, nbi_disp_intercept, nb_fixed}` (never `intercept`), then `ok=<n>/<n>`.

- [ ] **Step 2: Inspect the smoke-test training_summary.csv for the new columns and 3-stage-only behavior**

```bash
cd /project/cfRNA_NormativeModeling && python3 -c "
import pandas as pd
df = pd.read_csv('MixedEffectsModeling/engine_state_mixed/training_summary.csv')
print(df['stage'].value_counts())
print(df[['nbi_reject_reason', 'nbi_disp_intercept_reject_reason', 'nb_fixed_reject_reason']].head(10))
assert 'intercept' not in set(df['stage'].dropna()), 'intercept stage should no longer exist'
print('OK')
"
```

Expected: `stage` value counts only show `nbi`/`nbi_disp_intercept`/`nb_fixed`; reject-reason columns populated where a gene was demoted; `OK` printed.

- [ ] **Step 3: Run the full sweep (background — this refits ~20,097 genes via `glmmTMB`, expect it to take on the order of an hour or more even with 8 cores; `glmm_fit.R`'s chunked/checkpointed writes make it safe to re-run and resume if interrupted)**

```bash
cd /project/cfRNA_NormativeModeling
nohup python MixedEffectsModeling/core/run_engine.py > /tmp/run_engine_v2.log 2>&1 &
echo "started pid $!"
```

Run this in the background (`run_in_background: true` if using the Bash tool) and monitor `/tmp/run_engine_v2.log` for the chunk-progress lines `glmm_fit.R` prints (`elapsed=...min eta=...min`).

- [ ] **Step 4: Once finished, verify final artifacts and compare stage distribution to the v1 diagnostics baseline**

```bash
cd /project/cfRNA_NormativeModeling && python3 -c "
import pandas as pd
df = pd.read_csv('MixedEffectsModeling/engine_state_mixed/training_summary.csv')
print(df.shape)
print(df.groupby(['route', 'stage'])['ok'].agg(['size', 'sum']))
print('excluded:', int((~df['ok']).sum()))
"
```

Expected: 20,097 rows total; `stage` restricted to `{nbi, nbi_disp_intercept, nb_fixed}`; an `excluded` count that should be noticeably smaller than v1's `intercept_True`(495) + `intercept_False`(382) = 877 genes that either trivially "converged" on a 1-df model or failed outright — v2 should show most of those now landing honestly at `nb_fixed` (`ok=True`) or `excluded` (`ok=False`) instead.

- [ ] **Step 5: Commit the regenerated `engine_state_mixed/` artifacts**

```bash
git add MixedEffectsModeling/engine_state_mixed/
git commit -m "chore: regenerate engine_state_mixed/ with v2 3-stage cascade (full non-pooled sweep)"
```

---

## Explicitly out of scope this round

- Running `validation/cv_engine.py`'s actual 5-fold CV (multi-hour R refit).
- The comprehensive per-gene report (`gene_report.csv` with `z_mean/z_std/skew/kurtosis/quantile-coverage/naive_exceed/shash_exceed/corrected moments/FDR reject rates`, joined with gene symbol/chromosome metadata) — needs the CV run's held-out Z to exist first.
- `pooling` route (`nz_a_max` still undetermined per `[[project_mixed_effects_pooling_threshold]]`).
- Updating `demotion_dispersion_diagnostics.ipynb` or any other notebook against the new artifacts.
- Any downstream consumer (`pipeline/`, `Z_scores/`, GSEA) — this plan only touches `MixedEffectsModeling/`.
