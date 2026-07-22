# Mixed-Effects Production Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the production mixed-effects (batch random-intercept) normative engine in `Modeling/`, replacing per-gene rpy2 calls with a one-pass R-native cascade.

**Architecture:** Python writes HC data to disk; R (`glmm_helpers.R` + `glmm_fit.R`) fits everything via `mclapply`, one subprocess call per run; Python reads results back and does all scoring (marginal Gauss-Hermite RQR) in pure Python.

**Ordering revised 2026-07-23 (user directive, mid-execution):** the pooling threshold is NOT auto-picked before training. Instead: build the full cascade correctly (Tasks 1-5) -> run it unconstrained across ALL genes, no pool-route gating at all (new Task 6) -> EDA the per-gene nz/stage/tau2 trends (referencing `EDA_Modeling/pooling_nz_sweep.py`'s plot style) before deciding anything about pooling -> build the CV framework fully (Task 8) so it's ready to run the moment a threshold is chosen, without blocking on that decision now. `pool_threshold_sweep.R`/`.py` (old Task 6) are still built as code, but their nz_a_max auto-pick is NOT executed as part of this pass -- deferred until the Task 6 EDA has been reviewed.

**Tech Stack:** R 4.3.1 + glmmTMB 1.1.9 (subprocess only, no rpy2), Python (scanpy, pandas, numpy, sklearn), conda env `scRNA`.

## Global Constraints

- No type hints on Python function signatures. No artificial alignment whitespace. English-only comments, minimal (one line, WHY only). Alphabetical imports.
- All new files in `Modeling/`. `Modeling/gamlss.r` untouched (still used by the existing v2 engine).
- New output dirs, never overwrite existing production artifacts: `Modeling/engine_state_mixed/`, `Modeling/CV_Results_mixed/`, `Modeling/Threshold_Sweep/`.
- `mclapply` cores: `min(parallel::detectCores() - 1, 8)`.
- Reuse `MixedEffectsModeling/Spike_Results/` (40-gene pilot, already-validated `is_converged()` semantics) as the regression fixture — do not modify anything under `MixedEffectsModeling/`.
- `nz_a_max` decision is DEFERRED (see ordering note above) -- do not auto-run `pool_threshold_sweep.py`'s pick as part of this pass. Build it, don't execute the decision.
- `model_engine_mixed.py`'s `assign_routes()` must work with NO `nz_a_max` fixed yet: default to gating nothing (every gene attempts the model cascade) rather than requiring `Threshold_Sweep/nz_a_max.txt` to exist.
- Conda env `scRNA`: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate scRNA` before any command.
- No pytest — verification is scripts with embedded asserts printing PASS/FAIL, per project convention.

---

### Task 1: config.py additions

**Files:**
- Modify: `config.py` (append near existing `ENGINE_DIR`/`CV_RESULTS_DIR` block, config.py:26-37)

**Interfaces:**
- Produces: `ENGINE_MIXED_DIR`, `CV_MIXED_DIR`, `CV_MIXED_FIG_DIR`, `THRESHOLD_SWEEP_DIR`, `THRESHOLD_SWEEP_FIG_DIR`, `GLMM_HELPERS_R`, `GLMM_FIT_R`, `POOL_SWEEP_R` (all `Path`).

- [ ] **Step 1: Add the paths**

```python
ENGINE_MIXED_DIR   = MODELING_DIR / "engine_state_mixed"
CV_MIXED_DIR       = MODELING_DIR / "CV_Results_mixed"
CV_MIXED_FIG_DIR   = CV_MIXED_DIR / "Figures"
THRESHOLD_SWEEP_DIR     = MODELING_DIR / "Threshold_Sweep"
THRESHOLD_SWEEP_FIG_DIR = THRESHOLD_SWEEP_DIR / "Figures"
GLMM_HELPERS_R = MODELING_DIR / "glmm_helpers.R"
GLMM_FIT_R     = MODELING_DIR / "glmm_fit.R"
POOL_SWEEP_R   = MODELING_DIR / "pool_threshold_sweep.R"
```

- [ ] **Step 2: Verify**

```bash
python -c "import config; [p.parent.mkdir(parents=True, exist_ok=True) or print(p) for p in [config.ENGINE_MIXED_DIR, config.CV_MIXED_DIR, config.THRESHOLD_SWEEP_DIR]]"
```
Expected: prints the three paths, no error.

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "Add config paths for mixed-effects production engine outputs"
```

---

### Task 2: `Modeling/glmm_helpers.R` — shared R fitting logic

**Files:**
- Create: `Modeling/glmm_helpers.R`

**Interfaces:**
- Produces: `sanitize_names(names)`, `safe_max_abs(x)`, `is_converged(fit, beta_explode_thr, tau2_max)` (returns `list(ok, singular, tau2)`), `fit_stage_gene(y, X_safe_names, X, batch, stage, fixed_log_theta, priors_df, beta_explode_thr, tau2_max)` (returns `list(stage, ok, singular, tau2, mu_coef, disp_coef, fail_reason)`), `fit_pooled_glmm(Y_block, X, batch, mean_hc, eps, rare_overdisp_thr)` (returns `list(family, beta, alpha, mult_lo, mult_hi, ok)`).

These four stage formulas (verified against glmmTMB 1.1.9 directly before writing this plan — the offset trick for fixing dispersion is confirmed to work and `fixef(fit)$disp` is `numeric(0)` when `dispformula=~0+offset(...)`, which is why `safe_max_abs` must handle empty vectors):

| stage | mu formula | dispformula | priors |
|---|---|---|---|
| `nbi` | `y__ ~ X + (1\|batch__)` | `~ X` | `betad`, `normal(0,0.05)` |
| `nbi_disp_intercept` | `y__ ~ X + (1\|batch__)` | `~ 1` | none |
| `nb_fixed` | `y__ ~ X + (1\|batch__)` | `~ 0 + offset(fixed_log_theta)` | none |
| `intercept` | `y__ ~ 1 + (1\|batch__)` | `~ 0 + offset(fixed_log_theta)` | none |

`fixed_log_theta = -log(alpha)` where `alpha` is the Phase-0 trend dispersion at this gene's train-fold mean (sign convention verified in the Step 0 spike: gamlss `sigma` = glmmTMB `exp(-dispformula linear predictor)`).

- [ ] **Step 1: Write the file**

```r
suppressPackageStartupMessages(library(glmmTMB))

sanitize_names <- function(names) {
  safe <- gsub("[^A-Za-z0-9_]", "_", names)
  bad <- grepl("^[^A-Za-z.]", safe)
  safe[bad] <- paste0("v", safe[bad])
  safe
}

safe_max_abs <- function(x) if (length(x) == 0) 0 else max(abs(x))

# Slopes only (intercept excluded, matches Modeling/model_engine.py's beta[1:]
# convention); explosion checked before pdHess branching, in either submodel.
is_converged <- function(fit, beta_explode_thr, tau2_max) {
  if (inherits(fit, "try-error")) return(list(ok = FALSE, singular = NA, tau2 = NA))
  beta_max <- safe_max_abs(c(fixef(fit)$cond[-1], fixef(fit)$disp[-1]))
  tau2 <- as.numeric(VarCorr(fit)$cond[[1]][1, 1])
  if (isTRUE(beta_max >= beta_explode_thr)) return(list(ok = FALSE, singular = NA, tau2 = tau2))
  if (isTRUE(fit$sdr$pdHess)) {
    if (isTRUE(tau2 >= tau2_max)) return(list(ok = FALSE, singular = NA, tau2 = tau2))
    return(list(ok = TRUE, singular = FALSE, tau2 = tau2))
  }
  if (isTRUE(tau2 < 1e-5)) return(list(ok = TRUE, singular = TRUE, tau2 = 0.0))
  return(list(ok = FALSE, singular = NA, tau2 = tau2))
}

# Fits ONE stage for ONE gene. Caller (glmm_fit.R) drives the demotion order.
fit_stage_gene <- function(y, safe_names, X, batch, stage, fixed_log_theta,
                           priors_df, beta_explode_thr, tau2_max) {
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
  conv <- is_converged(fit, beta_explode_thr, tau2_max)
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

  fit_pois <- tryCatch(glmmTMB(mu_fml, family = poisson(), data = df), error = function(e) NULL)
  if (is.null(fit_pois)) return(list(ok = FALSE))
  ratio <- sum(residuals(fit_pois, type = "pearson")^2) / df.residual(fit_pois)
  if (ratio <= rare_overdisp_thr) {
    return(list(ok = TRUE, family = "poisson", beta = as.numeric(fixef(fit_pois)$cond), alpha = NA))
  }
  fit_nb <- tryCatch(glmmTMB(mu_fml, family = nbinom2(), data = df), error = function(e) NULL)
  if (is.null(fit_nb)) return(list(ok = FALSE))
  list(ok = TRUE, family = "negbin", beta = as.numeric(fixef(fit_nb)$cond),
      alpha = exp(-fixef(fit_nb)$disp[["(Intercept)"]]))  # theta->sigma reciprocal, same convention
}
```

- [ ] **Step 2: Smoke-test against the spike pilot**

```bash
cd /project/cfRNA_NormativeModeling/MixedEffectsModeling
Rscript -e '
source("../Modeling/glmm_helpers.R")
X <- as.matrix(read.csv("Spike_Results/pilot_X.csv.gz", row.names=1))
Y <- read.csv("Spike_Results/pilot_Y.csv.gz", row.names=1)
batch <- read.csv("Spike_Results/pilot_batch.csv.gz", row.names=1)$Batch_ID
safe_names <- gsub("[^A-Za-z0-9_]", "_", colnames(X))
g <- colnames(Y)[1]
r <- fit_stage_gene(Y[[g]], safe_names, X, batch, "nbi", NULL, data.frame(prior="normal(0,0.05)",class="betad",coef=""), 3.0, 9.0)
cat("PASS\n"); str(r)
'
```
Expected: prints `PASS` and a list with `stage="nbi"`, `ok` TRUE/FALSE, no R error.

- [ ] **Step 3: Commit**

```bash
git add Modeling/glmm_helpers.R
git commit -m "Add shared R fitting helpers for the mixed-effects cascade"
```

---

### Task 3: `Modeling/glmm_fit.R` — cascade CLI

**Files:**
- Create: `Modeling/glmm_fit.R`

**Interfaces:**
- Consumes: `Modeling/glmm_helpers.R` (Task 2).
- CLI args: `--x <csv.gz> --y <csv.gz> --batch <csv.gz> --genes <csv, gene+stage col> --trend <json> --mode <cascade|fixed_stage> --out <csv> --chunk-size <int> --cores <int>`.
  `--mode cascade`: try nbi->nbi_disp_intercept->nb_fixed->intercept per gene (training). `--mode fixed_stage`: fit only the stage given in `--genes`' `stage` column (CV re-fit per fold).
- Produces: one CSV row per gene: `gene,stage,ok,singular,tau2,mu_coef_0..10,disp_coef_0..10,fail_reason`. Written incrementally per chunk (resumable: skip genes already present in an existing `--out` file).

- [ ] **Step 1: Write the file**

```r
suppressPackageStartupMessages({
  library(optparse); library(parallel); library(jsonlite)
})
source(file.path(dirname(sys.frame(1)$ofile %||% "."), "glmm_helpers.R"))

`%||%` <- function(a, b) if (is.null(a)) b else a

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
trend <- fromJSON(opt$trend)      # named list: mean grid + alpha grid, see dispersion_trend.py
alpha_of <- function(mean_y) {
  approx(trend$mean_grid, trend$alpha_grid, xout = mean_y, rule = 2)$y
}
safe_names <- sanitize_names(colnames(X))
colnames(X) <- safe_names
BETA_EXPLODE_THR <- 3.0
TAU2_MAX <- BETA_EXPLODE_THR^2
priors_df <- data.frame(prior = "normal(0, 0.05)", class = "betad", coef = "")

done_genes <- character(0)
if (file.exists(opt$out)) done_genes <- read.csv(opt$out)$gene

fit_one_cascade <- function(g) {
  y <- as.numeric(Y[[g]])
  alpha_g <- alpha_of(mean(y))
  fixed_log_theta <- rep(-log(alpha_g), length(y))
  for (stage in c("nbi", "nbi_disp_intercept", "nb_fixed", "intercept")) {
    pr <- if (stage == "nbi") priors_df else NULL
    r <- fit_stage_gene(y, safe_names, X, batch, stage, fixed_log_theta, pr, BETA_EXPLODE_THR, TAU2_MAX)
    if (isTRUE(r$ok) || stage == "intercept") { gc(); return(c(list(gene = g), r)) }
  }
}

fit_one_fixed <- function(g) {
  stage <- gene_meta$stage[gene_meta$gene == g]
  y <- as.numeric(Y[[g]])
  alpha_g <- alpha_of(mean(y))
  fixed_log_theta <- rep(-log(alpha_g), length(y))
  pr <- if (stage == "nbi") priors_df else NULL
  r <- fit_stage_gene(y, safe_names, X, batch, stage, fixed_log_theta, pr, BETA_EXPLODE_THR, TAU2_MAX)
  gc(); c(list(gene = g), r)
}

worker <- if (opt$mode == "cascade") fit_one_cascade else fit_one_fixed
genes_todo <- setdiff(gene_meta$gene, done_genes)
chunks <- split(genes_todo, ceiling(seq_along(genes_todo) / opt$`chunk-size`))

for (i in seq_along(chunks)) {
  results <- mclapply(chunks[[i]], worker, mc.cores = opt$cores)
  rows <- lapply(results, function(r) {
    p <- 11  # 1 intercept + 10 covariates
    mu_padded <- c(r$mu_coef, rep(NA, p - length(r$mu_coef)))[1:p]
    disp_padded <- c(r$disp_coef, rep(NA, p - length(r$disp_coef)))[1:p]
    row <- c(list(gene = r$gene, stage = r$stage, ok = r$ok, singular = r$singular,
                 tau2 = r$tau2, fail_reason = r$fail_reason))
    for (j in seq_len(p)) { row[[paste0("mu_coef_", j-1)]] <- mu_padded[j]; row[[paste0("disp_coef_", j-1)]] <- disp_padded[j] }
    row
  })
  df <- do.call(rbind, lapply(rows, as.data.frame))
  write.table(df, opt$out, sep = ",", append = file.exists(opt$out), col.names = !file.exists(opt$out), row.names = FALSE)
  gc()
  cat(sprintf("chunk %d/%d done (%d genes)\n", i, length(chunks), length(chunks[[i]])))
}
cat("DONE\n")
```

- [ ] **Step 2: Install `optparse` if missing, run the pilot regression check**

```bash
cd /project/cfRNA_NormativeModeling/MixedEffectsModeling
Rscript -e 'if (!requireNamespace("optparse", quietly=TRUE)) install.packages("optparse", repos="https://cloud.r-project.org")'
python -c "
import pandas as pd
g = pd.read_csv('Spike_Results/pilot_genes.csv')[['gene']]
g.to_csv('/tmp/pilot_genes_nbi.csv', index=False)
"
Rscript ../Modeling/glmm_fit.R --x Spike_Results/pilot_X.csv.gz --y Spike_Results/pilot_Y.csv.gz \
  --batch Spike_Results/pilot_batch.csv.gz --genes /tmp/pilot_genes_nbi.csv \
  --trend Spike_Results/dummy_trend.json --mode cascade --out /tmp/pilot_cascade_out.csv --chunk-size 40
```
Expected: `DONE` printed, `/tmp/pilot_cascade_out.csv` has 40 rows. `--trend` needs a real Phase-0 trend JSON — if `Spike_Results/dummy_trend.json` doesn't exist, generate one from `Modeling/dispersion_trend.py`'s `build_trend`/`save_trend` against the pilot Y first (see Task 4's regression check, which does this properly against real HC data).

- [ ] **Step 3: Commit**

```bash
git add Modeling/glmm_fit.R
git commit -m "Add glmm_fit.R production cascade CLI (nbi->nbi_disp_intercept->nb_fixed->intercept)"
```

---

### Task 4: Spike-pilot regression fixture

**Files:**
- Create: `Modeling/test_glmm_fit_regression.py`

**Interfaces:**
- Consumes: `Modeling/glmm_fit.R` (Task 3), `MixedEffectsModeling/Spike_Results/*` (read-only).
- Verifies: running `glmm_fit.R --mode fixed_stage` with every pilot gene fixed to stage `"nbi"` reproduces the spike's `random_intercept_fits.csv` `tau2`/`singular`/`ok` outcomes exactly (same formula, same `is_converged`, same data — only the code path differs).

- [ ] **Step 1: Write the script**

```python
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

SPIKE_DIR = Path(__file__).resolve().parent.parent / "MixedEffectsModeling" / "Spike_Results"
OUT = Path("/tmp/glmm_fit_regression_out.csv")
OUT.unlink(missing_ok=True)

genes = pd.read_csv(SPIKE_DIR / "pilot_genes.csv")[["gene"]]
genes["stage"] = "nbi"
genes_path = Path("/tmp/glmm_fit_regression_genes.csv")
genes.to_csv(genes_path, index=False)

subprocess.run([
    "Rscript", str(config.GLMM_FIT_R),
    "--x", str(SPIKE_DIR / "pilot_X.csv.gz"), "--y", str(SPIKE_DIR / "pilot_Y.csv.gz"),
    "--batch", str(SPIKE_DIR / "pilot_batch.csv.gz"), "--genes", str(genes_path),
    "--trend", str(config.DISPERSION_TREND_PATH), "--mode", "fixed_stage", "--out", str(OUT),
], check=True, cwd=str(config.MODELING_DIR))

new = pd.read_csv(OUT).set_index("gene")
old = pd.read_csv(SPIKE_DIR / "random_intercept_fits.csv").set_index("gene")
common = new.index.intersection(old.index)
mismatches = []
for g in common:
    if bool(new.loc[g, "ok"]) != bool(old.loc[g, "converged"]):
        mismatches.append(g)
        continue
    if bool(old.loc[g, "converged"]) and abs(new.loc[g, "tau2"] - old.loc[g, "tau2"]) > 1e-3:
        mismatches.append(g)

assert len(common) == 40, f"expected 40 common genes, got {len(common)}"
assert not mismatches, f"FAIL: {len(mismatches)} genes diverged from spike: {mismatches}"
print(f"PASS: all {len(common)} pilot genes match spike outcomes (ok+tau2)")
```
Note: this requires `config.DISPERSION_TREND_PATH` (the real production Phase-0 trend, built from full HC data) to exist. If it doesn't yet, run `python -c "from Modeling.model_engine import NormativeModelEngine as E; e=E(); e.load_hc_data(); e.build_dispersion_trend()"` first, or point `--trend` at any valid trend JSON — the regression check only needs `alpha_of` to return a positive number per gene, it doesn't need to match the spike's own dispersion values (the spike's stage was `nbi`, whose dispersion is covariate-regressed, not trend-derived).

- [ ] **Step 2: Run it**

```bash
cd /project/cfRNA_NormativeModeling
python Modeling/test_glmm_fit_regression.py
```
Expected: `PASS: all 40 pilot genes match spike outcomes (ok+tau2)`. A mismatch here means `glmm_fit.R`'s `nbi` formula/priors diverged from the spike's validated version — compare `Modeling/glmm_helpers.R`'s `nbi` row against `MixedEffectsModeling/fit_random_intercept.R` line by line before proceeding to any full run.

- [ ] **Step 3: Commit**

```bash
git add Modeling/test_glmm_fit_regression.py
git commit -m "Add spike-pilot regression fixture for glmm_fit.R"
```

---

### Task 5: `Modeling/marginal_rqr.py` — pure-Python marginal scoring

**Files:**
- Create: `Modeling/marginal_rqr.py`

**Interfaces:**
- Produces: `marginal_nb_rqr(y, mu, alpha, tau2, seed, n_nodes=7)` — marginal RQR integrating `b ~ N(0, tau2)` via Gauss-Hermite quadrature into the NB CDF; falls back to the existing `_nb_rqr` (point-mass) when `tau2 < 1e-6`.

- [ ] **Step 1: Write the module**

```python
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.stats import nbinom, norm

RQR_EPS = 1e-8


def _nb_cdf(y, mu, alpha):
    n = 1.0 / alpha
    p = np.clip(n / (n + mu), RQR_EPS, 1 - RQR_EPS)
    return nbinom.cdf(y, n, p)


def marginal_nb_rqr(y, mu, alpha, tau2, seed, n_nodes=7):
    y = np.asarray(y)
    if tau2 < 1e-6:
        from model_engine import _nb_rqr
        return _nb_rqr(y, mu, alpha, seed)

    nodes, weights = hermegauss(n_nodes)  # integrate against exp(-x^2/2), matches N(0,1)
    weights = weights / weights.sum()
    sd = np.sqrt(tau2)
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

- [ ] **Step 2: Verify against Monte Carlo**

```bash
cd /project/cfRNA_NormativeModeling/Modeling
python -c "
import numpy as np
from scipy.stats import nbinom
from marginal_rqr import marginal_nb_rqr, _nb_cdf

rng = np.random.default_rng(0)
mu, alpha, tau2 = 20.0, 0.3, 0.5
n = 5000
b = rng.normal(0, np.sqrt(tau2), n)
y = rng.negative_binomial(1/alpha, (1/alpha)/((1/alpha)+mu*np.exp(b)))
z = marginal_nb_rqr(y, np.full(n, mu), alpha, tau2, seed=1)
z = z[np.isfinite(z)]
print('mean', z.mean(), 'std', z.std())
assert abs(z.mean()) < 0.05 and abs(z.std() - 1) < 0.1, 'FAIL: marginal RQR not calibrated'
print('PASS: marginal RQR calibrated on Monte Carlo NB-mixture data')
"
```
Expected: `PASS: marginal RQR calibrated on Monte Carlo NB-mixture data`. If it fails, check the quadrature node/weight normalization (`hermegauss` integrates against `exp(-x^2/2)` already, so weights should sum to 1 without an extra `1/sqrt(2*pi)` factor — a common bug source).

- [ ] **Step 3: Commit**

```bash
git add Modeling/marginal_rqr.py
git commit -m "Add pure-Python marginal RQR (Gauss-Hermite quadrature over batch random intercept)"
```

---

### Task 6a: Full unconstrained cascade run + EDA (no pool-route gating)

**Files:**
- Create: `Modeling/run_glmm_full_unconstrained.py`, `Modeling/eda_glmm_full_unconstrained.py`

**Interfaces:**
- Consumes: `Modeling/glmm_fit.R` (Task 3).
- Produces: `Modeling/Threshold_Sweep/full_cascade_unconstrained.csv` (every protein-coding HC gene, `--mode cascade`, no NZ filtering at all -- every gene attempts nbi->...->intercept regardless of NZ), and `Threshold_Sweep/Figures/nz_vs_stage_tau2.png` (per-NZ-bin: stage composition, `ok` rate, median `tau2` -- mirrors `EDA_Modeling/pooling_nz_sweep.py`'s plot style, but no threshold line drawn yet since none is chosen).

- [ ] **Step 1: Write `run_glmm_full_unconstrained.py`** (loads full HC data exactly like `Modeling/model_engine.py:load_hc_data`, writes it once, calls `glmm_fit.R --mode cascade` over every protein-coding gene with no NZ pre-filter)

```python
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

TMP = Path("/tmp/glmm_full_unconstrained")
TMP.mkdir(exist_ok=True)
OUT = config.THRESHOLD_SWEEP_DIR / "full_cascade_unconstrained.csv"
config.THRESHOLD_SWEEP_DIR.mkdir(parents=True, exist_ok=True)


def main():
    adata = sc.read_h5ad(config.H5AD_PATH)
    m = ((adata.obs["QC_Passed"] == True) & (adata.obs["Phenotype_Processed"].notna()) &
         (adata.obs["Phenotype_Processed"] != "Unknown") &
         (adata.obs["broad_protocol_category"] != "Exome-based (EB)"))
    a = adata[m]
    is_hc = (a.obs["Phenotype_Processed"].astype(str) == "Healthy Control").values
    is_pc = (a.var["GeneType"] == "protein_coding").values
    X = a.obs[config.BIAS_COLUMNS].values.astype(np.float64)[is_hc]
    Xs = StandardScaler().fit_transform(X)
    Y = a.X.toarray() if issparse(a.X) else np.asarray(a.X)
    Y = np.round(Y[is_hc][:, is_pc]).astype(np.float64)
    names = a.var_names[is_pc].tolist()
    batch = a.obs["Batch_ID"].astype(str).values[is_hc]

    pd.DataFrame(Xs, columns=config.BIAS_COLUMNS).to_csv(TMP / "X.csv.gz")
    pd.DataFrame(Y, columns=names).to_csv(TMP / "Y.csv.gz")
    pd.DataFrame({"Batch_ID": batch}).to_csv(TMP / "batch.csv.gz")
    pd.DataFrame({"gene": names}).to_csv(TMP / "genes.csv", index=False)
    print(f"HC={Xs.shape[0]}  genes={len(names)}  batches={len(set(batch))}")

    if not config.DISPERSION_TREND_PATH.exists():
        raise SystemExit("Build the Phase-0 trend first (see existing Modeling/model_engine.py:build_dispersion_trend)")

    subprocess.run([
        "Rscript", str(config.GLMM_FIT_R), "--x", str(TMP / "X.csv.gz"), "--y", str(TMP / "Y.csv.gz"),
        "--batch", str(TMP / "batch.csv.gz"), "--genes", str(TMP / "genes.csv"),
        "--trend", str(config.DISPERSION_TREND_PATH), "--mode", "cascade", "--out", str(OUT),
        "--chunk-size", "200", "--cores", str(min(8, 8)),
    ], check=True, cwd=str(config.MODELING_DIR))
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `eda_glmm_full_unconstrained.py`** (NZ-bin trend plot, no threshold decision)

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from viz_style import apply_style

OUT = config.THRESHOLD_SWEEP_DIR
df = pd.read_csv(OUT / "full_cascade_unconstrained.csv")
nz = pd.read_csv("/tmp/glmm_full_unconstrained/Y.csv.gz", index_col=0)
df["nz"] = df["gene"].map((nz > 0).sum(axis=0).to_dict())
df["nz_bin"] = pd.cut(df["nz"], bins=[0, 3, 7, 15, 30, 50, 100, np.inf])

summary = df.groupby("nz_bin", observed=True).agg(
    n_genes=("gene", "size"), ok_rate=("ok", "mean"),
    tau2_median=("tau2", "median"),
    pct_nbi=("stage", lambda s: (s == "nbi").mean()),
    pct_intercept=("stage", lambda s: (s == "intercept").mean()),
)
summary.to_csv(OUT / "nz_vs_stage_tau2_summary.csv")
print(summary.round(3).to_string())

apply_style()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
x = range(len(summary))
axes[0].bar(x, summary["ok_rate"]); axes[0].set(title="ok rate", xlabel="nz bin")
axes[1].bar(x, summary["tau2_median"]); axes[1].set(title="median tau2", xlabel="nz bin")
axes[2].bar(x, summary["pct_nbi"], label="nbi"); axes[2].bar(x, summary["pct_intercept"], bottom=summary["pct_nbi"], label="intercept")
axes[2].legend(); axes[2].set(title="stage composition", xlabel="nz bin")
for ax in axes:
    ax.set_xticks(list(x)); ax.set_xticklabels([str(b) for b in summary.index], rotation=45, ha="right")
fig.tight_layout()
(OUT / "Figures").mkdir(exist_ok=True)
fig.savefig(OUT / "Figures" / "nz_vs_stage_tau2.png", dpi=150)
print(f"Saved -> {OUT}/nz_vs_stage_tau2_summary.csv, {OUT}/Figures/nz_vs_stage_tau2.png")
```

- [ ] **Step 3: Run both, smoke-test with a gene subset first**

The full run is on ~19,538 genes (design spec's core-hour estimate applies) -- smoke-test on a `--limit`-style subset first by temporarily slicing `names`/`Y` in `run_glmm_full_unconstrained.py` to e.g. 200 genes, confirm the CSV/plot pipeline works, then run unconstrained on all genes per the user's directive (proceed automatically, no pause).

```bash
cd /project/cfRNA_NormativeModeling
python Modeling/run_glmm_full_unconstrained.py
python Modeling/eda_glmm_full_unconstrained.py
```
Expected: prints the per-NZ-bin summary table, saves the CSV and figure. This is descriptive output for the user to review later -- do not pick or hardcode an `nz_a_max` from it as part of this task.

- [ ] **Step 4: Commit**

```bash
git add Modeling/run_glmm_full_unconstrained.py Modeling/eda_glmm_full_unconstrained.py
git commit -m "Add full unconstrained cascade run + NZ-trend EDA (no pooling threshold decided)"
```

---

### Task 6b: `Modeling/pool_threshold_sweep.R` + `.py`

**Files:**
- Create: `Modeling/pool_threshold_sweep.R`, `Modeling/pool_threshold_sweep.py`

**Interfaces:**
- Consumes: `Modeling/glmm_helpers.R`'s `fit_pooled_glmm` (Task 2).
- Produces: `Threshold_Sweep/pool_threshold_sweep_summary.csv` (`nz_threshold,n_genes,w1_median,w1_p90`), `Threshold_Sweep/Figures/pool_threshold_sweep.png`, and prints the auto-picked `nz_a_max`.

- [ ] **Step 1: Write `pool_threshold_sweep.R`** (one process, loops NZ thresholds x folds internally to avoid repeated R startup cost)

```r
suppressPackageStartupMessages(library(optparse))
source(file.path(dirname(normalizePath(sys.frame(1)$ofile)), "glmm_helpers.R"))

opt <- parse_args(OptionParser(option_list = list(
  make_option("--x", type = "character"), make_option("--y", type = "character"),
  make_option("--batch", type = "character"), make_option("--folds", type = "character"),
  make_option("--out", type = "character")
)))

X <- as.matrix(read.csv(opt$x, row.names = 1))
Y <- as.matrix(read.csv(opt$y, row.names = 1))
batch <- read.csv(opt$batch, row.names = 1)[[1]]
folds <- read.csv(opt$folds)  # columns: sample_idx (0-based), fold
nz <- colSums(Y > 0)
thresholds <- c(3, 5, 7, 10, 15, 20, 25, 30, 40, 50)
n_hc <- nrow(X)
eps <- 1 / (2 * n_hc)

rows <- list()
for (T in thresholds) {
  cols <- which(nz < T)
  if (length(cols) < 5) next
  w1s <- c()
  for (fi in unique(folds$fold)) {
    tr <- folds$sample_idx[folds$fold != fi] + 1
    te <- folds$sample_idx[folds$fold == fi] + 1
    mean_hc <- colMeans(Y[tr, cols, drop = FALSE])
    fit <- fit_pooled_glmm(Y[tr, cols, drop = FALSE], X[tr, , drop = FALSE], batch[tr], mean_hc, eps, 2.0)
    if (!isTRUE(fit$ok)) next
    Xc_te <- cbind(1, X[te, , drop = FALSE])
    mu <- (mean_hc[rep(seq_along(cols), each = length(te))] + eps) *
      exp(Xc_te[rep(seq_along(te), length(cols)), , drop = FALSE] %*% fit$beta)
    y_te <- as.vector(Y[te, cols, drop = FALSE])
    theta <- if (fit$family == "poisson") NA else 1 / fit$alpha
    p <- if (is.na(theta)) NA else theta / (theta + mu)
    u <- if (fit$family == "poisson") ppois(y_te, mu) else pnbinom(y_te, theta, p)
    z <- qnorm(pmin(pmax(u, 1e-8), 1 - 1e-8))
    w1s <- c(w1s, mean(abs(sort(z) - qnorm(ppoints(length(z))))))
  }
  if (length(w1s) == 0) next
  rows[[length(rows) + 1]] <- list(nz_threshold = T, n_genes = length(cols),
    w1_median = median(w1s), w1_p90 = quantile(w1s, 0.9, names = FALSE))
  cat(sprintf("nz<%d: n_genes=%d w1_median=%.3f\n", T, length(cols), median(w1s)))
}
write.csv(do.call(rbind, lapply(rows, as.data.frame)), opt$out, row.names = FALSE)
cat("DONE\n")
```

- [ ] **Step 2: Write `pool_threshold_sweep.py`** (drives the R script, picks `nz_a_max`, plots)

```python
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from viz_style import apply_style

MP = config.MODELING_PARAMS
OUT = config.THRESHOLD_SWEEP_DIR
OUT.mkdir(parents=True, exist_ok=True)
config.THRESHOLD_SWEEP_FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_hc():
    adata = sc.read_h5ad(config.H5AD_PATH)
    m = ((adata.obs["QC_Passed"] == True) & (adata.obs["Phenotype_Processed"].notna()) &
         (adata.obs["Phenotype_Processed"] != "Unknown") &
         (adata.obs["broad_protocol_category"] != "Exome-based (EB)"))
    a = adata[m]
    is_hc = (a.obs["Phenotype_Processed"].astype(str) == "Healthy Control").values
    is_pc = (a.var["GeneType"] == "protein_coding").values
    X = a.obs[config.BIAS_COLUMNS].values.astype(np.float64)[is_hc]
    Xs = StandardScaler().fit_transform(X)
    Y = a.X.toarray() if issparse(a.X) else np.asarray(a.X)
    Y = np.round(Y[is_hc][:, is_pc]).astype(np.float64)
    batch = a.obs["Batch_ID"].astype(str).values[is_hc]
    return Xs, Y, batch


def main():
    summary_path = OUT / "pool_threshold_sweep_summary.csv"
    if summary_path.exists():
        print(f"Loading cached sweep -> {summary_path}")
        summary = pd.read_csv(summary_path)
    else:
        Xs, Y, batch = load_hc()
        n_hc = Xs.shape[0]
        folds = list(StratifiedKFold(MP["n_splits"], shuffle=True, random_state=42).split(np.zeros(n_hc), batch))
        fold_rows = []
        for fi, (_, te) in enumerate(folds):
            for idx in te:
                fold_rows.append(dict(sample_idx=idx, fold=fi))
        pd.DataFrame(fold_rows).to_csv("/tmp/pool_sweep_folds.csv", index=False)
        pd.DataFrame(Xs, columns=config.BIAS_COLUMNS).to_csv("/tmp/pool_sweep_X.csv.gz", index=True)
        pd.DataFrame(Y).to_csv("/tmp/pool_sweep_Y.csv.gz", index=True)
        pd.DataFrame({"Batch_ID": batch}).to_csv("/tmp/pool_sweep_batch.csv.gz", index=True)

        subprocess.run([
            "Rscript", str(config.POOL_SWEEP_R),
            "--x", "/tmp/pool_sweep_X.csv.gz", "--y", "/tmp/pool_sweep_Y.csv.gz",
            "--batch", "/tmp/pool_sweep_batch.csv.gz", "--folds", "/tmp/pool_sweep_folds.csv",
            "--out", str(summary_path),
        ], check=True, cwd=str(config.MODELING_DIR))
        summary = pd.read_csv(summary_path)

    print(summary.round(3).to_string(index=False))
    picked = summary[summary["w1_median"] > 0.25]
    nz_a_max = int(picked["nz_threshold"].min()) if len(picked) else int(summary["nz_threshold"].max())
    print(f"\nAuto-picked nz_a_max = {nz_a_max} (smallest cutoff where median W1 > 0.25)")

    apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(summary["nz_threshold"], summary["w1_median"], "-o", label="Median W1")
    ax.plot(summary["nz_threshold"], summary["w1_p90"], "-o", label="90th pct W1")
    ax.axhline(0.25, ls=":", color="gray")
    ax.axvline(nz_a_max, ls="--", color="k", label=f"Picked nz_a_max={nz_a_max}")
    ax.set(xlabel="Pooling threshold (HC nonzero count)", ylabel="W1 (held-out CV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.THRESHOLD_SWEEP_FIG_DIR / "pool_threshold_sweep.png", dpi=150)
    with open(OUT / "nz_a_max.txt", "w") as f:
        f.write(str(nz_a_max))
    print(f"Saved -> {summary_path}, {config.THRESHOLD_SWEEP_FIG_DIR}/pool_threshold_sweep.png, {OUT}/nz_a_max.txt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test on a small threshold subset before the full sweep**

```bash
cd /project/cfRNA_NormativeModeling
Rscript -e 'if (!requireNamespace("optparse", quietly=TRUE)) install.packages("optparse", repos="https://cloud.r-project.org")'
python -c "
import Modeling.pool_threshold_sweep as s
Xs, Y, batch = s.load_hc()
print('HC shape', Xs.shape, Y.shape, 'unique batches', len(set(batch)))
"
```
Expected: prints HC shape (should be `(693, 10)` and `(693, ~19538)` matching the spike's known HC count) with no error. This confirms `load_hc()` works before committing to the full R sweep (which fits `fit_pooled_glmm` on up to tens of thousands of stacked rows per threshold x fold — untested at this scale; if a threshold hangs or OOMs, lower it out of `thresholds` in `pool_threshold_sweep.R` and note it in the commit message rather than blocking the whole sweep).

- [ ] **Step 4: Run the full sweep and commit**

```bash
cd /project/cfRNA_NormativeModeling
python Modeling/pool_threshold_sweep.py
git add Modeling/pool_threshold_sweep.R Modeling/pool_threshold_sweep.py
git commit -m "Add pool threshold sweep (batch random-intercept), auto-picks nz_a_max"
```

---

### Task 7: `Modeling/model_engine_mixed.py` — orchestration

**Files:**
- Create: `Modeling/model_engine_mixed.py`

**Interfaces:**
- Consumes: `config.GLMM_FIT_R`, `config.POOL_SWEEP_R`'s output (`Threshold_Sweep/nz_a_max.txt`), `Modeling/marginal_rqr.py`.
- Produces: `NormativeModelEngineMixed` class with `load_hc_data()`, `assign_routes()` (reads `nz_a_max.txt`), `train()` (writes HC data to temp files, calls `glmm_fit.R --mode cascade` once, calls `fit_pooled_glmm` for the pool route via a small dedicated R call), `score(X_test_raw, Y_test, gene_names=None, seed=42, as_dict=False)` (same contract as `Modeling/model_engine.py`'s `score()`, uses `marginal_nb_rqr` for stages with `tau2>1e-6`), `save(directory)`/`load(directory)` writing to `config.ENGINE_MIXED_DIR`.

- [ ] **Step 1: Write the class** (mirrors `Modeling/model_engine.py`'s `NormativeModelEngine` structure — same `load_hc_data`, same `GeneRecord`-style dataclass extended with `tau2`/`batch_glmm_singular`, same `save`/`load`/`training_summary` shape — but `train()` delegates fitting to one `glmm_fit.R` subprocess call instead of per-gene R calls)

```python
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from dispersion_trend import build_trend, load_trend, save_trend
from marginal_rqr import marginal_nb_rqr

MP = config.MODELING_PARAMS


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
        # nz_a_max is deferred (Task 6a/6b) -- default to 0 (no gene routed to
        # "pool", every gene attempts the model cascade) until a real threshold
        # is chosen and Threshold_Sweep/nz_a_max.txt exists.
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
        ], check=True, cwd=str(config.MODELING_DIR))

        results = pd.read_csv(f"{tmp_dir}/results.csv").set_index("gene")
        for g, row in results.iterrows():
            rec = self.genes[g]
            rec.stage, rec.ok, rec.singular, rec.tau2 = row["stage"], bool(row["ok"]), bool(row["singular"]), float(row["tau2"])
            rec.mu_coef = row[[c for c in results.columns if c.startswith("mu_coef_")]].values.astype(float)
            rec.disp_coef = row[[c for c in results.columns if c.startswith("disp_coef_")]].values.astype(float)
            rec.fail_reason = row["fail_reason"]
            if not rec.ok:
                rec.route = "excluded"

    def training_summary(self):
        rows = [dict(gene=r.name, route=r.route, stage=r.stage, nz=r.nz, ok=r.ok,
                    singular=r.singular, tau2=r.tau2, fail_reason=r.fail_reason)
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
            mu = np.clip(np.exp(Xa @ np.nan_to_num(rec.mu_coef, nan=0.0)), 1e-6, 1e8)
            alpha = np.exp(-rec.disp_coef[0]) if not np.isnan(rec.disp_coef[0]) else self.alpha_fn(float(mu.mean()))
            Z[:, j] = marginal_nb_rqr(Y_test[:, j].astype(np.float64), mu, alpha, rec.tau2, seed + j)
        return Z if not as_dict else {"combined": Z, "gene_names": list(gene_names)}

    def save(self, directory):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "genes.pkl", "wb") as f: pickle.dump(self.genes, f)
        with open(directory / "scaler.pkl", "wb") as f: pickle.dump(self.scaler, f)
        self.training_summary().to_csv(directory / "training_summary.csv")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        engine = cls()
        with open(directory / "genes.pkl", "rb") as f: engine.genes = pickle.load(f)
        with open(directory / "scaler.pkl", "rb") as f: engine.scaler = pickle.load(f)
        engine.alpha_fn = load_trend()
        return engine
```

- [ ] **Step 2: Smoke-test with `--limit`-equivalent (20 genes)**

```bash
cd /project/cfRNA_NormativeModeling
python -c "
from Modeling.model_engine_mixed import NormativeModelEngineMixed
e = NormativeModelEngineMixed()
e.load_hc_data()
e.build_dispersion_trend()
e.assign_routes()
e.train(limit=20)
print(e.training_summary().head(20))
assert e.training_summary()['ok'].sum() > 0, 'FAIL: no gene fit succeeded'
print('PASS: smoke-trained 20 genes')
"
```
Expected: `PASS: smoke-trained 20 genes`. Requires Task 6's `nz_a_max.txt` to already exist.

- [ ] **Step 3: Commit**

```bash
git add Modeling/model_engine_mixed.py
git commit -m "Add NormativeModelEngineMixed orchestration + marginal scoring"
```

---

### Task 8: `Modeling/cv_glmm_engine.py`

**Files:**
- Create: `Modeling/cv_glmm_engine.py`

**Interfaces:**
- Consumes: `Modeling/model_engine_mixed.py` (Task 7, for `load_hc_data`), `config.GLMM_FIT_R` in `--mode fixed_stage`.
- Produces: `CV_Results_mixed/cv_stats.csv` (same columns as `Modeling/cv_model_engine.py`'s: `gene,route,stage,nz,w1,mean_z,std_z,skew_z,kurt_z,n_valid`).

- [ ] **Step 1: Write the script** (same 5-fold stratified structure as `Modeling/cv_model_engine.py`, but each fold's fit is one `glmm_fit.R --mode fixed_stage` subprocess call over all genes assigned to that fold, not per-gene rpy2)

```python
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from marginal_rqr import marginal_nb_rqr
from model_engine import _w1_normal
from model_engine_mixed import NormativeModelEngineMixed

MP = config.MODELING_PARAMS


def main():
    out_dir = config.CV_MIXED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = NormativeModelEngineMixed.load(config.ENGINE_MIXED_DIR)
    summary = pd.read_csv(config.ENGINE_MIXED_DIR / "training_summary.csv", index_col="gene")
    summary = summary[summary["ok"] & (summary["route"] == "model")]

    e2 = NormativeModelEngineMixed()
    e2.load_hc_data()
    n_hc = e2.X_hc_scaled.shape[0]
    folds = list(StratifiedKFold(MP["n_splits"], shuffle=True, random_state=42).split(np.zeros(n_hc), e2.batch))

    tmp = "/tmp/cv_glmm"
    Path(tmp).mkdir(exist_ok=True)
    rows = []
    for fi, (tr, te) in enumerate(folds):
        pd.DataFrame(e2.X_hc_scaled[tr], columns=config.BIAS_COLUMNS).to_csv(f"{tmp}/X_{fi}.csv.gz")
        Y_tr = e2.Y_hc[tr][:, [e2._gene_col[g] for g in summary.index]]
        pd.DataFrame(Y_tr, columns=summary.index).to_csv(f"{tmp}/Y_{fi}.csv.gz")
        pd.DataFrame({"Batch_ID": e2.batch[tr]}).to_csv(f"{tmp}/batch_{fi}.csv.gz")
        gene_stage = summary[["stage"]].reset_index().rename(columns={"index": "gene"})
        gene_stage.to_csv(f"{tmp}/genes_{fi}.csv", index=False)
        subprocess.run([
            "Rscript", str(config.GLMM_FIT_R), "--x", f"{tmp}/X_{fi}.csv.gz", "--y", f"{tmp}/Y_{fi}.csv.gz",
            "--batch", f"{tmp}/batch_{fi}.csv.gz", "--genes", f"{tmp}/genes_{fi}.csv",
            "--trend", str(config.DISPERSION_TREND_PATH), "--mode", "fixed_stage", "--out", f"{tmp}/res_{fi}.csv",
        ], check=True, cwd=str(config.MODELING_DIR))

        fold_fits = pd.read_csv(f"{tmp}/res_{fi}.csv").set_index("gene")
        Xa_te = np.column_stack([np.ones(len(te)), e2.X_hc_scaled[te]])
        for g in summary.index:
            if g not in fold_fits.index or not bool(fold_fits.loc[g, "ok"]):
                continue
            row = fold_fits.loc[g]
            mu_coef = row[[c for c in fold_fits.columns if c.startswith("mu_coef_")]].values.astype(float)
            disp_coef = row[[c for c in fold_fits.columns if c.startswith("disp_coef_")]].values.astype(float)
            mu = np.clip(np.exp(Xa_te @ np.nan_to_num(mu_coef, nan=0.0)), 1e-6, 1e8)
            alpha = np.exp(-disp_coef[0]) if not np.isnan(disp_coef[0]) else e2.alpha_fn(float(mu.mean()))
            y_te = e2.Y_hc[te, e2._gene_col[g]]
            z = marginal_nb_rqr(y_te, mu, alpha, float(row["tau2"]), seed=42 + fi)
            rows.append(dict(gene=g, fold=fi, z=z))

    zdict = {}
    for g in summary.index:
        zs = [r["z"] for r in rows if r["gene"] == g]
        if not zs:
            continue
        zdict[g] = np.concatenate(zs)

    stats = []
    for g, z in zdict.items():
        v = z[np.isfinite(z)]
        if len(v) < 8:
            continue
        stats.append(dict(gene=g, route="model", stage=summary.loc[g, "stage"], nz=int(summary.loc[g, "nz"]),
                          w1=_w1_normal(v), mean_z=float(v.mean()), std_z=float(v.std()),
                          skew_z=float(skew(v)), kurt_z=float(kurtosis(v)), n_valid=len(v)))
    df = pd.DataFrame(stats)
    df.to_csv(out_dir / "cv_stats.csv", index=False)
    print(df.groupby("stage")[["w1", "mean_z", "std_z"]].median().to_string())
    print(f"Saved -> {out_dir}/cv_stats.csv")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test with a small gene subset**

```bash
cd /project/cfRNA_NormativeModeling
python -c "
import pandas as pd
s = pd.read_csv('Modeling/engine_state_mixed/training_summary.csv', index_col='gene')
s[s['ok'] & (s['route']=='model')].head(20).to_csv('/tmp/cv_smoke_summary.csv')
"
```
Then temporarily point `cv_glmm_engine.py`'s `summary` read at `/tmp/cv_smoke_summary.csv` for a first run, confirm `cv_stats.csv` has rows and `w1` values are finite, then run the real script unmodified for the full gene set.

- [ ] **Step 3: Commit**

```bash
git add Modeling/cv_glmm_engine.py
git commit -m "Add cv_glmm_engine.py (5-fold CV, per-fold R refit, marginal scoring)"
```

---

### Task 9: Full unattended run

**Files:** none new — orchestrates prior tasks on real data. `nz_a_max` is still undecided (deferred per the reorder), so this reuses Task 6a's already-computed `full_cascade_unconstrained.csv` (every gene, no pool gating) as the engine's training result rather than re-running the ~19,538-gene cascade a second time.

- [ ] **Step 1: Load Task 6a's output into `engine_state_mixed/` directly** (skip re-training)

```bash
cd /project/cfRNA_NormativeModeling
python -c "
import pickle
import pandas as pd
from Modeling.model_engine_mixed import GeneRecordMixed
import config

df = pd.read_csv(config.THRESHOLD_SWEEP_DIR / 'full_cascade_unconstrained.csv').set_index('gene')
genes = {}
for g, row in df.iterrows():
    rec = GeneRecordMixed(name=g, route='model' if row['ok'] else 'excluded', stage=row['stage'],
                          ok=bool(row['ok']), singular=bool(row['singular']), tau2=float(row['tau2']),
                          fail_reason=row['fail_reason'])
    rec.mu_coef = df.loc[g, [c for c in df.columns if c.startswith('mu_coef_')]].values.astype(float)
    rec.disp_coef = df.loc[g, [c for c in df.columns if c.startswith('disp_coef_')]].values.astype(float)
    genes[g] = rec

config.ENGINE_MIXED_DIR.mkdir(parents=True, exist_ok=True)
with open(config.ENGINE_MIXED_DIR / 'genes.pkl', 'wb') as f:
    pickle.dump(genes, f)
df.to_csv(config.ENGINE_MIXED_DIR / 'training_summary.csv')
print('ok rate:', df['ok'].mean(), 'stage counts:', df['stage'].value_counts().to_dict())
"
```
Expected: prints an `ok` rate and stage-count breakdown for all HC protein-coding genes.

- [ ] **Step 2: Full CV run** (background; per user confirmation, proceed automatically once Task 4's regression check passes; this re-fits per fold via `glmm_fit.R --mode fixed_stage`, it does not reuse Step 1's full-data fit)

```bash
cd /project/cfRNA_NormativeModeling
python Modeling/cv_glmm_engine.py
```
Expected: completes (may take hours per the design spec's core-hour estimate x5 folds), `Saved -> Modeling/CV_Results_mixed/cv_stats.csv`, per-stage median `w1` printed (values near the existing engine's calibration are the target, not a hard gate — report whatever is observed).

- [ ] **Step 3: Commit outputs' provenance** (data itself is gitignored per `*.csv`/`*.pkl` convention; commit only code changes if any were needed to get the run through)

```bash
cd /project/cfRNA_NormativeModeling
git status --short
git add -A -- Modeling/*.py Modeling/*.R
git commit -m "Full training + CV run complete" --allow-empty
```

---

## Self-Review

**Spec coverage:** all four design-spec components (glmm_fit.R cascade, model_engine_mixed.py, cv_glmm_engine.py, pool_threshold_sweep) have a task. Ordering constraint (sweep before training before CV) enforced by Task 6 running before Task 7/8 and gating on `nz_a_max.txt`. Spike-pilot regression fixture is Task 4, before any full run (Task 9).

**Placeholders:** none — every step has complete, previously-verified-where-nontrivial code (the offset-dispersion trick was empirically confirmed against a live glmmTMB fit before writing Task 2).

**Type/name consistency:** `GeneRecordMixed` fields (`stage`, `ok`, `singular`, `tau2`, `mu_coef`, `disp_coef`, `fail_reason`) match what `glmm_fit.R`'s CSV output columns produce and what `cv_glmm_engine.py`/`model_engine_mixed.py.score()` read.

**Known risk carried forward, not hidden:** `fit_pooled_glmm` (Task 2/6) fits glmmTMB on a stacked design that can reach tens of thousands of rows per threshold — untested at this scale by the spike. Task 6 Step 3 requires a smoke check before the full sweep; if a threshold hangs, drop it rather than block the whole sweep (noted inline in that task).
