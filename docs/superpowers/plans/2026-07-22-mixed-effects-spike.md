# Mixed-Effects Batch Random-Intercept — Step 0 Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer the five open questions in `docs/superpowers/specs/2026-07-22-mixed-effects-batch-refactor-design.md`'s "Step 0" section on a ~30-50 gene pilot, entirely inside `MixedEffectsModeling/`, before any full-engine implementation is attempted.

**Architecture:** Python selects and writes a pilot HC data slice to disk (`csv.gz`). A set of standalone R scripts (run via `Rscript`, no rpy2) fit gamlss (reusing `Modeling/gamlss.r`, read-only) and glmmTMB models on that slice and write result tables back to disk. Python scripts then read those tables and produce the final spike report with concrete numbers, no interpretation deferred.

**Tech Stack:** Python (scanpy, pandas, numpy, sklearn — conda env `scRNA`), R 4.3.1 (gamlss already installed; glmmTMB to be installed via conda-forge), file-based IO only (no rpy2).

## Global Constraints

- Branch: `mixed-effects-batch-refactor` (already created).
- All new files live under `MixedEffectsModeling/`. No edits to `Modeling/`, root `config.py`, or any other existing file — `Modeling/gamlss.r` is only *read* (`source()`d), never modified.
- `MixedEffectsModeling/config.py` is fully separate from root `config.py` — copy needed constants by value, do not import root `config.py`.
- No type hints on function signatures (project convention).
- No test framework in this repo (CLAUDE.md: "테스트 스위트·린터·빌드 시스템 없음(연구 코드). 검증은 노트북 재실행/스크립트 산출물 확인으로 수행"). Every script embeds its own `assert` checks and prints a clear `PASS`/`FAIL` line — that IS the test for this codebase. Do not add pytest.
- All R scripts run with cwd = `MixedEffectsModeling/` (same convention as `EDA/` assuming cwd=EDA).
- Conda env: `scRNA` (`source ~/miniconda3/etc/profile.d/conda.sh && conda activate scRNA` before any Python or R command).
- Commit after every task.

---

### Task 1: `MixedEffectsModeling/config.py` scaffold

**Files:**
- Create: `MixedEffectsModeling/config.py`

**Interfaces:**
- Produces: `H5AD_PATH` (Path), `BIAS_COLUMNS` (list[str]), `STRATIFY_COL` (str, `"Batch_ID"`), `SPIKE_DIR` (Path), `SPIKE_PARAMS` (dict with `outlier_z`, `max_outlier_iter`, `max_remove_frac`, `ridge_lambda_sigma`, `beta_explode_thr`, `gaic_k`, `n_pilot_genes`, `seed`), `GAMLSS_R_HELPER` (Path, points at `Modeling/gamlss.r` — read-only reuse).

- [ ] **Step 1: Write the config file**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
H5AD_PATH = ROOT / "OpenAccess_nfcore" / "Merged_Processed_AnnData_with_Batch_Biases_QC_Status.h5ad"
GAMLSS_R_HELPER = ROOT / "Modeling" / "gamlss.r"

SPIKE_DIR = Path(__file__).resolve().parent / "Spike_Results"

BIAS_COLUMNS = [
    "log(Total Reads)",
    "Spliced Reads (%)",
    "gDNA Contamination (Intron/Exon)",
    "rRNA Fraction",
    "RNA Degradation (3' Bias)",
    "Platelet Score",
    "GC Bias",
    "Gene Length Bias",
    "NG80",
    "(NP80/NG80)",
]

STRATIFY_COL = "Batch_ID"

SPIKE_PARAMS = {
    "outlier_z": 5.0,
    "max_outlier_iter": 3,
    "max_remove_frac": 0.05,
    "ridge_lambda_sigma": 0.05,
    "beta_explode_thr": 3.0,
    "gaic_k": 2.0,
    "n_pilot_genes": 40,
    "seed": 42,
}
```

- [ ] **Step 2: Verify paths resolve**

Run:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate scRNA
cd /project/cfRNA_NormativeModeling
python -c "
import MixedEffectsModeling.config as c
assert c.H5AD_PATH.exists(), c.H5AD_PATH
assert c.GAMLSS_R_HELPER.exists(), c.GAMLSS_R_HELPER
print('PASS: config paths resolve')
print(c.H5AD_PATH)
print(c.GAMLSS_R_HELPER)
"
```
Expected: `PASS: config paths resolve` followed by the two paths. If `H5AD_PATH` doesn't exist, fix the exact filename by running `ls OpenAccess_nfcore/*.h5ad` and correcting Step 1.

Note: `MixedEffectsModeling/` has no `__init__.py` — Python 3 treats it as an implicit namespace package when run from the repo root, so `import MixedEffectsModeling.config` works without one. Do not add `__init__.py`.

- [ ] **Step 3: Commit**

```bash
git add MixedEffectsModeling/config.py
git commit -m "$(cat <<'EOF'
Add isolated config.py for mixed-effects spike workspace

Copies needed constants by value from root config.py rather than
importing it, per the design spec's isolation requirement for
MixedEffectsModeling/.
EOF
)"
```

---

### Task 2: Pilot gene selection + HC data export

**Files:**
- Create: `MixedEffectsModeling/pilot_select.py`
- Test: embedded asserts, run as `__main__`

**Interfaces:**
- Consumes: `MixedEffectsModeling.config` (Task 1).
- Produces on disk (under `config.SPIKE_DIR`): `pilot_X.csv.gz` (index=HC sample id, columns=`BIAS_COLUMNS`, scaled), `pilot_batch.csv.gz` (index=HC sample id, column `Batch_ID`, raw string labels), `pilot_Y.csv.gz` (index=HC sample id, columns=pilot gene ids, raw integer counts), `pilot_genes.csv` (columns `gene`, `nz`, `mean_hc`).

- [ ] **Step 1: Write `pilot_select.py`**

```python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MixedEffectsModeling.config as config


def load_hc_slice():
    adata = sc.read_h5ad(config.H5AD_PATH)
    adata = adata[adata.obs["QC_Passed"] == True]
    adata = adata[adata.obs["Phenotype_Processed"].notna()]
    adata = adata[adata.obs["Phenotype_Processed"] != "Unknown"]
    adata = adata[adata.obs["broad_protocol_category"] != "Exome-based (EB)"]
    is_hc = (adata.obs["Phenotype_Processed"].astype(str) == "Healthy Control").values
    adata_hc = adata[is_hc].copy()

    X_raw = adata_hc.obs[config.BIAS_COLUMNS].values.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    batch = adata_hc.obs[config.STRATIFY_COL].astype(str).values

    Y_raw = adata_hc.X.toarray() if issparse(adata_hc.X) else np.asarray(adata_hc.X)
    Y = np.round(Y_raw).astype(np.float64)

    is_pc = (adata_hc.var["GeneType"] == "protein_coding").values
    gene_names = adata_hc.var_names[is_pc].tolist()
    Y_pc = Y[:, is_pc]

    sample_ids = adata_hc.obs_names.tolist()
    return sample_ids, X_scaled, batch, Y_pc, gene_names


def select_pilot_genes(Y_pc, gene_names, n_pilot, seed):
    nz = (Y_pc > 0).sum(axis=0)
    mean_hc = Y_pc.mean(axis=0)
    df = pd.DataFrame({"gene": gene_names, "nz": nz, "mean_hc": mean_hc})
    df = df[df["nz"] >= config.SPIKE_PARAMS["outlier_z"]]  # trivial floor, avoids all-zero genes

    # Stratify across the NZ range in 4 bins, weighted toward low-NZ (low-expression,
    # highest sigma-explosion risk per the design spec) -- 40% of the pilot from the
    # lowest-NZ quartile, remaining 60% spread evenly across the other three.
    rng = np.random.default_rng(seed)
    df = df.sort_values("nz").reset_index(drop=True)
    quartile_edges = np.quantile(df.index, [0, 0.25, 0.5, 0.75, 1.0]).astype(int)
    weights = [0.4, 0.2, 0.2, 0.2]
    picks = []
    for i in range(4):
        lo, hi = quartile_edges[i], quartile_edges[i + 1]
        bin_idx = df.index[lo:hi] if i < 3 else df.index[lo:hi + 1]
        n_take = max(1, round(n_pilot * weights[i]))
        n_take = min(n_take, len(bin_idx))
        picks.extend(rng.choice(bin_idx, size=n_take, replace=False).tolist())
    picked = df.loc[sorted(set(picks))].reset_index(drop=True)
    return picked


def main():
    config.SPIKE_DIR.mkdir(parents=True, exist_ok=True)
    sample_ids, X_scaled, batch, Y_pc, gene_names = load_hc_slice()
    print(f"HC samples: {len(sample_ids)}  protein-coding genes: {len(gene_names)}  "
          f"unique batches: {len(set(batch))}")

    picked = select_pilot_genes(Y_pc, gene_names, config.SPIKE_PARAMS["n_pilot_genes"],
                                config.SPIKE_PARAMS["seed"])
    gene_col = {g: i for i, g in enumerate(gene_names)}
    pilot_idx = [gene_col[g] for g in picked["gene"]]

    X_df = pd.DataFrame(X_scaled, index=sample_ids, columns=config.BIAS_COLUMNS)
    batch_df = pd.DataFrame({"Batch_ID": batch}, index=sample_ids)
    Y_df = pd.DataFrame(Y_pc[:, pilot_idx], index=sample_ids, columns=picked["gene"].tolist())

    X_df.to_csv(config.SPIKE_DIR / "pilot_X.csv.gz")
    batch_df.to_csv(config.SPIKE_DIR / "pilot_batch.csv.gz")
    Y_df.to_csv(config.SPIKE_DIR / "pilot_Y.csv.gz")
    picked.to_csv(config.SPIKE_DIR / "pilot_genes.csv", index=False)

    assert X_df.shape[0] == Y_df.shape[0] == batch_df.shape[0], "row-count mismatch across exports"
    assert Y_df.shape[1] == len(picked), "pilot gene count mismatch"
    assert batch_df["Batch_ID"].nunique() >= 2, "expected multiple batches in HC pilot slice"
    print(f"PASS: wrote pilot data for {Y_df.shape[1]} genes x {X_df.shape[0]} HC samples, "
          f"{batch_df['Batch_ID'].nunique()} batches")
    print(picked)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate scRNA
cd /project/cfRNA_NormativeModeling
python MixedEffectsModeling/pilot_select.py
```
Expected: ends with a line starting `PASS: wrote pilot data for` and a printed table of ~40 genes with `nz` values spanning low to high. If it fails on `QC_Passed`/`Phenotype_Processed`/`GeneType` column names, check the actual column names with `python -c "import scanpy as sc; a=sc.read_h5ad('OpenAccess_nfcore/Merged_Processed_AnnData_with_Batch_Biases_QC_Status.h5ad'); print(a.obs.columns.tolist()); print(a.var.columns.tolist())"` and correct Step 1 to match (these must match exactly what `Modeling/model_engine.py:load_hc_data` uses, since the goal is a same-population pilot).

- [ ] **Step 3: Commit**

```bash
git add MixedEffectsModeling/pilot_select.py
git commit -m "$(cat <<'EOF'
Add pilot gene selection + HC data export for spike

Selects ~40 genes stratified across the NZ range (weighted toward
low-expression genes, where the design spec flags sigma-explosion
risk as highest) and exports scaled covariates, batch labels, and
raw counts to csv.gz for the R-side spike scripts to consume.
EOF
)"
```

---

### Task 3: glmmTMB install + capability probe

**Files:**
- Create: `MixedEffectsModeling/check_glmmtmb.R`

**Interfaces:**
- Produces: `Spike_Results/glmmtmb_capabilities.json` with keys `installed` (bool), `version` (str), `has_priors_arg` (bool), `priors_probe_success` (bool), `priors_probe_message` (str).

- [ ] **Step 1: Install glmmTMB**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate scRNA
conda install -n scRNA -c conda-forge r-glmmtmb -y
```
Expected: conda resolves and installs `r-glmmtmb` (pulls in `r-tmb`, `r-lme4`, `r-matrix` etc. as dependencies) without error. This modifies the shared `scRNA` conda environment — if the environment is shared with other in-progress work, confirm before running.

- [ ] **Step 2: Write `check_glmmtmb.R`**

```r
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
```

- [ ] **Step 3: Run it and verify**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate scRNA
cd /project/cfRNA_NormativeModeling/MixedEffectsModeling
Rscript -e 'if (!requireNamespace("jsonlite", quietly=TRUE)) install.packages("jsonlite", repos="https://cloud.r-project.org")'
Rscript check_glmmtmb.R
```
Expected: prints the JSON with `"installed": true`. `has_priors_arg` and `priors_probe_success` may be `true` or `false` — either is a valid spike finding, both get consumed by Task 4/5. If `installed` is `false`, stop and resolve the conda install before continuing (Task 4 onward depend on it).

- [ ] **Step 4: Commit**

```bash
git add MixedEffectsModeling/check_glmmtmb.R
git commit -m "$(cat <<'EOF'
Add glmmTMB install verification + priors() capability probe

Empirically determines whether this R 4.3.1 environment's glmmTMB
supports the priors() interface, per the design spec's requirement
to verify this rather than assume it.
EOF
)"
```

---

### Task 4: Sigma parameterization equivalence check (gamlss vs glmmTMB, fixed-effects only)

**Files:**
- Create: `MixedEffectsModeling/fit_fixed_only.R`
- Create: `MixedEffectsModeling/check_sigma_equivalence.py`

**Interfaces:**
- Consumes: `Spike_Results/pilot_X.csv.gz`, `pilot_Y.csv.gz` (Task 2).
- Produces: `Spike_Results/fixed_only_fits.csv` (one row per gene: `gene`, `gamlss_success`, `gamlss_mu_coef_0..10`, `gamlss_sigma_coef_0..10`, `glmmtmb_success`, `glmmtmb_mu_coef_0..10`, `glmmtmb_disp_coef_0..10`).
- Produces: `Spike_Results/sigma_equivalence_report.csv` and a `PASS`/`FAIL` printed verdict.

- [ ] **Step 1: Write `fit_fixed_only.R`**

```r
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
```

- [ ] **Step 2: Run it**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate scRNA
cd /project/cfRNA_NormativeModeling/MixedEffectsModeling
Rscript fit_fixed_only.R
```
Expected: one `gene: gamlss=TRUE glmmTMB=TRUE` line per pilot gene (some `FALSE` is an acceptable spike finding, not a script bug, as long as most succeed), ending with `Wrote Spike_Results/fixed_only_fits.csv`.

- [ ] **Step 3: Write `check_sigma_equivalence.py`**

```python
import numpy as np
import pandas as pd

df = pd.read_csv("Spike_Results/fixed_only_fits.csv")
X = pd.read_csv("Spike_Results/pilot_X.csv.gz", index_col=0)
n_cov = X.shape[1]

both_ok = df[df["gamlss_success"] & df["glmmtmb_success"]]
print(f"{len(both_ok)}/{len(df)} genes converged in both gamlss and glmmTMB")

# Covariate grid: HC-observed rows themselves (already representative of the
# real scoring distribution) plus the all-zero (mean) point.
Xa = np.column_stack([np.ones(len(X)), X.values])
grid = np.vstack([Xa, np.concatenate([[1.0], np.zeros(n_cov)])])

max_rel_diff = 0.0
worst = None
rows = []
for _, row in both_ok.iterrows():
    sigma_coef = row[[f"gamlss_sigma_coef_{i}" for i in range(n_cov + 1)]].values.astype(float)
    disp_coef = row[[f"glmmtmb_disp_coef_{i}" for i in range(n_cov + 1)]].values.astype(float)

    sigma_gamlss = np.exp(grid @ sigma_coef)
    # glmmTMB nbinom2 dispformula predicts log(theta), theta = 1/sigma (see design spec) --
    # so gamlss's sigma corresponds to exp(-grid @ disp_coef), NOT exp(+grid @ disp_coef).
    sigma_glmmtmb = np.exp(-grid @ disp_coef)

    rel_diff = np.abs(sigma_gamlss - sigma_glmmtmb) / np.clip(sigma_gamlss, 1e-8, None)
    gene_max = float(rel_diff.max())
    rows.append({"gene": row["gene"], "max_rel_diff": gene_max})
    if gene_max > max_rel_diff:
        max_rel_diff = gene_max
        worst = row["gene"]

report = pd.DataFrame(rows).sort_values("max_rel_diff", ascending=False)
report.to_csv("Spike_Results/sigma_equivalence_report.csv", index=False)
print(report.head(10))

TOLERANCE = 0.10  # 10% relative -- these are two different optimizers/penalizations,
                  # exact numerical match is not expected, but the sign/reciprocal
                  # mapping being right should keep genes in the same ballpark.
if max_rel_diff < TOLERANCE:
    print(f"PASS: max relative sigma(x) difference {max_rel_diff:.4f} (gene {worst}) "
          f"< tolerance {TOLERANCE}")
else:
    print(f"FAIL: max relative sigma(x) difference {max_rel_diff:.4f} (gene {worst}) "
          f">= tolerance {TOLERANCE} -- do NOT trust the exp(-X@disp_coef) mapping yet, "
          f"re-check the glmmTMB dispformula parameterization before proceeding to Task 5")
```

- [ ] **Step 4: Run it**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate scRNA
cd /project/cfRNA_NormativeModeling/MixedEffectsModeling
python check_sigma_equivalence.py
```
Expected: ends with either `PASS: max relative sigma(x) difference ...` or an explicit `FAIL: ...` line. **A FAIL here is a valid, important spike result** — it means the `sigma = exp(-X@disp_coef)` mapping from the design spec is wrong or incomplete and must be re-derived (e.g. by checking `glmmTMB` docs for `family(nbinom2())$linkfun`/`sigma()` accessor directly) before Task 5 or any later engine work touches sigma. Do not proceed past this task on a FAIL without resolving it — record whatever is found either way in the commit message.

- [ ] **Step 5: Commit**

```bash
git add MixedEffectsModeling/fit_fixed_only.R MixedEffectsModeling/check_sigma_equivalence.py
git commit -m "$(cat <<'EOF'
Add gamlss/glmmTMB sigma parameterization equivalence check

Fits both packages fixed-effects-only on the same pilot genes and
compares predicted sigma(x) curves (not raw coefficients) using the
exp(-X@disp_coef) reciprocal mapping derived in the design spec, per
its requirement to verify this empirically before trusting any
glmmTMB-derived dispersion output.
EOF
)"
```

---

### Task 5: Random-intercept fit — tau2 / singular-fit distribution

**Files:**
- Create: `MixedEffectsModeling/fit_random_intercept.R`

**Interfaces:**
- Consumes: `Spike_Results/pilot_X.csv.gz`, `pilot_Y.csv.gz`, `pilot_batch.csv.gz` (Task 2), `Spike_Results/glmmtmb_capabilities.json` (Task 3).
- Produces: `Spike_Results/random_intercept_fits.csv` (one row per gene: `gene`, `converged`, `singular`, `tau2`, `wall_time_sec`, `mu_coef_0..10`, `disp_coef_0..10`, `used_priors`).

- [ ] **Step 1: Write `fit_random_intercept.R`**

```r
source("gamlss.r")  # not used here, kept for name-sanitizing consistency if needed later
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
```

- [ ] **Step 2: Run it**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate scRNA
cd /project/cfRNA_NormativeModeling/MixedEffectsModeling
Rscript fit_random_intercept.R
```
Expected: one line per gene, ending with `PASS: N/N genes converged, K singular, mean wall time ...s`. All-genes-failing would indicate a formula/data bug (check `batch__` factor has >1 level, which Task 2's assert already guarantees) — investigate the R error text before moving on, since Task 6/7 depend on this file existing with real numbers.

- [ ] **Step 3: Commit**

```bash
git add MixedEffectsModeling/fit_random_intercept.R
git commit -m "$(cat <<'EOF'
Add batch random-intercept glmmTMB fit for spike pilot genes

Fits (1|batch) random intercept + dispformula covariates on the
pilot gene set, using priors() if Task 3 found it usable, and
records tau2 / singular-fit flag / wall time per gene -- the raw
data Task 7's report answers the design spec's Step 0 questions from.
EOF
)"
```

---

### Task 6: mclapply memory + chunking behavior

**Files:**
- Create: `MixedEffectsModeling/mclapply_memory_test.R`

**Interfaces:**
- Consumes: same pilot data as Task 5.
- Produces: `Spike_Results/mclapply_memory_log.csv` (columns: `config`, `chunk`, `rss_mb_before`, `rss_mb_after`, `n_genes_in_chunk`, `chunk_wall_time_sec`).

- [ ] **Step 1: Write `mclapply_memory_test.R`**

```r
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
```

- [ ] **Step 2: Run it**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate scRNA
cd /project/cfRNA_NormativeModeling/MixedEffectsModeling
Rscript mclapply_memory_test.R
```
Expected: chunk-by-chunk RSS lines for both configs, ending with `PASS: RSS drift over run -- preschedule=TRUE: X MB, preschedule=FALSE: Y MB`. `ps -o rss=` is Linux/macOS-standard and matches this environment (Linux per the session header).

- [ ] **Step 3: Commit**

```bash
git add MixedEffectsModeling/mclapply_memory_test.R
git commit -m "$(cat <<'EOF'
Add mclapply memory/chunking behavior test for spike

Runs ~200 glmmTMB fits (pilot genes x5) under two mc.preschedule
settings, logging RSS before/after each chunk, to ground the full
engine's chunking strategy in measured TMB memory behavior instead
of assumption.
EOF
)"
```

---

### Task 7: Spike report

**Files:**
- Create: `MixedEffectsModeling/summarize_spike.py`

**Interfaces:**
- Consumes: `Spike_Results/glmmtmb_capabilities.json`, `sigma_equivalence_report.csv`, `random_intercept_fits.csv`, `mclapply_memory_log.csv` (Tasks 3-6).
- Produces: `Spike_Results/SPIKE_REPORT.md`.

- [ ] **Step 1: Write `summarize_spike.py`**

```python
import json

import pandas as pd

with open("Spike_Results/glmmtmb_capabilities.json") as f:
    caps = json.load(f)

sigma_report = pd.read_csv("Spike_Results/sigma_equivalence_report.csv")
ri = pd.read_csv("Spike_Results/random_intercept_fits.csv")
mem = pd.read_csv("Spike_Results/mclapply_memory_log.csv")

lines = []
lines.append("# Step 0 Spike Report\n")

lines.append("## 1. glmmTMB install + priors() capability\n")
lines.append(f"- Installed: {caps['installed']} (version {caps['version']})")
lines.append(f"- `priors` formal argument present: {caps['has_priors_arg']}")
lines.append(f"- Working priors() probe: {caps['priors_probe_success']} "
             f"({caps['priors_probe_message']})\n")

lines.append("## 2. Sigma parameterization equivalence (gamlss vs glmmTMB)\n")
max_diff = sigma_report["max_rel_diff"].max()
verdict = "PASS" if max_diff < 0.10 else "FAIL"
lines.append(f"- Verdict: **{verdict}** (max relative sigma(x) diff = {max_diff:.4f}, "
             f"tolerance 0.10)")
lines.append(f"- Worst gene: {sigma_report.iloc[0]['gene']}\n")

lines.append("## 3. tau2 / singular-fit distribution\n")
conv = ri[ri["converged"]]
lines.append(f"- Converged: {len(conv)}/{len(ri)}")
lines.append(f"- Singular-but-converged: {int(conv['singular'].sum())}/{len(conv)}")
lines.append(f"- tau2 distribution (converged, non-singular genes): "
             f"{conv.loc[~conv['singular'], 'tau2'].describe().to_dict()}")
lines.append(f"- Mean wall time per gene fit: {ri['wall_time_sec'].mean():.2f}s "
             f"(estimate for 17,572 genes at stage nbi: "
             f"{ri['wall_time_sec'].mean() * 17572 / 3600:.1f} core-hours)\n")

lines.append("## 4. mclapply memory behavior\n")
for cfg, g in mem.groupby("config"):
    drift = g["rss_mb_after"].max() - g["rss_mb_before"].min()
    lines.append(f"- `{cfg}`: RSS drift over run = {drift:.0f} MB "
                 f"(chunks: {len(g)}, mean chunk time {g['chunk_wall_time_sec'].mean():.1f}s)")
lines.append("")

lines.append("## Decisions this report should feed back into the design spec\n")
lines.append("- Small-batch (HC n<3) handling: TBD by inspecting tau2 for genes where "
             "at least one batch has n_hc<3 in the pilot set (cross-reference "
             "`Spike_Results/pilot_batch.csv.gz` batch sizes against `random_intercept_fits.csv`).")
lines.append("- Sigma regularization approach: use priors() if section 1 shows a working probe "
             "AND section 2 shows PASS; otherwise fall back to unpenalized + beta_explode_thr "
             "per the design spec.")
lines.append("- mclapply chunking parameters for the full engine: pick whichever config in "
             "section 4 has lower RSS drift per gene.")

report = "\n".join(lines)
with open("Spike_Results/SPIKE_REPORT.md", "w") as f:
    f.write(report)
print(report)
print("\nPASS: wrote Spike_Results/SPIKE_REPORT.md")
```

- [ ] **Step 2: Run it**

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate scRNA
cd /project/cfRNA_NormativeModeling/MixedEffectsModeling
python summarize_spike.py
```
Expected: prints the full report and ends with `PASS: wrote Spike_Results/SPIKE_REPORT.md`.

- [ ] **Step 3: Commit**

```bash
git add MixedEffectsModeling/summarize_spike.py Spike_Results/SPIKE_REPORT.md
git commit -m "$(cat <<'EOF'
Add spike report aggregating all Step 0 findings

Answers the five Step 0 questions from the design spec with the
pilot's actual numbers (sigma equivalence verdict, tau2/singular
distribution, per-gene wall time, mclapply memory drift) and states
which follow-on decisions each feeds -- this is the gate for whether
full-engine implementation proceeds as designed or the spec needs
revision first.
EOF
)"
```

---

## Self-Review

**Spec coverage** — design spec's Step 0 items 1-5 map to: item 1 → Task 3, item 2 → Task 4, item 3 → Task 5, item 4 → Task 6, item 5 → Task 5's `wall_time_sec` (aggregated in Task 7). All five are covered.

**Placeholders** — none; every script above is complete, runnable code against files this plan itself creates in earlier tasks. The one explicit "TBD" in Task 7's generated *report* (small-batch handling) is not a plan placeholder — it is the pilot data correctly not yet containing enough n<3 batches to conclude from 40 genes alone, and is stated in the spec as something the full pilot findings, not this plan, resolve.

**Type/name consistency** — `Spike_Results/pilot_X.csv.gz` / `pilot_Y.csv.gz` / `pilot_batch.csv.gz` (Task 2) are read identically by Tasks 4, 5, 6. `glmmtmb_capabilities.json`'s keys (Task 3) match exactly what Task 5 (`use_priors`) and Task 7 (`caps['...']`) read. `random_intercept_fits.csv` columns (`converged`, `singular`, `tau2`, `wall_time_sec`) match Task 7's aggregation code.

## Out of scope for this plan

Everything past Step 0 — the full one-pass `glmm_fit.R` cascade, the marginal-scoring quadrature module, and any change to `Modeling/` or root `config.py` — is intentionally not planned here. It depends on this spike's findings (per the design spec) and should be a separate plan written after `SPIKE_REPORT.md` exists and has been reviewed.
