## Phase Objective
This phase focuses exclusively on testing and validating Mixed Effect Modeling. To address the limitations of previous models that omitted Batch IDs, this approach explicitly incorporates batch effects into the model to evaluate its structural validity and goodness-of-fit.

## Current Modeling Methodology (v2, 2026-07-25)

### 1. Standard Gene Modeling (Three-Tier Demotion Cascade)
**Step 4 ("intercept", a covariate-free 1-df fallback) was removed.** It violated the normative-modeling premise (a model must use the sample's own covariates to define "expected"), and it always trivially "converged," letting information-poor genes into the deployed gene set as if they had a real fit. A gene that now fails Step 3 (`nb_fixed`) is excluded outright (`route="excluded"`) instead of silently falling back. Empirically this removes exactly the same 877 genes that used to occupy the old Step 4 (`intercept_True`=495 + `intercept_False`=382), confirmed low-`nz` (median 5, 96% with `nz<=30`) and consistently failing convergence at both `nbi` and `nb_fixed` — honest exclusion of genuinely unmodelable genes, not a regression.

**Step 1: nbi (Full Model)**
Assumes covariates linearly influence both the mean expression and overdispersion.
* **Mean Submodel (Log-link):**
$$\log(\mu_i) = \beta_0 + \sum_{k=1}^{10} \beta_k X_{ik} + b_j$$
* **Dispersion Submodel:**
$$\log(\theta_i) = \gamma_0 + \sum_{k=1}^{10} \gamma_k X_{ik}$$

**Step 2: nbi_disp_intercept (Fixed Dispersion Model)**
Implemented when Step 1 fails to converge. The dispersion is treated as a single constant independent of covariates to secure degrees of freedom.
* **Mean Submodel:**
$$\log(\mu_i) = \beta_0 + \sum_{k=1}^{10} \beta_k X_{ik} + b_j$$
* **Dispersion Submodel:**
$$\log(\theta_i) = \gamma_0$$

**Step 3: nb_fixed (Trend-Based Forced Dispersion Model, final stage)**
Triggered when gene-specific dispersion estimation collapses (e.g., boundary errors). A pre-calculated fixed dispersion value ($\text{fixed\_log\_theta}$), derived from a Lowess trend line of the entire healthy control (HC) dataset, is injected as an offset. The dispersion submodel is not estimated; coefficients are set to 0. **If this stage also fails to converge, the gene is excluded** — there is no further fallback.

* **Mean Submodel:**
$$\log(\mu_i) = \beta_0 + \sum_{k=1}^{10} \beta_k X_{ik} + b_j$$
* **Dispersion Submodel:**
$$\log(\theta_i) = \text{fixed\_log\_theta}$$

Every stage's own reject reason is recorded (`nbi_reject_reason` / `nbi_disp_intercept_reject_reason` / `nb_fixed_reject_reason` in `training_summary.csv`), not just the reason for the final accepted/rejected stage — full demotion history is auditable per gene.

Full sweep across all 20,097 protein-coding genes (`nz_a_max=0`, no pooling): 19,220 ok (`nbi`=19,080, `nbi_disp_intercept`=5, `nb_fixed`=135), 877 excluded.

---

### 1b. Per-Gene SHASH Calibration (`core/calibration.py`)
Normative modeling requires held-out HC Z-scores to be N(0,1); genes with real skew/kurtosis in their RQR distribution break FDR calibration downstream (theoretical-null p-values become wrong). `core/calibration.py` fits a SHASH (sinh-arcsinh, Jones & Pewsey 2009) distribution per gene to its held-out Z-scores and reports naive-vs-SHASH-corrected 95%-exceedance rates, corrected skew/kurtosis, and naive-vs-corrected BH-FDR reject rates (computed on held-out HC, a true null, so any nonzero reject rate is exactly the false-positive inflation the correction should remove). This module is built and unit-verified on synthetic data; wiring it into a real per-gene report against actual CV output is deferred (see below).

---

### 2. Pooled GLMM (For Low-Expression Genes)
Designed for rare genes with a severe deficit of non-zero observations. Genes are stacked into a single tensor, sharing fixed effect coefficients ($\beta$) and batch variance ($\sigma^2_{batch}$) at the group level.
For a given gene $g$ with HC mean expression $\bar{Y}_{g, HC}$ and stabilization constant $\epsilon$, the expected value for sample $i$ in batch $j$ within the pooled tensor is:

* **Mean Submodel (with Normalization Offset):**
$$\log(\mu_{i,g}) = \log(\bar{Y}_{g, HC} + \epsilon) + \beta_0 + \sum_{k=1}^{10} \beta_k X_{ik} + b_j$$

**Distribution Selection Logic:**
1. The model is initially fitted using a Poisson distribution ($\text{Var} = \mu$).
2. The overdispersion ratio of the resulting Pearson residuals is evaluated. If the ratio is below a defined threshold (e.g., 2.0), the Poisson coefficients are retained.
3. If the threshold is exceeded, the model is refitted using a Negative Binomial (NB2) distribution to estimate an additional dispersion constant $\log(\theta) = \gamma_0$.

**Currently unused**: `nz_a_max` (the NZ cutoff routing a gene to pooling vs. the per-gene cascade) is still undetermined, so it defaults to 0 and every gene routes to the cascade above — `fit_pooled_glmm`/`glmm_fit_pool.R` code stays in place for when the threshold is picked.
---

## Development & Repository Constraints

* **Iterative Refinement:** The methodologies outlined above are experimental and subject to structural revisions. The optimal normative modeling strategy will be determined iteratively by modifying and debugging the core logic.
* **Directory Enforcement:**
* `/core`: Exclusively reserved for the core modeling engine logic. `glmm_helpers.R`(per-gene fit + pooled-GLM fit primitives) · `glmm_fit.R`(cascade/fixed-stage CLI, `--mode cascade|fixed_stage`) · `glmm_fit_pool.R`(pooled-GLM CLI, unused this round) · `dispersion_trend.py`(Phase 0 covariate-free NB2 trend) · `marginal_rqr.py`(tau2-marginalized RQR/log-likelihood, Gauss-Hermite) · `model_engine_mixed.py`(`NormativeModelEngineMixed`/`GeneRecordMixed`) · `calibration.py`(per-gene SHASH calibration) · `run_engine.py`(entry point: `python core/run_engine.py [--limit N] [--resume]`, writes `engine_state_mixed/`)
* `/validation`: Designated for large-scale validation pipelines and auditing scripts. `cv_engine.py`(5-fold CV with explicit per-`(gene, fold)` success/failure logging to `fold_stats.csv` — written this round, **not yet executed**: the actual CV run and the SHASH-based per-gene comprehensive report are deferred to a follow-up plan)
* `/_legacy/core_v1`, `/_legacy/validation_v1`: pre-2026-07-25 4-stage-cascade engine and its validation scripts, kept for reference/reproducibility only — do not import from or modify.
* **Root Directory (`.ipynb`):** All visualizations, tabular summaries, and analytical reviews must be conducted in Jupyter notebooks within the current working directory to allow immediate evaluation.
* *Strictly no modifications are permitted to external code outside these defined scopes.*


* **Coding Standards:**
* Prioritize concise, space-efficient, and highly optimized code implementations.
* Minimize comments; restrict them to core functionalities or essential comparisons with prior code, written exclusively in English.
* Notebook headers must remain strictly minimalist, utilizing standard numerical indexing and brief titles only.

