## Phase Objective
This phase focuses exclusively on testing and validating Mixed Effect Modeling. To address the limitations of previous models that omitted Batch IDs, this approach explicitly incorporates batch effects into the model to evaluate its structural validity and goodness-of-fit.

## Current Modeling Methodology (v3, 2026-07-27)

Design record: `docs/superpowers/specs/2026-07-27-eb-dispersion-cook-outlier-design.md`.

### 1. Standard Gene Modeling (Two-Tier Demotion Cascade)

v2's middle stage (`nbi_disp_intercept`) and its final hard-fixed stage
(`nb_fixed`) are both gone. Two measured facts drove this:

* **The v2 dispersion prior was crushing the slopes, not the intercept.**
  glmmTMB 1.1.9 with `coef=""` already excludes `"(Intercept)"` from the prior
  (`proc_priors` sets `prior_elstart=1`; verified `elstart=1, elend=2` on a
  3-coefficient dispersion model), and the real v2 run confirms the intercept was
  untouched (n=19,085, mean -0.358, sd 1.754, 1-99% range [-3.83, 2.86]). What
  `normal(0, 0.05)` did crush were the slopes: a true slope of 0.40 collapsed to
  0.04-0.12 on synthetic data, and only 27% of real fitted slopes exceeded |0.1|.
  Stage `nbi` was therefore nearly identical to `nbi_disp_intercept`, which is
  why the latter only ever claimed 5 genes out of 19,220.
* **A zero-df trend-fixed dispersion is unjustifiable** when the trend is a weak
  predictor of individual-gene dispersion (measured: gene dispersion scatters
  ~6.6x around the lowess curve, `tau_d^2 = 0.54`).

Both remaining stages carry the same EB intercept shrinkage, hence the names.

**Step 1: `nbi_full_eb`** -- covariates enter both submodels.
* Mean: $\log(\mu_i) = \beta_0 + \sum_{k=1}^{10}\beta_k X_{ik} + b_j$
* Dispersion: $\log(\theta_i) = \gamma_0 + \sum_{k=1}^{10}\gamma_k X_{ik}$, slopes
  under an EB prior `normal(0, tau_k)`, intercept squeezed toward the trend.

**Step 2: `nbi_intercept_eb`** -- dispersion loses its covariate slopes.
* Mean: unchanged.
* Dispersion: $\log(\theta_i) = \gamma_0$, squeezed toward the trend.

**A gene failing both is `route="excluded"`** -- no further fallback (unchanged
v2 policy: a covariate-free 1-df model violates the normative premise and always
trivially "converges"). Per-stage reject reasons
(`nbi_full_eb_reject_reason` / `nbi_intercept_eb_reject_reason`) are kept for
every gene, so the full demotion history stays auditable.

### 1a. EB Dispersion Shrinkage (`core/eb_shrinkage.py`)

Both shrinkage targets use the same limma/edgeR moment decomposition: for a
per-gene MLE with standard error `SE_g`, `Var(phi_hat) = tau^2 + mean(SE^2)`, so
`tau^2 = max(0, (1.4826*MAD(phi_hat))^2 - median(SE^2))`. MAD/median rather than
variance/mean, because a few near-divergent genes would otherwise inflate `tau`
and silently disable the shrinkage.

* **Slopes.** `--mode pilot` fits `nbi_full_eb` with *no* dispersion prior on an
  HC-mean-stratified gene subsample (`EB_PARAMS["pilot_n_genes"]`, 10 strata,
  seed 42), and `tau_k` is read off the across-gene slope spread per covariate.
  Cached to `<engine_dir>/disp_prior.json` (pilot skipped when it exists).
  Measured range: 0.10-0.36, i.e. 2-7x looser than v2's blanket 0.05.
* **Intercept.** Not penalized in the fit. A one-pass analytic squeeze toward
  `-log(alpha_of(mean_g))` is applied in Python afterwards, pooled over both
  stages (`nbi_intercept_eb` alone is too small for a stable `tau_d`):
  `log_theta_post = (hat/SE_0^2 + trend/tau_d^2)/(1/SE_0^2 + 1/tau_d^2)`.
  `SE_0 = NaN` (unusable `sdreport`) means `SE^2 = inf`, i.e. exactly the trend
  value -- v2's hard-fixed stage survives as the limiting case of the EB rule.
  The result overwrites `disp_coef[0]`, so `score()` needs no stage branch;
  `log_theta_raw` keeps the pre-squeeze value.

The squeeze is deliberately adaptive and, on this dataset, numerically small for
well-measured genes: `SE_0` runs 0.058 (high nz) to 0.29 (low nz) against
`tau_d ~ 0.73`, so the shrink weight `SE^2/(SE^2+tau_d^2)` spans ~0.006 to ~0.14.
That is EB working correctly, not a no-op -- the data say each gene's own
dispersion is far better evidence than the trend, which is exactly why the v2
hard-fixed stage was harmful.

*Known approximation:* mean coefficients and `tau2` are not refit under the
squeezed dispersion, so `(mu, alpha)` is not the joint posterior mode. limma and
edgeR do the same, and it is `alpha` entering the RQR that governs Z calibration.

### 1a-2. Dispersion Trend (`core/dispersion_trend.py:build_trend_from_fits`)

The trend is both the prior mean of the intercept squeeze and the variance Cook's
distance is measured against, so it gates both features. v2's `build_trend` used
raw-count MoM `(var-mean)/mean^2` with covariates and batch ignored, which
measures `alpha + tau^2 + CV^2_X(exp(beta'X))`, not the conditional dispersion
(verified: at mu~819 the three terms sum to 1.541 vs an observed 1.495, split
alpha 7.1% / batch 5.7% / covariates 82.3%; at mu~0.19, alpha 45.5% / covariates
37.0%). The covariate term is what the mean submodel already explains, so the
trend double-counted it -- over-estimating by 2.30x at mu~0.12 rising
monotonically to 16.71x at mu~1193 across 19,085 genes.

`build_trend_from_fits` restores edgeR/DESeq2's order of operations: lowess of
`log(alpha_fit)` on `log(mean)` over the pilot's own covariate-adjusted
dispersions (frac=0.3, it=3 robustifying iterations). Per-bin |median residual|
falls to <= 0.067, overall 1.520 -> 0.008. Two knock-on effects: `tau_d^2` drops
0.460 -> 0.115 so the intercept squeeze gets 3-5x stronger (it was a biased trend
that had been disabling its own shrinkage), and Cook's distance is restored at
high expression where a 16x-too-large alpha made it structurally unable to fire.

**Batch is explicit in the mean submodel only** (`(1|batch__)`), not in the
dispersion submodel, so alpha_fit -- and therefore the trend -- is conditional on
batch mean shifts but absorbs batch dispersion heterogeneity. HC has 31 batches
over 693 samples with 5 singletons (0.7%) and 9 batches of n<=3 (2.5%); the
random intercept partially pools them (measured BLUP shrinkage 0.366 at n_j=1,
0.663 at 3, 0.884 at 9, matching `n_j tau^2/(n_j tau^2 + v)`). A fixed batch
factor would instead give each singleton a free parameter that fits its one
sample exactly, deflating precisely the dispersion the trend is built from.
`score()` never uses batch: it marginalizes `u ~ N(0, tau^2)` by Gauss-Hermite,
which is why unseen batches (CV folds, LOBO, disease samples) need no BLUP.

`core/trend_report.py` runs on every pilot and writes
`Figures/dispersion_trend.png` (trend fit / bias vs expression / residual spread
with tau_d / tau_k bars) plus `trend_residuals.csv`, so no trend is ever deployed
without its calibration record.

`training_summary.csv` gains `tau2_collapsed` (`tau2 < 1e-4`): the R-side
`singular` flag only fires when `pdHess` is FALSE and so never fired once in
19,220 v2 genes, while 30.0% actually had a collapsed batch variance (32.3% had
`tau2 > 0.1`; percentiles 0/0/0.034/0.161/0.627/3.836 at 10/25/50/75/90/99).

### 1b. Cook's Distance Outlier Removal (`core/glmm_helpers.R:cook_outliers`)

One-step Pregibon (1981) Cook's distance on the fitted NB2 log-link model, with
the random intercept absorbed into `mu` and the hat matrix built from the
fixed-effect design only. Samples with `D_i > qf(0.99, p, n-p)` are dropped
(largest first, at most `floor(0.05*n)`) and the stage is refit once -- DESeq2's
cutoff rule. Observations are **removed**, not replaced by a trimmed mean, which
would fabricate counts and bias dispersion downward.

**The variance uses the lowess TREND dispersion, not the gene's own fitted one.**
With a freely estimated dispersion an outlier masks itself: three injected 20x
outliers inflated a near-Poisson gene's alpha 0.004 -> 0.147 (36x), which shrank
their Pearson residuals enough that `D` fell from 4.5-9.4 to 0.6-1.1, below the
2.29 cutoff -- nothing was flagged at *any* outlier magnitude. With the trend
alpha the same three are caught exactly and alpha recovers to 0.0042; a genuinely
overdispersed gene given a 4x-too-small trend alpha still flags nothing, and a
clean gene flags nothing.

If the refit fails to converge the pre-removal fit is kept and
`outlier_refit_failed=True` is logged. Cook's distance is computed even when the
first fit failed the convergence gate, since outliers can be the cause.
Measured on an nz-stratified 300-gene sample: 2.7% of genes drop exactly 1
sample, concentrated in the mid-nz bins. Sensitivity is conservative -- 5x
outliers on a near-Poisson gene were not caught.

---

### 1c. Per-Gene SHASH Calibration (`core/calibration.py`)
Unchanged from v2. Normative modeling requires held-out HC Z-scores to be
N(0,1); genes with real skew/kurtosis in their RQR distribution break FDR
calibration downstream (theoretical-null p-values become wrong).
`core/calibration.py` fits a SHASH (sinh-arcsinh, Jones & Pewsey 2009) per gene
to its held-out Z-scores and reports naive-vs-SHASH-corrected 95%-exceedance
rates, corrected skew/kurtosis, and naive-vs-corrected BH-FDR reject rates
(computed on held-out HC, a true null, so any nonzero reject rate is exactly the
false-positive inflation the correction should remove).

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
* `/core`: Exclusively reserved for the core modeling engine logic. `glmm_helpers.R`(per-gene fit + Cook's-distance outlier removal + pooled-GLM fit primitives) · `glmm_fit.R`(CLI, `--mode cascade|pilot|fixed_stage`, `--disp-prior`) · `eb_shrinkage.py`(EB prior sd estimation + dispersion-intercept squeeze) · `glmm_fit_pool.R`(pooled-GLM CLI, unused this round) · `dispersion_trend.py`(`build_trend_from_fits` = canonical covariate-adjusted trend; `build_trend` = covariate-free diagnostic reference) · `trend_report.py`(always-on trend/prior calibration figure) · `marginal_rqr.py`(tau2-marginalized RQR/log-likelihood, Gauss-Hermite) · `model_engine_mixed.py`(`NormativeModelEngineMixed`/`GeneRecordMixed`) · `calibration.py`(per-gene SHASH calibration) · `run_engine.py`(entry point: `python core/run_engine.py [--limit N] [--pilot-genes N] [--resume]`, writes `engine_state_mixed/`: adds `disp_prior.json` and `eb_meta.json` to the saved state)
* `/validation`: Designated for large-scale validation pipelines and auditing scripts. `cv_engine.py`(5-fold CV with explicit per-`(gene, fold)` success/failure logging to `fold_stats.csv` — written this round, **not yet executed**: the actual CV run and the SHASH-based per-gene comprehensive report are deferred to a follow-up plan)
* `/_legacy/core_v1`, `/_legacy/validation_v1`: pre-2026-07-25 4-stage-cascade engine and its validation scripts, kept for reference/reproducibility only — do not import from or modify.
* **Root Directory (`.ipynb`):** All visualizations, tabular summaries, and analytical reviews must be conducted in Jupyter notebooks within the current working directory to allow immediate evaluation.
* *Strictly no modifications are permitted to external code outside these defined scopes.*


* **Coding Standards:**
* Prioritize concise, space-efficient, and highly optimized code implementations.
* Minimize comments; restrict them to core functionalities or essential comparisons with prior code, written exclusively in English.
* Notebook headers must remain strictly minimalist, utilizing standard numerical indexing and brief titles only.

