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

### 1b. PCIS Outlier Removal (`core/glmm_helpers.R:pcis_outliers`)

**PCIS = Prior-Conditioned Impact Score.** Cook-shaped, deliberately NOT Cook's
distance, and named apart from it so the difference cannot be silently lost:

```
w_i    = mu_i / (1 + alpha_trend * mu_i)               NB2 log-link IRLS weight
M      = [Xf Z],  P = blkdiag(0_p, I/tau^2)            fixed design + batch design
H      = W^1/2 M (M'WM + P)^-1 M' W^1/2,  p_eff = tr(H)
r_i    = (y_i - mu_i) / sqrt(mu_i + alpha_trend * mu_i^2)
PCIS_i = r_i^2 / p_eff * h_ii / (1 - h_ii)^2
```

Both departures from Cook's distance are prior-conditioned, and both were forced
by measurement:

1. **Variance conditioned on the trend prior, not the gene's own fitted
   dispersion.** A freely estimated dispersion lets an outlier mask itself: three
   20x outliers on a synthetic near-Poisson gene inflated alpha 0.004 -> 0.147
   (36x), dropping the statistic from 4.5-9.4 to 0.6-1.1 and flagging nothing at
   *any* outlier magnitude. The EB squeeze cannot fix this -- it weights by
   precision (`SE^2/(SE^2+tau_d^2)`, median w = 0.037 on real fits), and a
   contaminated dispersion has a *small* SE, so 36x contamination survives the
   squeeze as 32x. EB shrinkage defends against noise; PCIS defends against
   contamination.
2. **Leverage conditioned on the prior-penalised mixed design.** `mu` already
   contains the BLUP, so a fixed-effect-only hat matrix mixes two different
   models inside one statistic. Measured: `p_eff` 18.4 vs `p` 11 (40% of the
   model's effective complexity ignored) and singleton-batch leverage 0.017 ->
   0.165 (10x). `tau2 -> 0` sends the penalty to infinity, so `p_eff -> p`
   automatically for the 30% of genes with a collapsed batch variance.

Because the variance is not the fitted model's, the one-step deletion
approximation does not hold: **`qf(0.99, p_eff, n-p_eff)` is an inherited DESeq2
threshold convention, not a distributional result.** PCIS has no F reference
distribution, and the empirical null (below) shows the convention resolves to a
per-observation rate of 1.16e-4, not 0.01. Observations are dropped (largest PCIS first, at most
`floor(0.05*n)`) and the stage is refit once with `droplevels` -- HC has 5
singleton batches, so removing one observation can empty a random-effect level.
Replacement by a trimmed mean was rejected: it fabricates counts and biases
dispersion downward.

**Measured behaviour.** Real data, 120 nz-stratified genes: 19 genes (15.8%) drop
observations, 37 observations total, mostly 1 per gene (max 5), 1 refit failure.
Not depth-confounded -- `corr(n_flag, library depth) = -0.034`, against `+0.619`
for a raw ">30x gene median" filter, which was rejected for exactly that reason
(its top-flagged samples were all in the top depth decile). Residual covariate
associations are mild: Gene Length Bias -0.257, gDNA Contamination +0.213, GC
Bias -0.212, everything else |r| < 0.13.

**Known blind spot, measured.** Contamination is absorbed by the dispersion
SLOPES, not the intercept: at 100x contamination on high-leverage samples the
disp intercept moved only -0.06 while slope estimates moved far more, so alpha_i
at the outlier positions stayed inflated ~13x. Consequently PCIS is sensitive for
low-alpha high-expression genes (3 of 3 detected at 20x) and weak for high-alpha
low-expression genes (1 of 3 even at 100x) -- arguably correct, since a large
count genuinely is plausible under a high-dispersion model, but it means PCIS is
a conservative filter for gross contamination in well-expressed genes, not a
general outlier detector. A proposal to anchor only the dispersion intercept
while keeping the fitted slopes was tested and **refuted**: 0 of 3 detected,
worse than the trend scalar.

**Empirical null, full run (`PCIS_Calibration/`, see its README).** Each gene's
own fitted `(beta, gamma, tau^2)` regenerate clean counts on the real design, the
same stage is refit under the same prior, and PCIS is recomputed -- 19,158 genes x
693 observations. The current threshold's realized null rates are per-observation
1.16e-4 and per-gene FWER 6.94%, against a real-data 7.78% of genes with any
removal (0.096/gene vs 0.081/gene null): **84% of current removals are
attributable to the null**, i.e. the threshold is strict enough to have almost no
power. It is also mis-shaped in the tail -- null max PCIS peaks at `log_mu ~ 2-3`
while `qf` decreases monotonically, giving realized FWER 0.7%-11.4% across
expression deciles. Two calibrated replacements are derived and stored: a pooled
constant (2.7054 at FWER 0.05) which does not fix the shape, and a smooth
`bs(log_mu, df=6)` quantile regression which flattens realized FWER to 4.0-6.6%
(`tau2` and `p_eff` add nothing). **The threshold is not yet changed in
`config.FIT_PARAMS`** -- the target rate form/value is an open decision, and the
real per-observation PCIS distribution (needed for an empirical FDR curve) has not
been extracted yet.

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
* `/core`: Exclusively reserved for the core modeling engine logic. `glmm_helpers.R`(per-gene fit + Cook's-distance outlier removal + pooled-GLM fit primitives) · `glmm_fit.R`(CLI, `--mode cascade|pilot|fixed_stage`, `--disp-prior`) · `eb_shrinkage.py`(EB prior sd estimation + dispersion-intercept squeeze) · `glmm_fit_pool.R`(pooled-GLM CLI, unused this round) · `dispersion_trend.py`(`build_trend_from_fits` = canonical covariate-adjusted trend; `build_trend` = covariate-free diagnostic reference) · `trend_report.py`(always-on trend/prior calibration figure) · `marginal_rqr.py`(tau2-marginalized RQR/log-likelihood, Gauss-Hermite) · `model_engine_mixed.py`(`NormativeModelEngineMixed`/`GeneRecordMixed`) · `calibration.py`(per-gene SHASH calibration) · `pcis_null.R`(PCIS empirical-null simulator: regenerates clean counts from each gene's own fit, refits, recomputes PCIS) · `pcis_calibration.py`(`run_all()` -> `PCIS_Calibration/`: null rates, threshold A/B derivation, figures) · `run_engine.py`(entry point: `python core/run_engine.py [--limit N] [--pilot-genes N] [--resume]`, writes `engine_state_mixed/`: adds `disp_prior.json` and `eb_meta.json` to the saved state)
* `/validation`: Designated for large-scale validation pipelines and auditing scripts. `cv_engine.py`(5-fold CV with explicit per-`(gene, fold)` success/failure logging to `fold_stats.csv` — written this round, **not yet executed**: the actual CV run and the SHASH-based per-gene comprehensive report are deferred to a follow-up plan)
* `/_legacy/core_v1`, `/_legacy/validation_v1`: pre-2026-07-25 4-stage-cascade engine and its validation scripts, kept for reference/reproducibility only — do not import from or modify.
* **Root Directory (`.ipynb`):** All visualizations, tabular summaries, and analytical reviews must be conducted in Jupyter notebooks within the current working directory to allow immediate evaluation.
* *Strictly no modifications are permitted to external code outside these defined scopes.*


* **Coding Standards:**
* Prioritize concise, space-efficient, and highly optimized code implementations.
* Minimize comments; restrict them to core functionalities or essential comparisons with prior code, written exclusively in English.
* Notebook headers must remain strictly minimalist, utilizing standard numerical indexing and brief titles only.

