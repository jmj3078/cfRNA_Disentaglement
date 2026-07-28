## Phase Objective
Testing and validating Mixed Effect Modeling. Explicitly incorporates batch effects (omitted in prior models) via a random intercept, to evaluate structural validity and goodness-of-fit.

## Current Modeling Methodology (v3, 2026-07-28)

Design record (motivation, measurements): `docs/superpowers/specs/2026-07-27-eb-dispersion-cook-outlier-design.md`.
Full mathematical derivation: `docs/superpowers/specs/2026-07-28-v3-mathematical-reference.md`.

### 1. Standard Gene Modeling (Two-Stage Cascade)

Per-gene loop: fit stage -> PCIS outlier removal + 1 refit (inside the stage, before any squeeze) -> demote on failure. After ALL genes finish the cascade, one global pass squeezes every gene's dispersion intercept toward the trend -- squeeze depends on the full-cohort residual spread, so it cannot run per-gene mid-cascade.

**`nbi_full_eb`**: $\log\mu_i=\beta_0+\sum_k\beta_kX_{ik}+b_j$; $\log\theta_i=\gamma_0+\sum_k\gamma_kX_{ik}$, slopes under EB prior `normal(0,tau_k)`, intercept squeezed post-cascade.
**`nbi_intercept_eb`** (on stage-1 failure): same mean submodel; $\log\theta_i=\gamma_0$ only, squeezed post-cascade.
Failing both -> `route="excluded"`. Reject reasons kept per stage (`nbi_full_eb_reject_reason` / `nbi_intercept_eb_reject_reason`) for auditability.

v2's `nbi_disp_intercept` (3rd stage) and `nb_fixed` (hard trend-fixed stage) are gone: with a correctly-calibrated slope prior, `nbi_disp_intercept` was indistinguishable from `nbi_full_eb`, and `nb_fixed` is now just the `SE->inf` limit of the EB squeeze below, not a separate code path.

### 1a. EB Dispersion Shrinkage (`core/eb_shrinkage.py`)

limma/edgeR moment decomposition: `tau^2 = max(0, (1.4826*MAD(phi_hat))^2 - median(SE^2))` (MAD/median, not variance/mean, so a few near-divergent genes can't inflate tau and silently disable shrinkage).

* **Slopes**: `--mode calib` fits `nbi_full_eb` with no dispersion prior on a stratified gene subsample (`EB_PARAMS`, `calib_fits.csv`), `tau_k` read off the across-gene slope spread per covariate. Cached to `disp_prior.json`. Measured 0.10-0.36 (v2's blanket 0.05 was too tight).
* **Intercept**: not penalized during the fit. Squeezed in Python after the full cascade: `log_theta_post = (hat/SE_0^2 + trend/tau_d^2)/(1/SE_0^2 + 1/tau_d^2)`, pooled across both stages for a stable `tau_d`. `SE_0=NaN` -> `SE^2=inf` -> squeeze collapses to the trend exactly. Overwrites `disp_coef[0]`; `log_theta_raw` keeps the pre-squeeze value.

*Known approximation:* mean coefficients / `tau2` are not refit under the squeezed dispersion (not the joint posterior mode) -- same as limma/edgeR, and it's `alpha` entering the RQR that governs Z calibration.

### 1a-2. Dispersion Trend (`core/dispersion_trend.py:build_trend_from_fits`)

Prior mean for the intercept squeeze AND the variance PCIS is measured against -- both features are gated by this being correct. `build_trend_from_fits`: lowess of `log(alpha_fit)` on `log(mean)` over the calibration run's own **covariate-adjusted** dispersions (frac=0.3, it=3 bisquare robustifying iterations). `build_trend` (raw-count MoM, ignores covariates/batch) is kept only as a diagnostic reference -- it conflates true dispersion with covariate-driven mean variance and over-estimates badly at high expression (see math reference for the decomposition).

Batch enters the mean submodel only (`(1|batch__)`), not dispersion -- a fixed batch factor would give HC's 5 singleton batches a free parameter each, deflating exactly the dispersion the trend measures. `score()` never needs batch: `u~N(0,tau^2)` is marginalized by Gauss-Hermite, so unseen batches (CV, LOBO, disease) need no BLUP.

`core/trend_report.py` runs on every calibration fit, writes `Figures/dispersion_trend.png` + `trend_residuals.csv` -- no trend is ever deployed without its calibration record.

`training_summary.csv` has `tau2_collapsed` (`tau2 < 1e-4`, ~30% of genes) -- the R-side `singular` flag (`pdHess` FALSE) does not catch this.

### 1a-3. `tau2_max = 3.0` (2026-07-28)

Decoupled from `beta_explode_thr ** 2` (=9). That value was a numeric divergence guard; this one is a **power** criterion, and conflating them hid the distinction. Both stages gate on it (`is_converged`, `glmm_helpers.R:155,166`), and `fit_one_cascade` returns at `nbi_intercept_eb` regardless of `ok`, so failing both -> `route="excluded"`.

Motivation: a large `tau2` does **not** break calibration. Per-tau2-bin held-out-style checks (`0_insample_analysis.ipynb` sec. 3) give `rqr_msq` 0.98-1.03, `cov_95` 0.957-0.963, 95%-exceedance 0.046-0.051 at *every* bin including `tau2 > 5` -- the marginal `pred_var` blows up as `exp(2*tau2)` while the quantiles barely move, so the PPC variance panel is a moment artifact, not miscalibration. What a large `tau2` actually costs is detection power: the predictive distribution widens until the count needed for `|z|=3` leaves the observable range. Section 13 of that notebook computes `y_crit` (the z=3 count) per gene and compares it to the largest count ever observed for that gene; the undetectable rate is 0% below tau2~0.5 and climbs to 15-34% above 3.

Caveats kept deliberately: the cut removes 305 genes (1.6%) of which only ~21% are demonstrably undetectable -- no `tau2` threshold separates the two populations cleanly (precision never exceeds 30% at any cutoff). It is accepted because the collateral is low-expression (median mu ~0.05), not because the threshold is sharp. A direct `y_crit`-based filter was evaluated and **rejected**: it is cohort-bound (it cannot tell "model too wide" from "no such sample in this cohort") and overlaps a depth-based criterion on only 27 of 113 genes.

### 1b. PCIS Outlier Removal (`core/glmm_helpers.R:pcis_outliers`)

**PCIS = Prior-Conditioned Impact Score.** Cook-shaped, deliberately not Cook's distance:

```
w_i    = mu_i / (1 + alpha_trend * mu_i)               NB2 log-link IRLS weight
M      = [Xf Z],  P = blkdiag(0_p, I/tau^2)            fixed design + batch design
H      = W^1/2 M (M'WM + P)^-1 M' W^1/2,  p_eff = tr(H)
r_i    = (y_i - mu_i) / sqrt(mu_i + alpha_trend * mu_i^2)
PCIS_i = r_i^2 / p_eff * h_ii / (1 - h_ii)^2
```

Two departures from Cook's distance, both forced by measurement (full derivation + numbers in the math reference):
1. **Variance = trend alpha, not the gene's own fit.** A freely-estimated dispersion lets 20x-scale outliers inflate their own gene's alpha and mask themselves (self-masking, same mechanism externally studentized residuals fix in OLS). Trend alpha is "external" to the whole gene (drawn from ~19,000 other genes), which also survives multiple simultaneous outliers within one gene -- something per-observation deletion diagnostics (ESR, Cook's D) cannot.
2. **Leverage from a prior-penalized mixed design**, not fixed-effects-only -- `mu` already contains the BLUP, so using only `X` for the hat matrix ignores ~40% of effective model complexity. `tau2 -> 0` sends the penalty to infinity, so `p_eff -> p` automatically (the ~30% of genes with collapsed batch variance).

Because the variance isn't the fitted model's own, PCIS has no F reference distribution -- **the cut is a fixed constant, `config.FIT_PARAMS["pcis_cut"] = 2.28`**, read off an empirical null (`PCIS_Calibration/`, README + math reference sec. 5): each gene's own fitted params regenerate clean counts, refit under the same prior, recompute PCIS; the cut is the value where null-driven removals (noise) fall below the observed real removal rate, at a target per-observation population false-alarm rate of 1e-4. Not `qf(0.99, p_eff, n-p_eff)` (DESeq2 convention, numerically close by coincidence but has no theoretical basis for this statistic).

Observations exceeding the cut are **dropped** (largest first, at most `floor(0.05*n)`) and the stage refit once with `droplevels` (HC has 5 singleton batches). Not replaced by a trimmed mean -- fabricates counts, biases dispersion downward.

**Known blind spot**: contamination gets absorbed into dispersion *slopes*, not the intercept, so PCIS is weak for low-expression/high-alpha genes (a large count is plausible under high dispersion) and strongest for well-expressed, low-alpha genes. Conservative filter for gross contamination, not a general outlier detector.

### 1c. Per-Gene SHASH Calibration (`core/calibration.py`)
Unchanged from v2. Held-out HC Z-scores must be N(0,1) for downstream FDR calibration; genes with real skew/kurtosis in their RQR distribution break it. Fits a SHASH (sinh-arcsinh, Jones & Pewsey 2009) per gene to held-out Z-scores, reports naive-vs-corrected 95%-exceedance / skew-kurtosis / BH-FDR reject rate on held-out HC (a true null).

---

### 2. Pooled GLMM (For Low-Expression Genes)
Rare genes with a severe deficit of non-zero observations, stacked into a single tensor sharing fixed effects ($\beta$) and batch variance ($\sigma^2_{batch}$) at the group level:

$$\log(\mu_{i,g}) = \log(\bar{Y}_{g,HC} + \epsilon) + \beta_0 + \sum_{k=1}^{10}\beta_k X_{ik} + b_j$$

Poisson first; if Pearson-residual overdispersion ratio exceeds a threshold (e.g. 2.0), refit NB2 with $\log\theta=\gamma_0$.

**`nz_a_max = 25`** (2026-07-28, provisional -- pending CV, see below). Set empirically from the completed `nz_a_max=0` run (`engine_state_mixed_tau9_nopool_20260728/`, every gene forced through the individual cascade), which makes the pooling boundary measurable rather than assumed: the cutoff should sit where individual fitting actually stops working.

Convergence failure by nz: 92% (1-5) / 62% (6-10) / 42% (11-15) / 18% (16-20) / 10% (21-25) / 8% (26-30) / 4.5% (31-40) / 1.5% (41-50). The knee is at nz~20-25; above it individual fits essentially always converge.

Raising the cutoff past the knee is a losing trade, because pooling *downgrades* a gene that fits individually -- it loses its own $\beta$ to the shared one, which is the whole premise of per-gene normative modeling. Rescued (would fail individually) vs downgraded (fits fine) at each cutoff: 4.70 at nz 10, 2.41 at 15, 1.57 at 20, **1.21 at 25**, 0.97 at 30, 0.74 at 40. At 40, 1,423 of the 2,475 pooled genes (58%) were fitting fine. Going 25 -> 40 buys 90 more rescues for 626 more downgrades.

Failures that remain above nz~25 are mostly `tau2 > tau2_max`, not non-convergence, and **pooling does not fix those** -- the pooled model keeps the same $b_j$ random intercept. So there is nothing to gain there.

**Unverified assumption**: all of the above counts a low-nz gene as "rescued" on the premise that the pooled fit succeeds and is calibrated. `train_pool`/`glmm_fit_pool.R` have never been run on real HC data. Until CV confirms the pool route's held-out Z is calibrated, the threshold argument is conditional.

**Tension with the earlier estimate**: a 2026-07-24 v2-engine analysis (`individual_fit_success.csv`, see [[project_mixed_effects_pooling_threshold]]) put the boundary near nz~50 from *per-fold* convergence (all 5 CV folds converging, ~554 training samples each) -- a strictly harder bar than the single full-cohort fit measured here (676 samples), and on a different cascade. The upcoming CV run measures the fold-level rate on the current engine directly and is what should settle the gap; if fold-level convergence at nz 25-50 turns out poor, raise the cutoff.

---

## Development & Repository Constraints

* **Iterative Refinement:** methodology is experimental; expect structural revisions.
* **Directory Enforcement:**
  * `/core`: modeling engine logic only. `glmm_helpers.R`(per-gene fit + PCIS outlier removal + pooled-GLM primitives) · `glmm_fit.R`(CLI, `--mode cascade|calib|fixed_stage`, `--disp-prior`, `--fit-params`) · `eb_shrinkage.py`(EB prior sd + intercept squeeze) · `glmm_fit_pool.R`(pooled-GLM CLI, unused this round) · `dispersion_trend.py`(`build_trend_from_fits` canonical; `build_trend` diagnostic-only) · `trend_report.py`(always-on trend/prior figure) · `marginal_rqr.py`(tau2-marginalized RQR/log-likelihood, Gauss-Hermite) · `model_engine_mixed.py`(`NormativeModelEngineMixed`/`GeneRecordMixed`) · `calibration.py`(per-gene SHASH) · `pcis_null.R`(PCIS empirical-null simulator) · `pcis_calibration.py`(`run_all()` -> `PCIS_Calibration/`: null PCIS rate table + histogram) · `run_engine.py`(entry point: `python core/run_engine.py [--limit N] [--calib-genes N] [--resume]`, writes `engine_state_mixed/`)
  * `/validation`: large-scale validation/auditing. `cv_engine.py`(5-fold CV, per-`(gene,fold)` logging to `fold_stats.csv` -- written, **not yet executed**)
  * `/_legacy/core_v1`, `/_legacy/validation_v1`: pre-2026-07-25 4-stage engine, reference only -- do not import/modify.
  * **Root (`.ipynb`):** all visualizations/tabular summaries/analytical reviews go in notebooks in the cwd.
  * No modifications outside these scopes.
* **Coding Standards:** concise/space-efficient code; minimal English-only comments (core functionality or comparisons with prior code only); notebook headers stay minimal (numeric index + brief title).
