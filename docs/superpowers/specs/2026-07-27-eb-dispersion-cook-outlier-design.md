# Mixed-Effects Engine v3: EB Dispersion Shrinkage + Cook's Distance Outlier Removal

Date: 2026-07-27
Scope: `MixedEffectsModeling/core/`, `MixedEffectsModeling/validation/`

## Motivation

v2 ran a 3-stage cascade (`nbi` -> `nbi_disp_intercept` -> `nb_fixed`) with a
`normal(0, 0.05)` prior on the dispersion submodel and a hard trend-fixed
dispersion as the last stage. Two problems were established empirically:

1. **The dispersion slope prior is far too tight.** glmmTMB 1.1.9 with
   `coef=""` already excludes the intercept from the prior
   (`proc_priors` sets `prior_elstart=1` when `"(Intercept)"` is present;
   verified `elstart=1, elend=2` for a 3-coefficient dispersion model). So the
   dispersion *intercept* was never shrunk -- the real fitted run confirms it
   (n=19,085, mean -0.358, sd 1.754, 1-99% range [-3.83, 2.86]). What *is*
   crushed are the slopes: on synthetic data a true slope of 0.40 collapses to
   0.04-0.12 across dispersion regimes, and in the real run only 27% of slopes
   exceed |0.1|. Stage `nbi` was therefore behaving almost identically to
   `nbi_disp_intercept` -- which explains why the latter only ever claimed 5
   genes out of 19,220.
2. **A hard-fixed dispersion is unjustifiably rigid.** `nb_fixed` injected
   `-log(alpha_trend)` as an offset with zero degrees of freedom, so a gene whose
   true dispersion genuinely departs from the trend had no recourse.

## Design

### 1. Cascade reduced to two stages

`nbi_full_eb` -> `nbi_intercept_eb`; a gene failing both is `route="excluded"` (no further
fallback, unchanged from v2's policy). `nbi_disp_intercept` is removed: with a
properly-scaled slope prior it is no longer a meaningfully distinct model, and
`nbi_intercept_eb` now occupies its structural position (`dispformula = ~1`) with the
addition of trend shrinkage.

| stage | mean submodel | dispersion submodel |
|---|---|---|
| `nbi_full_eb` | `log(mu) = b0 + sum_k b_k X_k + u_batch` | `log(theta) = g0 + sum_k g_k X_k`, slopes get an EB prior, `g0` squeezed toward trend |
| `nbi_intercept_eb` | same | `log(theta) = g0`, `g0` squeezed toward trend |

v2's `nbi` becomes `nbi_full_eb` and `nb_fixed` becomes `nbi_intercept_eb`: both
stages now carry the EB intercept squeeze, and nothing is fixed any more, so the
old names were actively misleading. `training_summary.csv` reject-reason columns become
`nbi_full_eb_reject_reason` / `nbi_intercept_eb_reject_reason`.

### 2. EB prior SD estimation (moment decomposition)

Both shrinkage targets use the same limma/edgeR-style moment decomposition. For
a per-gene MLE `phi_hat_g` with standard error `SE_g`,
`Var(phi_hat) = tau^2 + mean(SE^2)`, so

```
tau^2 = max(0, (1.4826 * MAD(phi_hat))^2 - median(SE^2))
```

MAD/median rather than sample variance/mean, because a handful of
near-divergent genes would otherwise inflate `tau` and silently disable the
shrinkage.

**Dispersion slopes.** A `--mode calib` run fits stage `nbi_full_eb` with *no*
dispersion prior on a subsample of genes (default 2,000, stratified into 10 HC
mean-expression deciles, seed 42) and reports coefficients and their SEs. Per
covariate k:

```
tau_k = sqrt(max(0, (1.4826 * MAD_g(gamma_hat_gk))^2 - median_g(SE_gk^2)))
```

The full run then applies `normal(0, tau_k)` per dispersion slope
(`class="betad"`, one row per covariate name). Cached to
`<engine_dir>/disp_prior.json`; the calibration run is skipped when that file exists.

**Dispersion intercept.** Not penalized during the fit. Instead a one-pass
analytic squeeze toward the Phase-0 lowess trend is applied in Python after the
R run, over all `ok` genes from both stages pooled (nbi_intercept_eb alone has too few
genes for a stable `tau_d`):

```
resid_g  = log_theta_hat_g - log_theta_trend_g,  log_theta_trend_g = -log(alpha_of(mean_g))
tau_d^2  = max(0, (1.4826 * MAD(resid))^2 - median(SE_0^2))
log_theta_post_g = (log_theta_hat_g/SE_0g^2 + log_theta_trend_g/tau_d^2) / (1/SE_0g^2 + 1/tau_d^2)
```

`SE_0g = NaN` (unusable `sdreport`) is treated as `SE^2 = inf`, giving exactly
`log_theta_post = log_theta_trend` -- v2's hard-fixed `nb_fixed` is recovered as
the limiting special case of the EB rule rather than as a separate stage.

The squeezed value is written back into `disp_coef[0]`, so `score()` needs no
new branch: `alpha = exp(-Xa @ disp_coef)` already yields the squeezed constant
for `nbi_intercept_eb` (NaN slopes -> 0) and the squeezed intercept plus real slopes for
`nbi_full_eb`. The pre-squeeze value is retained as `log_theta_raw` for auditing.

*Known approximation:* the mean coefficients and `tau2` are not refit under the
squeezed dispersion, so `(mu, alpha)` is not the joint posterior mode. This is
exactly what limma/edgeR do, and it is the `alpha` entering the RQR that governs
Z calibration.

### 2b. The dispersion trend itself was biased, and had to be rebuilt

The trend is the prior mean of the intercept squeeze AND the variance used by
Cook's distance below, so its correctness gates both features. `build_trend`
computed raw-count MoM `(var-mean)/mean^2` ignoring covariates and batch. That
quantity is not the conditional dispersion; the variance decomposition is
`sigma_MoM ~= alpha + tau^2 + CV^2_X(exp(beta'X))`, verified on real fits (at
mu~819 the three terms sum to 1.541 against an observed 1.495, split
alpha 7.1% / tau^2 5.7% / covariates 82.3%; at mu~0.19 it is alpha 45.5% /
covariates 37.0%). The covariate term is exactly what the mean submodel already
explains, so the trend was double-counting it.

Measured against 19,085 fitted dispersions the MoM trend over-estimated by 2.30x
at mu~0.12 rising monotonically to 16.71x at mu~1193 (median log residual -0.834
-> -2.816, `frac_pos` <= 12.4% in every one of 12 bins).

Fix: `build_trend_from_fits` -- lowess of `log(alpha_fit)` on `log(mean)` over the
calibration run's own covariate-adjusted dispersions, i.e. edgeR/DESeq2's order of
operations. Bias falls to |median residual| <= 0.067 per bin, overall 1.520 ->
0.008. `build_trend` is retained as an unconditional diagnostic reference only.

Two consequences beyond the trend itself:
* `tau_d^2` drops 0.460 -> 0.115, so the intercept squeeze gets 3-5x stronger
  (shrink weight at SE=0.29 goes 0.135 -> 0.423; at SE=0.7, 0.477 -> 0.810). A
  biased trend was inflating the residual spread and thereby disabling its own
  shrinkage.
* Cook's distance below is restored at high expression, where a 16x-too-large
  alpha had made it structurally unable to fire.

Ordering: the calibration run has no trend yet, so it runs with no dispersion prior and no
outlier removal (`alpha_of` returns NA in `--mode calib`, and `cook_outliers`
no-ops on a non-finite alpha). Its fits are hyperparameters only, never deployed.

`core/trend_report.py` runs on every calibration fit and writes
`Figures/dispersion_trend.png` + `trend_residuals.csv`, so a trend is never
deployed without its calibration record.

### 3. PCIS outlier removal (was: Cook's distance)

One-step Pregibon (1981) Cook's distance on the fitted NB2 log-link model, with
the estimated random intercept absorbed into `mu` (standard approximation --
the hat matrix uses the fixed-effect design only):

```
mu_i    = predict(fit, "response")            # includes u_batch
alpha_i = 1 / predict(fit, "disp")
V_i     = mu_i + alpha_i * mu_i^2
r_i     = (y_i - mu_i) / sqrt(V_i)
w_i     = mu_i / (1 + alpha_i * mu_i)         # NB2 log-link IRLS weight
h_i     = diag( W^1/2 Xa (Xa' W Xa)^-1 Xa' W^1/2 )
D_i     = r_i^2 * h_i / (p * (1 - h_i)^2),   p = 11
```

Samples with `D_i > qf(0.99, p, n - p)` are dropped and the stage is refit once
(DESeq2's cutoff rule). At most `floor(0.05 * n)` samples are dropped, largest
`D_i` first. Observations are **removed**, not replaced by a trimmed mean:
replacement fabricates counts and biases the dispersion downward, and with
n(HC) in the hundreds the lost information is negligible.

If the refit fails to converge the pre-removal fit is kept and
`outlier_refit_failed=True` is logged; if the *first* fit did not converge,
Cook's distance is still computed from it and the refit is accepted when it
converges (outliers can be the cause of non-convergence). Applied at both
stages, so the worst case is 2 fits per gene per stage. `n_outliers` is logged
per gene.

*Known limitation:* an outlier can partially mask itself by inflating the
gene's own fitted `alpha` (a heavy-tailed gene makes a large count genuinely
plausible). Cook's distance uses the gene's own fitted dispersion, matching
DESeq2, so the empirical flag rate must be read off `n_outliers` rather than
assumed.

### 4. SHASH recalibration unchanged

`core/calibration.py` and its CV wiring are untouched. `validation/cv_engine.py`
is updated only for the new stage names, the new output columns, passing
`--disp-prior` so folds use the same slope prior, and applying the intercept
squeeze per fold (with that fold's own `tau_d`) so CV Z-scores reflect the
deployed model.

*Documented leakage:* `tau_k` comes from a calibration run over all HC samples, so it is
shared across CV folds. It is a single hyperparameter pooled over ~20k genes;
per-fold re-estimation would double the calibration cost for no measurable change.

## Files

| file | change |
|---|---|
| `config.py` | `EB_PARAMS`, `DISP_PRIOR_PATH` |
| `core/eb_shrinkage.py` | new: `estimate_slope_prior`, `squeeze_log_theta` |
| `core/dispersion_trend.py` | new `build_trend_from_fits`; `build_trend` demoted to diagnostic |
| `core/trend_report.py` | new: always-on trend/prior calibration figure + residual CSV |
| `core/glmm_helpers.R` | 2 stages, `disp_se`, `pcis_outliers` (mixed-model leverage), outlier refit in `fit_stage_gene` |
| `core/glmm_fit.R` | `--mode calib`, `--disp-prior`, 2-stage cascade, `disp_se_*`/`n_outliers` columns |
| `core/model_engine_mixed.py` | `calib_genes`/`prepare_hyperparams`, squeeze in `train`, record fields, summary columns |
| `core/run_engine.py` | calibration cache step |
| `validation/cv_engine.py` | stage names, per-fold squeeze, `--disp-prior` |
| `MixedEffectsModeling/CLAUDE.md` | methodology section rewrite |

## Verification

1. `--limit` smoke run end-to-end (calibration + cascade + squeeze + save).
2. Confirm `disp_prior.json` `tau_k` are materially larger than 0.05 and that
   fitted slope spread widens versus the v2 run.
3. Confirm the squeeze is a genuine partial shrink (`log_theta_eb` strictly
   between `log_theta_raw` and `log_theta_trend`, not pinned to either).
4. Report the empirical `n_outliers` distribution.
5. Full refit with `nz_a_max=0` (no pooling; every gene routes to the cascade).
