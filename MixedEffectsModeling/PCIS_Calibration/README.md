# PCIS Threshold Calibration (2026-07-28)

Empirical null distribution of PCIS and the two candidate replacement thresholds
for the inherited `qf(0.99, p_eff, n-p_eff)` convention (which has no
distributional justification -- see `MixedEffectsModeling/CLAUDE.md` section 1b).

Reproduce with:

```
python MixedEffectsModeling/core/run_engine.py                  # -> cascade fits
Rscript MixedEffectsModeling/core/pcis_null.R <results.csv> <workdir> <out.csv> 1 12 <disp_prior.json>
python MixedEffectsModeling/core/pcis_calibration.py <out.csv>  # -> everything here
```

## How the null was generated (`core/pcis_null.R`)

For every converged gene, its own fitted `(beta, gamma, tau^2)` regenerate clean
counts on the real design matrix: `u_j ~ N(0, tau^2)`, `y* ~ NB2(mu_0 e^u, 1/alpha_i)`.
The same stage is then refit with the same EB slope prior, and PCIS is recomputed
by the exact `pcis_outliers` formula (`pcis_vec` mirrors it). Top 50 PCIS values
per gene are kept -- the top 7.2% of 693 observations, which resolves the upper
tail down to a per-observation rate of 1e-5. 1 replicate per gene suffices because
one replicate already yields 693 null PCIS draws for that gene.

19,158 genes x 693 observations = 13,276,494 null observations. No gene saturated
the top-50 cap at any threshold considered.

## Headline result

The current threshold sits where it cannot distinguish signal from noise:

| | genes with >=1 removal | mean removals/gene |
|---|---|---|
| real data | 7.78% | 0.0960 |
| **null simulation** | **6.94%** | **0.0806** |

84% of current removals are attributable to the null. Its realized rates are a
per-observation false-alarm rate of 1.16e-4 (1/86 of the nominal 0.01) and a
per-gene FWER of 6.94%.

The threshold is also mis-shaped in the FWER view: null max PCIS peaks at
`log_mu ~ 2-3` (q95 = 4.74) and is small at both ends (0.65 at the lowest decile),
while `qf` decreases monotonically in `log_mu`. Realized FWER therefore spans
0.7% to 11.4% (17x). In the per-observation view the current threshold is already
near-uniform (1.0e-4 to 2.2e-4 over 20 bins), so the two target forms disagree
about whether the current shape is broken.

## Candidate thresholds

Target per-gene FWER 0.05:
* **A (pooled constant)** = 2.7054. Overall exactly 0.05 but retains the shape
  problem (0.6% to 9.3% across deciles).
* **B (smooth)** = `exp(QuantReg(log max PCIS ~ bs(log_mu, df=6), q=0.95))`.
  Overall 0.0501, per-decile 4.0-6.6%. Adding `tau2` and `bs(log p_eff)` changes
  realized FWER by <0.0002 and pinball loss by 2.6%, so `log_mu` alone is used.

Target per-observation rate: pooled empirical quantile within 20 `log_mu` bins,
lowess-smoothed on the log scale (`frac=0.5, it=3`).

**Deployment caveat:** the `bs` basis extrapolates catastrophically below the
observed `log_mu` support -- the fitted cut reaches 1.5e-7 for the 20 genes below
the 0.1th percentile. Any deployed threshold function needs a floor / clamp to the
`grid_log_mu_range` recorded in `summary.json`.

## Files

| file | contents |
|---|---|
| `summary.json` | all scalar results: current-threshold null rates, real-vs-null comparison, A cuts (both target forms), B fit params + realized rates + support range |
| `null_pcis_top50_raw.csv.gz` | raw simulation output, 957,900 rows (gene x rank 1-50) with `log_mu`, `trend_alpha`, `tau2`, `p_eff`, `cut_current`, `pcis` |
| `null_max_pcis_per_gene.csv.gz` | one row per gene: null max PCIS + fitted B cuts at each FWER target |
| `fwer_by_expression_decile.csv` | per-decile null q95/q99 and realized FWER for current / A / B |
| `covariate_dependence.csv` | Spearman of log max PCIS vs each candidate covariate (all \|r\| <= 0.13; `log_mu` is 0.016 because the dependence is non-monotone, peaked at mid-expression) |
| `threshold_B_model_comparison.csv` | 4 candidate B formulas: params, realized FWER, pinball loss, cut range |
| `threshold_B_curve_fwer{0.05,0.01}.csv` | 200-point `log_mu` -> cut lookup for B, FWER targets |
| `threshold_B_curve_per_obs_rate{1e-3,1e-4}.csv` | same, per-observation-rate targets |
| `per_obs_bins_rate{1e-3,1e-4}.csv` | the 20 `log_mu` bins behind those curves: empirical cut, smoothed cut, current cut, current realized rate |
| `cascade_fit_results.csv` | the R-side cascade output the null was generated from (20,097 genes, all coefficients) |
| `fit_params.json` | `config.FIT_PARAMS` as actually used by the run |
| `Figures/null_max_pcis.png` | null max PCIS distribution; PCIS vs `log_mu` with null q95, current cut, B curve |
| `Figures/fwer_by_expression.png` | realized null FWER by expression decile, current / A / B vs target |
| `Figures/per_obs_thresholds.png` | per-observation-rate cuts: empirical bins, lowess B, current cut |

## Still open

The empirical FDR of an outlier call cannot be computed yet: the cascade saves
only `n_outliers`, not the real per-observation PCIS values, so "real excess /
total exceedances" as a function of the threshold is unavailable. Extracting the
real PCIS distribution requires the same harness with the `y*` regeneration
removed (~4 min), and would let the target rate be read off an FDR curve instead
of chosen by convention.
