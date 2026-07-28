# PCIS Threshold Calibration

Measures how much of PCIS outlier removal is just noise, by building an
empirical null: does the same statistic still fire this often when there is
nothing to detect. Used to fix `config.FIT_PARAMS["pcis_cut"]`, replacing the
inherited-but-unjustified `qf(0.99, p_eff, n-p_eff)` convention (PCIS has no F
reference distribution -- see `MixedEffectsModeling/CLAUDE.md` section 1b).

Reproduce:

```
python MixedEffectsModeling/core/run_engine.py                  # -> cascade fits
Rscript MixedEffectsModeling/core/pcis_null.R <results.csv> <workdir> <out.csv> 1 12 <disp_prior.json>
python MixedEffectsModeling/core/pcis_calibration.py <out.csv>  # -> pcis_rate_table.csv + figure
```

## Method (`core/pcis_null.R`)

For every converged gene, its own fitted `(beta, gamma, tau^2)` regenerate
clean counts on the real design matrix: `u_j ~ N(0, tau^2)`, `y* ~ NB2(mu_0
e^u, 1/alpha_i)`. The same stage is refit under the same EB slope prior, and
PCIS is recomputed by the exact `pcis_outliers` formula. Since these y* have
no injected contamination, any PCIS value that clears a threshold there is by
construction a false positive. Top 50 PCIS values per gene are kept (top 7.2%
of 693 observations per gene).

19,158 genes x 693 observations = 13,276,494 null draws (`null_pcis_top50_raw.csv.gz`).

## Result (`pcis_rate_table.csv`)

Target rates are **population-level per-observation false-alarm rates**
(`n_genes * n_obs` = 13,276,494 as the denominator), not a quantile of the
retained top-50-per-gene pool -- an earlier version of this table picked cuts
via `np.quantile` on that pool directly, which mislabels percentiles badly
(its "0.90" corresponded to roughly the population's 99.3rd percentile,
because the pool is already restricted to each gene's most extreme 7.2%).
`max_removed_per_gene` confirms no row here hits the 50-per-gene retention cap
(no truncation bias):

| target rate | population %ile | cut | null removed/gene | real removed/gene | null share |
|---|---|---|---|---|---|
| 1e-3 | 99.90% | 0.305 | 0.693 | 0.096 | 7.2x |
| 5e-4 | 99.95% | 0.574 | 0.346 | 0.096 | 3.6x |
| 3e-4 | 99.97% | 0.899 | 0.208 | 0.096 | 2.2x |
| 2e-4 | 99.98% | 1.293 | 0.139 | 0.096 | 1.4x |
| 1.5e-4 | 99.985% | 1.628 | 0.104 | 0.096 | 1.08 (breakeven) |
| **1e-4** | **99.99%** | **2.282** | **0.069** | 0.096 | **0.72** |
| 5e-5 | 99.995% | 4.239 | 0.035 | 0.096 | 0.36 |
| 1e-5 | 99.999% | 18.61 | 0.007 | 0.096 | 0.07 |

Null-driven removals cross below the observed real removal rate between rate
1.5e-4 and 1e-4. **`pcis_cut = 2.28`** (rate 1e-4) is deployed, giving headroom
above that breakeven. This range brackets the old `qf(0.99,...)` cut (~1.98) --
that convention turned out to be numerically close to the calibrated value
despite having no theoretical basis for this statistic.

## Files

| file | contents |
|---|---|
| `pcis_rate_table.csv` | the table above |
| `null_pcis_top50_raw.csv.gz` | raw null draws, 957,900 rows (gene x rank 1-50) |
| `cascade_fit_results.csv` | cascade fit this null was generated from (20,097 genes) |
| `fit_params.json` | `config.FIT_PARAMS` as used for this run (predates the `pcis_cut` rename; its `pcis_f_q` here is the old key name) |
| `Figures/null_pcis_distribution.png` | histogram of null PCIS with cut=0.5/1.0/2.0 marked |

## Still open

`real removed/gene` above is a single count (`training_summary.csv`'s
`n_outliers`, averaged) anchored to the old `qf(0.99,...)` cut (~1.98) -- the
real per-observation PCIS values were never saved during the cascade run, so
the true real-vs-null crossover as a function of cut is inferred from one
reference point, not measured directly. Deliberately left unresolved: the
constant threshold is the simpler, sufficient answer for now.
