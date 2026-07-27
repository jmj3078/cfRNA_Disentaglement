# Gene-level Diagnostics + SHASH Calibration in the Mixed-Effects Engine

## Problem

`insample_analysis.ipynb` currently recomputes, in a notebook cell, a full
per-gene diagnostic suite (PPC posterior-predictive checks, marginal
log-likelihood, Pearson chi2, and SHASH calibration params) against
`engine_state_mixed/genes.pkl`, caching the result only as a standalone
notebook artifact (`engine_state_mixed/train_fitness_results.pkl`).

This is not part of the deployed engine (`GeneRecordMixed`), so:
- `NormativeModelEngineMixed.score()` cannot apply SHASH correction to new
  (disease) sample Z-scores — the params exist only in a notebook-side cache.
- Any other pipeline entry point wanting these diagnostics has to recompute
  them from scratch against the raw HC data.

The model engine (`core/model_engine_mixed.py`) is the pipeline we actually
deploy and reuse; these values must live there, computed once at training
time, not duplicated in downstream notebooks.

## Scope

1. Add in-sample gene diagnostics (PPC, loglik, SHASH) as `GeneRecordMixed`
   fields, computed once inside the training pipeline.
2. Add a separate, currently-empty CV(held-out) SHASH field group to
   `GeneRecordMixed`, and build (but do not run) the `cv_engine.py` logic
   that will populate it in a follow-up session.
3. Extend `score()` to return both the existing raw Z and a SHASH-corrected
   Z, using in-sample SHASH params by default.
4. Reduce `insample_analysis.ipynb` to a thin reader of the new engine
   fields (no recomputation).

Out of scope (explicitly deferred): actually running `cv_engine.py`'s
5-fold CV sweep; wiring CV-based SHASH into `score()`'s default path.

## `GeneRecordMixed` field additions

All new fields default to `None` (unset / not yet computed), matching the
existing style of the dataclass.

**In-sample diagnostics** (computed once, from the full HC training data —
the same data `train()` fits on):

```
obs_mean, obs_var, obs_zero_frac
pred_mean, pred_var, pred_zero_frac, zero_diff
pearson_chi2, avg_loglik
cov_25, cov_50, cov_90, cov_95, cov_99
ppc_p_mean, ppc_p_var, ppc_p_zero, ppc_p_chi2
```

**In-sample SHASH** (fit on in-sample marginal RQR Z, subsampled to
`SHASH_MAX_N` for genes with more HC samples than that):

```
shash_ok, shash_xi, shash_eta, shash_eps, shash_delta
shash_z_lo, shash_z_hi
raw_skew, raw_kurtosis, corrected_skew, corrected_kurtosis
naive_exceed, shash_exceed
naive_fdr_reject_rate, corr_fdr_reject_rate
```

**CV (held-out) SHASH** — a fully separate namespace, `cv_`-prefixed
mirror of the SHASH block only (PPC/loglik are not duplicated for CV):

```
cv_shash_ok, cv_shash_xi, cv_shash_eta, cv_shash_eps, cv_shash_delta
cv_shash_z_lo, cv_shash_z_hi
cv_raw_skew, cv_raw_kurtosis, cv_corrected_skew, cv_corrected_kurtosis
cv_naive_exceed, cv_shash_exceed
cv_naive_fdr_reject_rate, cv_corr_fdr_reject_rate
```

These stay `None` until a follow-up session actually runs `cv_engine.py`.

## New method: `compute_gene_diagnostics()`

Added to `NormativeModelEngineMixed`, called after `train()` + `train_pool()`
and before `save()`. For every gene with `rec.ok`:

1. Reconstruct `mu, alpha, tau2` over `self.X_hc_scaled` / `self.Y_hc` using
   the same per-route formula `score()` already uses (model-route: `mu_coef`/
   `disp_coef`/`fixed_alpha`/`alpha_fn`; pool-route: `rare_glm` mean/mult
   formula) — this is genuinely in-sample (full HC data, matches what's
   deployed).
2. Compute `pred_mean/var`, `pred_zero_frac` (Gauss-Hermite marginalization,
   reusing the closed-form / quadrature helpers currently inlined in the
   notebook — moved into `core/model_engine_mixed.py` or a small new
   `core/gene_diagnostics.py` module), `avg_loglik` (`marginal_nb_loglik`),
   `pearson_chi2`.
3. Run posterior-predictive simulation (`PPC_N_REPS=500`) to get
   `cov_*` and `ppc_p_*`.
4. Compute in-sample RQR (`marginal_nb_rqr`), subsample to `SHASH_MAX_N`,
   fit SHASH (`gene_shash_calibration`), and populate the in-sample SHASH
   fields.
5. Write all values onto `self.genes[g]` in place.

Because this is O(20k genes x 500 reps), it reuses the notebook's existing
checkpoint pattern: partial results checkpointed every `CKPT_EVERY` genes to
`engine_state_mixed/_diagnostics_partial.pkl`, removed on successful
completion, and skipped entirely (`if os.path.isfile(...)`) if
`engine_state_mixed/genes.pkl` already has diagnostics filled in for a gene —
this satisfies the project's cache-first-loading convention.

`run_engine.py` calls `engine.compute_gene_diagnostics()` right after
`engine.train()`, gated by a new `--skip-diagnostics` flag (for fast smoke
tests with `--limit`).

## `score()` changes

`score(..., as_dict=True)` gains a `"combined_shash"` key alongside the
existing `"combined"` (raw Z): for each scored gene, if `rec.shash_ok`,
`combined_shash[:, j] = shash_transform_to_z(combined[:, j], rec.shash_xi,
rec.shash_eta, rec.shash_eps, rec.shash_delta)`; otherwise it falls back to
the raw value. Raw `"combined"` is unchanged, so existing downstream
consumers (`Z_scores/`, GSEA, etc.) are unaffected unless they opt in to the
new key.

Default SHASH source is in-sample (`rec.shash_*`). A `shash_source="cv"`
parameter is accepted but only usable once the CV fields are populated —
this is a forward-compatible hook, not implemented behavior yet.

## CV-based SHASH: pipeline only, not executed

`validation/cv_engine.py` already builds `zdict[g]` (concatenated held-out
RQR Z across folds) — this is exactly the input `gene_shash_calibration`
expects. Add to `cv_engine.py`:

1. Load the trained engine via `NormativeModelEngineMixed.load(config.ENGINE_MIXED_DIR)`.
2. For each gene in `zdict`, subsample to `SHASH_MAX_N`, run
   `gene_shash_calibration`, and write results into `engine.genes[g].cv_*`
   fields.
3. Re-save via `engine.save(config.ENGINE_MIXED_DIR)` (this overwrites
   `genes.pkl`/`training_summary.csv` with the same in-sample values plus
   the newly-populated `cv_*` fields — safe since `save()` serializes the
   full `self.genes` dict either way).

This code path is added now but **not executed** — running the actual
5-fold CV sweep (`python validation/cv_engine.py`) is a separate follow-up
task.

## Notebook simplification

`insample_analysis.ipynb`'s cells that recompute PPC/SHASH (currently cells
6-7) are replaced with a direct read of `engine_state_mixed/training_summary.csv`
(extended with the new columns via `training_summary()`) or
`engine_state_mixed/genes.pkl`. No recomputation happens in the notebook —
this satisfies the "no duplicate computation" requirement.

## Non-goals

- No changes to the demotion cascade, R-side fitting, or `dispersion_trend.py`.
- No change to `pool`-route usage decision (`nz_a_max` stays deferred).
- No production wiring of CV-based SHASH into `score()`'s default path yet.
