# Mixed-Effects Production Engine — Design

Branch: `mixed-effects-batch-refactor`. Builds on
`2026-07-22-mixed-effects-batch-refactor-design.md` (architecture decisions,
per-stage table, marginal-scoring rationale — not repeated here) and the now-closed
Step 0 spike (`MixedEffectsModeling/`, validated `is_converged()` logic, glmmTMB+
`priors()` confirmed working).

## Scope

Four new files in `Modeling/`, existing files untouched except where noted:

1. `Modeling/glmm_fit.R` — production cascade, generalizing the spike's
   `fit_random_intercept.R`/`fit_fixed_only.R` into one per-gene worker that tries
   nbi → nbi_disp_intercept → nb_fixed → intercept in sequence (spec's demotion
   table), using the spike's validated `is_converged()` (slopes-only explosion
   check, positional `VarCorr` access, tau2 upper bound) at every stage.
   `mclapply` + chunked, checkpointed I/O per the architecture spec.
2. `Modeling/model_engine_mixed.py` — orchestrates one call to `glmm_fit.R`,
   extends `GeneRecord` (`tau2`, `batch_glmm_singular`), implements marginal
   Gauss-Hermite scoring. `score()` keeps the existing `as_dict` contract.
3. `Modeling/cv_glmm_engine.py` — same shape as `cv_model_engine.py` (stratified
   5-fold, stage held fixed from full-data training, `_w1_normal` diagnostics),
   with per-stage CV functions for nbi/nbi_disp_intercept/nb_fixed/intercept/pool.
   Random intercept refit per fold (no leakage); scoring always marginal.
4. `Modeling/pool_threshold_sweep.py` — reproduces `EDA_Modeling/pooling_nz_sweep.py`
   (NZ-cutoff sweep, 5-fold held-out W1 calibration) but with the pooled-GLM
   extended to include a batch random intercept, run across ALL genes with no
   cutoff pre-applied, to pick the new `nz_a_max`.

## Ordering (this is load-bearing, not incidental)

`pool_threshold_sweep.py` runs **first**, unconstrained by any NZ cutoff — it
determines `nz_a_max` before that gate exists anywhere else. Only after
`nz_a_max` is fixed does `glmm_fit.R`/`model_engine_mixed.py` run a full
training pass (which needs the gate to route pool vs. model candidates), and
only after that does `cv_glmm_engine.py` evaluate the trained engine. Running
CV before the threshold is fixed would validate a gate that's still moving.

## Testing

The spike's 40-gene pilot (`MixedEffectsModeling/Spike_Results/`) is the
regression fixture: `glmm_fit.R`'s cascade output on those same genes must match
the spike's per-gene stage/tau2/singular outcomes (smoke check before any full
run). `cv_glmm_engine.py` supports `--limit N` per existing convention.

## Out of scope

Full LOBO re-validation, downstream `pipeline/` consumers (`scoring.py`,
`data_prep.py`), and migrating `Z_scores/`/`GSEA/` artifacts — all deferred
until this engine is trained and its own CV/threshold results are reviewed.
