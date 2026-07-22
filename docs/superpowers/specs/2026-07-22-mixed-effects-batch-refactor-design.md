# Mixed-Effects (Batch Random-Intercept) Normative Engine — Design

Branch: `mixed-effects-batch-refactor`
Scratch workspace: `MixedEffectsModeling/` (fully separate from `Modeling/`, own `config.py`)

## Motivation

`Modeling/model_engine.py`'s demotion chain (nbi → nb_fixed → intercept) and the
rare-pool GLM fit mu on 10 `BIAS_COLUMNS` covariates only. `Batch_ID` is not a
covariate. Prior EDA established meaningful `Batch_ID` unique-R² on top of the
existing covariates (see memory `project_hc_calibration_batch_confound.md`),
i.e. part of what looks like covariate-driven expression variance is actually
batch-driven and currently unmodeled, contaminating the HC reference curve.

A **fixed-effect** `Batch_ID` dummy was already rejected (memory, point 6):
most diseases live in exactly one batch, so a batch fixed effect and the
disease-relevant signal become unidentifiable, and a fixed-effect dummy has no
coefficient for a batch never seen during training — which breaks the
project's core scoring goal (score samples from unseen future batches).

This design instead adds a **batch random intercept** per gene: it purges
batch confound from the fixed-effect (covariate) coefficients during fitting
via partial pooling, but scoring always predicts marginally (batch integrated
out), so an unseen batch scores identically to a seen one and no batch-level
average is ever treated as a new "normal" baseline for that batch. This is the
same principle the project applied when rejecting the HC-PCA-latent-covariate
route (contamination via reuse of a fitted quantity at score time), moved to
the batch axis: fit-time confound removal, never score-time reuse.

## Scope

- Engine internals (fit + score) **and** downstream contracts
  (`GeneRecord`, `training_summary.csv`, `engine_state/` save format).
- Developed and tested entirely inside `MixedEffectsModeling/` with its own
  `config.py` (no changes to root `config.py` or `Modeling/` during this
  phase). Migration path (replace `Modeling/` vs. patch results back in) is
  an explicit open decision deferred until the design is validated —
  not part of this spec.
- CV (`cv_model_engine.py`-equivalent) and LOBO validation are out of scope
  for the initial implementation but must be rerunnable against the new
  engine once it exists (full LOBO re-run is ~6-8h per existing project
  experience and is expected, not a blocker).

## Per-stage design

| stage | mu | sigma / dispersion | on failure |
|---|---|---|---|
| nbi | fixed covariates + batch random intercept (glmmTMB `nbinom2`) | fixed covariates only, no batch (glmmTMB `dispformula`) | demote to nb_fixed |
| nb_fixed | fixed covariates + batch random intercept | fixed from Phase-0 trend (unchanged) | demote to intercept |
| intercept | population intercept + batch random intercept only (no covariates) | fixed from Phase-0 trend | **exclude** (no closed-form fallback below this — accepted tradeoff) |
| pool (rare, NZ<7) | shared beta (unchanged) + batch random intercept | existing Poisson/NB overdispersion-ratio choice (unchanged) | on GLMM non-convergence/singularity-collapse, **fall back to the current non-batch pooled GLM** — pool must always succeed |

sigma/dispersion never gets a batch random effect at any stage — batch-level
sample counts are too small (median 9 HC/batch, 25th pct = 1) for a second
variance component to be identifiable on top of the mean random effect.

### Failure vs. singular-fit distinction (demotion chain change)

The existing chain treats any R-side non-success as "demote." With random
intercepts this is too coarse: a gene with no real batch heterogeneity will
often converge to τ²≈0 with a boundary/singular-fit warning from glmmTMB —
that is a valid result, not a failure, and must not be a demotion trigger.

- **True failure (demote)**: optimizer non-convergence, exception, or
  fixed-effect coefficient explosion (`beta_explode_thr`, unchanged
  criterion, applied to fixed-effect coefficients only — τ² is exempt).
- **Singular but converged (accept + flag)**: record `batch_glmm_singular`
  on `GeneRecord`; does not demote.

### Conditional vs. marginal residuals (new distinction)

The in-sample outlier-removal loop and `w1_train` calibration diagnostic
should use **conditional** residuals (batch BLUP included) — they're
model-adequacy diagnostics for the fit that was actually estimated, run
during training, not the scoring rule.

The score-time RQR is always **marginal** (batch integrated out, see below).
These are two different residual computations sharing the same fitted
parameters; today's code does not have this distinction and must add it.

## Fitting architecture: one-pass, R-native, no rpy2

Per-gene rpy2 calls will not scale once random-intercept Laplace fits replace
today's IRLS/gamlss fits (19,538 model-candidate genes currently, 17,572 of
which reach stage nbi). Fitting moves to a **fully R-native, file-IO-based
subprocess**, invoked once per training run (not once per stage):

- `MixedEffectsModeling/glmm_fit.R`: takes HC `X` (covariates), `Batch_ID`,
  and `Y` (candidate genes × samples) via file input (`csv.gz` — data sizes
  here are small enough that no binary/arrow format is needed). Internally,
  a single `mclapply`-parallelized worker function **encapsulates the full
  nbi → nb_fixed → intercept cascade per gene** (tries nbi, falls through
  to nb_fixed on true failure, falls through to intercept on true failure)
  and returns one final row per gene: `stage`, fixed-effect coefficients,
  `tau2`, `batch_glmm_singular`, `fail_reason`. Python calls this script
  once for all model-candidates and reads back a single results table —
  no repeated cross-language round trips per stage.
- Pool route stays a separate call (shared-beta structure needs the full
  gene×sample block at once, not a per-gene worker).
- rpy2 is removed entirely; R is invoked only via `subprocess` at fit time.
  Scoring remains R-free, as today (now true for the whole engine, not just
  scoring — R involvement is limited to one `glmm_fit.R` subprocess call).

### Memory safety (mclapply + TMB)

TMB-based model objects (glmmTMB) hold external C++ pointers that don't
reliably release across `mclapply` fork boundaries; fitting ~17k genes
naively risks unbounded memory growth and OOM kill over a multi-hour run.
Mitigations, in `glmm_fit.R`:

- Explicit `rm(model); gc()` inside each per-gene worker call.
- Gene list processed in **chunks** (e.g. 500 genes); `gc()` called in the
  master process between chunks, not just inside workers.
- Chunk-size vs. `mc.preschedule` tradeoff (fewer forks vs. bounded memory
  per fork) determined empirically in the Step 0 spike, not assumed.
- Results written incrementally per chunk, not held in memory until the end.
  A completed chunk is skipped on restart (same resume-safety philosophy as
  `run_lobo_validation.py`'s meta.json-exists skip) — necessary given the
  expected multi-hour runtime.

## Marginal (out-of-batch) scoring

Random intercepts exist only to correct fixed-effect estimation during
fitting. **Batch-specific BLUPs are never used at score time** — every
sample (train-batch HC, train-batch disease, or a future unseen batch) is
scored against the same marginal (population-average) predictive
distribution. This is what preserves both the "score an unseen batch"
requirement and avoids batch-correlated disease signal being silently
absorbed into a batch-specific baseline.

- Marginal RQR computed via 1-D Gauss-Hermite quadrature over
  `b ~ N(0, τ²)`, in pure Python, using only the fixed-effect coefficients
  and `τ²` returned by the R fit (no BLUPs stored or used).
- **Fast path**: genes with `τ² ≈ 0` skip quadrature entirely and use the
  existing point-mass NB RQR (`_nb_rqr`) — expected to cover the majority
  of genes, since most genes are expected to show little batch
  heterogeneity once the fixed effects properly separate covariate signal.
- Implementation order: plain numpy first, benchmark at realistic scale
  (~3,000 samples × ~20,000 genes), and only add Numba (`@vectorize`) if the
  benchmark misses a wall-time budget (target: full scoring run <30 min).
  Not committing to JIT/GPU tooling before measuring actual throughput.

## Sigma/dispersion regularization (nbi stage) — requires empirical validation

gamlss's `ridgeVec` L2 penalty on sigma slope coefficients has no direct
glmmTMB equivalent. Planned approach, contingent on the Step 0 spike:

1. Try glmmTMB's `priors()` (available glmmTMB ≥ 1.1.8, experimental) to put
   independent N(0, λ) priors on dispformula slope coefficients (intercept
   unpenalized, mirroring the existing `ridgeVec` structure).
2. If unavailable or unstable in the installed R 4.3.1 environment, fall
   back to unpenalized dispformula regression relying solely on
   `beta_explode_thr` as the safety net, accepting a higher nbi→nb_fixed
   demotion rate (tracked via the existing `route_demotion_summary.png`).

### Parameterization equivalence — mandatory verification, not an assumption

glmmTMB's `nbinom2` models `Var(Y) = mu + mu²/theta`, i.e. `theta = 1/sigma`
in gamlss's own `Var(Y) = mu + sigma·mu²` parameterization, and glmmTMB's
`dispformula` predicts **`log(theta)`**, not `log(sigma)`. Both packages
happen to call their dispersion output "sigma" in places, which makes a
naive coefficient-name mapping silently invert the dispersion (this project
has one prior "critical statistical error" commit from exactly this class of
mistake — see git history). `β_disp(glmmTMB) = -β_σ(gamlss)`; the ridge
penalty magnitude is sign-invariant (penalizes β², not β) so `priors()` can
reuse the same `λ` scale for the shrinkage *strength*, but this must be
confirmed empirically, not assumed from the algebra alone, since the two
packages' penalty scale conventions (penalized deviance vs. prior variance)
are not guaranteed numerically identical.

Verification required before trusting any glmmTMB-derived sigma output:
1. Confirm `dispformula` parameterizes `log(theta)` from glmmTMB
   documentation/source directly (not from memory/assumption).
2. Fit gamlss NBI and glmmTMB `nbinom2` **with no random effect** (fixed
   effects only, directly comparable) on the same gene subset and covariate
   data. Compare **predicted `sigma(x) = 1/theta(x)` curves** across a grid
   of covariate values — not raw coefficients — since coefficient-level
   comparison is exactly the trap being guarded against.
3. Only extend to the random-intercept + `priors()` configuration once step
   2 passes.
4. Re-calibrate `lambda_sigma` empirically (e.g. against effective df or
   in-sample `w1` calibration) rather than reusing the existing numeric
   value unchanged.

## Pool route (rare pooling)

Extended to shared-beta (unchanged, gene-pooled GLM) **plus** a batch random
intercept in the same fit. This is the highest-risk route for identifiability
(NZ<7 genes already have very few nonzero observations per batch), so it
carries an explicit fallback: if the batch-augmented fit fails to converge or
collapses to a degenerate/singular fit, silently fall back to today's
non-batch pooled GLM for that call — pool is the one route that must always
succeed, by existing invariant.

## Small HC-batch handling (n<3) — deferred to Step 0 spike

Whether batches with very few HC samples need special handling (dedicated
random-effect level vs. pooling into an "other" catch-all level) is not
decided in this spec. The Step 0 spike will report observed τ² distribution,
singular-fit frequency, and convergence behavior on real data, and that
evidence — not a priori assumption — determines this choice.

## Step 0: feasibility spike (required before broader implementation)

A pilot run on ~30-50 genes (spanning the NZ range, weighted toward
low-expression genes where sigma-explosion risk is highest) must confirm,
before the full engine build:

1. glmmTMB is installed (`conda-forge r-glmmtmb`, not present in `scRNA` env
   as of this writing; `lme4`/`TMB` also absent) and its `priors()` behavior
   on this R 4.3.1 environment.
2. The sigma parameterization equivalence checks above.
3. Observed τ² distribution and `batch_glmm_singular` frequency across the
   pilot genes.
4. `mclapply` memory behavior (chunk size vs. `mc.preschedule`) under
   repeated glmmTMB fits, to fix the chunking/gc strategy before a
   multi-hour full run.
5. Rough per-gene fit wall-time, to estimate full-run duration and chunk
   checkpoint interval.

Findings from this spike determine: the small-batch handling policy, the
final sigma regularization approach, and the mclapply chunking parameters —
all currently open in this spec.

## Out of scope for this spec

- Final migration of validated results back into `Modeling/`/root `config.py`.
- CV and LOBO re-validation runs (expected next steps once the engine is
  built and passes its own diagnostics, not part of this design).
- Classifier-based utility evaluation (mentioned in memory as a possible
  future re-introduction of discrimination-control-style checks) — unrelated
  to this refactor.
