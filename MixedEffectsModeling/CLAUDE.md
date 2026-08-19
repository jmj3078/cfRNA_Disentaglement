## Phase Objective
Mixed Effect Modeling: normative model with an explicit random-intercept batch term, validated via CV/LOBO/PPC before use for disease Z-scores.

## Current Methodology (v3)

Params live in `config.py`: `FIT_PARAMS` (`tau2_max=3.0`, `pcis_cut=2.25`, `disp_intercept_max=10.0`, `max_outlier_frac=0.05`), `NZ_A_MAX=31` (individual-vs-pooled routing cutoff), `EB_PARAMS`, `SPIKE_PARAMS`.

### 1. Standard Gene Modeling (Two-Stage Cascade, `core/glmm_fit.R` + `core/glmm_helpers.R`)
Per-gene: fit `nbi_full_eb` (mean submodel `log(mu)=Xb+b_j[batch]`, dispersion slopes under EB prior, intercept squeezed post-cascade) -> on failure fall back to `nbi_intercept_eb` (dispersion intercept-only) -> PCIS outlier removal + 1 refit -> failing both routes = `route="excluded"`.

### 1a. EB Dispersion Shrinkage (`core/eb_shrinkage.py`)
Slope priors (`tau_k`) fit via limma/edgeR-style moment decomposition on a calibration subsample, cached to `disp_prior.json`. Intercept squeezed toward `core/dispersion_trend.py`'s lowess trend (`build_trend_from_fits`, covariate-adjusted) after the full cascade — `build_trend` (raw-count, no covariates) is diagnostic-only, do not deploy it.

### 1b. PCIS Outlier Removal (`core/glmm_helpers.R:pcis_outliers`)
Cook-shaped statistic using trend alpha (not the gene's own fit, avoids self-masking) and mixed-model leverage. Cut is a fixed empirical constant (`pcis_cut`, calibrated in `PCIS_Calibration/`), not an F-distribution quantile. Drops flagged observations (largest first, capped at `max_outlier_frac`) and refits once. Known blind spot: weak for low-expression/high-alpha genes.

### 1c. Per-Gene SHASH Calibration (`core/calibration.py`)
Fits SHASH to held-out HC Z-scores per gene so downstream FDR calibration holds; reports naive-vs-corrected exceedance/skew-kurtosis/BH-FDR on held-out HC.

### 2. Pooled GLMM (`core/glmm_fit_pool.R`, low-`nz` genes only)
Genes below `NZ_A_MAX` share fixed effects + batch variance in one stacked tensor (Poisson, NB2 fallback on overdispersion). Route chosen because individual-fit convergence collapses below this `nz`; do not raise the cutoff without re-checking per-fold CV convergence.

---

## Development & Repository Constraints
* **Directory scopes** — stay within these, no modifications outside:
  * `/core`: modeling engine only (`glmm_helpers.R`, `glmm_fit.R`, `eb_shrinkage.py`, `glmm_fit_pool.R`, `dispersion_trend.py`, `trend_report.py`, `marginal_rqr.py`, `model_engine_mixed.py`, `calibration.py`, `shash.py`, `ood_filter.py`, `pcis_null.R`, `pcis_calibration.py`, `run_engine.py`). This is the set of code that actually drives the model — treat it as load-bearing, not scratch space: no edits without an explicit reason and no exploratory/analysis code mixed in. Write it as if it will be packaged and imported standalone later (no notebook-only globals, no hardcoded paths outside `config.py`, no cwd-relative assumptions) even though it isn't packaged yet.
  * `/validation`: CV/LOBO/PPC (`cv_engine.py`, `lobo_engine.py`, `lobo_mmd.py`, `ppc_simulate.py`, `pool_threshold_sweep.py`)
  * `/_legacy`: pre-2026-07-25 engine, reference only — do not import/modify.
  * Root `.ipynb`: numbered pipeline notebooks (`1_cv_analysis` ... `6_sankey_convergence`), thin runners only.
* **Coding standards:** concise, space-efficient; minimal English-only comments; notebook headers stay minimal (numeric index + brief title).

## Directory & Naming Convention

This directory is the reproducibility root: numbered root notebooks are the only entry points a reader replays; everything else is engine code, one-off scripts, or a section's output cache.

* **Root notebooks (`N_topic.ipynb`)**: one per pipeline stage, numbered in run order (currently `1_cv_analysis` ... `6_sankey_convergence`). Thin runners only — import from `core/`/`validation/`, no inline modeling logic. A new analysis stage = next integer + a new section directory (below), never inserted mid-sequence (renumber only if truly reordering the pipeline).
* **Section output directories (`PascalCase`, e.g. `PerSamplePathwayAnalysis`, `SignalTrendAnalysis`, `PCIS_Calibration`, `Benchmark`)**: one per analysis topic, holding that topic's figures (`Figures/` subdir), cached csv/pkl, and any one-off scripts specific to it (`run_*.py`, e.g. `PerSamplePathwayAnalysis/run_pathway_convergence_batch.py`). One-off/exploratory scripts live in the section dir they support, never in `core/` or `validation/`. Every such directory must be registered as a path constant in `config.py` (`_HERE / "DirName"`) — no hardcoded path strings in notebooks.
* **Engine raw-output directories (`<Thing>_mixed`, e.g. `CV_Results_mixed`, `LOBO_Results_mixed`, `Z_scores_mixed`, `engine_state_mixed`)**: fixed outputs of `core/run_engine.py` / `validation/cv_engine.py` / `validation/lobo_engine.py` — the `_mixed` suffix tags the mixed-effects engine version. This set is closed; do not add ad-hoc `_mixed` dirs for analysis — those go under the PascalCase convention above instead.
* Do not create new top-level directories outside these two families without updating this section.

## Visualization Workflow

* **Style:** every figure goes through `apply_style()` (see root `CLAUDE.md`). Never override fontsize, spines, or other theme parameters ad hoc in a plotting call — if the current theme is wrong for a figure, fix it in `viz_style.py` itself so the change applies everywhere, don't patch around it locally.
* **Draft-first, user-directed:** Claude cannot see the rendered plot, and figure design here is the user's call, not an autonomous one. Before writing any non-trivial visualization code: propose a concrete plan (chart type, axes/encoding, what's highlighted) and get explicit user sign-off — do not guess and ship a "final" version. Treat the first implementation as a draft; keep iterating on the user's concrete feedback until they confirm it's settled, rather than assuming agreement from silence or moving on after one round.
* **Ask, don't assume:** where the plan has a real design choice (chart type, what to emphasize, grouping/ordering, color/legend use, what to omit), ask the user directly instead of picking silently — these are subjective calls the user needs to make, not defaults to fill in. Keep questions concrete and answerable (offer options) rather than open-ended.
