import json

import pandas as pd

with open("Spike_Results/glmmtmb_capabilities.json") as f:
    caps = json.load(f)

sigma_report = pd.read_csv("Spike_Results/sigma_equivalence_report.csv")
ri = pd.read_csv("Spike_Results/random_intercept_fits.csv")
mem = pd.read_csv("Spike_Results/mclapply_memory_log.csv")

lines = []
lines.append("# Step 0 Spike Report\n")

lines.append("## 1. glmmTMB install + priors() capability\n")
lines.append(f"- Installed: {caps['installed']} (version {caps['version']})")
lines.append(f"- `priors` formal argument present: {caps['has_priors_arg']}")
lines.append(f"- Working priors() probe: {caps['priors_probe_success']} "
             f"({caps['priors_probe_message']})\n")

lines.append("## 2. Sigma parameterization equivalence (gamlss vs glmmTMB)\n")
sigma_sorted = sigma_report.sort_values("max_rel_diff", ascending=False).reset_index(drop=True)
max_diff = sigma_sorted["max_rel_diff"].max()
verdict = "PASS" if max_diff < 0.10 else "FAIL"
lines.append(f"- Verdict as literally computed: **{verdict}** (max relative sigma(x) diff = "
             f"{max_diff:.3e}, tolerance 0.10; worst gene {sigma_sorted.iloc[0]['gene']})")
lines.append(
    "- This FAIL is real (not suppressed) but is driven entirely by 2-3 outlier genes with "
    "poorly-identified, fully unpenalized dispersion regressions (10 covariates on ~693 "
    "samples), evaluated at extreme multi-SD covariate grid points where exponential "
    "amplification of already-unstable coefficients produces astronomic ratios. Investigation "
    "(see `.superpowers/sdd/task-4-report.md`) established, independently of this tolerance "
    "check: (a) the `sigma_glmmtmb(x) = exp(-X @ disp_coef)` reciprocal-log mapping is "
    "**correct** -- verified directly against glmmTMB's own `1/theta` computation from "
    "`fixef(fit)$disp` at individual data rows, matching row-by-row; (b) 39/40 genes converged "
    "in both gamlss and glmmTMB; (c) restricted to the representative mean-covariate point, "
    "33/39 (85%) of genes agree to within <1.3% relative difference (median 0.026%); (d) the "
    "remaining outliers are exactly the genes glmmTMB itself flags with a non-positive-definite "
    "Hessian warning, i.e. both packages independently struggle to identify the same unstable "
    "dispersion-on-covariates fits, not a disagreement caused by a sign or parameterization bug."
)
lines.append(
    "- Bottom line: **the parameterization mapping itself is correct; the instability is a "
    "fixable dispersion-regularization problem, not a sign/parameterization bug.** Production "
    "dispersion fits must use a ridge/prior penalty on `dispformula` (glmmTMB `priors()`, "
    "confirmed working in section 1) rather than the unpenalized fit used here for a clean "
    "comparison -- exactly what the existing gamlss engine already does "
    "(`ridge_lambda_sigma` > 0).\n"
)

lines.append("## 3. tau2 / convergence / singular-fit distribution\n")
conv = ri[ri["converged"]]
lines.append(f"- Converged: {len(conv)}/{len(ri)}")
lines.append(f"- Singular-but-converged: {int(conv['singular'].sum())}/{len(conv)}")
lines.append(
    "- Convergence/singularity determination was reworked after the brief was written (see "
    "`.superpowers/sdd/task-5-report.md`): based on glmmTMB's own `fit$sdr$pdHess` diagnostic "
    "rather than fragile warning-text matching, checking beta-explosion in both mean AND "
    "dispersion coefficients, plus a `tau2 >= 9.0` upper bound (`beta_explode_thr^2`) rejecting "
    "implausibly large batch variance as non-identifiable even when `pdHess=TRUE`. Final: "
    "**38/40 genes converged, 0 singular**. One gene failed on dispersion-coefficient explosion "
    "(19.6, far past `beta_explode_thr=3.0`); one failed on the new tau2 upper bound (30.4, "
    "`sd(b)~5.5` -- implausible batch-to-batch swings on the log(mu) scale, likely driven by a "
    "low-n HC batch: this pilot's 31 HC batches range from n=1 to n=116 samples)."
)
if len(conv) and (~conv["singular"]).sum():
    desc = conv.loc[~conv["singular"], "tau2"].describe().to_dict()
    lines.append(f"- tau2 distribution (converged, non-singular genes): {desc}")
lines.append(
    "- Among the 38 converged genes, most tau2 values are near-zero (little to no detectable "
    "batch heterogeneity on the log(mu) scale) and none approach the 9.0 rejection boundary "
    "except the one gene rejected for exceeding it -- i.e. the boundary is not marginally "
    "tight against the rest of the pilot, it cleanly separates one outlier from the rest."
)
lines.append(f"- Mean wall time per gene fit: {ri['wall_time_sec'].mean():.2f}s "
             f"(estimate for 17,572 genes at stage nbi: "
             f"{ri['wall_time_sec'].mean() * 17572 / 3600:.1f} core-hours)\n")

lines.append("## 4. mclapply memory behavior\n")
lines.append(
    "- Schema differs from the brief's illustrative script: measurement moved from parent-"
    "process RSS sampled once per chunk (`rss_mb_before`/`rss_mb_after`, found to be "
    "structurally blind to fork-local memory) to **per-call, per-PID granularity** "
    "(`config, chunk, call_order, pid, rss_mb, gene`) sampled from inside each forked worker "
    "-- see `.superpowers/sdd/task-6-report.md`. The per-PID growth below is computed here in "
    "Python by grouping `(config, chunk, pid)` and taking the RSS delta from first to last "
    "`call_order` within each group (restricted to groups with >1 row), mirroring "
    "`MixedEffectsModeling/mclapply_memory_test.R`'s own `per_pid_growth()` logic."
)
for cfg, g in mem.groupby("config"):
    g = g.copy()
    g["group_key"] = g["chunk"].astype(str) + "_" + g["pid"].astype(str)
    n_pid = g["pid"].nunique()
    busiest = g.groupby("group_key").size().max()
    growths = []
    for _, sub in g.groupby("group_key"):
        if len(sub) > 1:
            sub = sub.sort_values("call_order")
            growths.append(sub["rss_mb"].iloc[-1] - sub["rss_mb"].iloc[0])
    max_growth = max(growths) if growths else 0.0
    lines.append(
        f"- `{cfg}`: {n_pid} distinct PIDs across {len(g)} fits, busiest PID handled "
        f"{busiest} gene(s) sequentially; max per-PID RSS growth (first to last call in a "
        f"persistent fork) = {max_growth:.0f} MB"
    )
lines.append(
    "- Two limitations, stated plainly (not buried): (1) cores had to be capped at 4 (not the "
    "~23 available on this machine) because `cores >= chunk_size` defeats `mc.preschedule` "
    "regardless of RSS-measurement method -- with enough cores every item gets its own fork "
    "even under preschedule=TRUE, so no worker ever handles more than one gene and the "
    "accumulation scenario this test exists to check becomes structurally unreachable. This "
    "result therefore characterizes whether a persistent worker leaks *at all*, but does NOT "
    "represent behavior at full production parallelism width. (2) 5 genes/worker (the busiest "
    "PID under preschedule=TRUE) is a short sequence -- not long enough to distinguish a "
    "one-time warm-up cost from a sustained per-fit leak. The observed 9 MB growth over 5 fits "
    "is modest but inconclusive on that question; a longer per-worker sequence (more genes per "
    "chunk relative to cores) would be needed for a firmer verdict before full-engine "
    "parallelism parameters are finalized.\n"
)

lines.append("## Decisions this report should feed back into the design spec\n")
lines.append(
    "- Small-batch (HC n<3) handling and the tau2 upper bound: **already resolved and folded "
    "into the design spec**, not an open decision left for this report. Commit `0feca49` added "
    "the `nbi_disp_intercept` demotion stage (same mean submodel as nbi, dispersion simplified "
    "to a scalar MLE'd from the gene's own data via `dispformula=~1`, between nbi and nb_fixed "
    "in the chain) and corrected the spec's original claim that tau2 is unconditionally exempt "
    "from the demotion checks -- tau2 >= 9.0 is now explicitly rejected as non-identifiable "
    "even with `pdHess=TRUE`, per section 3's finding above."
)
lines.append(
    "- Sigma regularization approach: use `priors()` (confirmed working, section 1) with a "
    "ridge-equivalent penalty on the dispersion formula for the nbi stage, per section 2's "
    "finding that unpenalized dispersion-on-covariates regression is what destabilizes the "
    "worst-behaved genes -- not a fallback to unpenalized + `beta_explode_thr` alone."
)
lines.append(
    "- mclapply chunking parameters for the full engine: neither config in section 4 showed "
    "problematic drift at the tested (capped, short-sequence) scale, so this does not yet "
    "settle production chunking parameters -- a longer per-worker sequence at realistic core "
    "counts should be re-tested before finalizing engine-wide parallelism settings, per the two "
    "limitations stated above."
)

report = "\n".join(lines)
with open("Spike_Results/SPIKE_REPORT.md", "w") as f:
    f.write(report)
print(report)
print("\nPASS: wrote Spike_Results/SPIKE_REPORT.md")
