import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

SPIKE_DIR = Path(__file__).resolve().parent.parent / "MixedEffectsModeling" / "Spike_Results"
OUT = Path("/tmp/glmm_fit_regression_out.csv")
OUT.unlink(missing_ok=True)

genes = pd.read_csv(SPIKE_DIR / "pilot_genes.csv")[["gene"]]
genes["stage"] = "nbi"
genes_path = Path("/tmp/glmm_fit_regression_genes.csv")
genes.to_csv(genes_path, index=False)

# config.DISPERSION_TREND_PATH (production Phase-0 trend) uses the lowess
# a0/a1/lowess_logmu/lowess_logsigma schema from dispersion_trend.py, not the
# mean_grid/alpha_grid schema glmm_fit.R's alpha_of() reads. All pilot genes
# land on stage "nbi", which regresses dispersion on covariates directly and
# never touches fixed_log_theta/alpha_of, so any valid mean_grid/alpha_grid
# trend unblocks this check without affecting the result (see task-4 brief).
trend_path = Path("/tmp/glmm_fit_regression_trend.json")
trend_path.write_text('{"mean_grid": [0.1, 1, 10, 100, 1000], "alpha_grid": [0.5, 0.4, 0.3, 0.2, 0.1]}')

subprocess.run([
    "Rscript", str(config.GLMM_FIT_R),
    "--x", str(SPIKE_DIR / "pilot_X.csv.gz"), "--y", str(SPIKE_DIR / "pilot_Y.csv.gz"),
    "--batch", str(SPIKE_DIR / "pilot_batch.csv.gz"), "--genes", str(genes_path),
    "--trend", str(trend_path), "--mode", "fixed_stage", "--out", str(OUT),
], check=True, cwd=str(config.MODELING_DIR))

new = pd.read_csv(OUT).set_index("gene")
old = pd.read_csv(SPIKE_DIR / "random_intercept_fits.csv").set_index("gene")
common = new.index.intersection(old.index)
mismatches = []
for g in common:
    if bool(new.loc[g, "ok"]) != bool(old.loc[g, "converged"]):
        mismatches.append(g)
        continue
    if bool(old.loc[g, "converged"]) and abs(new.loc[g, "tau2"] - old.loc[g, "tau2"]) > 1e-3:
        mismatches.append(g)

assert len(common) == 40, f"expected 40 common genes, got {len(common)}"
assert not mismatches, f"FAIL: {len(mismatches)} genes diverged from spike: {mismatches}"
print(f"PASS: all {len(common)} pilot genes match spike outcomes (ok+tau2)")
