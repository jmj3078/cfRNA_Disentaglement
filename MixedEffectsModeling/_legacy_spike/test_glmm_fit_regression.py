import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MixedEffectsModeling.config as config

SPIKE_DIR = Path(__file__).resolve().parent / "Spike_Results"
GLMM_FIT_R = Path(__file__).resolve().parent / "glmm_fit.R"
OUT = Path("/tmp/glmm_fit_regression_out.csv")
OUT.unlink(missing_ok=True)

genes = pd.read_csv(SPIKE_DIR / "pilot_genes.csv")[["gene"]]
genes["stage"] = "nbi"
genes_path = Path("/tmp/glmm_fit_regression_genes.csv")
genes.to_csv(genes_path, index=False)

# glmm_fit.R's alpha_of() reads dispersion_trend.py's real saved schema
# (lowess_logmu/lowess_logsigma/alpha_floor/alpha_cap). All pilot genes land on
# stage "nbi", which regresses dispersion on covariates directly and never
# touches fixed_log_theta/alpha_of, so any valid trend of this shape unblocks
# the check without affecting the result.
trend_path = Path("/tmp/glmm_fit_regression_trend.json")
trend_path.write_text(
    '{"alpha_floor": 0.01, "alpha_cap": 5.0, '
    '"lowess_logmu": [-2.3, 0, 2.3, 4.6, 6.9, 9.2], '
    '"lowess_logsigma": [-0.7, -0.9, -1.2, -1.6, -1.9, -2.3]}'
)

subprocess.run([
    "Rscript", str(GLMM_FIT_R),
    "--x", str(SPIKE_DIR / "pilot_X.csv.gz"), "--y", str(SPIKE_DIR / "pilot_Y.csv.gz"),
    "--batch", str(SPIKE_DIR / "pilot_batch.csv.gz"), "--genes", str(genes_path),
    "--trend", str(trend_path), "--mode", "fixed_stage", "--out", str(OUT),
], check=True, cwd=str(GLMM_FIT_R.parent))

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
