import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MixedEffectsModeling.config as config
from viz_style import apply_style

OUT = config.THRESHOLD_SWEEP_DIR
df = pd.read_csv(OUT / "full_cascade_unconstrained.csv")
nz = pd.read_csv("/tmp/glmm_full_unconstrained/Y.csv.gz", index_col=0)
df["nz"] = df["gene"].map((nz > 0).sum(axis=0).to_dict())
df["nz_bin"] = pd.cut(df["nz"], bins=[0, 3, 7, 15, 30, 50, 100, np.inf])

summary = df.groupby("nz_bin", observed=True).agg(
    n_genes=("gene", "size"), ok_rate=("ok", "mean"),
    tau2_median=("tau2", "median"),
    pct_nbi=("stage", lambda s: (s == "nbi").mean()),
    pct_intercept=("stage", lambda s: (s == "intercept").mean()),
)
summary.to_csv(OUT / "nz_vs_stage_tau2_summary.csv")
print(summary.round(3).to_string())

apply_style()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
x = range(len(summary))
axes[0].bar(x, summary["ok_rate"]); axes[0].set(title="ok rate", xlabel="nz bin")
axes[1].bar(x, summary["tau2_median"]); axes[1].set(title="median tau2", xlabel="nz bin")
axes[2].bar(x, summary["pct_nbi"], label="nbi"); axes[2].bar(x, summary["pct_intercept"], bottom=summary["pct_nbi"], label="intercept")
axes[2].legend(); axes[2].set(title="stage composition", xlabel="nz bin")
for ax in axes:
    ax.set_xticks(list(x)); ax.set_xticklabels([str(b) for b in summary.index], rotation=45, ha="right")
fig.tight_layout()
(OUT / "Figures").mkdir(exist_ok=True)
fig.savefig(OUT / "Figures" / "nz_vs_stage_tau2.png", dpi=150)
print(f"Saved -> {OUT}/nz_vs_stage_tau2_summary.csv, {OUT}/Figures/nz_vs_stage_tau2.png")
