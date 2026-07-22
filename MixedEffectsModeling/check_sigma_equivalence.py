import numpy as np
import pandas as pd

df = pd.read_csv("Spike_Results/fixed_only_fits.csv")
X = pd.read_csv("Spike_Results/pilot_X.csv.gz", index_col=0)
n_cov = X.shape[1]

both_ok = df[df["gamlss_success"] & df["glmmtmb_success"]]
print(f"{len(both_ok)}/{len(df)} genes converged in both gamlss and glmmTMB")

# Covariate grid: HC-observed rows themselves (already representative of the
# real scoring distribution) plus the all-zero (mean) point.
Xa = np.column_stack([np.ones(len(X)), X.values])
grid = np.vstack([Xa, np.concatenate([[1.0], np.zeros(n_cov)])])

max_rel_diff = 0.0
worst = None
rows = []
for _, row in both_ok.iterrows():
    sigma_coef = row[[f"gamlss_sigma_coef_{i}" for i in range(n_cov + 1)]].values.astype(float)
    disp_coef = row[[f"glmmtmb_disp_coef_{i}" for i in range(n_cov + 1)]].values.astype(float)

    sigma_gamlss = np.exp(grid @ sigma_coef)
    # glmmTMB nbinom2 dispformula predicts log(theta), theta = 1/sigma (see design spec) --
    # so gamlss's sigma corresponds to exp(-grid @ disp_coef), NOT exp(+grid @ disp_coef).
    sigma_glmmtmb = np.exp(-grid @ disp_coef)

    rel_diff = np.abs(sigma_gamlss - sigma_glmmtmb) / np.clip(sigma_gamlss, 1e-8, None)
    gene_max = float(rel_diff.max())
    rows.append({"gene": row["gene"], "max_rel_diff": gene_max})
    if gene_max > max_rel_diff:
        max_rel_diff = gene_max
        worst = row["gene"]

report = pd.DataFrame(rows).sort_values("max_rel_diff", ascending=False)
report.to_csv("Spike_Results/sigma_equivalence_report.csv", index=False)
print(report.head(10))

TOLERANCE = 0.10  # 10% relative -- these are two different optimizers/penalizations,
                  # exact numerical match is not expected, but the sign/reciprocal
                  # mapping being right should keep genes in the same ballpark.
if max_rel_diff < TOLERANCE:
    print(f"PASS: max relative sigma(x) difference {max_rel_diff:.4f} (gene {worst}) "
          f"< tolerance {TOLERANCE}")
else:
    print(f"FAIL: max relative sigma(x) difference {max_rel_diff:.4f} (gene {worst}) "
          f">= tolerance {TOLERANCE} -- do NOT trust the exp(-X@disp_coef) mapping yet, "
          f"re-check the glmmTMB dispformula parameterization before proceeding to Task 5")
