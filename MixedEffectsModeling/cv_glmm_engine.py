import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.marginal_rqr import marginal_nb_rqr
from MixedEffectsModeling.model_engine_mixed import NormativeModelEngineMixed

MP = config.SPIKE_PARAMS


# Copied from Modeling/model_engine.py's _w1_normal (mean abs deviation vs
# theoretical N(0,1) quantiles) -- duplicate, not an import, per isolation.
def _w1_normal(z):
    v = z[np.isfinite(z)]
    n = len(v)
    if n < 8:
        return np.nan
    ref = norm.ppf(np.linspace(1 / (2 * n), 1 - 1 / (2 * n), n))
    return float(np.mean(np.abs(np.sort(v) - ref)))


def main():
    out_dir = config.CV_MIXED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(config.ENGINE_MIXED_DIR / "training_summary.csv", index_col="gene")
    summary = summary[summary["ok"] & (summary["route"] == "model")]

    e2 = NormativeModelEngineMixed()
    e2.load_hc_data()
    n_hc = e2.X_hc_scaled.shape[0]
    folds = list(StratifiedKFold(MP["n_splits"], shuffle=True, random_state=42).split(np.zeros(n_hc), e2.batch))

    tmp = "/tmp/cv_glmm"
    Path(tmp).mkdir(exist_ok=True)
    rows = []
    for fi, (tr, te) in enumerate(folds):
        pd.DataFrame(e2.X_hc_scaled[tr], columns=config.BIAS_COLUMNS).to_csv(f"{tmp}/X_{fi}.csv.gz")
        Y_tr = e2.Y_hc[tr][:, [e2._gene_col[g] for g in summary.index]]
        pd.DataFrame(Y_tr, columns=summary.index).to_csv(f"{tmp}/Y_{fi}.csv.gz")
        pd.DataFrame({"Batch_ID": e2.batch[tr]}).to_csv(f"{tmp}/batch_{fi}.csv.gz")
        gene_stage = summary[["stage"]].reset_index().rename(columns={"index": "gene"})
        gene_stage.to_csv(f"{tmp}/genes_{fi}.csv", index=False)
        subprocess.run([
            "Rscript", str(config.GLMM_FIT_R), "--x", f"{tmp}/X_{fi}.csv.gz", "--y", f"{tmp}/Y_{fi}.csv.gz",
            "--batch", f"{tmp}/batch_{fi}.csv.gz", "--genes", f"{tmp}/genes_{fi}.csv",
            "--trend", str(config.DISPERSION_TREND_PATH), "--mode", "fixed_stage", "--out", f"{tmp}/res_{fi}.csv",
        ], check=True, cwd=str(config.GLMM_FIT_R.parent))

        fold_fits = pd.read_csv(f"{tmp}/res_{fi}.csv").set_index("gene")
        Xa_te = np.column_stack([np.ones(len(te)), e2.X_hc_scaled[te]])
        for g in summary.index:
            if g not in fold_fits.index or not bool(fold_fits.loc[g, "ok"]):
                continue
            row = fold_fits.loc[g]
            mu_coef = row[[c for c in fold_fits.columns if c.startswith("mu_coef_")]].values.astype(float)
            disp_coef = row[[c for c in fold_fits.columns if c.startswith("disp_coef_")]].values.astype(float)
            mu = np.clip(np.exp(Xa_te @ np.nan_to_num(mu_coef, nan=0.0)), 1e-6, 1e8)
            # Full per-sample linear predictor (intercept + covariate slopes),
            # matching model_engine_mixed.py's score() fix -- NOT disp_coef[0]
            # alone, which silently flattens dispersion for stage nbi genes
            # (see task-7-report.md's "Finding 1" post-review fix).
            if not np.all(np.isnan(disp_coef)):
                alpha = np.exp(-Xa_te @ np.nan_to_num(disp_coef, nan=0.0))
            else:
                alpha = np.full(len(te), e2.alpha_fn(float(mu.mean())))
            y_te = e2.Y_hc[te, e2._gene_col[g]]
            z = marginal_nb_rqr(y_te, mu, alpha, float(row["tau2"]), seed=42 + fi)
            rows.append(dict(gene=g, fold=fi, z=z))

    zdict = {}
    for g in summary.index:
        zs = [r["z"] for r in rows if r["gene"] == g]
        if not zs:
            continue
        zdict[g] = np.concatenate(zs)

    stats = []
    for g, z in zdict.items():
        v = z[np.isfinite(z)]
        if len(v) < 8:
            continue
        stats.append(dict(gene=g, route="model", stage=summary.loc[g, "stage"], nz=int(summary.loc[g, "nz"]),
                          w1=_w1_normal(v), mean_z=float(v.mean()), std_z=float(v.std()),
                          skew_z=float(skew(v)), kurt_z=float(kurtosis(v)), n_valid=len(v)))
    df = pd.DataFrame(stats)
    df.to_csv(out_dir / "cv_stats.csv", index=False)
    print(df.groupby("stage")[["w1", "mean_z", "std_z"]].median().to_string())
    print(f"Saved -> {out_dir}/cv_stats.csv")


if __name__ == "__main__":
    main()
