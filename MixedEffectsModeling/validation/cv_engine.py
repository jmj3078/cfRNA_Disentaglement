import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.dispersion_trend import load_trend
from MixedEffectsModeling.core.marginal_rqr import marginal_nb_rqr
from MixedEffectsModeling.core.model_engine_mixed import NormativeModelEngineMixed

MP = config.SPIKE_PARAMS


def _w1_normal(z):
    v = z[np.isfinite(z)]
    n = len(v)
    if n < 8:
        return np.nan
    ref = norm.ppf(np.linspace(1 / (2 * n), 1 - 1 / (2 * n), n))
    return float(np.mean(np.abs(np.sort(v) - ref)))


# Every (gene, fold) pair is recorded here regardless of convergence -- the
# v1 CV script silently dropped non-converging folds, which hid exactly the
# fold-level failure information this module exists to keep.
def cv_model_route(e2, model_genes, stage_of, folds, tmp):
    if not model_genes:
        return {}, {}, []
    rows, fold_stat_rows = [], []
    for fi, (tr, te) in enumerate(folds):
        pd.DataFrame(e2.X_hc_scaled[tr], columns=config.BIAS_COLUMNS).to_csv(f"{tmp}/X_{fi}.csv.gz")
        Y_tr = e2.Y_hc[tr][:, [e2._gene_col[g] for g in model_genes]]
        pd.DataFrame(Y_tr, columns=model_genes).to_csv(f"{tmp}/Y_{fi}.csv.gz")
        pd.DataFrame({"Batch_ID": e2.batch[tr]}).to_csv(f"{tmp}/batch_{fi}.csv.gz")
        pd.DataFrame({"gene": model_genes, "stage": [stage_of[g] for g in model_genes]}).to_csv(
            f"{tmp}/genes_{fi}.csv", index=False)
        subprocess.run([
            "Rscript", str(config.GLMM_FIT_R), "--x", f"{tmp}/X_{fi}.csv.gz", "--y", f"{tmp}/Y_{fi}.csv.gz",
            "--batch", f"{tmp}/batch_{fi}.csv.gz", "--genes", f"{tmp}/genes_{fi}.csv",
            "--trend", str(config.DISPERSION_TREND_PATH), "--mode", "fixed_stage", "--out", f"{tmp}/res_{fi}.csv",
        ], check=True, cwd=str(config.GLMM_FIT_R.parent))

        fold_fits = pd.read_csv(f"{tmp}/res_{fi}.csv").set_index("gene")
        Xa_te = np.column_stack([np.ones(len(te)), e2.X_hc_scaled[te]])
        for g in model_genes:
            if g not in fold_fits.index:
                fold_stat_rows.append(dict(gene=g, fold=fi, stage=stage_of[g], ok=False,
                                           singular=None, tau2=np.nan, fail_reason="fold_output_missing", n_test=len(te)))
                continue
            row = fold_fits.loc[g]
            ok = bool(row["ok"])
            fold_stat_rows.append(dict(gene=g, fold=fi, stage=row["stage"], ok=ok,
                                       singular=bool(row["singular"]) if not pd.isna(row["singular"]) else None,
                                       tau2=float(row["tau2"]) if not pd.isna(row["tau2"]) else np.nan,
                                       fail_reason=row["fail_reason"] if not pd.isna(row["fail_reason"]) else "",
                                       n_test=len(te)))
            if not ok:
                continue
            mu_coef = row[[c for c in fold_fits.columns if c.startswith("mu_coef_")]].values.astype(float)
            disp_coef = row[[c for c in fold_fits.columns if c.startswith("disp_coef_")]].values.astype(float)
            mu = np.clip(np.exp(Xa_te @ np.nan_to_num(mu_coef, nan=0.0)), 1e-6, 1e8)
            if not np.all(np.isnan(disp_coef)):
                alpha = np.exp(-Xa_te @ np.nan_to_num(disp_coef, nan=0.0))
            elif "fixed_alpha" in row.index and not pd.isna(row["fixed_alpha"]):
                alpha = np.full(len(te), float(row["fixed_alpha"]))
            else:
                alpha = np.full(len(te), e2.alpha_fn(float(mu.mean())))
            y_te = e2.Y_hc[te, e2._gene_col[g]]
            tau2 = float(row["tau2"])
            z = marginal_nb_rqr(y_te, mu, alpha, tau2, seed=42 + fi)
            rows.append(dict(gene=g, fold=fi, y=y_te.astype(np.float32), mu=mu.astype(np.float32),
                             alpha=np.asarray(alpha, dtype=np.float32),
                             tau2=np.full(len(te), tau2, dtype=np.float32), z=z))

    zdict, ppc_dict = {}, {}
    for g in model_genes:
        grecs = [r for r in rows if r["gene"] == g]
        if not grecs:
            continue
        zdict[g] = np.concatenate([r["z"] for r in grecs])
        ppc_dict[g] = dict(
            y=np.concatenate([r["y"] for r in grecs]),
            mu=np.concatenate([r["mu"] for r in grecs]),
            alpha=np.concatenate([r["alpha"] for r in grecs]),
            tau2=np.concatenate([r["tau2"] for r in grecs]),
            family="negbin", stage=stage_of[g])
    return zdict, ppc_dict, fold_stat_rows


def main():
    out_dir = config.CV_MIXED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(config.ENGINE_MIXED_DIR / "training_summary.csv", index_col="gene")
    summary = summary[summary["ok"]]
    model_genes = summary.index[summary["route"] == "model"].tolist()
    stage_of = summary["stage"].to_dict()

    e2 = NormativeModelEngineMixed()
    e2.load_hc_data()
    e2.alpha_fn = load_trend()
    n_hc = e2.X_hc_scaled.shape[0]
    folds = list(StratifiedKFold(MP["n_splits"], shuffle=True, random_state=42).split(np.zeros(n_hc), e2.batch))

    tmp = "/tmp/cv_glmm_v2"
    Path(tmp).mkdir(exist_ok=True)

    print(f"CV: {len(model_genes)} model-route genes (pool route not run this round)")
    zdict, ppc_dict, fold_stat_rows = cv_model_route(e2, model_genes, stage_of, folds, tmp)

    fold_stats = pd.DataFrame(fold_stat_rows)
    fold_stats.to_csv(out_dir / "fold_stats.csv", index=False)
    print(f"fold success rate: {fold_stats['ok'].mean():.3f} ({int(fold_stats['ok'].sum())}/{len(fold_stats)})")

    stats = []
    for g, z in zdict.items():
        v = z[np.isfinite(z)]
        if len(v) < 8:
            continue
        nz = int((e2.Y_hc[:, e2._gene_col[g]] > 0).sum())
        stats.append(dict(gene=g, route="model", stage=stage_of[g], nz=nz,
                          w1=_w1_normal(v), mean_z=float(v.mean()), std_z=float(v.std()),
                          skew_z=float(skew(v)), kurt_z=float(kurtosis(v)), n_valid=len(v)))
    df = pd.DataFrame(stats)
    df.to_csv(out_dir / "cv_stats.csv", index=False)
    with open(out_dir / "cv_zscores.pkl", "wb") as f:
        pickle.dump(zdict, f)
    with open(out_dir / "cv_ppc.pkl", "wb") as f:
        pickle.dump(ppc_dict, f)
    print(df.groupby("stage")[["w1", "mean_z", "std_z"]].median().to_string())
    print(f"Saved -> {out_dir}/fold_stats.csv, cv_stats.csv, cv_zscores.pkl, cv_ppc.pkl")


if __name__ == "__main__":
    main()
