import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.dispersion_trend import build_trend, save_trend

TMP = Path("/tmp/glmm_full_unconstrained")
TMP.mkdir(exist_ok=True)
OUT = config.THRESHOLD_SWEEP_DIR / "full_cascade_unconstrained.csv"
config.THRESHOLD_SWEEP_DIR.mkdir(parents=True, exist_ok=True)


def main():
    adata = sc.read_h5ad(config.H5AD_PATH)
    m = ((adata.obs["QC_Passed"] == True) & (adata.obs["Phenotype_Processed"].notna()) &
         (adata.obs["Phenotype_Processed"] != "Unknown") &
         (adata.obs["broad_protocol_category"] != "Exome-based (EB)"))
    a = adata[m]
    is_hc = (a.obs["Phenotype_Processed"].astype(str) == "Healthy Control").values
    is_pc = (a.var["GeneType"] == "protein_coding").values
    X = a.obs[config.BIAS_COLUMNS].values.astype(np.float64)[is_hc]
    Xs = StandardScaler().fit_transform(X)
    Y = a.X.toarray() if issparse(a.X) else np.asarray(a.X)
    Y = np.round(Y[is_hc][:, is_pc]).astype(np.float64)
    names = a.var_names[is_pc].tolist()
    batch = a.obs["Batch_ID"].astype(str).values[is_hc]

    pd.DataFrame(Xs, columns=config.BIAS_COLUMNS).to_csv(TMP / "X.csv.gz")
    pd.DataFrame(Y, columns=names).to_csv(TMP / "Y.csv.gz")
    pd.DataFrame({"Batch_ID": batch}).to_csv(TMP / "batch.csv.gz")
    pd.DataFrame({"gene": names}).to_csv(TMP / "genes.csv", index=False)
    print(f"HC={Xs.shape[0]}  genes={len(names)}  batches={len(set(batch))}")

    if not config.DISPERSION_TREND_PATH.exists():
        print("Building Phase-0 dispersion trend from full HC data...")
        trend = build_trend(Y, min_nz=config.SPIKE_PARAMS["trend_min_nz"])
        save_trend(trend)
        print(f"Trend built: n_reliable={trend['n_reliable']} n_bins_used={trend['n_bins_used']}")

    subprocess.run([
        "Rscript", str(config.GLMM_FIT_R), "--x", str(TMP / "X.csv.gz"), "--y", str(TMP / "Y.csv.gz"),
        "--batch", str(TMP / "batch.csv.gz"), "--genes", str(TMP / "genes.csv"),
        "--trend", str(config.DISPERSION_TREND_PATH), "--mode", "cascade", "--out", str(OUT),
        "--chunk-size", "200", "--cores", str(min(8, 8)),
    ], check=True, cwd=str(config.GLMM_FIT_R.parent))
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
