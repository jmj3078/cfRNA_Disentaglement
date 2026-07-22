import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import MixedEffectsModeling.config as config
from viz_style import apply_style

MP = config.SPIKE_PARAMS
OUT = config.THRESHOLD_SWEEP_DIR
OUT.mkdir(parents=True, exist_ok=True)
config.THRESHOLD_SWEEP_FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_hc():
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
    batch = a.obs["Batch_ID"].astype(str).values[is_hc]
    return Xs, Y, batch


def main():
    summary_path = OUT / "pool_threshold_sweep_summary.csv"
    if summary_path.exists():
        print(f"Loading cached sweep -> {summary_path}")
        summary = pd.read_csv(summary_path)
    else:
        Xs, Y, batch = load_hc()
        n_hc = Xs.shape[0]
        folds = list(StratifiedKFold(MP["n_splits"], shuffle=True, random_state=42).split(np.zeros(n_hc), batch))
        fold_rows = []
        for fi, (_, te) in enumerate(folds):
            for idx in te:
                fold_rows.append(dict(sample_idx=idx, fold=fi))
        pd.DataFrame(fold_rows).to_csv("/tmp/pool_sweep_folds.csv", index=False)
        pd.DataFrame(Xs, columns=config.BIAS_COLUMNS).to_csv("/tmp/pool_sweep_X.csv.gz", index=True)
        pd.DataFrame(Y).to_csv("/tmp/pool_sweep_Y.csv.gz", index=True)
        pd.DataFrame({"Batch_ID": batch}).to_csv("/tmp/pool_sweep_batch.csv.gz", index=True)

        subprocess.run([
            "Rscript", str(config.POOL_SWEEP_R),
            "--x", "/tmp/pool_sweep_X.csv.gz", "--y", "/tmp/pool_sweep_Y.csv.gz",
            "--batch", "/tmp/pool_sweep_batch.csv.gz", "--folds", "/tmp/pool_sweep_folds.csv",
            "--out", str(summary_path),
        ], check=True, cwd=str(config.GLMM_FIT_R.parent))
        summary = pd.read_csv(summary_path)

    print(summary.round(3).to_string(index=False))
    picked = summary[summary["w1_median"] > 0.25]
    nz_a_max = int(picked["nz_threshold"].min()) if len(picked) else int(summary["nz_threshold"].max())
    print(f"\nAuto-picked nz_a_max = {nz_a_max} (smallest cutoff where median W1 > 0.25)")

    apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(summary["nz_threshold"], summary["w1_median"], "-o", label="Median W1")
    ax.plot(summary["nz_threshold"], summary["w1_p90"], "-o", label="90th pct W1")
    ax.axhline(0.25, ls=":", color="gray")
    ax.axvline(nz_a_max, ls="--", color="k", label=f"Picked nz_a_max={nz_a_max}")
    ax.set(xlabel="Pooling threshold (HC nonzero count)", ylabel="W1 (held-out CV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.THRESHOLD_SWEEP_FIG_DIR / "pool_threshold_sweep.png", dpi=150)
    with open(OUT / "nz_a_max.txt", "w") as f:
        f.write(str(nz_a_max))
    print(f"Saved -> {summary_path}, {config.THRESHOLD_SWEEP_FIG_DIR}/pool_threshold_sweep.png, {OUT}/nz_a_max.txt")


if __name__ == "__main__":
    main()
