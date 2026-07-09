#!/usr/bin/env python
"""Post-hoc OOD flagging for each LOBO_Results/<batch>/ held-out sample.

run_lobo_validation.py scores every held-out (HC+disease) sample in a batch
regardless of how far its covariate profile sits from the training HC
distribution. A sample whose covariates are far outside what the model was
trained on can show a large |Z| purely from extrapolation, not disease
biology -- exactly the same problem sample_filter.MahalanobisFilter already
guards against for the main disease-scoring pipeline. This script applies the
identical filter here, but fit on the TRAIN-FOLD HC only (all HC minus the
held-out batch), never on the held-out batch itself, since fitting on data
that includes the held-out batch would let the batch define its own normal
range and defeat the point of the check.

Writes ood_mask.npy (True = inlier, kept) and appends ood threshold info to
meta.json for each batch dir. Does not touch Z_test.npy or any existing file.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from pipeline import data_prep
from sample_filter import MahalanobisFilter

MP = config.MODELING_PARAMS
LOBO_DIR = config.MODELING_DIR / "LOBO_Results"


def main():
    adata = data_prep.load_adata()
    obs = adata.obs
    is_hc = (obs["Phenotype_Processed"].astype(str) == "Healthy Control").values
    X_raw = data_prep.bias_matrix(adata)
    names = np.array(adata.obs_names.astype(str))
    batch = obs[MP["stratify_col"]].astype(str).values
    name2row = {n: i for i, n in enumerate(names)}

    percentile = MP["ood_percentile"]

    for bdir in sorted(LOBO_DIR.iterdir()):
        meta_path = bdir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.load(open(meta_path))
        batch_id = meta["batch_id"]
        test_names = meta["test_names"]

        tr_mask = is_hc & (batch != batch_id)   # same train-fold HC as run_lobo_validation
        filt = MahalanobisFilter(percentile=percentile).fit(X_raw[tr_mask])

        te_rows = np.array([name2row[n] for n in test_names])
        X_te = X_raw[te_rows]
        d = filt.distances(X_te)
        keep = d <= filt.threshold_

        np.save(bdir / "ood_mask.npy", keep)
        np.save(bdir / "ood_distance.npy", d)
        meta["ood_percentile"] = percentile
        meta["ood_threshold"] = filt.threshold_
        meta["n_ood_removed"] = int((~keep).sum())
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"{batch_id:40s} n_test={len(keep):4d}  "
             f"removed_OOD={int((~keep).sum()):3d} ({(~keep).mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
