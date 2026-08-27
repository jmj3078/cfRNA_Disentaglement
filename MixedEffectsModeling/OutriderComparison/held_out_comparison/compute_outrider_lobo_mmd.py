import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.validation.lobo_mmd import _mmd_permutation_test, _ref_centroid_direction

DIR = Path(__file__).parent
IN_DIR = config.OUTRIDER_COMPARISON_DIR  # insample_comparison, shares lobo_test_meta.json
OUT = DIR / "outrider_mmd_summary.csv"
RAW_OUT = DIR / "outrider_mmd_raw.pkl"

if OUT.exists() and RAW_OUT.exists():
    print(f"already cached -> {OUT}, {RAW_OUT}")
    raise SystemExit

test_meta = json.load(open(IN_DIR / "lobo_test_meta.json"))
rows = []
raw = {}
for b, m in test_meta.items():
    safe = b.replace(" ", "_")
    df = pd.read_csv(DIR / f"z_test_{safe}.csv", index_col=0)
    df = df.loc[m["test_names"]]
    Z = np.nan_to_num(df.values, nan=0.0, posinf=10.0, neginf=-10.0)
    is_hc = np.array(m["test_is_hc"])
    hc_Z, dis_Z = Z[is_hc], Z[~is_hc]

    rng = np.random.default_rng(42)
    max_n = 150
    hc_s = hc_Z if len(hc_Z) <= max_n else hc_Z[rng.choice(len(hc_Z), max_n, replace=False)]
    dis_s = dis_Z if len(dis_Z) <= max_n else dis_Z[rng.choice(len(dis_Z), max_n, replace=False)]
    mmd2, p, mmd2_null = _mmd_permutation_test(hc_s, dis_s, n_perm=1000, seed=42)
    d_hc, d_dis, p_dir = _ref_centroid_direction(Z, is_hc)
    rows.append(dict(batch=b, n_genes=df.shape[1], n_hc=len(hc_Z), n_dis=len(dis_Z), mmd2=mmd2, perm_p=p,
                     hc_ref_dist=d_hc.mean(), dis_ref_dist=d_dis.mean(),
                     disease_farther=d_dis.mean() > d_hc.mean(), p_direction=p_dir))
    raw[b] = dict(d_hc=d_hc, d_dis=d_dis, mmd2_obs=mmd2, mmd2_null=mmd2_null)

out = pd.DataFrame(rows).sort_values("n_dis", ascending=False)
out.to_csv(OUT, index=False)
with open(RAW_OUT, "wb") as f:
    pickle.dump(raw, f)
print(f"saved -> {OUT}, {RAW_OUT}")
print(out.to_string())
