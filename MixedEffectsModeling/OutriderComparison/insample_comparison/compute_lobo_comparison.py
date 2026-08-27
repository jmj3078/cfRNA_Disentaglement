import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from MixedEffectsModeling.validation.lobo_mmd import _mmd_permutation_test, _ref_centroid_direction, mmd_raw_cached

DIR = Path(__file__).parent


def null_stats(mmd2_obs, mmd2_null):
    null_mean, null_std = mmd2_null.mean(), mmd2_null.std()
    z_vs_null = (mmd2_obs - null_mean) / null_std if null_std > 0 else np.nan
    pct_rank = float((mmd2_null < mmd2_obs).mean())
    return null_mean, null_std, z_vs_null, pct_rank


def outrider_lobo_mmd():
    out_csv = DIR / "outrider_mmd_summary_v2.csv"
    out_pkl = DIR / "outrider_mmd_raw_v2.pkl"
    if out_csv.exists() and out_pkl.exists():
        return pd.read_csv(out_csv), pickle.load(open(out_pkl, "rb"))

    test_meta = json.load(open(DIR / "lobo_test_meta.json"))
    rows, raw = [], {}
    for b, m in test_meta.items():
        safe = b.replace(" ", "_")
        df = pd.read_csv(DIR / f"z_test_{safe}.csv", index_col=0).loc[m["test_names"]]
        Z = np.nan_to_num(df.values, nan=0.0, posinf=10.0, neginf=-10.0)
        is_hc = np.array(m["test_is_hc"])
        hc_Z, dis_Z = Z[is_hc], Z[~is_hc]

        rng = np.random.default_rng(42)
        max_n = 150
        hc_s = hc_Z if len(hc_Z) <= max_n else hc_Z[rng.choice(len(hc_Z), max_n, replace=False)]
        dis_s = dis_Z if len(dis_Z) <= max_n else dis_Z[rng.choice(len(dis_Z), max_n, replace=False)]
        mmd2, p, mmd2_null = _mmd_permutation_test(hc_s, dis_s, n_perm=1000, seed=42)
        d_hc, d_dis, p_dir = _ref_centroid_direction(Z, is_hc)
        null_mean, null_std, z_vs_null, pct_rank = null_stats(mmd2, mmd2_null)

        rows.append(dict(batch=b, n_genes=df.shape[1], n_hc=len(hc_Z), n_dis=len(dis_Z),
                         mmd2_obs=mmd2, mmd2_null_mean=null_mean, mmd2_null_std=null_std,
                         z_vs_null=z_vs_null, pct_rank_of_obs=pct_rank, perm_p=p,
                         hc_ref_dist=d_hc.mean(), dis_ref_dist=d_dis.mean(),
                         disease_farther=d_dis.mean() > d_hc.mean(), p_direction=p_dir))
        raw[b] = dict(mmd2_obs=mmd2, mmd2_null=mmd2_null, d_hc=d_hc, d_dis=d_dis)

    out = pd.DataFrame(rows).sort_values("n_dis", ascending=False)
    out.to_csv(out_csv, index=False)
    with open(out_pkl, "wb") as f:
        pickle.dump(raw, f)
    return out, raw


def our_engine_lobo_mmd(shash=False):
    """Production LOBO/MMD (validation/lobo_mmd.py), no ood_filter. shash=False: raw Z, exactly
    as scored. shash=True: per-batch SHASH fit on that batch's own train-fold in-sample Z, applied
    to the held-out LOBO Z (lobo_engine.run_one_batch) -- OUTRIDER has no equivalent second-stage
    correction, so it only ever has a raw-Z counterpart to this."""
    raw = mmd_raw_cached(shash=shash)
    rows = []
    for b, d in raw.items():
        null_mean, null_std, z_vs_null, pct_rank = null_stats(d["mmd2_obs"], d["mmd2_null"])
        rows.append(dict(batch=b, mmd2_obs=d["mmd2_obs"], mmd2_null_mean=null_mean,
                         mmd2_null_std=null_std, z_vs_null=z_vs_null, pct_rank_of_obs=pct_rank))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    outr_df, _ = outrider_lobo_mmd()
    eng_df = our_engine_lobo_mmd()
    print("OUTRIDER:\n", outr_df[["batch", "mmd2_obs", "mmd2_null_mean", "mmd2_null_std",
                                    "z_vs_null", "pct_rank_of_obs", "perm_p", "p_direction"]].to_string())
    print("\nour engine:\n", eng_df.to_string())
