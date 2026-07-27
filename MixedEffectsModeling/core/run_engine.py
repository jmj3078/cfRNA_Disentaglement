import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.model_engine_mixed import NormativeModelEngineMixed

TMP_DIR = "/tmp/glmm_train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pilot-genes", type=int, default=None,
                    help="smoke test: shrink the EB-prior pilot from its default gene count")
    ap.add_argument("--resume", action="store_true",
                    help="resume from an interrupted run's chunked results.csv in "
                         f"{TMP_DIR} instead of starting fresh (only safe against the same code version)")
    ap.add_argument("--out-dir", type=Path, default=config.ENGINE_MIXED_DIR,
                    help="directory to save the trained engine into (defaults to the production engine_state_mixed)")
    args = ap.parse_args()

    if not args.resume:
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trend_path = args.out_dir / "dispersion_trend.json"
    disp_prior_path = args.out_dir / "disp_prior.json"

    engine = NormativeModelEngineMixed()
    engine.load_hc_data()
    engine.assign_routes()
    n_pool = sum(1 for r in engine.genes.values() if r.route == "pool")
    n_model = sum(1 for r in engine.genes.values() if r.route == "model")
    print(f"HC={engine.X_hc_scaled.shape[0]} genes={len(engine.genes)} nz_a_max={engine.nz_a_max} pool_route={n_pool} model_route={n_model}")

    ran = engine.prepare_hyperparams(trend_path, disp_prior_path, TMP_DIR, args.pilot_genes)
    prior = engine.disp_prior
    print(f"hyperparams {'from pilot' if ran else 'cached'}: trend -> {trend_path}, prior -> {disp_prior_path}")
    print(f"disp slope EB prior sd (n_pilot={prior['n_pilot_genes']})")
    for cov, tau in zip(prior["covariates"], prior["tau_slope"]):
        print(f"  tau_slope[{cov}] = {tau:.4f}")

    engine.train(limit=args.limit, tmp_dir=TMP_DIR, disp_prior_path=disp_prior_path)
    engine.save(args.out_dir)

    summary = engine.training_summary()
    print(summary.groupby(["route", "stage"]).size().to_string())
    print(f"ok={int(summary['ok'].sum())}/{len(summary)}")
    ok = summary[summary["ok"]]
    print(f"disp intercept EB: tau_d={engine.disp_tau_d2 ** 0.5:.4f} "
          f"shrink_frac_median={np.median(_shrink_frac(ok)):.3f}")
    print(f"Cook outliers: genes_with_any={int((ok['n_outliers'] > 0).sum())}/{len(ok)} "
          f"mean_removed={ok['n_outliers'].mean():.2f} max={int(ok['n_outliers'].max())} "
          f"refit_failed={int(ok['outlier_refit_failed'].sum())}")
    print(f"Saved -> {args.out_dir}")


def _shrink_frac(ok):
    """Fraction of the way from the raw MLE to the trend that the EB squeeze moved
    log(theta). 0 = kept the gene's own estimate, 1 = pinned to the trend."""
    raw = ok["log_theta_raw"].to_numpy(float)
    eb = ok["log_theta_eb"].to_numpy(float)
    trend = -np.log(ok["trend_alpha"].to_numpy(float))
    gap = raw - trend
    frac = np.where(np.abs(gap) > 1e-9, (eb - raw) / np.where(np.abs(gap) > 1e-9, -gap, 1.0), 0.0)
    return frac[np.isfinite(frac)]


if __name__ == "__main__":
    main()
