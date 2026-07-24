import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import MixedEffectsModeling.config as config
from MixedEffectsModeling.core.model_engine_mixed import NormativeModelEngineMixed

TMP_DIR = "/tmp/glmm_train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="resume from an interrupted run's chunked results.csv in "
                         f"{TMP_DIR} instead of starting fresh (only safe against the same code version)")
    args = ap.parse_args()

    if not args.resume:
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    engine = NormativeModelEngineMixed()
    engine.load_hc_data()
    if config.DISPERSION_TREND_PATH.exists():
        from MixedEffectsModeling.core.dispersion_trend import load_trend
        engine.alpha_fn = load_trend()
        print(f"Reusing cached trend -> {config.DISPERSION_TREND_PATH}")
    else:
        engine.build_dispersion_trend()
        print(f"Built trend -> {config.DISPERSION_TREND_PATH}")

    engine.assign_routes()
    n_pool = sum(1 for r in engine.genes.values() if r.route == "pool")
    n_model = sum(1 for r in engine.genes.values() if r.route == "model")
    print(f"HC={engine.X_hc_scaled.shape[0]} genes={len(engine.genes)} nz_a_max={engine.nz_a_max} pool_route={n_pool} model_route={n_model}")

    engine.train(limit=args.limit, tmp_dir=TMP_DIR)
    engine.save(config.ENGINE_MIXED_DIR)

    summary = engine.training_summary()
    print(summary.groupby(["route", "stage"]).size().to_string())
    print(f"ok={int(summary['ok'].sum())}/{len(summary)}")
    print(f"Saved -> {config.ENGINE_MIXED_DIR}")


if __name__ == "__main__":
    main()
