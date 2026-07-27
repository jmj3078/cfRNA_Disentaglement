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
    ap.add_argument("--out-dir", type=Path, default=config.ENGINE_MIXED_DIR,
                    help="directory to save the trained engine into (defaults to the production engine_state_mixed)")
    args = ap.parse_args()

    if not args.resume:
        shutil.rmtree(TMP_DIR, ignore_errors=True)

    trend_path = args.out_dir / "dispersion_trend.json"

    engine = NormativeModelEngineMixed()
    engine.load_hc_data()
    if trend_path.exists():
        from MixedEffectsModeling.core.dispersion_trend import load_trend
        engine.alpha_fn = load_trend(trend_path)
        print(f"Reusing cached trend -> {trend_path}")
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        engine.build_dispersion_trend(path=trend_path)
        print(f"Built trend -> {trend_path}")

    engine.assign_routes()
    n_pool = sum(1 for r in engine.genes.values() if r.route == "pool")
    n_model = sum(1 for r in engine.genes.values() if r.route == "model")
    print(f"HC={engine.X_hc_scaled.shape[0]} genes={len(engine.genes)} nz_a_max={engine.nz_a_max} pool_route={n_pool} model_route={n_model}")

    engine.train(limit=args.limit, tmp_dir=TMP_DIR)
    engine.save(args.out_dir)

    summary = engine.training_summary()
    print(summary.groupby(["route", "stage"]).size().to_string())
    print(f"ok={int(summary['ok'].sum())}/{len(summary)}")
    print(f"Saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
