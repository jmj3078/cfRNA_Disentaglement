#!/usr/bin/env bash
# Full refit then 5-fold CV, both on config defaults (nz_a_max=25, tau2_max=3.0).
# CV reads its route assignment from the engine's training_summary.csv, so it must
# run after the refit and against the same engine dir.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname "$HERE")"
OUT="${OUT_DIR:-$BASE/engine_state_mixed}"
CV_OUT="${CV_DIR:-$BASE/CV_Results_mixed}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$BASE/Logs/refit_cv_$STAMP.log"
mkdir -p "$BASE/Logs" "$OUT"
rm -rf /tmp/cv_glmm_v2   # cv_engine reuses this dir; drop any earlier run's fold inputs

cd "$BASE/.."
{
  echo "=== refit $(date -Is) -> $OUT"
  conda run --no-capture-output -n scRNA python MixedEffectsModeling/core/run_engine.py \
    --out-dir "$OUT" "$@"
  echo "=== cv $(date -Is) -> $CV_OUT"
  conda run --no-capture-output -n scRNA python MixedEffectsModeling/validation/cv_engine.py \
    --engine-dir "$OUT" --out-dir "$CV_OUT"
  echo "=== done $(date -Is)"
} 2>&1 | tee "$LOG"

echo "log -> $LOG"
