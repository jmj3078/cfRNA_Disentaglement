#!/usr/bin/env bash
# Full refit under the MIN_HC_BATCH_SIZE HC filter, no pooling cut (nz_a_max=0):
# dispersion trend + EB slope prior calibration, then the cascade over every gene.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname "$HERE")"
OUT="${OUT_DIR:-$BASE/engine_state_mixed}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$BASE/Logs/refit_nopool_$STAMP.log"
mkdir -p "$BASE/Logs"

# prepare_hyperparams short-circuits on a cached trend+prior, so an empty OUT
# forces the trend/calibration to be rebuilt on the filtered HC set.
mkdir -p "$OUT"

cd "$BASE/.."
conda run --no-capture-output -n scRNA python MixedEffectsModeling/core/run_engine.py \
  --nz-a-max 0 --out-dir "$OUT" "$@" 2>&1 | tee "$LOG"

echo "log -> $LOG"
