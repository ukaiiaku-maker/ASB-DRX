#!/usr/bin/env bash
set -euo pipefail
ROOT="${DRX_OUTROOT:-results_v34_rate_sweep}"
RUNNER="${RUNNER:-run_v34_candidate_drx_asb.sh}"
DRIVER="${DRIVER:-drx_var_v34_candidate_drx_asb_sweep.py}"
RATE_LIST="${RATE_LIST:-100 300 1000 3000 10000 30000}"
SEED_LIST="${SEED_LIST:-42}"
TARGET_STRAIN="${TARGET_STRAIN:-0.5}"
DT_STRAIN_STEP="${DT_STRAIN_STEP:-1.0e-4}"
BRANCH_LIST="${BRANCH_LIST:-coupled}"
MODE="${ACTIVITY_MODE:-crystallographic_local}"
PYTHON="${PYTHON:-python3}"
mkdir -p "$ROOT"
for BRANCH in $BRANCH_LIST; do
  for RATE in $RATE_LIST; do
    TAG="$($PYTHON - <<PY
r=float('$RATE')
print(str(int(round(r))) if abs(r-round(r)) < 1e-12 else ("%.4g" % r).replace('.','p'))
PY
)"
    for SEED in $SEED_LIST; do
      OUTDIR="$ROOT/${BRANCH}_rate_${TAG}_${MODE}_seed${SEED}"
      LOG="$ROOT/${BRANCH}_rate_${TAG}_${MODE}_seed${SEED}.log"
      echo
      echo "=== V34 BRANCH=$BRANCH RATE=$RATE SEED=$SEED TARGET_STRAIN=$TARGET_STRAIN OUTDIR=$OUTDIR ==="
      DRX_OUTDIR="$OUTDIR" DRIVER="$DRIVER" RATE="$RATE" TARGET_STRAIN="$TARGET_STRAIN" \
      DT_STRAIN_STEP="$DT_STRAIN_STEP" BRANCH="$BRANCH" ACTIVITY_MODE="$MODE" \
      POLY_SEED="$SEED" NUC_SEED="$((271828 + SEED))" bash "$RUNNER" 2>&1 | tee "$LOG"
    done
  done
done

echo "Sweep finished. Summarize with: python3 summarize_v34_drx_asb_rate_sweep.py $ROOT"
