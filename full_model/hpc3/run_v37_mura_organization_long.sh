#!/usr/bin/env bash
set -euo pipefail

source_root="${V37_SOURCE_ROOT:?immutable staged source root required}"
source_sha="${V37_SOURCE_SHA:?exact immutable source SHA required}"
run_root="${V37_MURA_RUN_ROOT:?durable result root required}"
case_dir="$run_root/mechanical_heterogeneity_topology_off_n64"
mkdir -p "$case_dir"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONPATH="$source_root:$source_root/src"
export V37_SOURCE_SHA="$source_sha"

actual_sha="$(git -C "$source_root" rev-parse HEAD)"
test "$actual_sha" = "$source_sha"
git -C "$source_root" diff --quiet
git -C "$source_root" diff --cached --quiet

python3 -m py_compile \
  "$source_root/full_model/production/v24_mechanical_wall.py" \
  "$source_root/full_model/analysis/run_v30_mura_tier_b1_case.py" \
  >"$case_dir/preflight.log" 2>&1

python3 "$source_root/full_model/analysis/run_v30_mura_tier_b1_case.py" \
  --case-dir "$case_dir" --grid 64 \
  --condition mechanical_heterogeneity --seed 42 \
  --length-m 1e-5 --temperature-K 1100 --strain-rate-s 10000 \
  --initial-strain .01 --target-strain .05 --trial-dt-s 2e-9 \
  --progress-checkpoint-strain .005 --wall-checkpoint-s 840 \
  --history-interval 25 --max-wall-s 50400 \
  --mura-work-budget-mode energy_limited_feasible_extents \
  >"$case_dir/runner.log" 2>"$case_dir/runner.err"

