#!/usr/bin/env bash
set -u -o pipefail

root="${HPC3_WORK_DIR:?}"
out="$root/full_model/production/output/v37-mura-organization"
source_sha="${V37_MURA_SOURCE_SHA:?}"
case_dir="$out/mechanical_heterogeneity_topology_off_n64"
mkdir -p "$case_dir"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONPATH="$root:$root/src" V37_SOURCE_SHA="$source_sha"
cd "$root" || exit 2
python3 -m py_compile \
  full_model/production/v24_mechanical_wall.py \
  full_model/analysis/run_v30_mura_tier_b1_case.py \
  >"$case_dir/preflight.log" 2>&1 || exit $?
python3 full_model/analysis/run_v30_mura_tier_b1_case.py \
  --case-dir "$case_dir" --grid 64 \
  --condition mechanical_heterogeneity --seed 42 \
  --length-m 1e-5 --temperature-K 1100 --strain-rate-s 10000 \
  --initial-strain .01 --target-strain .05 --trial-dt-s 2e-9 \
  --progress-checkpoint-strain .005 --wall-checkpoint-s 840 \
  --history-interval 25 --max-wall-s 50400 \
  --mura-work-budget-mode energy_limited_feasible_extents \
  >"$case_dir/runner.log" 2>"$case_dir/runner.err"
rc=$?
printf '{"production_source_commit":"%s","hpc3_run_id":"%s","exit_code":%s}\n' \
  "$source_sha" "${HPC3_RUN_ID:?}" "$rc" >"$case_dir/v37_hpc_terminal.json"
exit "$rc"
