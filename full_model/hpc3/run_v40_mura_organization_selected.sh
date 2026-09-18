#!/usr/bin/env bash
set -Eeuo pipefail

: "${V40_SOURCE_ARCHIVE:?immutable source archive required}"
: "${V40_SOURCE_SHA:?source identity required}"
: "${V40_RUN_ROOT:?persistent run root required}"
: "${SLURM_ARRAY_TASK_ID:?array task required}"

conditions=(homogeneous mechanical_heterogeneity)
condition="${conditions[$SLURM_ARRAY_TASK_ID]}"
scratch_root="${TMPDIR:?}/v40-mura-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
source_root="$scratch_root/source"
output_root="$scratch_root/output"
case_dir="$output_root/${condition}_topology_off_n64"
mkdir -p "$source_root" "$case_dir" "$V40_RUN_ROOT/results" \
  "$V40_RUN_ROOT/status" "$V40_RUN_ROOT/logs"
tar -xzf "$V40_SOURCE_ARCHIVE" -C "$source_root"
export PYTHONPATH="$source_root:$source_root/src"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export V37_SOURCE_SHA="$V40_SOURCE_SHA"

finish() {
  code=$?
  if [[ -d "$output_root" ]]; then
    partial="$V40_RUN_ROOT/results/${condition}.tar.gz.partial"
    final="$V40_RUN_ROOT/results/${condition}.tar.gz"
    tar -czf "$partial" -C "$output_root" .
    mv "$partial" "$final"
    sha256sum "$final" >"$V40_RUN_ROOT/results/${condition}.sha256"
  fi
  tmp="$V40_RUN_ROOT/status/task-${SLURM_ARRAY_TASK_ID}.json.partial"
  printf '{"array_job_id":"%s","task_id":%s,"condition":"%s","source_sha":"%s","exit_code":%d}\n' \
    "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$condition" \
    "$V40_SOURCE_SHA" "$code" >"$tmp"
  mv "$tmp" "$V40_RUN_ROOT/status/task-${SLURM_ARRAY_TASK_ID}.json"
  exit "$code"
}
trap finish EXIT

python "$source_root/full_model/analysis/run_v30_mura_tier_b1_case.py" \
  --case-dir "$case_dir" --grid 64 --condition "$condition" --seed 42 \
  --length-m 1e-5 --temperature-K 1100 --strain-rate-s 10000 \
  --initial-strain .01 --target-strain .02 --trial-dt-s 2e-9 \
  --progress-checkpoint-strain .0025 --wall-checkpoint-s 840 \
  --history-interval 25 --max-wall-s 18000 \
  --mura-work-budget-mode energy_limited_feasible_extents \
  --ordering-integration-method v40_stiff_dispatch \
  >"$V40_RUN_ROOT/logs/${condition}.out" \
  2>"$V40_RUN_ROOT/logs/${condition}.err"
