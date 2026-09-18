#!/usr/bin/env bash
set -Eeuo pipefail

: "${V40_SOURCE_ARCHIVE:?immutable source archive required}"
: "${V40_SOURCE_SHA:?source identity required}"
: "${V40_RUN_ROOT:?persistent run root required}"
: "${SLURM_ARRAY_TASK_ID:?array task required}"

case_ids=(0 4 5)
case_id="${case_ids[$SLURM_ARRAY_TASK_ID]}"
scratch_root="${TMPDIR:?}/v40-conduction-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
source_root="$scratch_root/source"
output_root="$scratch_root/output"
mkdir -p "$source_root" "$output_root" "$V40_RUN_ROOT/results" \
  "$V40_RUN_ROOT/status" "$V40_RUN_ROOT/logs"
tar -xzf "$V40_SOURCE_ARCHIVE" -C "$source_root"
export PYTHONPATH="$source_root:$source_root/src"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HPC3_RUN_ID="${V40_RUN_ID:?}"
export HPC3_INPUT_DIR="$(dirname "$V40_SOURCE_ARCHIVE")"

finish() {
  code=$?
  if [[ -d "$output_root" ]]; then
    partial="$V40_RUN_ROOT/results/case-${case_id}.tar.gz.partial"
    final="$V40_RUN_ROOT/results/case-${case_id}.tar.gz"
    tar -czf "$partial" -C "$output_root" .
    mv "$partial" "$final"
    sha256sum "$final" >"$V40_RUN_ROOT/results/case-${case_id}.sha256"
  fi
  tmp="$V40_RUN_ROOT/status/task-${SLURM_ARRAY_TASK_ID}.json.partial"
  printf '{"array_job_id":"%s","task_id":%s,"case_id":%s,"source_sha":"%s","exit_code":%d}\n' \
    "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$case_id" \
    "$V40_SOURCE_SHA" "$code" >"$tmp"
  mv "$tmp" "$V40_RUN_ROOT/status/task-${SLURM_ARRAY_TASK_ID}.json"
  exit "$code"
}
trap finish EXIT

python "$source_root/full_model/hpc3/run_v37_conduction_case.py" \
  --case-id "$case_id" \
  --case-table "$source_root/full_model/hpc3/v37_conduction_cases.json" \
  --source-root "$source_root" --expected-source-sha "$V40_SOURCE_SHA" \
  --run-root "$output_root" --grid 128 --target-step 2500 \
  >"$V40_RUN_ROOT/logs/case-${case_id}.out" \
  2>"$V40_RUN_ROOT/logs/case-${case_id}.err"
