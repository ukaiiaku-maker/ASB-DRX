#!/usr/bin/env bash
set -Eeuo pipefail

: "${V40_SOURCE_ARCHIVE:?immutable source archive required}"
: "${V40_SOURCE_SHA:?source identity required}"
: "${V40_RUN_ROOT:?persistent run root required}"

scratch_root="${TMPDIR:?}/v40-canary-${SLURM_JOB_ID:?}"
source_root="$scratch_root/source"
output_root="$scratch_root/output"
mkdir -p "$source_root" "$output_root" "$V40_RUN_ROOT/results" \
  "$V40_RUN_ROOT/status" "$V40_RUN_ROOT/logs"
tar -xzf "$V40_SOURCE_ARCHIVE" -C "$source_root"
export PYTHONPATH="$source_root:$source_root/src"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

finish() {
  code=$?
  partial="$V40_RUN_ROOT/results/canary.tar.gz.partial"
  final="$V40_RUN_ROOT/results/canary.tar.gz"
  if [[ -d "$output_root" ]]; then
    tar -czf "$partial" -C "$scratch_root" output
    mv "$partial" "$final"
    sha256sum "$final" >"$V40_RUN_ROOT/results/canary.sha256"
  fi
  tmp="$V40_RUN_ROOT/status/final.json.partial"
  printf '{"job_id":"%s","source_sha":"%s","exit_code":%d}\n' \
    "$SLURM_JOB_ID" "$V40_SOURCE_SHA" "$code" >"$tmp"
  mv "$tmp" "$V40_RUN_ROOT/status/final.json"
  exit "$code"
}
trap finish EXIT

python "$source_root/full_model/analysis/run_v40_archive_canary.py" \
  --output-dir "$output_root" \
  >"$V40_RUN_ROOT/logs/canary.out" \
  2>"$V40_RUN_ROOT/logs/canary.err"
