#!/usr/bin/env bash
set -Eeuo pipefail

: "${V40_SOURCE_ARCHIVE:?immutable source archive required}"
: "${V40_SOURCE_SHA:?source identity required}"
: "${V40_SEED_ARCHIVE:?verified predecessor archive required}"
: "${V40_SEED_SHA256:?predecessor checksum required}"
: "${V40_RUN_ROOT:?persistent run root required}"

scratch_root="${TMPDIR:?}/v40-mura-continuation-${SLURM_JOB_ID}"
source_root="$scratch_root/source"
output_root="$scratch_root/output"
case_dir="$output_root/mechanical_heterogeneity_topology_off_n64"
mkdir -p "$source_root" "$output_root" "$V40_RUN_ROOT/results" \
  "$V40_RUN_ROOT/status" "$V40_RUN_ROOT/logs"
tar -xzf "$V40_SOURCE_ARCHIVE" -C "$source_root"
actual_seed=$(sha256sum "$V40_SEED_ARCHIVE" | awk '{print $1}')
test "$actual_seed" = "$V40_SEED_SHA256"
tar -xzf "$V40_SEED_ARCHIVE" -C "$output_root"
test -s "$case_dir/status.json"
export PYTHONPATH="$source_root:$source_root/src"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export V37_SOURCE_SHA="$V40_SOURCE_SHA"

finish() {
  code=$?
  partial="$V40_RUN_ROOT/results/mechanical_heterogeneity_to_5pct.tar.gz.partial"
  final="$V40_RUN_ROOT/results/mechanical_heterogeneity_to_5pct.tar.gz"
  tar -czf "$partial" -C "$output_root" .
  mv "$partial" "$final"
  sha256sum "$final" >"$V40_RUN_ROOT/results/mechanical_heterogeneity_to_5pct.sha256"
  tmp="$V40_RUN_ROOT/status/terminal.json.partial"
  printf '{"job_id":"%s","source_sha":"%s","seed_sha256":"%s","exit_code":%d}\n' \
    "$SLURM_JOB_ID" "$V40_SOURCE_SHA" "$V40_SEED_SHA256" "$code" >"$tmp"
  mv "$tmp" "$V40_RUN_ROOT/status/terminal.json"
  exit "$code"
}
trap finish EXIT

printf '{"predecessor_source_sha":"%s","predecessor_archive_sha256":"%s","continuation_source_sha":"%s","target_strain":0.05}\n' \
  "62e424f2eb01deced5e14125f4fcff26b869cb9f" "$V40_SEED_SHA256" \
  "$V40_SOURCE_SHA" >"$case_dir/v40_continuation_provenance.json"

python "$source_root/full_model/analysis/run_v30_mura_tier_b1_case.py" \
  --case-dir "$case_dir" --grid 64 --condition mechanical_heterogeneity \
  --seed 42 --length-m 1e-5 --temperature-K 1100 --strain-rate-s 10000 \
  --initial-strain .01 --target-strain .05 --trial-dt-s 2e-9 \
  --progress-checkpoint-strain .0025 --wall-checkpoint-s 840 \
  --history-interval 25 --max-wall-s 21600 \
  --mura-work-budget-mode energy_limited_feasible_extents \
  --ordering-integration-method v40_stiff_dispatch \
  >"$V40_RUN_ROOT/logs/continuation.out" \
  2>"$V40_RUN_ROOT/logs/continuation.err"
