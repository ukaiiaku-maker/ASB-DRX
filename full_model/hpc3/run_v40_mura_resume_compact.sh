#!/usr/bin/env bash
# Resume a one-grain continuation while publishing compact restart snapshots.
set -Eeuo pipefail

: "${V40_SOURCE_ARCHIVE:?immutable source archive required}"
: "${V40_SOURCE_SHA:?source identity required}"
: "${V40_SEED_ARCHIVE:?verified predecessor archive required}"
: "${V40_SEED_SHA256:?predecessor checksum required}"
: "${V40_RUN_ROOT:?persistent run root required}"

scratch_root="${TMPDIR:?}/v40-mura-resume-${SLURM_JOB_ID}"
source_root="$scratch_root/source"
output_root="$scratch_root/output"
case_dir="$output_root/mechanical_heterogeneity_topology_off_n64"
mkdir -p "$source_root" "$output_root" "$V40_RUN_ROOT/checkpoints" \
  "$V40_RUN_ROOT/status" "$V40_RUN_ROOT/logs"
tar -xzf "$V40_SOURCE_ARCHIVE" -C "$source_root"
actual_seed=$(sha256sum "$V40_SEED_ARCHIVE" | awk '{print $1}')
test "$actual_seed" = "$V40_SEED_SHA256"
tar -xzf "$V40_SEED_ARCHIVE" -C "$output_root"
test -s "$case_dir/status.json"

export PYTHONPATH="$source_root:$source_root/src"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export V37_SOURCE_SHA="$V40_SOURCE_SHA"

snapshot() {
  lock="$V40_RUN_ROOT/status/snapshot.lock"
  if ! mkdir "$lock" 2>/dev/null; then
    return 0
  fi
  latest=$(find "$case_dir" -maxdepth 1 -type f \
    -name 'checkpoint_step_*.npz' | sort | tail -1)
  if [[ -s "$latest" ]]; then
    stage=$(mktemp -d "$scratch_root/snapshot.XXXXXX")
    compact="$stage/mechanical_heterogeneity_topology_off_n64"
    mkdir -p "$compact"
    for name in case_config.json history.jsonl status.json \
      v40_continuation_provenance.json; do
      test ! -f "$case_dir/$name" || cp "$case_dir/$name" "$compact/$name"
    done
    cp "$latest" "$compact/"
    archive="$V40_RUN_ROOT/checkpoints/latest.tar.gz"
    tar -czf "$archive.partial" -C "$stage" .
    mv "$archive.partial" "$archive"
    sha256sum "$archive" >"$V40_RUN_ROOT/checkpoints/latest.sha256.partial"
    mv "$V40_RUN_ROOT/checkpoints/latest.sha256.partial" \
      "$V40_RUN_ROOT/checkpoints/latest.sha256"
    printf '{"job_id":"%s","source_sha":"%s","checkpoint":"%s","utc":"%s"}\n' \
      "$SLURM_JOB_ID" "$V40_SOURCE_SHA" "$(basename "$latest")" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >"$V40_RUN_ROOT/status/latest_snapshot.json.partial"
    mv "$V40_RUN_ROOT/status/latest_snapshot.json.partial" \
      "$V40_RUN_ROOT/status/latest_snapshot.json"
  fi
  rmdir "$lock"
}

solver_pid=""
monitor_pid=""
stop_for_signal() {
  snapshot
  test -z "$solver_pid" || kill -TERM "$solver_pid" 2>/dev/null || true
  test -z "$solver_pid" || wait "$solver_pid" 2>/dev/null || true
  exit 99
}
trap stop_for_signal USR1 TERM

snapshot
python "$source_root/full_model/analysis/run_v30_mura_tier_b1_case.py" \
  --case-dir "$case_dir" --grid 64 --condition mechanical_heterogeneity \
  --seed 42 --length-m 1e-5 --temperature-K 1100 --strain-rate-s 10000 \
  --initial-strain .01 --target-strain .05 --trial-dt-s 2e-9 \
  --progress-checkpoint-strain .0025 --wall-checkpoint-s 840 \
  --history-interval 25 --max-wall-s 21600 \
  --mura-work-budget-mode energy_limited_feasible_extents \
  --ordering-integration-method v40_stiff_dispatch \
  >"$V40_RUN_ROOT/logs/continuation.out" \
  2>"$V40_RUN_ROOT/logs/continuation.err" &
solver_pid=$!
(
  while kill -0 "$solver_pid" 2>/dev/null; do
    sleep 600
    snapshot
  done
) &
monitor_pid=$!
set +e
wait "$solver_pid"
code=$?
set -e
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
snapshot
printf '{"job_id":"%s","source_sha":"%s","exit_code":%d}\n' \
  "$SLURM_JOB_ID" "$V40_SOURCE_SHA" "$code" \
  >"$V40_RUN_ROOT/status/terminal.json.partial"
mv "$V40_RUN_ROOT/status/terminal.json.partial" \
  "$V40_RUN_ROOT/status/terminal.json"
exit "$code"
