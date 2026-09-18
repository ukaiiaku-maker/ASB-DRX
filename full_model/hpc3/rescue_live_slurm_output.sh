#!/usr/bin/env bash
# Snapshot a running job's node-local output into its persistent HPC3 run record.
# This is a bounded local monitor: all filesystem work runs inside the allocation.
set -Eeuo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 JOB_ID RUN_ID SCRATCH_RELATIVE_OUTPUT REMOTE_RUN_DIR" >&2
  exit 64
fi

job_id=$1
run_id=$2
scratch_relative_output=$3
remote_run_dir=$4
login_host=${HPC3_LOGIN_HOST:-uci-hpc3}
poll_seconds=${HPC3_RESCUE_POLL_SECONDS:-1200}

while true; do
  state=$(ssh -o BatchMode=yes "$login_host" \
    "squeue -h -j '$job_id' -o %T | head -n 1" || true)
  [[ "$state" == "RUNNING" || "$state" == "COMPLETING" ]] || break

  if ! ssh -o BatchMode=yes "$login_host" \
    "srun --jobid='$job_id' --overlap --ntasks=1 bash -s -- '$run_id' '$scratch_relative_output' '$remote_run_dir'" <<'REMOTE'
set -Eeuo pipefail
run_id=$1
scratch_relative_output=$2
remote_run_dir=$3
source_dir="${TMPDIR:?}/hpc3-${run_id}-single/${scratch_relative_output}"
[[ -d "$source_dir" ]] || { echo "rescue source missing: $source_dir" >&2; exit 2; }
mkdir -p "$remote_run_dir/results"
partial="$remote_run_dir/results/live-rescue-single.tar.gz.partial"
final="$remote_run_dir/results/live-rescue-single.tar.gz"
tar -czf "$partial" -C "$(dirname "$source_dir")" "$(basename "$source_dir")"
mv "$partial" "$final"
if command -v sha256sum >/dev/null; then
  sha256sum "$final" >"$remote_run_dir/results/live-rescue-single.sha256"
else
  shasum -a 256 "$final" >"$remote_run_dir/results/live-rescue-single.sha256"
fi
REMOTE
  then
    # A live history file can change while tar is reading it.  That is an
    # expected snapshot race, not a reason to abandon all future rescues.
    printf 'rescue attempt failed at %s; retaining previous verified archive\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
  fi
  sleep "$poll_seconds"
done
