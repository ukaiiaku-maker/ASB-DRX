#!/usr/bin/env bash
# Fetch and checksum the single V42 topology job after scheduler termination.
set -Eeuo pipefail

: "${V42_JOB_ID:?job id required}"
: "${V42_REMOTE_ROOT:?remote run root required}"
: "${V42_LOCAL_FETCH_ROOT:?local fetch root required}"

mkdir -p "$V42_LOCAL_FETCH_ROOT"
while ssh uci-hpc3 "squeue -h -j $V42_JOB_ID" | grep -q .; do
  sleep 60
done
state=$(ssh uci-hpc3 "sacct -j $V42_JOB_ID -X -n -P --format=State | head -n 1" | cut -d'|' -f1)
printf '%s\n' "$state" >"$V42_LOCAL_FETCH_ROOT/scheduler_state.txt.partial"
mv "$V42_LOCAL_FETCH_ROOT/scheduler_state.txt.partial" \
  "$V42_LOCAL_FETCH_ROOT/scheduler_state.txt"
if [[ "$state" != COMPLETED* ]]; then
  exit 2
fi

ssh uci-hpc3 "cd '$V42_REMOTE_ROOT' && tar -czf results.tar.gz mechanical_heterogeneity_topology_on_n64 logs && sha256sum results.tar.gz > results.tar.gz.sha256"
scp "uci-hpc3:$V42_REMOTE_ROOT/results.tar.gz" \
  "$V42_LOCAL_FETCH_ROOT/results.tar.gz.partial"
mv "$V42_LOCAL_FETCH_ROOT/results.tar.gz.partial" \
  "$V42_LOCAL_FETCH_ROOT/results.tar.gz"
scp "uci-hpc3:$V42_REMOTE_ROOT/results.tar.gz.sha256" \
  "$V42_LOCAL_FETCH_ROOT/results.tar.gz.sha256.remote"
expected=$(awk '{print $1}' "$V42_LOCAL_FETCH_ROOT/results.tar.gz.sha256.remote")
actual=$(shasum -a 256 "$V42_LOCAL_FETCH_ROOT/results.tar.gz" | awk '{print $1}')
test "$actual" = "$expected"
printf '%s  results.tar.gz\n' "$actual" \
  >"$V42_LOCAL_FETCH_ROOT/results.tar.gz.sha256.partial"
mv "$V42_LOCAL_FETCH_ROOT/results.tar.gz.sha256.partial" \
  "$V42_LOCAL_FETCH_ROOT/results.tar.gz.sha256"
mkdir -p "$V42_LOCAL_FETCH_ROOT/extracted"
tar -xzf "$V42_LOCAL_FETCH_ROOT/results.tar.gz" \
  -C "$V42_LOCAL_FETCH_ROOT/extracted"
printf '{"job_id":"%s","scheduler_state":"%s","archive_sha256":"%s","checksum_verified":true}\n' \
  "$V42_JOB_ID" "$state" "$actual" \
  >"$V42_LOCAL_FETCH_ROOT/fetch_status.json.partial"
mv "$V42_LOCAL_FETCH_ROOT/fetch_status.json.partial" \
  "$V42_LOCAL_FETCH_ROOT/fetch_status.json"
