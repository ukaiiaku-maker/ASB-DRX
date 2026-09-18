#!/usr/bin/env bash
# Durable fetcher for a compact-snapshot V40 continuation.
set -Eeuo pipefail

: "${V40_JOB_ID:?Slurm job id required}"
: "${V40_REMOTE_ROOT:?remote compact run root required}"
: "${V40_LOCAL_RESULT_ROOT:?local destination required}"
poll_s="${V40_POLL_SECONDS:-300}"
mkdir -p "$V40_LOCAL_RESULT_ROOT"
while true; do
  stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  active=$(ssh -o ConnectTimeout=10 -o ConnectionAttempts=1 uci-hpc3 \
    "squeue -h -j '$V40_JOB_ID' -o '%i %T'" || true)
  printf '%s active=%s\n' "$stamp" "${active:-none}"
  [[ -z "$active" ]] && break
  sleep "$poll_s"
done
for stem in checkpoints status logs; do
  scp -r -o ConnectTimeout=10 -o ConnectionAttempts=1 \
    "uci-hpc3:$V40_REMOTE_ROOT/$stem" "$V40_LOCAL_RESULT_ROOT/"
done
ssh -o ConnectTimeout=10 -o ConnectionAttempts=1 uci-hpc3 \
  "sacct -j '$V40_JOB_ID' --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS -P" \
  >"$V40_LOCAL_RESULT_ROOT/sacct.txt"
expected=$(awk '{print $1}' "$V40_LOCAL_RESULT_ROOT/checkpoints/latest.sha256")
actual=$(shasum -a 256 "$V40_LOCAL_RESULT_ROOT/checkpoints/latest.tar.gz" | \
  awk '{print $1}')
printf 'archive=%s expected=%s actual=%s\n' \
  "$V40_LOCAL_RESULT_ROOT/checkpoints/latest.tar.gz" "$expected" "$actual"
test "$expected" = "$actual"
