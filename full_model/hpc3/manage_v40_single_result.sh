#!/usr/bin/env bash
# Durable local monitor/fetcher for one V40 Slurm job.
set -Eeuo pipefail

: "${V40_JOB_ID:?Slurm job id required}"
: "${V40_REMOTE_ROOT:?remote result root required}"
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
for stem in results status logs; do
  scp -r -o ConnectTimeout=10 -o ConnectionAttempts=1 \
    "uci-hpc3:$V40_REMOTE_ROOT/$stem" "$V40_LOCAL_RESULT_ROOT/"
done
ssh -o ConnectTimeout=10 -o ConnectionAttempts=1 uci-hpc3 \
  "sacct -j '$V40_JOB_ID' --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS -P" \
  >"$V40_LOCAL_RESULT_ROOT/sacct.txt"
status=0
while IFS= read -r checksum_file; do
  expected=$(awk '{print $1}' "$checksum_file")
  archive="${checksum_file%.sha256}.tar.gz"
  actual=$(shasum -a 256 "$archive" | awk '{print $1}')
  printf '%s expected=%s actual=%s\n' "$archive" "$expected" "$actual"
  [[ "$expected" = "$actual" ]] || status=1
done < <(find "$V40_LOCAL_RESULT_ROOT" -type f -name '*.sha256' | sort)
exit "$status"
