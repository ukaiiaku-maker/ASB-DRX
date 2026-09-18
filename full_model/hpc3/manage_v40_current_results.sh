#!/usr/bin/env bash
# Durable local monitor/fetcher for the two V40 arrays.
set -Eeuo pipefail

: "${V40_REMOTE_ROOT:?remote V40 root required}"
: "${V40_LOCAL_RESULT_ROOT:?local durable result root required}"
: "${V40_CONDUCTION_JOB_ID:?thermal array id required}"
: "${V40_MURA_JOB_ID:?organization array id required}"
poll_s="${V40_POLL_SECONDS:-300}"
mkdir -p "$V40_LOCAL_RESULT_ROOT"

while true; do
  stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  active=$(ssh -o ConnectTimeout=10 -o ConnectionAttempts=1 uci-hpc3 \
    "squeue -h -j '$V40_CONDUCTION_JOB_ID,$V40_MURA_JOB_ID' -o '%i %T'" || true)
  printf '%s active=%s\n' "$stamp" "${active:-none}"
  [[ -z "$active" ]] && break
  sleep "$poll_s"
done

scp -r -o ConnectTimeout=10 -o ConnectionAttempts=1 \
  "uci-hpc3:$V40_REMOTE_ROOT/results/conduction" \
  "$V40_LOCAL_RESULT_ROOT/"
scp -r -o ConnectTimeout=10 -o ConnectionAttempts=1 \
  "uci-hpc3:$V40_REMOTE_ROOT/results/mura" \
  "$V40_LOCAL_RESULT_ROOT/"
ssh -o ConnectTimeout=10 -o ConnectionAttempts=1 uci-hpc3 \
  "sacct -j '$V40_CONDUCTION_JOB_ID,$V40_MURA_JOB_ID' --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS -P" \
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
