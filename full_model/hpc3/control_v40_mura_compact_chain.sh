#!/usr/bin/env bash
# Bounded preemption-resume controller for V40 compact Mura continuations.
set -Eeuo pipefail

: "${V40_INITIAL_JOB_ID:?initial retry job required}"
: "${V40_INITIAL_REMOTE_ROOT:?initial remote run root required}"
: "${V40_BASE_REMOTE_ROOT:?remote root prefix required}"
: "${V40_LOCAL_CHAIN_ROOT:?local chain destination required}"
: "${V40_SOURCE_ARCHIVE:?immutable source archive required}"
: "${V40_SOURCE_SHA:?source identity required}"
: "${V40_RESUME_WRAPPER:?remote wrapper required}"
: "${V40_SUBMIT_SCRIPT:?remote sbatch script required}"
: "${V40_HOMOGENEOUS_CASE:?homogeneous control required}"
: "${V40_ANALYSIS_SOURCE:?analysis source required}"
max_attempts="${V40_MAX_ATTEMPTS:-2}"
poll_s="${V40_POLL_SECONDS:-300}"
mkdir -p "$V40_LOCAL_CHAIN_ROOT"

attempt=1
job="$V40_INITIAL_JOB_ID"
remote_root="$V40_INITIAL_REMOTE_ROOT"
while true; do
  while true; do
    active=$(ssh -o ConnectTimeout=10 -o ConnectionAttempts=1 uci-hpc3 \
      "squeue -h -j '$job' -o '%i %T'" || true)
    printf '%s attempt=%d job=%s active=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$attempt" "$job" "${active:-none}"
    [[ -z "$active" ]] && break
    sleep "$poll_s"
  done
  snapshot_json=$(ssh uci-hpc3 "cat '$remote_root/status/latest_snapshot.json'")
  checkpoint=$(printf '%s\n' "$snapshot_json" | \
    sed -E 's/.*"checkpoint":"([^"]+)".*/\1/')
  strain=$(printf '%s\n' "$checkpoint" | \
    sed -E 's/.*_strain_([0-9.]+)\.npz/\1/')
  printf '%s attempt=%d job=%s checkpoint=%s strain=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$attempt" "$job" \
    "$checkpoint" "$strain" | tee -a "$V40_LOCAL_CHAIN_ROOT/attempts.log"
  complete=$(awk -v value="$strain" 'BEGIN {print(value >= 0.05 ? 1 : 0)}')
  if [[ "$complete" = 1 || "$attempt" -ge "$max_attempts" ]]; then
    break
  fi
  next=$((attempt+1))
  next_root="${V40_BASE_REMOTE_ROOT}-attempt${next}"
  seed="$remote_root/checkpoints/latest.tar.gz"
  seed_sha=$(ssh uci-hpc3 "awk '{print \$1}' '$remote_root/checkpoints/latest.sha256'")
  ssh uci-hpc3 "mkdir -p '$next_root'/logs '$next_root'/status '$next_root'/checkpoints"
  job=$(ssh uci-hpc3 "cd '$next_root' && sbatch --parsable \
    --job-name='v40-mura-chain${next}-${V40_SOURCE_SHA:0:7}' \
    --export=ALL,V40_SOURCE_ARCHIVE='$V40_SOURCE_ARCHIVE',V40_SOURCE_SHA='$V40_SOURCE_SHA',V40_SEED_ARCHIVE='$seed',V40_SEED_SHA256='$seed_sha',V40_RUN_ROOT='$next_root',V40_MURA_RESUME_WRAPPER='$V40_RESUME_WRAPPER' \
    '$V40_SUBMIT_SCRIPT'")
  printf '%s submitted_attempt=%d job=%s seed_sha256=%s remote_root=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$next" "$job" "$seed_sha" \
    "$next_root" | tee -a "$V40_LOCAL_CHAIN_ROOT/attempts.log"
  attempt="$next"
  remote_root="$next_root"
done

printf '%s\n' "$remote_root" >"$V40_LOCAL_CHAIN_ROOT/final_remote_root.txt"
mkdir -p "$V40_LOCAL_CHAIN_ROOT/final"
for stem in checkpoints status logs; do
  scp -r "uci-hpc3:$remote_root/$stem" "$V40_LOCAL_CHAIN_ROOT/final/"
done
expected=$(awk '{print $1}' "$V40_LOCAL_CHAIN_ROOT/final/checkpoints/latest.sha256")
actual=$(shasum -a 256 "$V40_LOCAL_CHAIN_ROOT/final/checkpoints/latest.tar.gz" | \
  awk '{print $1}')
test "$expected" = "$actual"
V40_FETCH_ROOT="$V40_LOCAL_CHAIN_ROOT/final" \
V40_ARCHIVE_PATH="$V40_LOCAL_CHAIN_ROOT/final/checkpoints/latest.tar.gz" \
V40_CHECKSUM_PATH="$V40_LOCAL_CHAIN_ROOT/final/checkpoints/latest.sha256" \
V40_HOMOGENEOUS_CASE="$V40_HOMOGENEOUS_CASE" \
V40_ANALYSIS_SOURCE="$V40_ANALYSIS_SOURCE" \
bash "$V40_ANALYSIS_SOURCE/full_model/hpc3/finalize_v40_mura_continuation.sh"
