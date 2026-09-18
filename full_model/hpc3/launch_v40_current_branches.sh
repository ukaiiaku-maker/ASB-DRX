#!/usr/bin/env bash
# Idempotent local launcher. A remote receipt is committed before this returns,
# so reconnects cannot duplicate either Slurm submission.
set -Eeuo pipefail

source_root="${V40_LOCAL_SOURCE_ROOT:?local clean worktree required}"
source_sha="${V40_SOURCE_SHA:-$(git -C "$source_root" rev-parse HEAD)}"
run_id="v40-current-20260917-${source_sha:0:7}"
remote_root="/pub/sdillon1/codex-runs/asb-drx-full-v34-recovery/$run_id"
local_stage="/tmp/$run_id"
receipt_root="${V40_LOCAL_RECEIPT_ROOT:-/Users/sdillon/HPC3/local-results/asb-drx-v40-current/$run_id}"
mkdir -p "$local_stage" "$receipt_root"
git -C "$source_root" diff --quiet
git -C "$source_root" diff --cached --quiet
git -C "$source_root" archive --format=tar.gz \
  --output="$local_stage/source.tar.gz" "$source_sha"
shasum -a 256 "$local_stage/source.tar.gz" >"$local_stage/source.tar.gz.sha256"

remote="ssh -o ConnectTimeout=10 -o ConnectionAttempts=1 uci-hpc3"
$remote "timeout 30 mkdir -p '$remote_root'/control '$remote_root'/logs \
  '$remote_root'/results/conduction '$remote_root'/results/mura \
  '$remote_root'/status/conduction '$remote_root'/status/mura"
scp -o ConnectTimeout=10 -o ConnectionAttempts=1 \
  "$local_stage/source.tar.gz" "$local_stage/source.tar.gz.sha256" \
  "$source_root/full_model/hpc3/run_v40_conduction_selected.sh" \
  "$source_root/full_model/hpc3/submit_v40_conduction_selected.sbatch" \
  "$source_root/full_model/hpc3/run_v40_mura_organization_selected.sh" \
  "$source_root/full_model/hpc3/submit_v40_mura_organization_selected.sbatch" \
  "uci-hpc3:$remote_root/control/"

$remote "timeout 45 bash -s" -- "$remote_root" "$source_sha" "$run_id" <<'REMOTE'
set -Eeuo pipefail
root="$1"; sha="$2"; run_id="$3"
cd "$root"
if [[ ! -s status/conduction/submission.txt ]]; then
  job=$(sbatch --parsable --job-name="v40-cond-${sha:0:7}" \
    --export=ALL,V40_SOURCE_ARCHIVE="$root/control/source.tar.gz",V40_SOURCE_SHA="$sha",V40_RUN_ROOT="$root/results/conduction",V40_RUN_ID="$run_id-conduction",V40_CONDUCTION_WRAPPER="$root/control/run_v40_conduction_selected.sh" \
    control/submit_v40_conduction_selected.sbatch)
  printf '%s\n' "$job" >status/conduction/submission.txt.partial
  mv status/conduction/submission.txt.partial status/conduction/submission.txt
fi
if [[ ! -s status/mura/submission.txt ]]; then
  job=$(sbatch --parsable --job-name="v40-mura-${sha:0:7}" \
    --export=ALL,V40_SOURCE_ARCHIVE="$root/control/source.tar.gz",V40_SOURCE_SHA="$sha",V40_RUN_ROOT="$root/results/mura",V40_RUN_ID="$run_id-mura",V40_MURA_WRAPPER="$root/control/run_v40_mura_organization_selected.sh" \
    control/submit_v40_mura_organization_selected.sbatch)
  printf '%s\n' "$job" >status/mura/submission.txt.partial
  mv status/mura/submission.txt.partial status/mura/submission.txt
fi
REMOTE

scp -o ConnectTimeout=10 -o ConnectionAttempts=1 \
  "uci-hpc3:$remote_root/status/conduction/submission.txt" \
  "$receipt_root/conduction-submission.txt"
scp -o ConnectTimeout=10 -o ConnectionAttempts=1 \
  "uci-hpc3:$remote_root/status/mura/submission.txt" \
  "$receipt_root/mura-submission.txt"
printf 'run_id=%s\nsource_sha=%s\nremote_root=%s\n' \
  "$run_id" "$source_sha" "$remote_root"
