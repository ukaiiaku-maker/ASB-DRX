#!/usr/bin/env bash
#SBATCH --account=SDILLON1
#SBATCH --partition=free
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --nodelist=hpc3-14-02
#SBATCH --output=logs/mura-rescue-%j.out
#SBATCH --error=logs/mura-rescue-%j.err

set -Eeuo pipefail
: "${V40_RESCUE_SOURCE:?preempted scratch case required}"
: "${V40_RESCUE_ROOT:?persistent rescue destination required}"
test -d "$V40_RESCUE_SOURCE"
mkdir -p "$V40_RESCUE_ROOT/results" "$V40_RESCUE_ROOT/status"
latest=$(find "$V40_RESCUE_SOURCE" -maxdepth 1 -type f \
  -name 'checkpoint_step_*.npz' | sort | tail -1)
test -s "$latest"
stage=$(mktemp -d "${TMPDIR:?}/v40-mura-rescue.XXXXXX")
case_dir="$stage/mechanical_heterogeneity_topology_off_n64"
mkdir -p "$case_dir"
for name in case_config.json history.jsonl status.json \
  v40_continuation_provenance.json; do
  test ! -f "$V40_RESCUE_SOURCE/$name" || \
    cp "$V40_RESCUE_SOURCE/$name" "$case_dir/$name"
done
cp "$latest" "$case_dir/"
archive="$V40_RESCUE_ROOT/results/preempted_checkpoint_rescue.tar.gz"
tar -czf "$archive.partial" -C "$stage" .
mv "$archive.partial" "$archive"
sha256sum "$archive" >"$V40_RESCUE_ROOT/results/preempted_checkpoint_rescue.sha256"
printf '{"rescued_job_id":"56147765","rescue_job_id":"%s","node":"%s","checkpoint":"%s"}\n' \
  "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$(basename "$latest")" \
  >"$V40_RESCUE_ROOT/status/rescue.json.partial"
mv "$V40_RESCUE_ROOT/status/rescue.json.partial" \
  "$V40_RESCUE_ROOT/status/rescue.json"
