#!/bin/bash
#SBATCH --job-name=asb-drx-nlmem-2efold
#SBATCH --account=SDILLON1
#SBATCH --partition=free
#SBATCH --qos=low
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -Eeuo pipefail
: "${HPC3_RUN_ID:?HPC3_RUN_ID must be exported by sbatch}"
: "${HPC3_RUN_DIR:?HPC3_RUN_DIR must be exported by sbatch}"
: "${SOURCE_COMMIT:?SOURCE_COMMIT must be exported by sbatch}"
mkdir -p "$HPC3_RUN_DIR/logs" "$HPC3_RUN_DIR/results/live" "$HPC3_RUN_DIR/status"
module load anaconda/2025.12
: "${TMPDIR:?Slurm must provide TMPDIR}"
work_dir="$TMPDIR/$HPC3_RUN_ID"
mkdir -p "$work_dir"
tar -xzf "$HPC3_RUN_DIR/input/source.tar.gz" -C "$work_dir"
cd "$work_dir"
export HPC3_LIVE_OUTPUT_DIR="$HPC3_RUN_DIR/results/live"
export PYTHONHASHSEED=0
complete=false
finalize() {
  exit_code=$?
  if [ "$exit_code" -eq 0 ]; then complete=true; fi
  printf '{"run_id":"%s","job_id":"%s","application_exit":%d,"complete":%s}\n' \
    "$HPC3_RUN_ID" "${SLURM_JOB_ID:-unknown}" "$exit_code" "$complete" \
    > "$HPC3_RUN_DIR/status/final.json"
}
trap finalize EXIT
bash run_v3_nonlinear_memory_pilot_hpc3.sh
