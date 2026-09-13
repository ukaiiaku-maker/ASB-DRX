#!/usr/bin/env bash
#SBATCH --account=SDILLON1
#SBATCH --partition=free
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
module load anaconda/2025.12
mkdir -p logs status output
start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

finish() {
  code=$?
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"case":"%s","job_id":"%s","start_utc":"%s","end_utc":"%s","exit_code":%d}\n' \
    "${CASE_NAME:-unset}" "${SLURM_JOB_ID:-unset}" "$start_utc" "$end_utc" "$code" \
    > "status/final-${CASE_NAME:-unset}.json"
}
trap finish EXIT

bash run_full_v34_case.sh
