#!/usr/bin/env bash
#SBATCH --account=SDILLON1
#SBATCH --partition=free
#SBATCH --qos=low
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v31-asb-anchor-%A_%a.out
#SBATCH --error=logs/v31-asb-anchor-%A_%a.err

set -euo pipefail
module load anaconda/2025.12

source_root="${V31_SOURCE_ROOT:?point to the immutable submitted source tree}"
source_sha="${V31_SOURCE_SHA:?set the immutable submitted source SHA}"
run_root="${V31_ASB_RUN_ROOT:?point to the V31 ASB anchor run root}"
mkdir -p "$run_root" "$source_root/logs"

python "$source_root/full_model/hpc3/run_v31_asb_anchor_case.py" \
  --case-id "${SLURM_ARRAY_TASK_ID}" \
  --source-root "$source_root" \
  --expected-source-sha "$source_sha" \
  --run-root "$run_root" \
  --target-steps 5000
