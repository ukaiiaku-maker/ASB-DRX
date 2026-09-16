#!/usr/bin/env bash
#SBATCH --account=SDILLON1
#SBATCH --partition=free
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=12-23%4
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm-v30-front-a1-delta-%A_%a.out
#SBATCH --error=slurm-v30-front-a1-delta-%A_%a.err

set -euo pipefail
module load anaconda/2025.12
: "${HPC3_WORK_DIR:?HPC3_WORK_DIR must name the staged source root}"
: "${V30_EXPECTED_SOURCE_SHA:?export the clean pushed source SHA at submission}"
: "${V30_EXPECTED_REMOTE_SHA:?export the verified remote branch SHA at submission}"
cd "$HPC3_WORK_DIR"
python full_model/hpc3/run_v30_front_case.py \
  --case-index "${SLURM_ARRAY_TASK_ID}"
