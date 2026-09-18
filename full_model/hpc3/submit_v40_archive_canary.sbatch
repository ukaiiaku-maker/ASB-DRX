#!/usr/bin/env bash
#SBATCH --account=SDILLON1
#SBATCH --partition=free
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/canary-slurm-%j.out
#SBATCH --error=logs/canary-slurm-%j.err

set -Eeuo pipefail
module load anaconda/2025.12
: "${V40_CANARY_WRAPPER:?staged wrapper required}"
bash "$V40_CANARY_WRAPPER"
