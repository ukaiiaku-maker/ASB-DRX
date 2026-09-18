#!/usr/bin/env bash
#SBATCH --account=SDILLON1
#SBATCH --partition=free
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/conduction-%A_%a.out
#SBATCH --error=logs/conduction-%A_%a.err

set -Eeuo pipefail
module load anaconda/2025.12
: "${V40_CONDUCTION_WRAPPER:?staged wrapper required}"
bash "$V40_CONDUCTION_WRAPPER"
