#!/usr/bin/env bash
#SBATCH --account=SDILLON1
#SBATCH --partition=free
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --array=0-1%1
#SBATCH --output=logs/mura-%A_%a.out
#SBATCH --error=logs/mura-%A_%a.err

set -Eeuo pipefail
module load anaconda/2025.12
: "${V40_MURA_WRAPPER:?staged wrapper required}"
bash "$V40_MURA_WRAPPER"
