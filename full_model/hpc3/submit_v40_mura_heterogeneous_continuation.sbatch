#!/usr/bin/env bash
#SBATCH --account=SDILLON1
#SBATCH --partition=free
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=07:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/mura-continuation-%j.out
#SBATCH --error=logs/mura-continuation-%j.err

set -Eeuo pipefail
module load anaconda/2025.12
: "${V40_MURA_CONTINUATION_WRAPPER:?staged wrapper required}"
bash "$V40_MURA_CONTINUATION_WRAPPER"
