#!/usr/bin/env bash
#SBATCH --account=SDILLON1
#SBATCH --partition=free
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/v37-mura-organization-%j.out
#SBATCH --error=logs/v37-mura-organization-%j.err

set -euo pipefail
module load anaconda/2025.12
: "${V37_SOURCE_ROOT:?}"
: "${V37_SOURCE_SHA:?}"
: "${V37_MURA_RUN_ROOT:?}"
mkdir -p "$V37_SOURCE_ROOT/logs" "$V37_MURA_RUN_ROOT"
bash "$V37_SOURCE_ROOT/full_model/hpc3/run_v37_mura_organization_long.sh"
