#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
python3 smoke.py
