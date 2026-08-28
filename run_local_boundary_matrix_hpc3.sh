#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="${PWD}/src"
python --version > output/python_version.txt 2>&1
python tools/run_local_boundary_matrix.py \
  --points 16 --steps 1000 --target-shear 0.9 \
  --source-commit 9b8f62de0e7b5f00f6d1040a756baa7fb9ac3de7 \
  --execution-site hpc3 \
  --output output/local_boundary_matrix.json
sha256sum \
  run_local_boundary_matrix_hpc3.sh \
  src/asb_drx/antiplane.py src/asb_drx/local_coupled.py \
  src/asb_drx/local_mechanism.py tools/run_local_boundary_matrix.py \
  > output/input_inventory.sha256
