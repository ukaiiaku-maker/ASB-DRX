#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="${PWD}/src"

python --version > output/python_version.txt 2>&1
python -m unittest -v tests.test_grains 2>&1 | tee output/unittest.txt
python tools/run_grain_metric_verification.py
sha256sum \
  run_grain_metric_hpc3.sh \
  src/asb_drx/grains.py \
  tests/test_grains.py \
  tools/run_grain_metric_verification.py \
  > output/input_inventory.sha256
