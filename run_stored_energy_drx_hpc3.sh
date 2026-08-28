#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="${PWD}/src"

python --version > output/python_version.txt 2>&1
python -m unittest -v tests.test_multi_order tests.test_stored_energy_drx 2>&1 | tee output/unittest.txt
python tools/run_stored_energy_drx_verification.py
sha256sum \
  run_stored_energy_drx_hpc3.sh \
  src/asb_drx/multi_order.py \
  src/asb_drx/stored_energy_drx.py \
  tests/test_multi_order.py \
  tests/test_stored_energy_drx.py \
  tools/run_stored_energy_drx_verification.py \
  > output/input_inventory.sha256
