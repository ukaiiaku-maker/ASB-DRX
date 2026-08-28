#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="${PWD}/src"

python --version > output/python_version.txt 2>&1
python -m unittest -v tests.test_exp_floor tests.test_material_point tests.test_shear_layer 2>&1 | tee output/unittest.txt
python tools/run_shear_layer_verification.py
sha256sum \
  run_shear_layer_hpc3.sh \
  src/asb_drx/analytical.py \
  src/asb_drx/thermodynamics.py \
  src/asb_drx/material_point.py \
  src/asb_drx/shear_layer.py \
  tests/test_shear_layer.py \
  tools/run_shear_layer_verification.py \
  > output/input_inventory.sha256
