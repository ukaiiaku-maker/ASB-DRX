#!/usr/bin/env bash
set -Eeuo pipefail

mkdir -p output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="${PWD}/src"

python --version > output/python_version.txt 2>&1
python -c 'import numpy; print(numpy.__version__)' > output/numpy_version.txt
python -m unittest -v tests.test_thermodynamics tests.test_phase_field_2d 2>&1 | tee output/unittest.txt
python tools/run_thermodynamic_verification.py
sha256sum \
  run_thermodynamic_hpc3.sh \
  src/asb_drx/thermodynamics.py \
  tests/test_thermodynamics.py \
  tests/test_phase_field_2d.py \
  tools/run_thermodynamic_verification.py \
  > output/input_inventory.sha256
