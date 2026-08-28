#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="${PWD}/src"
python --version > output/python_version.txt 2>&1
python -m unittest -v tests.test_shear_layer tests.test_coupled tests.test_spatial_coupled 2>&1 | tee output/unittest.txt
python tools/run_spatial_coupled_verification.py
sha256sum run_spatial_coupled_hpc3.sh src/asb_drx/spatial_coupled.py tests/test_spatial_coupled.py tools/run_spatial_coupled_verification.py > output/input_inventory.sha256
