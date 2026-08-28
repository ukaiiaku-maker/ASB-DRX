#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="${PWD}/src"
python --version > output/python_version.txt 2>&1
python -m unittest -v tests.test_exp_floor tests.test_ddd_fixture tests.test_analytical_boundary 2>&1 | tee output/unittest.txt
python tools/run_analytical_boundary_verification.py
sha256sum run_analytical_boundary_hpc3.sh src/asb_drx/analytical.py src/asb_drx/fixtures.py src/asb_drx/boundary.py tests/test_analytical_boundary.py tools/run_analytical_boundary_verification.py > output/input_inventory.sha256
