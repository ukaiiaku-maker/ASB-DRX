#!/bin/bash
set -Eeuo pipefail

mkdir -p output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="${PWD}/src"

python --version > output/python_version.txt 2>&1
python -c 'import scipy; print(scipy.__version__)' > output/scipy_version.txt
python -m unittest -v tests.test_exp_floor tests.test_peak_optimization 2>&1 | tee output/unittest.txt
sha256sum run_peak_optimization_hpc3.sh src/asb_drx/analytical.py src/asb_drx/optimization.py tests/test_exp_floor.py tests/test_peak_optimization.py > output/input_inventory.sha256
