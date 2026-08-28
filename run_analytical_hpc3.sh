#!/usr/bin/env bash
set -euo pipefail

mkdir -p output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="${PWD}/src"

python --version > output/python_version.txt 2>&1
python -c 'import scipy; print(scipy.__version__)' > output/scipy_version.txt
python -m unittest -v tests/test_exp_floor.py 2>&1 | tee output/unittest.txt

sha256sum \
  src/asb_drx/__init__.py \
  src/asb_drx/analytical.py \
  tests/test_exp_floor.py \
  > output/input_inventory.sha256
