#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="${PWD}/src"
python --version > output/python_version.txt 2>&1
python -m unittest -v tests.test_bertin_bcc 2>&1 | tee output/unittest.txt
python tools/run_gate_A_bertin.py --output output/gate_A_bertin_bcc.json
sha256sum \
  run_gate_A_bertin_hpc3.sh \
  src/asb_drx/bertin_bcc.py \
  tests/test_bertin_bcc.py \
  tools/run_gate_A_bertin.py > output/input_inventory.sha256
