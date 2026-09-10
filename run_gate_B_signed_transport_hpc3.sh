#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLCONFIGDIR="${PWD}/.matplotlib"
export PYTHONPATH="${PWD}/src"
python --version > output/python_version.txt 2>&1
python -m unittest -v tests.test_signed_transport 2>&1 | tee output/unittest.txt
python tools/run_gate_B_signed_transport.py \
  --output output/gate_B_signed_transport.json \
  --plot-directory output/gate_B_plots
sha256sum \
  run_gate_B_signed_transport_hpc3.sh \
  src/asb_drx/bertin_bcc.py \
  src/asb_drx/signed_transport.py \
  tests/test_signed_transport.py \
  tools/run_gate_B_signed_transport.py > output/input_inventory.sha256
