#!/bin/bash
set -euo pipefail

mkdir -p output
export PYTHONPATH=src
python -m pytest -q \
  tests/test_arrhenius_v3.py \
  tests/test_vector_topology_cdd_v3.py \
  tests/test_nonlinear_memory_cdd_v3.py \
  | tee output/nonlinear_memory_pilot_tests.txt
python tools/run_v3_nonlinear_memory_hold.py \
  --case fast \
  --cells 128 \
  --wavelengths 8 \
  --target-efolds 2 \
  --chunk-s 0.001 \
  --checkpoint output/nonlinear_memory_fast_pilot_checkpoint.npz \
  --output output/nonlinear_memory_fast_pilot.json
