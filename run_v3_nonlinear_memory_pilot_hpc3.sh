#!/bin/bash
set -euo pipefail

output_dir="${HPC3_LIVE_OUTPUT_DIR:-output}"
target_efolds="${TARGET_EFOLDS:-2}"
seed_type="${SEED_TYPE:-eigenvector}"
mkdir -p "$output_dir"
export PYTHONPATH=src
python -m pytest -q \
  tests/test_arrhenius_v3.py \
  tests/test_vector_topology_cdd_v3.py \
  tests/test_nonlinear_memory_cdd_v3.py \
  | tee "$output_dir/nonlinear_memory_pilot_tests.txt"
python tools/run_v3_nonlinear_memory_hold.py \
  --case fast \
  --cells 128 \
  --wavelengths 8 \
  --target-efolds "$target_efolds" \
  --seed-type "$seed_type" \
  --chunk-s 0.001 \
  --checkpoint "$output_dir/nonlinear_memory_fast_pilot_checkpoint.npz" \
  --output "$output_dir/nonlinear_memory_fast_pilot.json"
