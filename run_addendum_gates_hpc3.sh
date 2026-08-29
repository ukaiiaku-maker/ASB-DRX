#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
python --version > output/python_version.txt 2>&1
python -m unittest -v tests.test_addendum_verification 2>&1 | tee output/unittest.txt
python verification/gate_A_bertin_rotation.py > output/gate_A_kinematic_fixture.json
python verification/gate_B_signed_patterning.py > output/gate_B_linear_fixture.json
python verification/gate_C_polygonization.py > output/gate_C_mechanism_fixture.json
python verification/gate_D_frank_bilby_rotation.py > output/gate_D_static_fixture.json
sha256sum \
  run_addendum_gates_hpc3.sh \
  verification/gate_A_bertin_rotation.py \
  verification/gate_B_signed_patterning.py \
  verification/gate_C_polygonization.py \
  verification/gate_D_frank_bilby_rotation.py \
  tests/test_addendum_verification.py > output/input_inventory.sha256
