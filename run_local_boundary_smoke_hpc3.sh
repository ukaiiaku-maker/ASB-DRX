#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="${PWD}/src"
python --version > output/python_version.txt 2>&1
python -m unittest -v tests.test_antiplane tests.test_local_coupled tests.test_local_mechanism 2>&1 | tee output/unittest.txt
python tools/run_local_boundary_matrix.py \
  --points 16 --steps 1000 --target-shear 0.9 \
  --temperature 850 --rate 4.5 --density-ratio 0.5 \
  --source-commit 9b8f62de0e7b5f00f6d1040a756baa7fb9ac3de7 \
  --execution-site hpc3-single-smoke \
  --output output/local_boundary_single_smoke.json
sha256sum \
  run_local_boundary_smoke_hpc3.sh \
  src/asb_drx/antiplane.py src/asb_drx/local_coupled.py \
  src/asb_drx/local_mechanism.py tools/run_local_boundary_matrix.py \
  > output/input_inventory.sha256
