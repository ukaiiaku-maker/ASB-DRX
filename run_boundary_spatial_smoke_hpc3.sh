#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="${PWD}/src"
python --version > output/python_version.txt 2>&1
python -m unittest -v tests.test_analytical_boundary tests.test_boundary_campaign tests.test_spatial_coupled tests.test_mechanism_ladder 2>&1 | tee output/unittest.txt
python tools/run_boundary_spatial_smoke.py
sha256sum run_boundary_spatial_smoke_hpc3.sh src/asb_drx/boundary.py src/asb_drx/boundary_campaign.py src/asb_drx/fixtures.py src/asb_drx/spatial_coupled.py src/asb_drx/mechanism_ladder.py tests/test_boundary_campaign.py tools/run_boundary_spatial_smoke.py > output/input_inventory.sha256
