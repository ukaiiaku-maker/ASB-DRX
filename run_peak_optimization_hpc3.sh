#!/bin/bash
set -Eeuo pipefail

mkdir -p output
python --version > output/python_version.txt 2>&1
python -m unittest -v tests.test_exp_floor tests.test_peak_optimization > output/unittest.txt 2>&1
sha256sum run_peak_optimization_hpc3.sh src/asb_drx/analytical.py src/asb_drx/optimization.py tests/test_exp_floor.py tests/test_peak_optimization.py > output/input_inventory.sha256
