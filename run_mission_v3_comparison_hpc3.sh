#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src"
python -m unittest -q tests.test_arrhenius_v3 tests.test_cdd_flux_v3 tests.test_driven_cdd
python tools/run_v3_model_comparison.py
