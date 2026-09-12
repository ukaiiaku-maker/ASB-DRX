#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src"
python -m unittest -q \
  tests.test_arrhenius_v3 \
  tests.test_cdd_flux_v3 \
  tests.test_staggered_cdd \
  tests.test_integrated_cdd_v3 \
  tests.test_polygonization_v3
python tools/run_v3_integrated_screen.py
