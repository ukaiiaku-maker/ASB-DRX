#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src"
python -m unittest -q tests.test_driven_cdd tests.test_bertin_bcc tests.test_signed_transport
python tools/run_gate_B1_extended.py
