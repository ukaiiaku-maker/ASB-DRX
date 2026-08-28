#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p output; export PYTHONPATH="${PWD}/src"
python --version > output/python_version.txt 2>&1
python -m unittest -v tests.test_localization 2>&1 | tee output/unittest.txt
sha256sum run_localization_hpc3.sh src/asb_drx/localization.py tests/test_localization.py > output/input_inventory.sha256
