#!/usr/bin/env bash
set -euo pipefail

mkdir -p output
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="${PWD}/src"

python --version > output/python_version.txt 2>&1
python -m unittest -v tests/test_collective_diagnostics.py 2>&1 | tee output/unittest.txt
python tools/analyze_collective_context.py \
  --native-audit native_n2=context-inputs/native_n2_audit.jsonl \
  --native-audit native_n4=context-inputs/native_n4_audit.jsonl \
  --native-audit native_n8=context-inputs/native_n8_audit.jsonl \
  --native-audit native_n16=context-inputs/native_n16_audit.jsonl \
  --depin-counts v24_T1000_rho1e15_seed86=context-inputs/v24_T1000_rho1e15_seed86_depin_counts.csv \
  --depin-counts v24_T1000_rho1p5e16_seed86=context-inputs/v24_T1000_rho1p5e16_seed86_depin_counts.csv \
  --output output/collective_context_diagnostics.json

sha256sum \
  context-inputs/native_n2_audit.jsonl \
  context-inputs/native_n4_audit.jsonl \
  context-inputs/native_n8_audit.jsonl \
  context-inputs/native_n16_audit.jsonl \
  context-inputs/v24_T1000_rho1e15_seed86_depin_counts.csv \
  context-inputs/v24_T1000_rho1p5e16_seed86_depin_counts.csv \
  > output/context_input_inventory.sha256
