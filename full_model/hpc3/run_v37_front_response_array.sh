#!/usr/bin/env bash
set -u -o pipefail

root="${HPC3_WORK_DIR:?}"
out="$root/full_model/production/output/v37-front-response"
index="${HPC3_PARAMETER_INDEX:?}"
mkdir -p "$out"
export MPLBACKEND=Agg PYTHONPATH="$root:$root/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$root" || exit 2
python3 -m py_compile \
  full_model/analysis/run_v34_finite_coupled_response.py \
  full_model/analysis/run_v36_recurrent_physical_response.py \
  full_model/hpc3/run_v37_front_response_case.py \
  >"$out/preflight-${index}.log" 2>&1 || exit $?
python3 full_model/hpc3/run_v37_front_response_case.py \
  --case-index "$index" \
  --case-table full_model/hpc3/v37_front_response_cases.json \
  --run-root "$out" --grid 128 --intervals 40 \
  >"$out/runner-${index}.log" 2>"$out/runner-${index}.err"
