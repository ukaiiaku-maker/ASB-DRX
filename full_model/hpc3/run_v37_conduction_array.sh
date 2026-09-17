#!/usr/bin/env bash
set -u -o pipefail

root="${HPC3_WORK_DIR:?}"
out="$root/full_model/production/output/v37-conduction"
index="${HPC3_PARAMETER_INDEX:?}"
source_sha="${V37_SOURCE_SHA:?}"
mkdir -p "$out"
export MPLBACKEND=Agg PYTHONPATH="$root:$root/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$root" || exit 2
python3 -m py_compile full_model/hpc3/run_v37_conduction_case.py \
  >"$out/preflight-${index}.log" 2>&1 || exit $?
python3 full_model/hpc3/run_v37_conduction_case.py \
  --case-id "$index" \
  --case-table full_model/hpc3/v37_conduction_cases.json \
  --source-root "$root" --expected-source-sha "$source_sha" \
  --run-root "$out" --grid 128 --target-step 2500 \
  >"$out/runner-${index}.log" 2>"$out/runner-${index}.err"
