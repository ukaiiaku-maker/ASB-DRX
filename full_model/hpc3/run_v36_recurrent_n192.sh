#!/usr/bin/env bash
set -u -o pipefail

root="${HPC3_WORK_DIR:?}"
out="$root/full_model/production/output/v36-recurrent-n192"
mkdir -p "$out"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONPATH="$root:$root/src"
export V36_SOURCE_SHA="${V36_SOURCE_SHA:?exact frozen source SHA required}"

cd "$root" || exit 2
python3 -m py_compile \
  full_model/analysis/run_v36_recurrent_physical_response.py \
  full_model/analysis/run_v34_finite_coupled_response.py \
  >"$out/preflight.log" 2>&1 || exit $?

python3 full_model/analysis/run_v36_recurrent_physical_response.py \
  --output-dir "$out/case" \
  --protocol hold --grid 192 --intervals 10 --dt-s 5e-6 \
  --proposal-fraction .0625 --checkpoint-every 2 \
  >"$out/runner.log" 2>"$out/runner.err"
rc=$?
printf '{"source_commit":"%s","exit_code":%s}\n' \
  "$V36_SOURCE_SHA" "$rc" >"$out/terminal.json"
exit "$rc"
