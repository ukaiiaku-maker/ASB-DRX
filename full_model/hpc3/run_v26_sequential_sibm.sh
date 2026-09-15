#!/usr/bin/env bash
set -u -o pipefail
root="${HPC3_WORK_DIR:?}/full_model"
out="$root/production/output/v26-sequential-sibm"
mkdir -p "$out"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg
overall=0

cd "$root/production" || exit 2
python3 -m py_compile drx_full_v34_recovery.py moving_front.py symmetric_sibm.py \
  >"$out/preflight.log" 2>&1 || exit $?

for grid in 64 128; do
  python3 ../analysis/run_v26_sequential_sibm.py \
    --grid "$grid" --steps 1000 --stages S5 \
    --root "$out/grid${grid}" --output "$out/v26_sibm_${grid}.json" \
    >"$out/grid${grid}.log" 2>"$out/grid${grid}.err"
  rc=$?
  if [ "$rc" -ne 0 ]; then overall="$rc"; fi
done
exit "$overall"
