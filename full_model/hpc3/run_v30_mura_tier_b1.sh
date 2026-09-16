#!/usr/bin/env bash
set -u -o pipefail

root="${HPC3_WORK_DIR:?}/full_model"
out="$root/production/output/v30-mura-tier-b1"
cases="$out/cases"
mkdir -p "$cases"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONPATH="${HPC3_WORK_DIR}:${HPC3_WORK_DIR}/src"
export V30_SOURCE_SHA="${V30_SOURCE_SHA:?exact pushed source SHA required}"
if [ "$V30_SOURCE_SHA" = "SET_TO_PUSHED_SOURCE_SHA_BEFORE_SUBMISSION" ]; then
  echo "replace V30_SOURCE_SHA with the immutable pushed source SHA" >&2
  exit 2
fi

cd "$HPC3_WORK_DIR" || exit 2
python3 -m py_compile \
  full_model/production/mura_kinematics.py \
  full_model/production/v24_mechanical_wall.py \
  full_model/production/wall_topology_supply.py \
  full_model/analysis/run_v30_mura_tier_b1_case.py \
  full_model/analysis/postprocess_v30_mura_tier_b1.py \
  >"$out/preflight.log" 2>&1 || exit $?

pids=()
names=()
launch() {
  grid="$1" condition="$2"
  name="${condition}_${grid}"
  case_dir="$cases/$name"
  mkdir -p "$case_dir"
  python3 full_model/analysis/run_v30_mura_tier_b1_case.py \
    --case-dir "$case_dir" --grid "$grid" --condition "$condition" \
    --seed 42 --temperature-K 1100 --strain-rate-s 10000 \
    --initial-strain 0.01 --target-strain 0.2 --trial-dt-s 2e-9 \
    --progress-checkpoint-strain 0.01 --wall-checkpoint-s 840 \
    --history-interval 25 --max-wall-s 54000 \
    >"$case_dir/runner.log" 2>"$case_dir/runner.err" &
  pids+=("$!"); names+=("$name")
}

postprocess() {
  python3 full_model/analysis/postprocess_v30_mura_tier_b1.py \
    --cases "$cases" --output "$out/v30_mura_tier_b1_decision.json" \
    --index-csv "$out/v30_mura_tier_b1_index.csv" \
    >"$out/postprocess.log" 2>"$out/postprocess.err" || true
}

terminate_children() {
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  wait || true
  postprocess
  exit 143
}
trap terminate_children TERM INT USR1

for grid in 64 128; do
  for condition in homogeneous broadband_noise mechanical_heterogeneity; do
    launch "$grid" "$condition"
  done
done

overall=0
for index in "${!pids[@]}"; do
  wait "${pids[$index]}"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    printf '{"case":"%s","exit":%s}\n' "${names[$index]}" "$rc" \
      >"$cases/${names[$index]}/process_failure.json"
    overall=1
  fi
done
postprocess
exit "$overall"
