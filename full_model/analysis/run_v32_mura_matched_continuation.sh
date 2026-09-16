#!/usr/bin/env bash
set -u -o pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
expected="${V32_SOURCE_SHA:?set V32_SOURCE_SHA to the immutable repair commit}"
current="$(git -C "$root" rev-parse HEAD)"
if [ "$expected" != "$current" ]; then
  echo "V32 source mismatch: requested $expected, checked out $current" >&2
  exit 2
fi
if ! git -C "$root" diff --quiet || ! git -C "$root" diff --cached --quiet; then
  echo "V32 continuation requires a clean immutable source tree" >&2
  exit 2
fi

input="${V32_V31_CASES_ROOT:-/Users/sdillon/HPC3/v31-evidence/mura-b1-56070295/raw/cases}"
output="${V32_OUTPUT_ROOT:-$root/full_model/production/output/v32-mura-matched-continuation}"
mkdir -p "$output/cases"
export V32_SOURCE_SHA="$expected"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

run_case() {
  grid="$1" checkpoint="$2" checksum="$3"
  source_path="$input/mechanical_heterogeneity_${grid}/$checkpoint"
  actual="$(shasum -a 256 "$source_path" | awk '{print $1}')"
  if [ "$actual" != "$checksum" ]; then
    echo "frozen checkpoint checksum mismatch for grid $grid" >&2
    return 2
  fi
  case_dir="$output/cases/mechanical_heterogeneity_${grid}"
  mkdir -p "$case_dir"
  if ! find "$case_dir" -name 'checkpoint_step_*.npz' -print -quit | grep -q .; then
    cp -p "$source_path" "$case_dir/$checkpoint"
  fi
  /opt/anaconda3/bin/python "$root/full_model/analysis/run_v30_mura_tier_b1_case.py" \
    --case-dir "$case_dir" --grid "$grid" \
    --condition mechanical_heterogeneity --seed 42 \
    --temperature-K 1100 --strain-rate-s 10000 \
    --initial-strain .03 --target-strain .05 --trial-dt-s 2e-9 \
    --progress-checkpoint-strain .002 --wall-checkpoint-s 840 \
    --history-interval 25 --max-wall-s 54000 \
    --mura-work-budget-mode energy_limited
}

run_case 64 checkpoint_step_000001000_strain_0.03000000.npz \
  deb10fdeb921770b54e5fa4136f4e4b4e2689269b4a5cb2604029367512d02d0 || exit $?
run_case 128 checkpoint_step_000001000_strain_0.03000000.npz \
  fff0212845fb9d9017454edb4d9e5b1c7eb2e6c4650263c4a29688c3c6a48b17 || exit $?
