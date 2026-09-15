#!/usr/bin/env bash
set -u -o pipefail

# Zero-external-pressure full-model fallback.  Every case uses the same common
# variational phase energy and conservative v14 moving-front map.  Density
# contrast and curvature are inputs; no continuation pressure is applied.
root="${HPC3_WORK_DIR:?}/full_model"
out="${HPC3_WORK_DIR}/full_model/production/output"
mkdir -p "$out/cases" "$out/status"
export OMP_NUM_THREADS=1
overall=0

run_case() {
  name="$1"; source="$2"; amplitude="$3"; steps="$4"; rate="$5"; dt_strain="$6"
  case_out="$out/cases/$name"
  mkdir -p "$case_out"
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  PYTHON=python3 DRIVER=drx_full_v34_recovery.py \
    BRANCH=asb_only RATE="$rate" NSTEPS="$steps" DT_STRAIN_STEP="$dt_strain" \
    NX=128 NY=128 USE_SPARSE_COMMON_FRONT_STATE=true \
    USE_SIBM_EXISTING_BOUNDARY=true STORED_ENERGY_COUPLING_MODE=common_variational \
    RESTART_FILE="$source" SIBM_PIN_ENDPOINTS=true SIBM_PIN_RADIUS_UM=0.3 \
    SIBM_INITIAL_BULGE_RADIUS_UM="$amplitude" SIBM_SEED_HALF_CHORD_UM=1.5 \
    SIBM_PARENT_LABEL_OVERRIDE=0 SIBM_CHILD_LABEL_OVERRIDE=1 \
    SIBM_ACTIVE_WINDOW_RADIUS_UM=3.5 SIBM_PHYSICAL_DRAG_PRESSURE_PA=0 \
    SIBM_APPLIED_PRESSURE_PA=0 SIBM_FRONT_PROCESSING_ENABLED=true \
    DIAG_INTERVAL=10 SAVE_INTERVAL=10000 RESTART_INTERVAL=50 \
    SAVE_MAIN_PANELS=false DRX_OUTDIR="$case_out" \
    bash run_full_v34_recovery.sh >"$case_out/stdout.log" 2>"$case_out/stderr.log"
  rc=$?
  end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '{"case":"%s","source":"%s","amplitude_um":%s,"steps":%s,"rate_s-1":%s,"applied_pressure_Pa":0,"start":"%s","end":"%s","exit":%s}\n' \
    "$name" "$source" "$amplitude" "$steps" "$rate" "$start" "$end" "$rc" \
    >"$out/status/$name.json"
  if [ "$rc" -ne 0 ]; then overall="$rc"; fi
}

cd "$root/production" || exit 2
PYTHONPATH="$root/.." python3 -m unittest discover -s "$root/../tests" \
  -p 'test_full_model_moving_front.py' >"$out/front_invariants.log" 2>&1 || overall=$?
PYTHONPATH="$root/.." python3 -m unittest discover -s "$root/../tests" \
  -p 'test_full_model_sibm_boundary.py' >>"$out/front_invariants.log" 2>&1 || overall=$?

high_low=../../.hpc3/inputs/v13_canonical_bicrystal_128.npz
equal=../../.hpc3/inputs/v13_canonical_bicrystal_equal_128.npz
reverse=../../.hpc3/inputs/v13_canonical_bicrystal_reverse_128.npz
run_case equal_energy_curved "$equal" 0.60 200 0.001 1e-10
run_case parent_high_child_low "$high_low" 0.60 200 0.001 1e-10
run_case reversed_density_contrast "$reverse" 0.60 200 0.001 1e-10
run_case favorable_small_radius "$high_low" 0.30 200 0.001 1e-10
run_case favorable_large_radius "$high_low" 1.00 200 0.001 1e-10
run_case continued_deformation_rehardening "$high_low" 0.60 500 3000 1e-4

exit "$overall"
