#!/usr/bin/env bash
set -u -o pipefail

# One non-array v14 qualification bundle. Cases are deliberately fault-isolated
# so one failure cannot erase completed evidence from the other four.
root="${HPC3_WORK_DIR:?}/full_model"
out="${HPC3_WORK_DIR}/full_model/production/output"
mkdir -p "$out/cases" "$out/status"
export OMP_NUM_THREADS=1

overall=0
run_case() {
  name="$1"; grid="$2"; steps="$3"; rate="$4"; dt_strain="$5"
  source="$6"; amplitude="$7"; drag="$8"; applied="$9"
  case_out="$out/cases/$name"
  mkdir -p "$case_out"
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  PYTHON=python3 DRIVER=drx_full_v34_recovery.py \
    BRANCH=asb_only RATE="$rate" NSTEPS="$steps" \
    DT_STRAIN_STEP="$dt_strain" NX="$grid" NY="$grid" \
    USE_SPARSE_COMMON_FRONT_STATE=true USE_SIBM_EXISTING_BOUNDARY=true \
    STORED_ENERGY_COUPLING_MODE=common_variational RESTART_FILE="$source" \
    SIBM_PIN_ENDPOINTS=true SIBM_PIN_RADIUS_UM=0.3 \
    SIBM_INITIAL_BULGE_RADIUS_UM="$amplitude" \
    SIBM_SEED_HALF_CHORD_UM=1.5 SIBM_ACTIVE_WINDOW_RADIUS_UM=3.5 \
    SIBM_PHYSICAL_DRAG_PRESSURE_PA="$drag" \
    SIBM_APPLIED_PRESSURE_PA="$applied" SIBM_FRONT_PROCESSING_ENABLED=true \
    DIAG_INTERVAL=20 SAVE_INTERVAL=10000 RESTART_INTERVAL="$steps" \
    SAVE_MAIN_PANELS=false DRX_OUTDIR="$case_out" \
    bash run_full_v34_recovery.sh >"$case_out/stdout.log" 2>"$case_out/stderr.log"
  rc=$?
  end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '{"case":"%s","grid":%s,"steps":%s,"rate_s-1":%s,"source":"%s","amplitude_um":%s,"drag_Pa":%s,"applied_Pa":%s,"start":"%s","end":"%s","exit":%s}\n' \
    "$name" "$grid" "$steps" "$rate" "$source" "$amplitude" "$drag" "$applied" "$start" "$end" "$rc" \
    >"$out/status/$name.json"
  if [ "$rc" -ne 0 ]; then overall="$rc"; fi
}

cd "$root/production" || exit 2
PYTHONPATH="$root/.." python3 -m unittest discover -s "$root/../tests" \
  -p 'test_full_model_moving_front.py' >"$out/local_invariants.log" 2>&1 || overall=$?
PYTHONPATH="$root/.." python3 "$root/analysis/run_v10_sibm_fixtures.py" \
  "$out/v10_sibm_fixtures.json" >>"$out/local_invariants.log" 2>&1 || overall=$?

run_case pinned_subcritical_128 128 80 0.001 1e-10 \
  ../../.hpc3/inputs/v14_equilibrated_common_128.npz 0.6 4500000 0
run_case pinned_supercritical_128 128 80 0.001 1e-10 \
  ../../.hpc3/inputs/v14_equilibrated_common_128.npz 0.6 0 5500000
run_case production_unseeded_128 128 100 3000 1e-4 \
  ../../.hpc3/inputs/v13_canonical_bicrystal_128.npz 0.0 0 5500000
run_case production_bulge_128 128 100 3000 1e-4 \
  ../../.hpc3/inputs/v13_canonical_bicrystal_128.npz 0.6 0 5500000
run_case production_bulge_192 192 100 3000 1e-4 \
  ../../.hpc3/inputs/v13_canonical_bicrystal_192.npz 0.6 0 5500000

exit "$overall"
