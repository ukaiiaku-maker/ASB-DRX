#!/usr/bin/env bash
set -u -o pipefail
root="${HPC3_WORK_DIR:?}/full_model"
out="$root/production/output"
mkdir -p "$out/cases" "$out/status" "$out/common_state"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg
overall=0

run_case() {
  name="$1" source="$2" steps="$3" mobility="$4" processing="$5"
  case_out="$out/cases/$name"; mkdir -p "$case_out"
  params=$(python3 - "$source" "$steps" "$mobility" "$processing" <<'PY'
import json,sys
p={"Nx":128,"Ny":128,"nSteps":int(sys.argv[2]),"edot_app":.001,
   "dt_strain_step":1e-10,"restart_file":sys.argv[1],"restart_reset_clock":False,
   "use_sparse_common_front_state":True,"use_sibm_existing_boundary":True,
   "stored_energy_coupling_mode":"common_variational",
   "sibm_parent_label_override":0,"sibm_child_label_override":1,
   "sibm_initial_bulge_radius_um":0.,"sibm_pin_endpoints":False,
   "sibm_active_window_radius_um":4.5,"sibm_mobility_multiplier":float(sys.argv[3]),
   "sibm_physical_drag_pressure_Pa":0.,"sibm_applied_pressure_Pa":0.,
   "sibm_front_processing_enabled":sys.argv[4].lower()=="true",
   "disable_new_stochastic_creation_after_restart":True,
   "use_hazard_nucleation":False,"use_component_relabel":False,
   "diag_interval":10,"save_interval":100000,"plot_interval":100000,
   "restart_interval":250,"write_field_npz":False,
   "save_main_panels":False,"save_signed_panels":False}
print(json.dumps(p,separators=(",",":")))
PY
  )
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  DRX_PARAMS="$params" DRX_OUTDIR="$case_out" python3 drx_full_v34_recovery.py \
    >"$case_out/stdout.log" 2>"$case_out/stderr.log"
  rc=$?; end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '{"case":"%s","source":"%s","steps":%s,"mobility":%s,"processing":%s,"start":"%s","end":"%s","exit":%s}\n' \
    "$name" "$source" "$steps" "$mobility" "$processing" "$start" "$end" "$rc" \
    >"$out/status/$name.json"
  if [ "$rc" -ne 0 ]; then overall="$rc"; fi
}

cd "$root/production" || exit 2
source_equal=../../.hpc3/inputs/v13_canonical_bicrystal_equal_128.npz
# Profile equilibration fixture: no front transfer and negligible imposed rate.
run_case equilibrate_planar "$source_equal" 300 1 false
equilibrated=$(find "$out/cases/equilibrate_planar" -name 'drx_v25_restart_*.npz' | sort | tail -n 1)
if [ -z "$equilibrated" ]; then exit 3; fi
python3 ../analysis/assign_v24_planar_sibm_variants.py \
  "$equilibrated" "$out/common_state"
run_case equal "$out/common_state/equal.npz" 500 200 true
run_case parent_high_child_low "$out/common_state/parent_high_child_low.npz" 1200 200 true
run_case reversed "$out/common_state/reversed.npz" 1200 200 true
run_case mobility_off "$out/common_state/mobility_off.npz" 500 0 true
exit "$overall"
