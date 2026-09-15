#!/usr/bin/env bash
set -u -o pipefail

root="${HPC3_WORK_DIR:?}/full_model"
out="$root/production/output"
mode="${ASB_REFINEMENT_MODE:?}"
mkdir -p "$out/cases" "$out/status"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg
overall=0

run_case() {
  thermal="$1"; bath="$2" conductivity="$3" source="$4"
  case_name="${thermal}_128_seed43_${mode}"
  case_out="$out/cases/$case_name"
  mkdir -p "$case_out"
  if [ "$mode" = "base" ]; then
    steps=2000; strain_step=1e-4
  elif [ "$mode" = "dt_refined" ]; then
    steps=4000; strain_step=5e-5
  else
    printf 'unknown refinement mode: %s\n' "$mode" >&2; return 2
  fi
  params=$(python3 - "$steps" "$strain_step" "$bath" "$conductivity" "$source" <<'PY'
import json,sys
p={"Nx":128,"Ny":128,"poly_n":12,"poly_seed":43,
   "nSteps":int(sys.argv[1]),"edot_app":30000.,"dt_strain_step":float(sys.argv[2]),
   "T_bath_coupling":float(sys.argv[3]),"k_thermal":float(sys.argv[4]),
   "restart_file":sys.argv[5],"restart_reset_clock":False,
   "use_hazard_nucleation":False,"use_component_relabel":False,
   "disable_new_stochastic_creation_after_restart":True,
   "v22_common_tensorial_wall_enabled":False,
   "diag_interval":10,"save_interval":100000,"plot_interval":100000,
   "restart_interval":250,"restart_wallclock_interval_s":900.,
   "write_field_npz":False,"save_main_panels":False,"save_signed_panels":False}
print(json.dumps(p,separators=(",",":")))
PY
  )
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  DRX_PARAMS="$params" DRX_OUTDIR="$case_out" \
    python3 drx_full_v34_recovery.py >"$case_out/stdout.log" 2>"$case_out/stderr.log"
  rc=$?; end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '{"case":"%s","mode":"%s","source":"%s","steps":%s,"strain_step":%s,"start":"%s","end":"%s","exit":%s}\n' \
    "$case_name" "$mode" "$source" "$steps" "$strain_step" "$start" "$end" "$rc" \
    >"$out/status/$case_name.json"
  if [ "$rc" -ne 0 ]; then overall="$rc"; fi
}

cd "$root/production" || exit 2
python3 -m py_compile drx_full_v34_recovery.py asb_classifier.py \
  >"$out/preflight.log" 2>&1 || exit $?
source_root="../../hpc3-results/asb-drx-full-v34-recovery/20260915T144612Z-4f9eb55-2c3fe3/work/full_model/production/output/cases"
run_case adiabatic 2.0e8 0.015 \
  "$source_root/adiabatic_128_seed43/drx_v25_restart_002500.npz"
run_case isothermal 3.5e13 0.15 \
  "$source_root/isothermal_128_seed43/drx_v25_restart_002500.npz"
exit "$overall"
