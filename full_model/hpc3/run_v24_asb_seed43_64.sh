#!/usr/bin/env bash
set -u -o pipefail
root="${HPC3_WORK_DIR:?}/full_model"; out="$root/production/output"
mkdir -p "$out/cases" "$out/status"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg
overall=0
run_case() {
  thermal="$1" bath="$2" conductivity="$3"
  name="${thermal}_64_seed43_high_cadence"; case_out="$out/cases/$name"
  mkdir -p "$case_out"
  params=$(python3 - "$bath" "$conductivity" <<'PY'
import json,sys
p={"Nx":64,"Ny":64,"poly_n":12,"poly_seed":43,"nSteps":5000,
   "edot_app":30000.,"T0":1100.,"T_bath_coupling":float(sys.argv[1]),
   "k_thermal":float(sys.argv[2]),"use_hazard_nucleation":False,
   "use_component_relabel":False,"v22_common_tensorial_wall_enabled":False,
   "diag_interval":10,"save_interval":100000,"plot_interval":100000,
   "restart_interval":25,"restart_wallclock_interval_s":900.,
   "write_field_npz":False,"save_main_panels":False,"save_signed_panels":False}
print(json.dumps(p,separators=(",",":")))
PY
  )
  DRX_PARAMS="$params" DRX_OUTDIR="$case_out" python3 drx_full_v34_recovery.py \
    >"$case_out/stdout.log" 2>"$case_out/stderr.log"
  rc=$?; printf '{"case":"%s","exit":%s}\n' "$name" "$rc" >"$out/status/$name.json"
  if [ "$rc" -ne 0 ]; then overall="$rc"; fi
}
cd "$root/production" || exit 2
python3 -m py_compile drx_full_v34_recovery.py asb_classifier.py >"$out/preflight.log" 2>&1 || exit $?
run_case adiabatic 2.0e8 0.015
run_case isothermal 3.5e13 0.15
exit "$overall"
