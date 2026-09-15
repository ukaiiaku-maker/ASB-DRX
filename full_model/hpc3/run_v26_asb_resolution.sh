#!/usr/bin/env bash
set -u -o pipefail
root="${HPC3_WORK_DIR:?}/full_model"
out="$root/production/output/v26-asb-resolution"
mkdir -p "$out"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg
overall=0

run_case() {
  name="$1" grid="$2" grains="$3"
  case_out="$out/$name"; mkdir -p "$case_out"
  params=$(python3 - "$grid" "$grains" <<'PY'
import json,sys
n=int(sys.argv[1]); grains=int(sys.argv[2])
p={"Nx":n,"Ny":n,"grain_max":32,"poly_n":grains,"poly_seed":43,
   "nSteps":5000,"edot_app":30000.,"T0":1100.,
   "T_bath_coupling":2.0e8,"k_thermal":0.015,
   "use_hazard_nucleation":False,"use_component_relabel":False,
   "v22_common_tensorial_wall_enabled":False,"diag_interval":10,
   "save_interval":100000,"plot_interval":100000,"restart_interval":250,
   "restart_wallclock_interval_s":900.,"write_field_npz":False,
   "save_main_panels":False,"save_signed_panels":False}
print(json.dumps(p,separators=(",",":")))
PY
  )
  DRX_PARAMS="$params" DRX_OUTDIR="$case_out" python3 drx_full_v34_recovery.py \
    >"$case_out/stdout.log" 2>"$case_out/stderr.log"
  rc=$?; if [ "$rc" -ne 0 ]; then overall="$rc"; fi
}

cd "$root/production" || exit 2
python3 -m py_compile drx_full_v34_recovery.py asb_grid_audit.py \
  >"$out/preflight.log" 2>&1 || exit $?
run_case seed43_96 96 12
run_case seed43_128 128 12
run_case seed43_192 192 12
run_case homogeneous_128 128 1
if [ "$overall" -eq 0 ]; then
  cd "$root" || exit 2
  python3 full_model/analysis/postprocess_v26_asb_resolution.py \
    --root "$out" --output "$out/v26_asb_resolution.json" \
    >"$out/postprocess.log" 2>"$out/postprocess.err" || overall=$?
fi
exit "$overall"
