#!/usr/bin/env bash
set -u -o pipefail

root="${HPC3_WORK_DIR:?}/full_model"
out="$root/production/output"
mkdir -p "$out/cases" "$out/status"
export OMP_NUM_THREADS=1 MPLBACKEND=Agg
overall=0

run_case() {
  name="$1" grid="$2" domain_um="$3" seed="$4" heterogeneity="$5"
  multihit="$6" order_enabled="$7" recovery="$8"
  case_out="$out/cases/$name"
  mkdir -p "$case_out"
  start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  params=$(python3 - "$grid" "$domain_um" "$seed" "$heterogeneity" \
    "$multihit" "$order_enabled" "$recovery" <<'PY'
import json, sys
grid=int(sys.argv[1]); domain=float(sys.argv[2])*1e-6
seed=int(sys.argv[3]); heterogeneity=sys.argv[4]
multi=sys.argv[5].lower() == "true"; order=sys.argv[6].lower() == "true"
recovery=sys.argv[7].lower() == "true"
p={
 "v22_common_tensorial_wall_enabled": True,
 "v22_multi_hit_enabled": multi,
 "v22_wall_order_enabled": order,
 "v19_one_grain_mode": True,
 "v19_density_noise_fraction": .01,
 "v19_signed_noise_fraction": .01,
 "v19_noise_seed": seed,
 "v19_mechanical_heterogeneity": heterogeneity,
 "Nx": grid, "Ny": grid, "L_phys": domain,
 "poly_n": 1, "T0": 1100., "edot_app": 1e4,
 "dt": 1e-8, "dt_base": 1e-8, "dt_base_mode": "fixed",
 "nSteps": 2500, "diag_interval": 100, "save_interval": 500,
 "plot_interval": 100000, "restart_interval": 250,
 "restart_wallclock_interval_s": 900., "write_field_npz": False,
 "save_main_panels": False, "save_signed_panels": False,
 "restart_file": None, "restart_reset_clock": True,
}
if recovery:
 p.update(v22_multi_hit_enabled=True,
          v22_multi_hit_wall_log_factor=-1.,
          v22_multi_hit_junction_log_factor=-1.,
          v22_multi_hit_annihilation_log_factor=2.)
print(json.dumps(p, separators=(",", ":")))
PY
  )
  DRX_PARAMS="$params" DRX_OUTDIR="$case_out" \
    python3 drx_full_v34_recovery.py >"$case_out/stdout.log" \
    2>"$case_out/stderr.log"
  rc=$?
  end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '{"case":"%s","grid":%s,"domain_um":%s,"seed":%s,"heterogeneity":"%s","multi_hit":%s,"wall_order":%s,"recovery_control":%s,"start":"%s","end":"%s","exit":%s}\n' \
    "$name" "$grid" "$domain_um" "$seed" "$heterogeneity" "$multihit" \
    "$order_enabled" "$recovery" "$start" "$end" "$rc" \
    >"$out/status/$name.json"
  if [ "$rc" -ne 0 ]; then overall="$rc"; fi
}

cd "$root/production" || exit 2
python3 -m py_compile common_tensorial_wall.py tensorial_nye.py \
  nonlocal_elasticity.py drx_full_v34_recovery.py >"$out/preflight.log" 2>&1 \
  || exit $?

run_case homogeneous_off_64 64 10.0 2201 none false true false
run_case heterogeneous_off_64 64 10.0 2201 eigenstrain_particle false true false
run_case heterogeneous_on_64 64 10.0 2201 eigenstrain_particle true true false
run_case order_disabled_64 64 10.0 2201 eigenstrain_particle true false false
run_case recovery_dominant_64 64 10.0 2201 eigenstrain_particle true true true
run_case heterogeneous_on_domain_seed_64 64 12.7 2202 eigenstrain_particle true true false
run_case heterogeneous_on_128 128 10.0 2201 eigenstrain_particle true true false
run_case heterogeneous_on_domain_seed_128 128 12.7 2202 eigenstrain_particle true true false

exit "$overall"
