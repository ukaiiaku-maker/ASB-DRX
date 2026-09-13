#!/usr/bin/env bash
set -euo pipefail

# v32 runner for controlled DRX/ASB rate maps.
# v3 fixes the v2 early-failure issue by avoiding a hard 1e-3 Euler loading jump:
#   - start finite loading from the Arrhenius inverse stress
#   - use a smaller strain increment by default
#   - damp the explicit elastic loading update
#   - keep adaptive/local-slip dt safeguards available by default

DRIVER="${DRIVER:-drx_var_v32_gb_transmission_processzone_asb_sweep.py}"
OUT="${DRX_OUTDIR:-results_v32_gbtrans_processzone_ASB_try1}"
RATE="${RATE:-30000}"
NSTEPS="${NSTEPS:-2000}"
DT_STRAIN_STEP="${DT_STRAIN_STEP:-1.0e-4}"
MODE="${ACTIVITY_MODE:-crystallographic_local}"
RATE_WEIGHT="${ACTIVITY_RATE_WEIGHT:-1.0}"
POLY_SEED="${POLY_SEED:-42}"
NUC_SEED="${NUC_SEED:-271828}"
USE_HAZARD_NUCLEATION="${USE_HAZARD_NUCLEATION:-true}"
USE_ADAPTIVE_THERMAL_DT="${USE_ADAPTIVE_THERMAL_DT:-true}"
USE_LOCAL_SLIP_INCREMENT_DT="${USE_LOCAL_SLIP_INCREMENT_DT:-true}"
THERMAL_DT_MAX_DT_STEP="${THERMAL_DT_MAX_DT_STEP:-2.0}"
LOCAL_SLIP_MAX_INCREMENT="${LOCAL_SLIP_MAX_INCREMENT:-5.0e-3}"
FINITE_LOADING_INIT_FROM_INVERSE="${FINITE_LOADING_INIT_FROM_INVERSE:-true}"
FINITE_LOADING_DAMPING="${FINITE_LOADING_DAMPING:-0.20}"
MECH_VALIDITY_STOP="${MECH_VALIDITY_STOP:-true}"

PARAMS="$(python3 - <<PY
import json
rate=float('$RATE')
deps=float('$DT_STRAIN_STEP')
mode='$MODE'
use_hazard=str('$USE_HAZARD_NUCLEATION').lower() in ('1','true','yes','on')
use_adapt=str('$USE_ADAPTIVE_THERMAL_DT').lower() in ('1','true','yes','on')
use_slipdt=str('$USE_LOCAL_SLIP_INCREMENT_DT').lower() in ('1','true','yes','on')
init_inv=str('$FINITE_LOADING_INIT_FROM_INVERSE').lower() in ('1','true','yes','on')
mech_stop=str('$MECH_VALIDITY_STOP').lower() in ('1','true','yes','on')
params=dict(
    use_finite_elastic_loading=True,
    finite_loading_init_from_inverse=init_inv,
    finite_loading_Eeff=None,
    finite_loading_damping=float('$FINITE_LOADING_DAMPING'),
    finite_loading_nonnegative_stress=True,

    use_collective_taylor=True,
    use_rho_state_partition=True,
    use_collective_organization=True,
    use_collective_activity_memory=True,
    collective_activity_memory_mode=mode,
    collective_activity_tau=2.0e-6,
    collective_activity_D_parallel=0.0,
    collective_activity_D_perp=0.0,
    collective_activity_rate_weight=float('$RATE_WEIGHT'),
    collective_activity_rate_power=1.0,
    collective_heat_partition_weight=0.0,
    collective_heat_use_activity_memory=True,

    collective_rate_closure='domain_count',
    collective_domain_power=2.0,
    collective_min_suppression=1.0e-4,
    local_gdot_cap_factor=0.0,
    enforce_macro_rate_after_ms=False,
    use_energy_conserving_heat=True,
    use_finite_loading_work_budget=True,
    finite_loading_work_budget_safety=0.95,
    finite_loading_scale_gdot_to_budget=True,
    use_heat_process_zone_kernel=True,
    heat_process_zone_sigma_um=0.30,
    heat_process_zone_min_sigma_px=2.0,
    heat_process_zone_preserve_mean=True,
    use_gb_blocked_work_partition=True,
    gb_blocked_work_heat_fraction=0.25,
    gb_blocked_work_store_fraction=0.50,
    gb_blocked_work_store_to_rhoGB=True,

    use_hazard_nucleation=use_hazard,
    nuc_min_strain=0.10,
    nuc_interval=100,
    nuc_attempt_freq=1.0e2,
    nuc_hazard_site_floor=0.0,
    nuc_site_gb_weight=0.35,
    nuc_site_kappa_weight=0.30,
    nuc_site_grad_r_weight=0.10,
    nuc_require_organized_structure=True,
    nuc_wall_fraction_scale=0.03,
    nuc_gnd_fraction_scale=0.05,
    nuc_excess_to_rhoGB_fraction=0.25,
    nuc_comp_relief_factor=0.05,
    nuc_comp_relief_cap_factor=0.25,

    poly_seed=int('$POLY_SEED'),
    nuc_rng_seed=int('$NUC_SEED'),
    poly_spread_deg=70.0,
    poly_min_mis_deg=8.0,
    psi_max_deg=75.0,
    psi_plastic_max_deg=25.0,
    use_gb_slip_transmission_barrier=True,
    gb_trans_outgoing_mode='same_index',
    gb_trans_misorientation_barrier_eV=0.80,
    gb_trans_residual_barrier_eV=1.20,
    gb_trans_gdot_coupling=1.0,
    gb_trans_bres_ref=0.75,
    gb_trans_min_factor=1.0e-6,
    gb_trans_store_residual_scale=1.5,
    gb_trans_use_hard_grain_orientation=True,
    gb_trans_include_plastic_orientation=True,
    use_gb_residual_rotation=True,
    gb_residual_rotation_rate=2.0e3,
    gb_residual_rotation_cap_deg_step=0.01,
    gb_residual_rotation_requires_net_signed=True,
    gb_residual_rotation_smooth_um=0.35,

    T0=1100.0,
    edot_app=rate,
    dt_base=deps/max(abs(rate), 1e-300),
    dt_base_mode='strain_increment',
    dt_strain_step=deps,
    rho0_abs=3.5e17,
    nSteps=int('$NSTEPS'),

    k_thermal=0.015,
    T_bath_coupling=2.0e8,

    use_adaptive_thermal_dt=use_adapt,
    thermal_dt_max_dT_step=float('$THERMAL_DT_MAX_DT_STEP'),
    thermal_dt_log_change=0.05,
    thermal_dt_min=1.0e-11,
    use_local_slip_increment_dt=use_slipdt,
    local_slip_max_increment=float('$LOCAL_SLIP_MAX_INCREMENT'),

    use_thermal_validity_stop=True,
    thermal_validity_Tmax_K=1811.0,
    thermal_validity_Tmean_K=1700.0,
    use_mechanical_validity_stop=mech_stop,
    mechanical_validity_mode='fit_or_ideal',
    mechanical_validity_fit_fraction=1.0,
    mechanical_validity_ideal_mu_frac=0.12,

    plot_interval=250,
    diag_interval=25,
    save_interval=250,
)
print(json.dumps(params))
PY
)"

DRX_OUTDIR="$OUT" DRX_PARAMS="$PARAMS" python3 "$DRIVER"
