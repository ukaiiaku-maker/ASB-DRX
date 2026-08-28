#!/usr/bin/env bash
set -euo pipefail

# v34 runner: separates persistent DRX nuclei from transient hazard spikes.
# Branch options:
#   coupled        = DRX candidate incubation + ASB thermal branch
#   asb_only       = hazard off, topology relabel off; hotspot/ASB diagnostic
#   drx_isothermal = hazard candidate branch on, strong thermal bath; DRX diagnostic

DRIVER="${DRIVER:-drx_var_v34_candidate_drx_asb_sweep.py}"
OUT="${DRX_OUTDIR:-results_v34_candidate_drx_asb}"
RATE="${RATE:-3000}"
NSTEPS="${NSTEPS:-5000}"
TARGET_STRAIN="${TARGET_STRAIN:-}"
DT_STRAIN_STEP="${DT_STRAIN_STEP:-1.0e-4}"
BRANCH="${BRANCH:-coupled}"
MODE="${ACTIVITY_MODE:-crystallographic_local}"
POLY_SEED="${POLY_SEED:-42}"
NUC_SEED="${NUC_SEED:-271828}"
PYTHON="${PYTHON:-python3}"

# Conservative DRX candidate defaults.  These are deliberately much less eager
# than v33 and use candidate incubation before allocating a new grain ID.
USE_HAZARD_NUCLEATION="${USE_HAZARD_NUCLEATION:-true}"
NUC_ATTEMPT_FREQ="${NUC_ATTEMPT_FREQ:-1.0e4}"
NUC_INTERVAL="${NUC_INTERVAL:-20}"
NUC_MIN_STRAIN="${NUC_MIN_STRAIN:-0.02}"
NUC_HAZARD_SITE_FLOOR="${NUC_HAZARD_SITE_FLOOR:-0.0}"
NUC_SITE_GB_WEIGHT="${NUC_SITE_GB_WEIGHT:-0.45}"
NUC_SITE_KAPPA_WEIGHT="${NUC_SITE_KAPPA_WEIGHT:-0.45}"
NUC_SITE_GRAD_R_WEIGHT="${NUC_SITE_GRAD_R_WEIGHT:-0.15}"
NUC_REQUIRE_ORGANIZED_STRUCTURE="${NUC_REQUIRE_ORGANIZED_STRUCTURE:-true}"
NUC_COMP_RELIEF_FACTOR="${NUC_COMP_RELIEF_FACTOR:-0.12}"
NUC_COMP_RELIEF_CAP_FACTOR="${NUC_COMP_RELIEF_CAP_FACTOR:-0.50}"
NUC_EXCESS_TO_RHOGB_FRACTION="${NUC_EXCESS_TO_RHOGB_FRACTION:-0.20}"
NUC_HAZARD_RATE_CAP="${NUC_HAZARD_RATE_CAP:-1.0e6}"
NUC_MAX_RADIUS_UM="${NUC_MAX_RADIUS_UM:-0.30}"
NUC_MIN_RADIUS_CELLS="${NUC_MIN_RADIUS_CELLS:-2}"
NUC_CANDIDATE_HOLD_EVALS="${NUC_CANDIDATE_HOLD_EVALS:-8}"
NUC_CANDIDATE_MAX_BARRIER_EV="${NUC_CANDIDATE_MAX_BARRIER_EV:-1.20}"
NUC_CANDIDATE_DECAY_EVALS="${NUC_CANDIDATE_DECAY_EVALS:-2}"
NUC_CANDIDATE_DIAGNOSTIC_ONLY="${NUC_CANDIDATE_DIAGNOSTIC_ONLY:-false}"

# Keep topology relabel off by default; physical grain births are candidate-promoted hazard events.
USE_COMPONENT_RELABEL="${USE_COMPONENT_RELABEL:-false}"
COMPONENT_RELABEL_MAX_SPLITS="${COMPONENT_RELABEL_MAX_SPLITS:-0}"

# Finite-loading/thermal safeguards.
FINITE_LOADING_INIT_FROM_INVERSE="${FINITE_LOADING_INIT_FROM_INVERSE:-true}"
FINITE_LOADING_DAMPING="${FINITE_LOADING_DAMPING:-0.20}"
USE_ADAPTIVE_THERMAL_DT="${USE_ADAPTIVE_THERMAL_DT:-true}"
USE_LOCAL_SLIP_INCREMENT_DT="${USE_LOCAL_SLIP_INCREMENT_DT:-true}"
THERMAL_DT_MAX_DT_STEP="${THERMAL_DT_MAX_DT_STEP:-2.0}"
LOCAL_SLIP_MAX_INCREMENT="${LOCAL_SLIP_MAX_INCREMENT:-5.0e-3}"
MECH_VALIDITY_STOP="${MECH_VALIDITY_STOP:-true}"

T0="${T0:-1100.0}"
K_THERMAL="${K_THERMAL:-0.015}"
T_BATH_COUPLING="${T_BATH_COUPLING:-2.0e8}"
HEAT_PROCESS_ZONE_SIGMA_UM="${HEAT_PROCESS_ZONE_SIGMA_UM:-0.30}"
GB_BLOCKED_WORK_HEAT_FRACTION="${GB_BLOCKED_WORK_HEAT_FRACTION:-0.25}"
GB_BLOCKED_WORK_STORE_FRACTION="${GB_BLOCKED_WORK_STORE_FRACTION:-0.50}"
RHO0_ABS="${RHO0_ABS:-3.5e17}"

case "${BRANCH}" in
  asb_only)
    USE_HAZARD_NUCLEATION=false
    USE_COMPONENT_RELABEL=false
    COMPONENT_RELABEL_MAX_SPLITS=0
    ;;
  drx_isothermal)
    USE_HAZARD_NUCLEATION=true
    T_BATH_COUPLING="${T_BATH_COUPLING_ISOTHERMAL:-2.0e10}"
    K_THERMAL="${K_THERMAL_ISOTHERMAL:-0.15}"
    ;;
  coupled)
    ;;
  *) echo "Unknown BRANCH=${BRANCH}. Use coupled, asb_only, or drx_isothermal." >&2; exit 2;;
esac

mkdir -p "$OUT"
if [[ -n "$TARGET_STRAIN" ]]; then
  NSTEPS="$($PYTHON - <<PY
import math
S=float('$TARGET_STRAIN'); deps=float('$DT_STRAIN_STEP')
print(int(math.ceil(S/max(deps,1e-300))))
PY
)"
fi

PARAMS="$($PYTHON - <<PY
import json

def b(s): return str(s).lower() in ('1','true','yes','on')
rate=float('$RATE'); deps=float('$DT_STRAIN_STEP')
params=dict(
    use_finite_elastic_loading=True,
    finite_loading_init_from_inverse=b('$FINITE_LOADING_INIT_FROM_INVERSE'),
    finite_loading_Eeff=None,
    finite_loading_damping=float('$FINITE_LOADING_DAMPING'),
    finite_loading_nonnegative_stress=True,
    edot_app=rate,
    dt_base=deps/max(abs(rate), 1e-300),
    dt_base_mode='strain_increment',
    dt_strain_step=deps,
    nSteps=int('$NSTEPS'),
    rho0_mode='absolute', rho0_abs=float('$RHO0_ABS'),
    T0=float('$T0'), k_thermal=float('$K_THERMAL'), T_bath_coupling=float('$T_BATH_COUPLING'),
    use_heat_process_zone_kernel=True, heat_process_zone_sigma_um=float('$HEAT_PROCESS_ZONE_SIGMA_UM'),
    heat_process_zone_min_sigma_px=2.0, heat_process_zone_preserve_mean=True,
    use_gb_blocked_work_partition=True,
    gb_blocked_work_heat_fraction=float('$GB_BLOCKED_WORK_HEAT_FRACTION'),
    gb_blocked_work_store_fraction=float('$GB_BLOCKED_WORK_STORE_FRACTION'),
    gb_blocked_work_store_to_rhoGB=True,
    use_energy_conserving_heat=True,
    use_finite_loading_work_budget=True,
    finite_loading_work_budget_safety=0.95,
    finite_loading_scale_gdot_to_budget=True,
    use_adaptive_thermal_dt=b('$USE_ADAPTIVE_THERMAL_DT'), thermal_dt_max_dT_step=float('$THERMAL_DT_MAX_DT_STEP'),
    thermal_dt_log_change=0.05, thermal_dt_min=1.0e-11,
    use_local_slip_increment_dt=b('$USE_LOCAL_SLIP_INCREMENT_DT'), local_slip_max_increment=float('$LOCAL_SLIP_MAX_INCREMENT'),
    use_thermal_validity_stop=True, thermal_validity_Tmax_K=1811.0, thermal_validity_Tmean_K=1700.0,
    use_mechanical_validity_stop=b('$MECH_VALIDITY_STOP'), mechanical_validity_mode='fit_or_ideal',
    mechanical_validity_fit_fraction=1.0, mechanical_validity_ideal_mu_frac=0.12,
    use_collective_taylor=True, use_rho_state_partition=True, use_collective_organization=True,
    use_collective_activity_memory=True, collective_activity_memory_mode='$MODE',
    collective_activity_tau=2.0e-6, collective_activity_D_parallel=0.0, collective_activity_D_perp=0.0,
    collective_activity_rate_weight=1.0, collective_activity_rate_power=1.0,
    collective_heat_partition_weight=0.0, collective_heat_use_activity_memory=True,
    collective_rate_closure='domain_count', collective_domain_power=2.0, collective_min_suppression=1.0e-4,
    local_gdot_cap_factor=0.0, enforce_macro_rate_after_ms=False,
    use_hazard_nucleation=b('$USE_HAZARD_NUCLEATION'), disable_nucleation=False,
    nuc_min_strain=float('$NUC_MIN_STRAIN'), nuc_interval=int(float('$NUC_INTERVAL')),
    nuc_attempt_freq=float('$NUC_ATTEMPT_FREQ'), nuc_hazard_rate_cap=float('$NUC_HAZARD_RATE_CAP'),
    nuc_hazard_site_floor=float('$NUC_HAZARD_SITE_FLOOR'), nuc_site_gb_weight=float('$NUC_SITE_GB_WEIGHT'),
    nuc_site_kappa_weight=float('$NUC_SITE_KAPPA_WEIGHT'), nuc_site_grad_r_weight=float('$NUC_SITE_GRAD_R_WEIGHT'),
    nuc_require_organized_structure=b('$NUC_REQUIRE_ORGANIZED_STRUCTURE'), nuc_wall_fraction_scale=0.03,
    nuc_gnd_fraction_scale=0.05, nuc_gamma_GB=0.50, nuc_rs_theta_m_deg=15.0,
    nuc_min_field_mis_deg=3.0, nuc_barrier_thickness_b=2.0,
    nuc_comp_relief_factor=float('$NUC_COMP_RELIEF_FACTOR'), nuc_comp_relief_cap_factor=float('$NUC_COMP_RELIEF_CAP_FACTOR'),
    nuc_gnd_feed_efficiency=0.50, nuc_theta_candidates_frac=[-1.0,-0.5,0.5,1.0],
    nuc_min_radius_cells=int(float('$NUC_MIN_RADIUS_CELLS')), nuc_max_radius_um=float('$NUC_MAX_RADIUS_UM'),
    nuc_event_select='max_excess', nuc_excess_to_rhoGB_fraction=float('$NUC_EXCESS_TO_RHOGB_FRACTION'),
    nuc_reset_hazard_radius_factor=1.5, nuc_rng_seed=int('$NUC_SEED'),
    use_nuc_candidate_incubation=True, nuc_candidate_hold_evals=int(float('$NUC_CANDIDATE_HOLD_EVALS')),
    nuc_candidate_decay_evals=int(float('$NUC_CANDIDATE_DECAY_EVALS')),
    nuc_candidate_max_barrier_eV=float('$NUC_CANDIDATE_MAX_BARRIER_EV'),
    nuc_candidate_min_rate=0.0, nuc_candidate_min_dF_Jm3=0.0,
    nuc_candidate_promote_select='oldest', nuc_candidate_diagnostic_only=b('$NUC_CANDIDATE_DIAGNOSTIC_ONLY'),
    use_component_relabel=b('$USE_COMPONENT_RELABEL'), component_relabel_interval=100,
    component_relabel_min_px=64, component_relabel_max_splits_per_step=int(float('$COMPONENT_RELABEL_MAX_SPLITS')),
    component_relabel_require_pure=True, track_grain_provenance=True,
    poly_seed=int('$POLY_SEED'), poly_spread_deg=70.0, poly_min_mis_deg=8.0, psi_max_deg=75.0, psi_plastic_max_deg=25.0,
    use_gb_slip_transmission_barrier=True, gb_trans_outgoing_mode='same_index',
    gb_trans_misorientation_barrier_eV=0.80, gb_trans_residual_barrier_eV=1.20,
    gb_trans_gdot_coupling=1.0, gb_trans_bres_ref=0.75, gb_trans_min_factor=1.0e-6,
    gb_trans_store_residual_scale=1.5, gb_trans_use_hard_grain_orientation=True, gb_trans_include_plastic_orientation=True,
    use_gb_residual_rotation=True, gb_residual_rotation_rate=2.0e3, gb_residual_rotation_cap_deg_step=0.01,
    gb_residual_rotation_requires_net_signed=True, gb_residual_rotation_smooth_um=0.35,
    plot_interval=250, diag_interval=25, save_interval=250, restart_interval=250,
    save_main_panels=True, save_signed_panels=False, diag_print_extended=True,
)
print(json.dumps(params))
PY
)"
cat > "$OUT/v34_params.json" <<< "$PARAMS"
echo "Override: $PARAMS"
DRX_OUTDIR="$OUT" DRX_PARAMS="$PARAMS" "$PYTHON" "$DRIVER"
