#!/usr/bin/env bash
set -euo pipefail

# v33 runner: balanced DRX + ASB diagnostic branch for v32 driver.
# This runner does not impose a DRX gate.  It restores the finite-amplitude
# hazard model to a coarse-grained continuum-patch prefactor and disables
# topology relabeling by default so grain_id only changes through explicit
# hazard/spinodal birth mechanisms rather than bookkeeping segmentation.

DRIVER="${DRIVER:-drx_var_v32_gb_transmission_processzone_asb_sweep.py}"
OUT="${DRX_OUTDIR:-results_v33_drx_asb_balanced}"
RATE="${RATE:-3000}"
NSTEPS="${NSTEPS:-5000}"
TARGET_STRAIN="${TARGET_STRAIN:-}"
DT_STRAIN_STEP="${DT_STRAIN_STEP:-1.0e-4}"
MODE="${ACTIVITY_MODE:-crystallographic_local}"
RATE_WEIGHT="${ACTIVITY_RATE_WEIGHT:-1.0}"
POLY_SEED="${POLY_SEED:-42}"
NUC_SEED="${NUC_SEED:-271828}"
PYTHON="${PYTHON:-python3}"

# Nucleation controls.  These are rate-law/prefactor parameters, not hard gates.
USE_HAZARD_NUCLEATION="${USE_HAZARD_NUCLEATION:-true}"
NUC_ATTEMPT_FREQ="${NUC_ATTEMPT_FREQ:-1.0e6}"
NUC_INTERVAL="${NUC_INTERVAL:-20}"
NUC_MIN_STRAIN="${NUC_MIN_STRAIN:-0.0}"
NUC_HAZARD_SITE_FLOOR="${NUC_HAZARD_SITE_FLOOR:-0.02}"
NUC_SITE_GB_WEIGHT="${NUC_SITE_GB_WEIGHT:-1.0}"
NUC_SITE_KAPPA_WEIGHT="${NUC_SITE_KAPPA_WEIGHT:-0.7}"
NUC_SITE_GRAD_R_WEIGHT="${NUC_SITE_GRAD_R_WEIGHT:-0.3}"
NUC_REQUIRE_ORGANIZED_STRUCTURE="${NUC_REQUIRE_ORGANIZED_STRUCTURE:-true}"
NUC_WALL_FRACTION_SCALE="${NUC_WALL_FRACTION_SCALE:-0.03}"
NUC_GND_FRACTION_SCALE="${NUC_GND_FRACTION_SCALE:-0.05}"
NUC_GAMMA_GB="${NUC_GAMMA_GB:-0.50}"
NUC_BARRIER_THICKNESS_B="${NUC_BARRIER_THICKNESS_B:-2.0}"
NUC_COMP_RELIEF_FACTOR="${NUC_COMP_RELIEF_FACTOR:-0.50}"
NUC_COMP_RELIEF_CAP_FACTOR="${NUC_COMP_RELIEF_CAP_FACTOR:-2.0}"
NUC_GND_FEED_EFFICIENCY="${NUC_GND_FEED_EFFICIENCY:-0.50}"
NUC_EXCESS_TO_RHOGB_FRACTION="${NUC_EXCESS_TO_RHOGB_FRACTION:-1.0}"
NUC_HAZARD_RATE_CAP="${NUC_HAZARD_RATE_CAP:-1.0e7}"
NUC_MAX_RADIUS_UM="${NUC_MAX_RADIUS_UM:-0.35}"
NUC_MIN_RADIUS_CELLS="${NUC_MIN_RADIUS_CELLS:-2}"

# Disable bookkeeping relabel by default for clean grain-id/provenance diagnostics.
# Set USE_COMPONENT_RELABEL=true to restore the old component-splitting path.
USE_COMPONENT_RELABEL="${USE_COMPONENT_RELABEL:-false}"
COMPONENT_RELABEL_INTERVAL="${COMPONENT_RELABEL_INTERVAL:-100}"
COMPONENT_RELABEL_MIN_PX="${COMPONENT_RELABEL_MIN_PX:-64}"
COMPONENT_RELABEL_MAX_SPLITS="${COMPONENT_RELABEL_MAX_SPLITS:-0}"

# Mechanical / thermal controls retained from v32/v3.
USE_ADAPTIVE_THERMAL_DT="${USE_ADAPTIVE_THERMAL_DT:-true}"
USE_LOCAL_SLIP_INCREMENT_DT="${USE_LOCAL_SLIP_INCREMENT_DT:-true}"
THERMAL_DT_MAX_DT_STEP="${THERMAL_DT_MAX_DT_STEP:-2.0}"
LOCAL_SLIP_MAX_INCREMENT="${LOCAL_SLIP_MAX_INCREMENT:-5.0e-3}"
FINITE_LOADING_INIT_FROM_INVERSE="${FINITE_LOADING_INIT_FROM_INVERSE:-true}"
FINITE_LOADING_DAMPING="${FINITE_LOADING_DAMPING:-0.20}"
MECH_VALIDITY_STOP="${MECH_VALIDITY_STOP:-true}"

# Thermal branch controls for ASB.  Keep exposed for rate maps.
T0="${T0:-1100.0}"
K_THERMAL="${K_THERMAL:-0.015}"
T_BATH_COUPLING="${T_BATH_COUPLING:-2.0e8}"
HEAT_PROCESS_ZONE_SIGMA_UM="${HEAT_PROCESS_ZONE_SIGMA_UM:-0.30}"
GB_BLOCKED_WORK_HEAT_FRACTION="${GB_BLOCKED_WORK_HEAT_FRACTION:-0.25}"
GB_BLOCKED_WORK_STORE_FRACTION="${GB_BLOCKED_WORK_STORE_FRACTION:-0.50}"

# Initial structure.  Absolute mode avoids rate-dependent rho_c causing the
# high-rate cases to start at unphysical density scales.
RHO0_ABS="${RHO0_ABS:-3.5e17}"
RHO0_MODE="${RHO0_MODE:-absolute}"

mkdir -p "$OUT"

# If TARGET_STRAIN is supplied, override NSTEPS consistently.
if [[ -n "$TARGET_STRAIN" ]]; then
  NSTEPS="$($PYTHON - <<PY
import math
S=float('$TARGET_STRAIN')
deps=float('$DT_STRAIN_STEP')
print(int(math.ceil(S/max(deps,1e-300))))
PY
)"
fi

PARAMS="$($PYTHON - <<PY
import json

def b(s):
    return str(s).lower() in ('1','true','yes','on')
rate=float('$RATE')
deps=float('$DT_STRAIN_STEP')
params=dict(
    # loading / rate-map controls
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
    rho0_mode='$RHO0_MODE',
    rho0_abs=float('$RHO0_ABS'),

    # ASB / heat branch
    T0=float('$T0'),
    k_thermal=float('$K_THERMAL'),
    T_bath_coupling=float('$T_BATH_COUPLING'),
    use_heat_process_zone_kernel=True,
    heat_process_zone_sigma_um=float('$HEAT_PROCESS_ZONE_SIGMA_UM'),
    heat_process_zone_min_sigma_px=2.0,
    heat_process_zone_preserve_mean=True,
    use_gb_blocked_work_partition=True,
    gb_blocked_work_heat_fraction=float('$GB_BLOCKED_WORK_HEAT_FRACTION'),
    gb_blocked_work_store_fraction=float('$GB_BLOCKED_WORK_STORE_FRACTION'),
    gb_blocked_work_store_to_rhoGB=True,
    use_energy_conserving_heat=True,
    use_finite_loading_work_budget=True,
    finite_loading_work_budget_safety=0.95,
    finite_loading_scale_gdot_to_budget=True,

    # finite-time safeguards / validity diagnostics
    use_adaptive_thermal_dt=b('$USE_ADAPTIVE_THERMAL_DT'),
    thermal_dt_max_dT_step=float('$THERMAL_DT_MAX_DT_STEP'),
    thermal_dt_log_change=0.05,
    thermal_dt_min=1.0e-11,
    use_local_slip_increment_dt=b('$USE_LOCAL_SLIP_INCREMENT_DT'),
    local_slip_max_increment=float('$LOCAL_SLIP_MAX_INCREMENT'),
    use_thermal_validity_stop=True,
    thermal_validity_Tmax_K=1811.0,
    thermal_validity_Tmean_K=1700.0,
    use_mechanical_validity_stop=b('$MECH_VALIDITY_STOP'),
    mechanical_validity_mode='fit_or_ideal',
    mechanical_validity_fit_fraction=1.0,
    mechanical_validity_ideal_mu_frac=0.12,

    # collective slip / organization branch
    use_collective_taylor=True,
    use_rho_state_partition=True,
    use_collective_organization=True,
    use_collective_activity_memory=True,
    collective_activity_memory_mode='$MODE',
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

    # DRX hazard branch: restored v11-style coarse-grained patch prefactor.
    use_hazard_nucleation=b('$USE_HAZARD_NUCLEATION'),
    disable_nucleation=False,
    nuc_min_strain=float('$NUC_MIN_STRAIN'),
    nuc_interval=int(float('$NUC_INTERVAL')),
    nuc_attempt_freq=float('$NUC_ATTEMPT_FREQ'),
    nuc_hazard_rate_cap=float('$NUC_HAZARD_RATE_CAP'),
    nuc_hazard_site_floor=float('$NUC_HAZARD_SITE_FLOOR'),
    nuc_site_gb_weight=float('$NUC_SITE_GB_WEIGHT'),
    nuc_site_kappa_weight=float('$NUC_SITE_KAPPA_WEIGHT'),
    nuc_site_grad_r_weight=float('$NUC_SITE_GRAD_R_WEIGHT'),
    nuc_require_organized_structure=b('$NUC_REQUIRE_ORGANIZED_STRUCTURE'),
    nuc_wall_fraction_scale=float('$NUC_WALL_FRACTION_SCALE'),
    nuc_gnd_fraction_scale=float('$NUC_GND_FRACTION_SCALE'),
    nuc_gamma_GB=float('$NUC_GAMMA_GB'),
    nuc_rs_theta_m_deg=15.0,
    nuc_min_field_mis_deg=1.0,
    nuc_barrier_thickness_b=float('$NUC_BARRIER_THICKNESS_B'),
    nuc_comp_relief_factor=float('$NUC_COMP_RELIEF_FACTOR'),
    nuc_comp_relief_cap_factor=float('$NUC_COMP_RELIEF_CAP_FACTOR'),
    nuc_gnd_feed_efficiency=float('$NUC_GND_FEED_EFFICIENCY'),
    nuc_theta_candidates_frac=[-1.0,-0.5,0.5,1.0],
    nuc_min_radius_cells=int(float('$NUC_MIN_RADIUS_CELLS')),
    nuc_max_radius_um=float('$NUC_MAX_RADIUS_UM'),
    nuc_spinodal_barrier_factor=1.0,
    nuc_event_select='max_excess',
    nuc_excess_to_rhoGB_fraction=float('$NUC_EXCESS_TO_RHOGB_FRACTION'),
    nuc_reset_hazard_radius_factor=1.5,
    nuc_rng_seed=int('$NUC_SEED'),

    # Grain ID diagnostic: prevent bookkeeping relabeling from masquerading as DRX.
    use_component_relabel=b('$USE_COMPONENT_RELABEL'),
    component_relabel_interval=int(float('$COMPONENT_RELABEL_INTERVAL')),
    component_relabel_min_px=int(float('$COMPONENT_RELABEL_MIN_PX')),
    component_relabel_max_splits_per_step=int(float('$COMPONENT_RELABEL_MAX_SPLITS')),
    component_relabel_require_pure=True,
    track_grain_provenance=True,

    # Polycrystal / GB transmission branch
    poly_seed=int('$POLY_SEED'),
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

    # output
    plot_interval=250,
    diag_interval=25,
    save_interval=250,
    restart_interval=250,
    save_main_panels=True,
    save_signed_panels=False,
    diag_print_extended=True,
)
print(json.dumps(params))
PY
)"

cat > "$OUT/v33_params.json" <<< "$PARAMS"
echo "Override: $PARAMS"
DRX_OUTDIR="$OUT" DRX_PARAMS="$PARAMS" "$PYTHON" "$DRIVER"
