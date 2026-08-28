"""
drx_var_v32_gb_transmission_processzone_asb_sweep.py — Variational DRX/ASB simulation for BCC iron

Free energy:
  F = int[ Phi_dw(r) + kappa_r/2 |grad r|^2
         + sum_i kappa_eta/2 |grad eta_i|^2
         + W sum_{i!=j} eta_i^2 eta_j^2
         + F_comp(kappa, grad_psi, rho_GB) ] dV

  r = rho/rho_c  (normalised dislocation density)
  Phi_dw(r): double-well with wells at r_lo (interior) and r_hi (GB/wall)
  F_comp: compatibility penalty coupling GND, orientation, boundary content

Explicit kinetics (NOT from F):
  - Arrhenius slip via EXP-floor barrier
  - Moulinec-Suquet heterogeneous stress
  - Kocks-Mecking storage/recovery (recovery modulated by Phi'' curvature)
  - Orowan advection of signed dislocation populations
  - Taylor-Quinney adiabatic heating
  - Swept-front density cleaning on label change
  - Stochastic free-energy-gated nucleation
  - v9 Arrhenius Hall-Petch grain-boundary source/transmission kinetics
  - v10 connected-component topology relabeling for grain fields
  - v11 GND-bounded cumulative-hazard nucleation with trial misorientation budget
  - v12 comoving GB-GND projection so initialized GB GND does not trail
  - v13 purity-aware GB support and topology relabeling to prevent mixed-grain artifacts
  - v14 switchable Arrhenius–Taylor stress-as-energy variational potential modes
  - v15 provenance accounting for spinodal/topology vs hazard grain births
  - v16 Arrhenius-sigma defaults and disk-safe plotting/output controls
  - v17 local-temperature recovery and local plastic-dissipation heating
  - v18 work-conjugate Arrhenius-Taylor b*ell integral potential as default
  - v19 implicit spectral heat update for stable local heating
  - v20 lattice-diffusion-assisted local recovery added to Kocks-Mecking
  - v21 split Hall-Petch xi into source/transmission/sink channels
  - v22 thermodynamic stored-energy functional + Arrhenius kinetic instability diagnostics
  - v25 full restart/checkpoint workflow and ASB hot-band branch diagnostics
  - v27 dislocation-state partition: mobile/forest/wall density and collective organization
  - v26e collective Taylor with thermal + mechanical validity controls
  - v26c collective-domain Arrhenius-Taylor depinning hazard
  - v28 finite-elastic loading + persistent collective activity field for ASB
  - v30 Arrhenius GB slip-transmission barrier and residual-Burgers rotation
  - v31 finite-loading plastic-work budget + GB-core hard-orientation slip transfer
  - v32 finite ASB process-zone heat kernel + blocked-GB work partition
"""

import numpy as np
from scipy.optimize import brentq
from scipy import ndimage
from scipy.special import gammainc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import os, json, time as _wtime, csv

# ================================================================
# 1. PHYSICAL PARAMETERS — no gates, no sigmoid floors/caps
# ================================================================
P = dict(
    # -- Grid --
    Nx=128, Ny=128, L_phys=10.0e-6,

    # -- Material (BCC iron) --
    b=2.48e-10,
    C11=228e9, C12=132e9, C44=116.5e9,
    rho_min=1e8, rho_max=5e18,

    # -- Arrhenius barrier model --
    # Default is the fitted EXP/EXP-floor form:
    #   G*(sigma,T)=G0(T)[f+(1-f)exp(-a(sigma/sigc(T))^n)].
    # With expf_floor=0 this reduces to the uploaded median EXP fit.
    # The old bracketed Kocks/KAA form is retained with arrhenius_barrier_model='kaa'.
    arrhenius_barrier_model='exp_floor',
    expf_Tref=1100.0, expf_G00_eV=1.90819, expf_gT=1.24174,
    expf_sigc0=1497.04e6, expf_sT=0.108506,
    expf_a=2.20562, expf_n=2.52073, expf_floor=0.0,
    expf_sigma_ratio_cap=6.0,
    pTaylor=4.0, eta0=1.0e12,

    # -- v26 collective/multi-hit Taylor depinning --
    # The old independent Arrhenius-Taylor slip law is recovered with
    # use_collective_taylor=False.  When enabled, the old site prefactor
    # (b/X)^p is partitioned into independent correlated domains and a
    # within-domain Poisson-tail completion probability.  This prevents dense
    # forest junctions from being treated as independent strain-producing
    # events and adds diagnostics for cooperative length, hit number, and
    # suppression relative to the isolated-event rate.
    use_collective_taylor=True,
    collective_taylor_mode='multi_hit',     # 'multi_hit' or 'independent'/'off'
    # v26c: rate closure.  'domain_count' replaces the isolated-site prefactor
    # by the number of correlated domains and uses the Poisson tail only as an
    # activity/cooperativity diagnostic.  'poisson_tail' recovers the stronger
    # v26b simultaneous multi-hit rate closure, which can over-harden and drive
    # thermal runaway when used as the imposed-rate constitutive inverse.
    collective_rate_closure='domain_count', # 'domain_count' or 'poisson_tail'
    collective_domain_power=2.0,            # v27c: 2D correlated-domain count; N_eff=N_site/n_c^2
    collective_min_suppression=1.0e-4,      # v27c: tiny mobility floor; avoids singular rate inversion
    collective_tc_mode='fixed',             # 'fixed' or 'elastic'
    collective_tc=1.0e-9,                   # renewal/correlation window [s]
    collective_tc_min=1.0e-11,
    collective_tc_max=1.0e-7,
    collective_v_char=2.0e3,                # used only when tc_mode='elastic' [m/s]
    collective_C_el=1.0,                    # elastic triggering length prefactor
    collective_tau0_MPa=100.0,               # regularizes sigc-sigma margin [MPa]
    collective_ell_max_um=0.10,             # upper bound on local correlated segment [um]
    collective_nc_max=20.0,
    collective_eta_m=0.25,                  # fraction of local constraints needed as hits
    collective_m_min=1.0,
    collective_m_max=8.0,
    collective_m_round=False,               # keep smooth non-integer m via gamma tail
    collective_density_switch=None,         # optional [m^-2]; None -> no density gate
    collective_density_switch_power=4.0,
    collective_diag=True,
    collective_rhostate_enforce_domain_power=True,
    collective_rhostate_min_domain_power=2.0,
    collective_rhostate_disable_min_suppression=False,

    # -- v27 dislocation-state organization --
    # rho_mobile is represented by the signed glissile populations rp/rm.
    # rho_forest is a slip-system-resolved immobile/forest/junction reservoir.
    # rho_wall is an organized wall/cell/subgrain-boundary density.  These are
    # state partitions of one dislocation network, not separate critical densities.
    use_rho_state_partition=True,
    rho_state_mobile_fraction=0.65,        # initial fraction of total rho in rp/rm
    rho_state_forest_fraction=0.35,        # initial fraction in rho_forest
    rho_state_wall_fraction=0.00,          # initial organized wall density fraction
    rho_state_load_from_restart=True,
    rho_state_ref_mode='initial_structural', # v27c: CH scale is structural density, not total/mobile density
    rho_state_ref_abs=None,                # optional structural density scale [m^-2]
    rho_state_use_structural_scale_for_ch=True,
    rho_state_use_total_for_ch=False,
    rho_state_use_total_for_energy=True,
    rho_state_obstacle_mode='forest_wall_gnd',  # 'mobile_total', 'total', 'forest_wall_gnd'
    rho_state_ch_density_mode='structural',  # v27c: CH acts on forest+wall structural density
    rho_state_ch_redistribute_mode='structural_only',
    rho_state_forest_mix=0.35,             # mixture of local forest and total forest
    rho_state_mobile_obstacle_weight=0.10, # residual mobile contribution to Taylor obstacles
    rho_state_wall_obstacle_weight=0.35,      # v27c: wall density constrains motion, but is not all forest
    rho_state_gb_obstacle_weight=0.25,        # v27c: avoid GB-shell runaway in Taylor obstacles
    rho_state_gnd_obstacle_weight=0.10,       # v27c: GND contribution is a backstress-like correction
    rho_state_store_to_forest=True,
    rho_state_mobile_recovery_factor=1.0,
    rho_state_forest_recovery_factor=0.35,
    rho_state_wall_recovery_factor=0.05,
    rho_state_min_component=1e8,

    # Collective activity changes what plastic slip does to the network.
    # It does not by itself nucleate grains; DRX still comes from variational
    # instability or finite-amplitude nucleation hazard.
    use_collective_organization=True,
    collective_activity_mode='suppression_nc',  # 'pcomplete', 'suppression', 'suppression_nc'
    collective_activity_floor=0.0,
    collective_activity_power=1.0,
    collective_activity_smooth_um=0.10,    # v27c: coarse-grained elastic correlation width for A_coll
    collective_storage_boost=0.25,          # v27c: conservative organization strength
    collective_mobile_to_forest=0.02,      # v27c: slow mobile -> forest locking      # extra locking per unit |gamma_dot|
    collective_wall_conversion=0.03,       # v27c: slow forest -> wall organization       # forest -> wall per unit |gamma_dot|
    collective_wall_gnd_weight=0.50,
    collective_wall_grad_gamma_weight=0.25,
    collective_wall_max_frac_step=0.03,    # v27c: explicit state-transfer limiter    # explicit-update limiter only
    collective_wall_relax_tau=2.0e-6,
    collective_heat_partition_weight=0.0,        # v29 default: heat from local tau*gdot; no extra heat multiplier
    collective_heat_partition_power=1.0,

    # -- v29/v30 persistent activity memory and GB transmission physics for ASB organization --
    # Default is NOT the old fixed lab-frame band memory.  The production path is
    # local/slip-system resolved persistence: a site that was recently active is
    # slightly easier to keep active, but no spatial band direction is imposed.
    # Optional modes are controls/diagnostics:
    #   none                 : disable A memory
    #   local                : scalar local persistence only
    #   crystallographic_local: per-slip local persistence; no spatial transport
    #   isotropic            : scalar isotropic diffusion control
    #   fixed_lab_slip_control: old v28 fixed lab-frame anisotropic control
    #   crystallographic     : experimental per-slip, grain-orientation aligned diffusion
    use_collective_activity_memory=True,
    collective_activity_memory_mode='crystallographic_local',
    collective_activity_tau=2.0e-6,
    collective_activity_D_parallel=0.0,      # production default: no imposed spatial band memory
    collective_activity_D_perp=0.0,          # set only for explicit diagnostic controls
    collective_activity_memory_source_weight=1.0,
    collective_activity_memory_use_gdot=True,
    collective_activity_memory_gdot_ref=None, # None -> abs(edot_app)
    collective_activity_rate_weight=1.0,     # A_s increases local correlated-slip susceptibility
    collective_activity_rate_power=1.0,
    collective_heat_use_activity_memory=True,

    # -- Temperature --
    T0=1300.0,
    taylor_quinney=0.9, cp_rho_vol=3.8e6,
    k_thermal=0.03, T_bath_coupling=5.0e9,
    # v17: heat generation is local plastic dissipation, not a global scalar σ_bar*edot.
    # local_heat_stress='effective' uses (resolved shear - backstress) as the
    # thermodynamic driving stress; 'resolved' uses the resolved shear stress itself.
    use_local_heat_source=True,
    local_heat_stress='effective',      # 'effective' or 'resolved'
    local_heat_nonnegative=True,        # clip negative local power; prevents elastic unloading from cooling
    local_heat_floor_qdot=0.0,          # W/m^3; numerical floor only
    # Energy/rate consistency controls.  Under imposed total strain rate, the
    # volume-averaged plastic heat input should be tied to sigma_bar*edot_app,
    # while local activity can partition that heat spatially.  The raw
    # sum(tau_local*gdot_local) is useful diagnostically but can violate the
    # macroscopic work budget after the heterogeneous stress re-solve.
    enforce_macro_rate_after_ms=True,
    use_energy_conserving_heat=True,
    heat_partition_tiny=1.0e-300,
    # v31: finite-loading work-budget consistency.  Local tau*gdot is used as
    # the spatial partition of plastic work, but total heat cannot exceed the
    # external work plus released elastic energy during the step.
    use_finite_loading_work_budget=True,
    finite_loading_work_budget_safety=0.95,
    finite_loading_allow_elastic_unload=True,
    finite_loading_scale_gdot_to_budget=True,
    # v32: finite plastic/accommodation process-zone.  The finite-loading work
    # budget sets the total available heat, while this kernel prevents that
    # allowed work from collapsing into one or two grid cells.  It represents
    # the finite width of slip-transfer/avalanche/accommodation zones.
    use_heat_process_zone_kernel=True,
    heat_process_zone_sigma_um=0.30,
    heat_process_zone_min_sigma_px=2.0,
    heat_process_zone_preserve_mean=True,
    # v32: blocked GB slip should not all become local heat.  A fraction of the
    # incompatible GB work is stored as residual GB content / backstress proxy
    # or dissipated by GB sliding/accommodation; only gb_blocked_work_heat_fraction
    # remains in the local heat partition.
    use_gb_blocked_work_partition=True,
    gb_blocked_work_heat_fraction=0.25,
    gb_blocked_work_store_fraction=0.50,
    gb_blocked_work_support_power=2.0,
    gb_blocked_work_factor_power=1.0,
    gb_blocked_work_store_to_rhoGB=True,
    gb_blocked_work_rhoGB_max_frac_step=0.05,
    # v19: local heating creates high-k temperature structure, so explicit spectral
    # diffusion is unstable unless dt << dx^2/(4 alpha).  Use an implicit spectral
    # heat solve by default.
    heat_update_mode='implicit_spectral',  # 'implicit_spectral' or 'explicit'
    heat_enforce_finite=True,
    heat_min_K=1.0,

    # -- v26d physical thermal-validity / adaptive-time controls --
    # These are not heat gates or stress caps. They are numerical and model-validity
    # constraints for stiff thermo-Arrhenius coupling.
    use_adaptive_thermal_dt=True,
    dt_base=None,
    # v28b: choose base timestep by strain increment when dt_base is not supplied.
    # This is essential for rate sweeps: dt = dt_strain_step / |edot_app|.
    # Set dt_base_mode='legacy' to use the historical fixed dt value.
    dt_base_mode='strain_increment',  # 'strain_increment' or 'legacy'
    dt_strain_step=1.0e-3,
    dt_base_min=1.0e-11,
    dt_base_max=None,
    thermal_dt_max_dT_step=3.0,
    thermal_dt_log_change=0.05,
    thermal_dt_Q_eV=None,
    thermal_dt_min=1.0e-11,
    thermal_dt_print_changes=True,
    use_thermal_validity_stop=True,
    thermal_validity_Tmax_K=1811.0,
    thermal_validity_Tmean_K=1700.0,
    thermal_validity_save_restart=True,
    thermal_validity_stop_reason='solidus/model-validity boundary reached',

    # v26e: mechanical/rate-solvability validity check.  This is not a stress
    # gate used to continue the run; it stops before the code heats/updates a
    # solid-state model with stresses outside the Arrhenius fit/ideal-strength
    # regime.  For the EXP-floor fit, the local activation stress is meaningful
    # only up to expf_sigma_ratio_cap*sigc(T).  The macroscopic stress limit is
    # converted back through the Taylor/geometry drive factor.
    use_mechanical_validity_stop=True,
    mechanical_validity_mode='fit_or_ideal',  # 'fit', 'ideal', 'fit_or_ideal', 'off'
    mechanical_validity_sigma_MPa=None,       # optional explicit macro-stress validity limit
    mechanical_validity_fit_fraction=1.0,
    mechanical_validity_ideal_mu_frac=0.12,   # ideal shear/tensile scale O(mu/10)
    mechanical_validity_save_restart=True,
    mechanical_validity_stop_reason='rate-control stress exceeded constitutive validity range',

    # High-rate/low-T runs can make rho_c far above any physical density scale.
    # The initial density should then be specified or bounded by rho_max rather
    # than blindly initialized at 0.9*rho_c.
    rho0_mode='relative_to_rho_c',
    rho0_abs=None,
    rho0_rhoc_frac=0.90,
    rho0_cap_to_rho_max=True,
    rho0_max_frac_rho_max=0.75,

    potential_T_min=300.0,
    potential_T_max=3500.0,

    # -- Loading --
    edot_app=3.0e3, nSlip=2,
    # v28: finite elastic/viscoplastic loading for ASB.  This replaces the
    # algebraic imposed-rate bisection with sigma_dot=E_eff*(edot_app-<edot_p>).
    # Under this mode the model heats from actual local plastic work, not from a
    # globally renormalized sigma*edot_app heat budget.
    use_finite_elastic_loading=True,
    finite_loading_Eeff=None,          # None -> plane-strain modulus E_ps
    finite_loading_damping=1.0,        # explicit stress update damping; 1 = physical Euler
    finite_loading_nonnegative_stress=True,
    finite_loading_init_from_inverse=False,
    finite_loading_print=True,
    use_local_slip_increment_dt=True,
    local_slip_max_increment=2.0e-2,   # numerical resolution criterion for |Delta gamma| per step
    local_gdot_cap_factor=0.0,         # <=0 disables old local gdot cap
    slip_angles_deg=[25, -65],

    # -- Kocks-Mecking --
    KM_k1=5.0e8,
    KM_k2_0=65.0,     # pre-exponential for recovery coefficient
    KM_Q2_eV=0.45,
    KM_chi=0.5,        # forest mixing (0=per-slip, 1=total)
    # v17: dynamic recovery is evaluated using the local evolving temperature field T(x,y).
    # This keeps hot zones from retaining an artificially low, T0-based recovery rate.
    KM_recovery_local_T=True,
    KM_T_min=300.0,
    KM_T_max=3500.0,

    # -- v20 lattice-diffusion-assisted local recovery --
    # Added to the traditional Kocks-Mecking storage/recovery balance.
    # Form: dρ/dt = K (D_L/b) ρ_rec^(3/2) [exp(κ μ b^4 sqrt(ρ_rec)/(kT)) - 1]
    # D_L uses a conservative Fe-superalloy lattice/self-diffusion scale with an
    # alloy slowdown factor.  The max-fraction limiter only prevents a single
    # explicit step from deleting more density than is available.
    use_lattice_diffusive_recovery=True,
    diffrec_K=2.0e-2,
    diffrec_D0_m2_s=1.5e-4,
    diffrec_Q_eV=2.90,
    diffrec_alloy_D_factor=0.10,
    diffrec_stress_coeff=1.0,
    diffrec_rho_floor=0.0,
    diffrec_T_min=300.0,
    diffrec_T_max=3500.0,
    diffrec_exp_arg_cap=60.0,
    diffrec_max_frac_step=0.05,

    # -- Cahn-Hilliard (normalised r = rho/rho_c) --
    # M_ch [m^2/s], kappa_r [m^2]: set for ~3-cell interface, ~30-step e-folding
    use_ch_step=True,
    M_ch=5.0e-7,
    ch_base_scale=0.050,      # v27c: structural-density CH is slower than mobile kinetics
    kappa_r=5.0e-13,
    CH_Cs=8.0,          # stabilisation constant (raised for v7 low-rho penalty)

    # v6: variable-mobility CH and rho-eta coupling.
    # The CH mobility barrier suppresses density flux through current KWC/GB support
    # without pinning rho to a prescribed GB mask.  The rho-eta term is a true
    # free-energy coupling: high-rho/spinodal/GND-precursor regions lower the cost
    # of diffuse KWC support and add the conjugate contribution to mu_CH.
    use_ch_mobility_barrier=True,
    use_ch_plasticity_mobility=False, # v22: CH is thermodynamic regularization; glide mobility comes from Orowan velocity
    ch_plasticity_power=0.50,
    ch_plasticity_floor_frac=0.10,
    ch_plasticity_cap_frac=2.0,
    ch_mobility_gb_barrier=0.90,      # M -> (1-barrier)M on GB support
    ch_mobility_floor_frac=0.03,      # never zero; avoids stranded mass
    use_ch_increment_limiter=True,    # v7: prevent single-step CH runaway
    ch_max_frac_step=0.05,            # cap |Δr_CH| per global step
    use_rho_eta_coupling=True,
    rho_eta_mu_strength=0.01,         # v27c: very weak density/eta bias; wall field is not a grain         # dimensionless addition to mu_dw
    rho_eta_ac_strength=5.0e5,        # v27c: suppress eta-label fragmentation from rho noise        # J/m^3; KWC preference for GB support in high-rho/GND zones
    rho_eta_use_relative_r=True,      # v7: couple to locally high rho, not absolute rho/rho_c only
    rho_eta_r_lo=1.05,
    rho_eta_r_hi=1.65,
    rho_eta_use_grad_r=True,
    rho_eta_grad_weight=0.20,
    rho_eta_kappa_weight=1.00,
    rho_eta_gb_weight=0.05,
    rho_eta_precursor_floor=0.0,
    rho_eta_mobility_on_precursor=False,

    # v7: soft low-density penalty.  rho_min is numerical only; this keeps CH
    # from treating near-zero dislocation density as a free reservoir under load.
    use_lowrho_soft_penalty=True,
    rho_soft_floor_frac=0.03,
    lowrho_mu_strength=1.0,

    # v23 Arrhenius-Taylor kinetic diagnostic: dσ_AT/dρ<0 is reported, but
    # it is NOT a production hazard gate by default.  Arrhenius kinetics enter
    # the nucleation hazard through local slip activity / Orowan flux, while
    # the finite-amplitude barrier remains thermodynamic.
    use_arrhenius_kinetic_instability=False,
    arrhenius_hazard_gate_mode='diagnostic',  # 'diagnostic', 'hard', or 'soft' if explicitly enabled
    arrhenius_hazard_gate_floor=1.0,
    use_plastic_activity_hazard_prefactor=True,
    hazard_activity_prefactor_mode='gdot',    # 'gdot' or 'orowan_flux'
    hazard_activity_floor=1.0e-4,
    hazard_activity_cap=50.0,
    hazard_activity_power=1.0,
    hazard_activity_ref=None,                 # None -> edot_app

    # v22 mechanical-power cap on positive KM storage.  Stored dislocation
    # energy cannot be generated faster than the stored fraction of local plastic
    # mechanical power.
    use_storage_energy_cap=True,
    stored_work_fraction=None,             # None -> 1 - taylor_quinney
    storage_cap_apply_to_KM=True,
    storage_power_use_absolute=True,        # robust first implementation for sign conventions
    storage_cap_tiny=1e-300,


    # -- Variational thermodynamic density potential Φ_th(ρ,T) --
    # v22 production default: conventional stored-energy branch.
    #
    #   Φ_th = E*(ρ,T) + Φ_ent(ρ,T) + Φ_ord(ρ,T) + Φ_lowρ
    #   E*(ρ,T)=A_E(T)ρ, A_E(T)=0.5 μ(T)b² by default.
    #
    # The Arrhenius-Taylor curve is retained for local slip kinetics and the
    # kinetic instability diagnostic dσ_AT/dρ<0, but is not used as the production
    # thermodynamic free-energy density.  The Arrhenius modes remain available as
    # ablation/diagnostic modes.
    # choices:
    #   'thermo_stored'/'stored_energy' : E*(ρ,T) + entropy + ordering + low-rho (v22 default)
    #   'surrogate'                    : legacy alias for thermo_stored
    #   'arrhenius_sigma'              : diagnostic: scale*σ_AT(ρ)
    #   'arrhenius_sigma_plus_elastic' : diagnostic: scale*σ_AT(ρ) + E*(ρ,T)
    #   'arrhenius_integral_b2'        : diagnostic: scale*b²∫σ_AT dρ
    #   'arrhenius_work'               : diagnostic: scale*∫σ_AT bℓ dρ
    potential_mode='thermo_stored',
    arrhenius_phi_scale=1.0,
    arrhenius_work_length_mode='X_peak',   # diagnostic only: 'b', 'X_peak', or 'X_local'
    arrhenius_phi_use_taylor_concentration=True,  # σ_macro = σ_local*(b/X); gives Taylor peak
    arrhenius_phi_concentration_prefactor=1.0,
    use_potential_entropy=True,
    use_potential_ordering=True,
    use_potential_lowrho_penalty=True,
    alpha_Taylor=0.30,      # optional Taylor coefficient if Estar_use_alpha=True
    # v22 stored-energy thermodynamic coefficients.  Defaults use full line energy
    # E*=0.5*mu(T)*b^2*rho rather than alpha-weighted Taylor energy.
    use_temperature_dependent_Estar=True,
    Estar_use_alpha=False,
    Estar_mu0=82.0e9,
    Estar_dmu_dT=-3.7e7,
    Estar_mu_floor=5.0e9,
    C_ent_frac=0.06,        # entropy coeff as fraction of A_E(T)
    rho_ref_ent=1e14,       # reference density for entropy term [1/m²]
    r_ord=1.30,             # center of ordering dip: wall state (~rho_c, matches KM equil)
    w_r_ord=0.30,           # width of ordering dip in r-units
    A_ord_scale=1.2,        # A_ord = A_ord_scale * A1 * ρ_c * w_r (sets ~O(1) dip in μ̃)

    # -- Allen-Cahn --
    freeze_kwc_eta=False,            # ablation: keep eta/lab fixed while CH/KM run
    freeze_orientation=False,        # ablation: keep psi_plastic fixed
    freeze_rhoGB=False,              # ablation: keep rho_GB fixed
    disable_nucleation=False,
    L_ac=3.0e-3,          # v7: modestly slower GB/AC mobility; preserves growth but resolves it
    # Numerical explicit-AC limiter, analogous to the CH increment limiter.
    # It prevents one time step from reassigning most pixels/grains when mobility
    # is high, but does not create a thermodynamic gate or stop GB motion.
    use_ac_increment_limiter=True,
    ac_max_abs_step=0.02,

    # Temperature-dependent GB/KWC mobility.
    # Relative Arrhenius form keeps v25 unchanged at gb_mobility_Tref:
    #   M_GB(T) = M_GB(Tref) exp[-Q/kB (1/T - 1/Tref)]
    # This is not a gate/stop; it lets hot bands have faster GB migration.
    use_temperature_dependent_gb_mobility=False,
    gb_mobility_Q_eV=2.0,
    gb_mobility_Tref=None,          # None -> P['T0']
    gb_mobility_use_local_T=False,   # average-temperature mobility for production sweeps
    gb_mobility_exp_arg_clip=80.0,  # overflow guard only, not a physical cap
    gb_mobility_apply_to_rhoGB_relax=True,
    gb_mobility_diag=True,
    kappa_eta=5.0e-7,     # gradient energy (sets ~2-cell interface width)
    W_eta=5.0e6,          # multi-phase barrier height
    sweep_wake_rho=1e14,

    # -- Compatibility penalty (replaces gate fields) --
    A_alpha=5.0e-9,     # GND-orientation coupling [J*m]
    A_GB=2.0e-8,        # rho_GB-orientation coupling [J*m]
    c_alpha=1.0, c_GB=1.0,

    # v4 variational-closure controls
    # Orientation is reconstructed as psi_lat = sum_i eta_i*psi_i + psi_plastic.
    use_grain_slaved_orientation=True,
    psi_plastic_max_deg=25.0,
    M_psi_plastic=5.0e-7,
    plastic_spin_weight=0.07,

    # v30: residual Burgers content left in a GB by failed/partial slip
    # transmission can drive local lattice/grain rotation to reduce that residual.
    use_gb_residual_rotation=True,
    gb_residual_rotation_rate=2.0e3,     # 1/s; v31 slower cooperative GB residual relaxation
    gb_residual_rotation_rho_scale=None, # None -> rho_state_ref/runtime structural density
    gb_residual_rotation_cap_deg_step=0.01,
    gb_residual_rotation_requires_net_signed=True,
    gb_residual_rotation_net_power=1.0,
    gb_residual_rotation_smooth_um=0.35,
    gb_residual_rotation_smooth_passes=1,
    gb_residual_rotation_elastic_penalty=0.25,

    # Frank-Bilby boundary-density relaxation: rho_GB -> |grad psi|/b on diffuse GBs.
    use_frank_bilby_rhoGB=True,
    frank_bilby_coeff=1.0,
    rhoGB_relax_tau=2.0e-6,
    rhoGB_decay_tau=1.0e-6,
    rhoGB_absorb_mobile=False,  # v8: use Arrhenius HP sink instead of direct absorption by default
    rhoGB_absorb_fraction=0.75,
    gb_support_floor=0.02,

    # v12 GB-bound GND handling.  GND generated by the existence of a GB
    # should move/disappear with the GB instead of being left behind as an
    # artificial lattice nucleation seed.  This projection only acts where
    # diffuse GB support decreased between steps; it neutralizes sign imbalance
    # while preserving total mobile density.
    use_gb_comoving_gnd=True,
    gb_comoving_support_power=4.0,
    gb_comoving_relax_tau=2.5e-7,       # fast relative to GB motion, but finite
    gb_comoving_max_frac_step=0.35,     # numerical limiter on sign neutralization
    gb_comoving_hazard_reset_strength=4.0,

    # v21 Arrhenius Hall-Petch GB source + slip-transmission channel.
    # GBs are active sources and transmission sites, not strong density sinks.
    # Important correction: the pileup amplification xi~sqrt(d/X) is now used
    # only for pileup-assisted slip transmission.  GB source nucleation uses a
    # separate GB-step/triple-junction concentration that does not grow with the
    # total dislocation density; the weak residual-storage/sink branch uses no
    # strong HP amplification.  This avoids the old unphysical rho^(1/4) boost
    # of GB source activity.
    use_gb_hp_source_sink=True,
    gb_hp_A_mult=7.0,              # GB barrier height relative to Taylor barrier (5-10x suggested)
    gb_hp_xi_floor=1.0,
    gb_hp_xi_cap=80.0,             # legacy/global cap retained for diagnostics/backward compatibility
    gb_hp_rate_cap=5.0e6,          # 1/s; numerical cap on event probability per global step
    gb_hp_min_gb_support=0.25,     # restrict HP activity to real GB cores, not diffuse tails
    gb_hp_support_power=4.0,       # further localize HP activity to GB cores

    # Split stress-concentration factors for physically distinct GB processes.
    gb_hp_source_xi_prefactor=8.0, # source: GB step/TJ concentration, independent of positive rho scaling
    gb_hp_source_xi_floor=1.0,
    gb_hp_source_xi_cap=20.0,
    gb_hp_source_use_backstress_screen=True,
    gb_hp_source_rho_screen_frac=1.0,  # screen source xi as rho/rho_c grows
    gb_hp_trans_xi_prefactor=1.0,  # transmission: pileup xi=sqrt(d_g/X_pileup)
    gb_hp_trans_xi_floor=1.0,
    gb_hp_trans_xi_cap=80.0,
    gb_hp_sink_xi_prefactor=1.0,   # weak residual storage: no strong stress amplification
    gb_hp_sink_xi_floor=1.0,
    gb_hp_sink_xi_cap=5.0,
    gb_hp_source_strength=0.03,    # v27c: conservative GB replenishing source
    gb_hp_source_density_scale='initial_total', # v27c: never scale GB source by independent kinetic rho_c
    gb_hp_source_sat_frac=0.95,    # source shuts off near normal deforming density
    gb_hp_source_signed_bias=0.00, # neutral source by default; GND comes from compatibility feedback
    gb_hp_transmission_strength=0.60,            # conservative redistribution rate across slip systems
    gb_hp_transmission_residual_fraction=0.06,   # transmitted content stored as residual GB Burgers content

    # v30: Arrhenius slip-transmission barrier at GBs.
    # The added barrier depends on crystallographic mismatch and on the residual
    # Burgers vector that must remain in the boundary if slip crosses.  This is
    # deliberately applied to both local slip mobility at GB cores and to the
    # HP transmission hazard, so a thermal band cannot pass through arbitrary
    # high-misorientation boundaries for free.
    use_gb_slip_transmission_barrier=True,
    gb_transmission_mode='arrhenius_residual_burgers',
    gb_trans_misorientation_barrier_eV=0.80,
    gb_trans_residual_barrier_eV=1.20,
    gb_trans_barrier_power=2.0,
    gb_trans_mis_ref_deg=30.0,
    gb_trans_bres_ref=0.75,
    gb_trans_min_factor=1.0e-6,
    gb_trans_gdot_coupling=1.0,
    gb_trans_use_neighbor_worst=True,
    gb_trans_outgoing_mode='same_index', # 'same_index' blocks straight bands unless same slip transmits; 'best' uses best outgoing slip
    gb_trans_store_residual_scale=1.5,
    gb_trans_diag=True,
    gb_trans_use_hard_grain_orientation=True,
    gb_trans_include_plastic_orientation=True,

    gb_hp_weak_sink_strength=0.015,              # very weak mobile -> rho_GB storage, capacity limited
    gb_hp_sink_floor_rho=1e14,                   # lower bound for weak residual storage only
    gb_hp_sink_to_rhoGB_fraction=1.00,


    # Variational signed-GND feedback: dF/dkappa drives rho+ <-> rho- exchange.
    use_signed_gnd_feedback=True,
    signed_gnd_feedback_rate=2.0e4,   # 1/s; gentle because dt is small
    signed_gnd_max_frac_step=0.02,
    signed_gnd_slip_weight_floor=0.05,

    # Temperature-dependent potential update.
    update_potential_with_temperature=True,
    potential_update_interval=50,

    # -- Stress solver --
    ms_iters=3, hs_alpha=0.80,

    # -- Advection --
    use_advection=True, v_cfl_frac=0.25, rho_mobile_min=1e13,

    # -- Orientation --
    M_psi=5.0e-7, psi_max_deg=75.0,

    # -- Nucleation / finite-amplitude GB formation --
    # v11 replaces hard candidate gates (rho>threshold, kappa>threshold,
    # fixed radius, fixed misorientation) by a cumulative hazard.  The barrier is
    # estimated from a local classical-nucleation free-energy balance using the
    # same AT/KWC/compatibility fields.  Candidate misorientation is bounded by
    # the local residual Burgers/GND budget.  The CH/AC spinodal pathway remains
    # active independently when Phi''<0.
    use_hazard_nucleation=True,
    nuc_interval=100,                # v27c: finite-amplitude nucleation is slow/incubated                 # hazard integration/evaluation stride
    nuc_min_strain=0.05,             # v27c: require some organization before discrete nuclei              # keep for optional startup suppression only
    nuc_attempt_freq=1.0e2,         # v27c: conservative patch-scale attempt rate          # v16 default: conservative hazard attempt rate per local patch
    nuc_hazard_rate_cap=1.0e7,       # 1/s, numerical cap only
    nuc_hazard_site_floor=0.0,      # v27c: no bulk floor; organization supplies sites      # weak floor: interiors can nucleate, GB/walls faster
    nuc_site_gb_weight=0.35,
    nuc_site_kappa_weight=0.30,
    nuc_site_grad_r_weight=0.10,
    nuc_gamma_GB=0.50,               # high-angle GB energy at T0 [J/m^2]
    nuc_rs_theta_m_deg=15.0,         # Read-Shockley saturation angle
    nuc_min_field_mis_deg=3.0,      # v27c: sub-degree changes are wall recovery, not new eta fields      # numerical: below this, treat as recovery/subgrain, not new eta field
    nuc_barrier_thickness_b=2.0,     # converts 2-D barrier per-depth to event energy
    nuc_comp_relief_factor=0.05,
    nuc_comp_relief_cap_factor=0.25, # v27c: compatibility relief cannot dominate stored-energy relief  # cap compatibility relief to O(stored-energy relief)
    nuc_gnd_feed_efficiency=0.50,    # fraction of residual GND available to feed new GB
    nuc_theta_candidates_frac=[-1.0,-0.5,0.5,1.0],
    nuc_require_organized_structure=True, # v27c: discrete nuclei require wall/GND organization
    nuc_wall_fraction_scale=0.03,
    nuc_gnd_fraction_scale=0.05,
    nuc_min_radius_cells=2,          # numerical phase-field resolution limit
    nuc_max_radius_um=0.35,          # numerical cap to avoid inserting whole grains
    nuc_spinodal_barrier_factor=1.0, # spinodal handled by CH/AC; no artificial barrier collapse
    nuc_event_select='max_excess',   # max_excess or min_barrier
    nuc_excess_to_rhoGB_fraction=0.25,# v27c: avoid GB-shell density feedback# conserve removed mobile density into boundary shell
    nuc_reset_hazard_radius_factor=1.5,
    nuc_rng_seed=271828,

    # v34 candidate-nucleus incubation.  A cumulative-hazard event no longer
    # creates a permanent grain ID immediately.  It starts a local candidate
    # nucleus; the candidate must remain thermodynamically/kinetically viable for
    # several hazard evaluations before an eta field is allocated.  This prevents
    # transient hazard spikes or GB hot spots from masquerading as DRX grains.
    use_nuc_candidate_incubation=True,
    nuc_candidate_hold_evals=6,
    nuc_candidate_decay_evals=2,
    nuc_candidate_max_barrier_eV=1.5,
    nuc_candidate_min_rate=0.0,
    nuc_candidate_min_dF_Jm3=0.0,
    nuc_candidate_promote_select='oldest',      # oldest, max_excess, min_barrier
    nuc_candidate_diagnostic_only=False,

    # Legacy knobs retained only for compatibility with old parameter files; they
    # are not used by the v11 hazard-nucleation path unless use_hazard_nucleation=False.
    nuc_rho_ratio=1.5,
    nuc_min_kappa_frac=0.04,
    nuc_min_gradpsi_deg_um=0.20,
    nuc_radius_um=0.30,
    nuc_mis_deg=4.0,
    nuc_dF_margin=0.0,
    nuc_max_trials=8,

    # -- Polycrystal --
    poly_n=12, poly_spread_deg=70.0, poly_min_mis_deg=8.0,
    grain_max=320, poly_seed=42,     # v16 default: 10x original 32; override to 3200 for very large fields

    # -- Grain-topology bookkeeping --
    # Multiphase order-parameter fields are allowed to represent one connected
    # grain each.  Without this, a single eta_i can nucleate disconnected islands;
    # if those islands later touch, the code silently coalesces them because they
    # share an ID/orientation.  The splitter below promotes sufficiently large
    # disconnected components into unused eta fields without changing rho.
    use_component_relabel=True,
    component_relabel_interval=100,      # v27c: less eager; avoids relabeling transient ASB texture
    component_relabel_min_px=64,         # v13: split only resolved grains, not cell-scale label noise
    component_relabel_dilate_px=1,
    component_relabel_keep_orientation=True,
    component_relabel_jitter_deg=0.0,
    component_relabel_max_splits_per_step=1,
    component_relabel_min_strain=0.60,
    component_relabel_require_pure=True,
    component_relabel_eta_min=0.65,
    component_relabel_second_frac_max=0.35,

    # -- GB support / mixed-order-parameter regularisation --
    # v13 separates a true two-grain interface from a many-grain mixed patch.
    # The old support 2*(1-sum eta_i^2) becomes large for any multiphase mixture,
    # including pathological interpenetrating labels.  The top-two pair support
    # 4*eta_1*eta_2 is large for a normal two-grain diffuse boundary but remains
    # small when many eta fields are present with comparable tiny amplitudes.
    use_pairwise_gb_support=True,
    gb_pair_support_scale=4.0,
    use_purity_aware_hard_gb_edges=True,
    gb_hard_eta_min=0.65,
    gb_hard_second_frac_max=0.35,
    eta_active_thresh=0.05,
    eta_mixed_warning_entropy=1.0,
    eta_tail_zero=1.0e-12,

    # -- DRX provenance accounting --
    # v15 distinguishes physical/hazard nucleation from spinodal/topology growth.
    # grain_birth_mechanism: 0=initial, 1=topology/spinodal relabel, 2=hazard nucleus.
    # grain_origin_lineage: 0=initial lineage, 1=spinodal/topology lineage, 2=hazard lineage.
    # Topology descendants of a hazard nucleus inherit hazard lineage but retain topology birth mechanism.
    track_grain_provenance=True,
    provenance_save_arrays=True,

    # -- Time stepping --
    dt=3.333e-8, nSteps=30000,
    diag_interval=25, save_interval=250,   # v16: lower default output volume for 256x256+ runs

    # -- Diagnostics / audit trail --
    write_diag_csv=True,
    diag_csv_name='drx_v25_restart_asb_diagnostics.csv',
    diag_top_frac=0.10,          # top fraction of rho used for overlap metrics
    diag_gb_thresh=0.20,         # diffuse GB support threshold
    diag_print_extended=True,
    write_field_npz=True,
    # v16 disk-safe output controls.  Field NPZ saves still follow save_interval.
    # PNG plots are decoupled via plot_interval and save errors disable further plotting
    # rather than aborting a long run.  Signed panels are off by default because they
    # roughly double PNG output and caused the reported no-space crash.
    save_main_panels=True,
    save_signed_panels=False,
    plot_interval=500,
    plot_dpi=100,
    max_saved_png_frames=200,
    disable_plots_on_save_error=True,

    # -- v25 restart / checkpoint / ASB branch controls --
    # write_restart_npz saves exact continuation checkpoints containing eta,
    # psi_gv, plastic strain, temperature, density populations, hazard memory,
    # provenance, and RNG state.  restart_file can load either one of these
    # exact checkpoints or an older diagnostic fields_*.npz file; the latter is
    # reconstructed approximately from lab + psi_lat and should only be used for
    # short branch/basin-of-attraction tests.
    write_restart_npz=True,
    restart_interval=250,
    restart_prefix='drx_v25_restart',
    restart_file=None,
    restart_reset_clock=True,
    restart_reset_hazard=False,
    restart_reset_plastic_strain=False,
    restart_lab_to_eta_smooth_sigma=0.75,
    restart_temperature_perturb_scale=1.0,
    restart_temperature_offset_K=0.0,
    restart_E11=None,

    # Hot-band / ASB diagnostics.  These do not affect evolution; they only
    # quantify whether a hot/plastic-power band is also becoming a low-rho band.
    asb_diag_hot_frac=0.05,
    asb_diag_cold_frac=0.50,
    asb_print_thermal_scales=True,
    asb_target_wavelength_um=None,  # None -> L_phys, useful for k/bath tuning estimates
)
kB_J = 1.380649e-23
eV_J = 1.602176634e-19


def _gb_mobility_factor_from_T(Tfield):
    """Relative Arrhenius GB mobility factor M(T)/M(Tref).

    This is a physical mobility factor, not a stop/gate.  The exponential is
    clipped only to prevent floating-point overflow in pathological runs.
    """
    if not P.get('use_temperature_dependent_gb_mobility', False):
        return 1.0

    TT = np.asarray(Tfield if P.get('gb_mobility_use_local_T', True) else np.nanmean(Tfield), dtype=float)
    Tref = P.get('gb_mobility_Tref', None)
    if Tref is None:
        Tref = float(P.get('T0', 1300.0))
    Tref = max(float(Tref), 1.0)

    Q = float(P.get('gb_mobility_Q_eV', 2.0))
    arg = -Q / 8.617333262145e-5 * (1.0 / np.maximum(TT, 1.0) - 1.0 / Tref)

    clip = float(P.get('gb_mobility_exp_arg_clip', 80.0))
    if clip > 0.0:
        arg = np.clip(arg, -clip, clip)

    fac = np.exp(arg)
    fac = np.where(np.isfinite(fac), fac, 1.0)
    return fac

def _km_k2_from_T(Tfield):
    """Kocks-Mecking recovery coefficient k2(T).

    v17 uses this on the local evolving temperature field during the run.
    The clipping only protects the exponential from pathological numerical
    temperatures; it is not a physics gate.
    """
    TT = np.asarray(Tfield, dtype=float)
    TT = np.clip(TT, float(P.get('KM_T_min', 300.0)), float(P.get('KM_T_max', 3500.0)))
    return P['KM_k2_0'] * np.exp(-P['KM_Q2_eV']/(8.617333262145e-5*TT))

def _mu_shear_fe_local(Tfield):
    """Vectorized Fe shear modulus [Pa], matching ATPotential.mu_shear."""
    TT = np.asarray(Tfield, dtype=float)
    mu_0 = 82e9
    dmu_dT = -3.7e7
    return np.maximum(mu_0 + dmu_dT*(TT - 300.0), 5e9)

def _lattice_diffusive_recovery_rate(rho_total, Tfield):
    """Local lattice-diffusion-assisted recovery rate dρ/dt [m^-2 s^-1].

    Form requested for v20:
        dρ/dt = K (D_L/b) ρ_rec^(3/2) [exp(κ μ b^4 sqrt(ρ_rec)/(kT)) - 1]

    This is a thermally activated climb/recovery-like sink added to the
    traditional KM storage/recovery update.  It is evaluated from the local
    evolving temperature field and local total mobile density, then applied
    proportionally to the sign/slip populations so it does not create GND.
    """
    rho_arr = np.asarray(rho_total, dtype=float)
    T_arr = np.asarray(Tfield, dtype=float)
    T0v = float(P.get('T0', 1300.0))
    T_arr = np.where(np.isfinite(T_arr), T_arr, T0v)
    T_arr = np.clip(T_arr, float(P.get('diffrec_T_min', 300.0)), float(P.get('diffrec_T_max', 3500.0)))
    rho_floor = float(P.get('diffrec_rho_floor', 0.0))
    rho_rec = np.maximum(np.where(np.isfinite(rho_arr), rho_arr, 0.0) - rho_floor, 0.0)
    D0 = float(P.get('diffrec_D0_m2_s', 1.5e-4))
    QeV = float(P.get('diffrec_Q_eV', 2.90))
    alloy = float(P.get('diffrec_alloy_D_factor', 0.10))
    D_L = alloy * D0 * np.exp(np.clip(-QeV/(8.617333262145e-5*T_arr), -700.0, 100.0))
    mu_loc = _mu_shear_fe_local(T_arr)
    b = float(P.get('b', 2.48e-10))
    kappa_coeff = float(P.get('diffrec_stress_coeff', 1.0))
    sqrt_rho = np.sqrt(np.maximum(rho_rec, 0.0))
    arg = kappa_coeff * mu_loc * b**4 * sqrt_rho / np.maximum(kB_J*T_arr, 1e-300)
    arg = np.clip(np.where(np.isfinite(arg), arg, 0.0), 0.0, float(P.get('diffrec_exp_arg_cap', 60.0)))
    rate = float(P.get('diffrec_K', 0.0)) * (D_L / max(b, 1e-300)) * np.power(rho_rec, 1.5) * np.expm1(arg)
    rate = np.where(np.isfinite(rate), rate, 0.0)
    return np.maximum(rate, 0.0), D_L, arg

# env override
_ov = os.environ.get('DRX_PARAMS', '')
if _ov:
    try:
        P.update(json.loads(_ov))
        print(f"Override: {_ov[:120]}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid DRX_PARAMS JSON: {exc}. First 240 chars: {_ov[:240]!r}") from exc
    except Exception as exc:
        raise SystemExit(f"Could not apply DRX_PARAMS override: {exc}. First 240 chars: {_ov[:240]!r}") from exc

# ================================================================
# 2. GRID + SPECTRAL OPS
# ================================================================
Nx, Ny = P['Nx'], P['Ny']
Lx = Ly = P['L_phys']
dx = dy = Lx / Nx
kx = np.fft.fftfreq(Nx, d=dx/(2*np.pi))
ky = np.fft.fftfreq(Ny, d=dy/(2*np.pi))
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2
K2nz = K2.copy(); K2nz[0,0] = 1.0
K4 = K2**2

def ddx(f): return np.real(np.fft.ifft2(1j*KX*np.fft.fft2(f)))
def ddy(f): return np.real(np.fft.ifft2(1j*KY*np.fft.fft2(f)))
def lap(f): return np.real(np.fft.ifft2(-K2*np.fft.fft2(f)))

def finite_clipped_T_mean(Tfield):
    """Robust scalar temperature for constitutive/potential table refresh."""
    Tf = np.asarray(Tfield, dtype=float)
    finite = np.isfinite(Tf)
    if not np.any(finite):
        return float(P.get('T0', 1300.0))
    Tm = float(np.nanmean(Tf[finite]))
    return float(np.clip(Tm, float(P.get('potential_T_min', 300.0)),
                         float(P.get('potential_T_max', 3500.0))))

def update_temperature_field(Tfield, qdot_field):
    """Advance temperature using local plastic power.

    v17/v18 used an explicit spectral diffusion term.  With local heat generation,
    even small temperature heterogeneity contains high-k components; at the current
    dx and dt the explicit heat step is far beyond the diffusion stability limit.
    The default v19 update treats thermal diffusion and bath coupling implicitly in
    Fourier space while keeping the plastic heat source explicit.
    """
    cp = max(float(P.get('cp_rho_vol', 1.0)), 1.0)
    dt = float(P.get('dt', 1.0))
    alpha = float(P.get('k_thermal', 0.0)) / cp
    beta = float(P.get('T_bath_coupling', 0.0)) / cp
    T0v = float(P.get('T0', 1300.0))
    Twork = np.asarray(Tfield, dtype=float)
    qwork = np.asarray(qdot_field, dtype=float)
    Twork = np.where(np.isfinite(Twork), Twork, T0v)
    qwork = np.where(np.isfinite(qwork), qwork, 0.0)
    mode = str(P.get('heat_update_mode', 'implicit_spectral')).lower()
    if mode.startswith('impl'):
        rhs = Twork + dt * (qwork / cp + beta * T0v)
        denom = 1.0 + dt * (alpha * K2 + beta)
        Tnew = np.real(np.fft.ifft2(np.fft.fft2(rhs) / denom))
    else:
        Tnew = Twork + dt * (qwork / cp + alpha * lap(Twork) - beta * (Twork - T0v))
    if P.get('heat_enforce_finite', True):
        Tnew = np.where(np.isfinite(Tnew), Tnew, T0v)
    Tmin = P.get('heat_min_K', None)
    if Tmin is not None:
        Tnew = np.maximum(Tnew, float(Tmin))
    return Tnew

def grad_mag(f): return np.sqrt(ddx(f)**2 + ddy(f)**2)

def _thermal_control_Q_eV():
    """Activation/transport scale used only for adaptive thermal stepping."""
    q = P.get('thermal_dt_Q_eV', None)
    if q is not None:
        try:
            return max(float(q), 1.0e-6)
        except Exception:
            pass
    vals = [float(P.get('KM_Q2_eV', 0.0)), float(P.get('diffrec_Q_eV', 0.0)),
            float(P.get('gb_mobility_Q_eV', 0.0)), float(P.get('expf_G00_eV', 0.0))]
    vals = [v for v in vals if np.isfinite(v) and v > 0.0]
    return max(vals) if vals else 1.0

def _adaptive_thermal_dt(current_sigma_bar, Tfield):
    """Return a mechanics time step constrained by thermo-Arrhenius stiffness.

    This does not cap temperature or heat production. It reduces the explicit
    mechanics/update step so the imposed mechanical work cannot change T enough
    to move Arrhenius mobilities by a large unresolved factor in one step.
    """
    dt_base = P.get('dt_base', None)
    if dt_base is None:
        dt_base = P.get('_dt_base_runtime', P.get('dt', 1.0))
    dt_base = max(float(dt_base), 1e-300)
    if not P.get('use_adaptive_thermal_dt', False):
        return dt_base, dict(active=0, dt_base=dt_base, dt=dt_base, dT_allow=np.inf, dT_macro_pred=np.nan, n_sub_equiv=1.0, QeV=np.nan)
    Tmean = finite_clipped_T_mean(Tfield)
    QeV = _thermal_control_Q_eV()
    kB_eV = 8.617333262145e-5
    dT_log = float(P.get('thermal_dt_log_change', 0.08)) * kB_eV * max(Tmean, 1.0)**2 / max(QeV, 1.0e-12)
    dT_user = float(P.get('thermal_dt_max_dT_step', 5.0))
    dT_allow = max(min(dT_user, dT_log), 1e-9)
    qmacro = max(float(current_sigma_bar), 0.0) * abs(float(P.get('edot_app', 0.0))) * float(P.get('taylor_quinney', 1.0))
    if qmacro <= 0.0 or not np.isfinite(qmacro):
        return dt_base, dict(active=0, dt_base=dt_base, dt=dt_base, dT_allow=dT_allow, dT_macro_pred=0.0, n_sub_equiv=1.0, QeV=QeV)
    dt_heat = float(P.get('cp_rho_vol', 1.0)) * dT_allow / max(qmacro, 1e-300)
    dt_min = max(float(P.get('thermal_dt_min', 1.0e-11)), 1e-300)
    dt_eff = min(dt_base, max(dt_heat, dt_min))
    dT_pred = qmacro * dt_base / max(float(P.get('cp_rho_vol', 1.0)), 1.0)
    active = int(dt_eff < 0.999999*dt_base)
    return dt_eff, dict(active=active, dt_base=dt_base, dt=dt_eff, dT_allow=dT_allow,
                        dT_macro_pred=dT_pred, n_sub_equiv=dt_base/max(dt_eff, 1e-300), QeV=QeV)

def _thermal_validity_exceeded(Tfield):
    if not P.get('use_thermal_validity_stop', False):
        return False, ''
    Tmax = float(np.nanmax(Tfield))
    Tmean = float(np.nanmean(Tfield))
    Tmax_lim = float(P.get('thermal_validity_Tmax_K', np.inf))
    Tmean_lim = float(P.get('thermal_validity_Tmean_K', np.inf))
    reasons = []
    if np.isfinite(Tmax_lim) and Tmax > Tmax_lim:
        reasons.append(f'Tmax={Tmax:.1f}K > {Tmax_lim:.1f}K')
    if np.isfinite(Tmean_lim) and Tmean > Tmean_lim:
        reasons.append(f'Tmean={Tmean:.1f}K > {Tmean_lim:.1f}K')
    return bool(reasons), '; '.join(reasons)

def _mechanical_validity_limit(Tfield):
    """Return a physical macro-stress validity limit [Pa], or inf.

    This is a model-validity boundary, not a stress cap.  The run stops before
    applying heat/storage updates when the imposed-rate solve requires stresses
    outside the constitutive calibration/solid-strength regime.
    """
    mode = str(P.get('mechanical_validity_mode', 'fit_or_ideal')).lower()
    if mode in ['off', 'none', 'false', '0']:
        return np.inf, 'off'
    vals = []
    labels = []
    explicit = P.get('mechanical_validity_sigma_MPa', None)
    if explicit is not None:
        try:
            v = float(explicit)*1.0e6
            if np.isfinite(v) and v > 0:
                vals.append(v); labels.append(f'explicit={v/1e6:.0f}MPa')
        except Exception:
            pass
    Tmean = finite_clipped_T_mean(Tfield)
    if 'ideal' in mode:
        mu = float(np.nanmean(_mu_shear_fe_local(np.asarray(Tfield, dtype=float))))
        frac = float(P.get('mechanical_validity_ideal_mu_frac', 0.12))
        v = frac * max(mu, 1.0)
        if np.isfinite(v) and v > 0:
            vals.append(v); labels.append(f'{frac:.2g}mu={v/1e6:.0f}MPa')
    if 'fit' in mode:
        try:
            # Local activation-stress range of the EXP-floor fit mapped to macro stress.
            # drive_sc converts macro stress to local effective activation stress in this code.
            sigc = float(np.asarray(ATpot.sigc(Tmean)))
            cap = float(P.get('expf_sigma_ratio_cap', 6.0))
            frac = float(P.get('mechanical_validity_fit_fraction', 1.0))
            drv = max(float(globals().get('drive_sc', 1.0)), 1.0e-12)
            v = frac * cap * sigc / drv
            if np.isfinite(v) and v > 0:
                vals.append(v); labels.append(f'fit cap={v/1e6:.0f}MPa')
        except Exception:
            pass
    if not vals:
        return np.inf, 'no finite mechanical validity limit'
    i = int(np.nanargmin(vals))
    return float(vals[i]), labels[i]

def _mechanical_validity_exceeded(current_sigma_bar, Tfield):
    if not P.get('use_mechanical_validity_stop', False):
        return False, ''
    lim, lab = _mechanical_validity_limit(Tfield)
    sig = abs(float(current_sigma_bar))
    if np.isfinite(lim) and sig > lim:
        return True, f'|sigma|={sig/1e6:.1f}MPa > {lim/1e6:.1f}MPa ({lab})'
    return False, ''

def _thermal_asb_scale_report():
    """Return simple thermal length/time scales for ASB branch design.

    These are not used as a gate.  They give order-of-magnitude guidance for
    whether the cell can contain a thermally localized mode.  The two most
    useful estimates are:
      lambda_edot = 2*pi*sqrt(alpha/edot), with alpha=k/(rho cp), using the
                    imposed strain-rate time scale as the growth time.
      ell_bath    = sqrt(k/h), where h is T_bath_coupling [W/m^3/K].
    If these are far larger than L_phys, the natural thermal mode is likely
    box-size limited unless edot is increased or k/h is reduced.
    """
    cp = max(float(P.get('cp_rho_vol', 1.0)), 1.0)
    k = max(float(P.get('k_thermal', 0.0)), 0.0)
    ed = max(abs(float(P.get('edot_app', 1.0))), 1e-300)
    h = max(float(P.get('T_bath_coupling', 0.0)), 0.0)
    alpha = k / cp
    lam_edot = 2.0*np.pi*np.sqrt(max(alpha, 0.0)/ed) if alpha > 0 else 0.0
    ell_bath = np.sqrt(k/h) if (k > 0 and h > 0) else np.inf
    target_um = P.get('asb_target_wavelength_um', None)
    target = float(target_um)*1e-6 if target_um is not None else float(P.get('L_phys', Lx))
    k_for_target = cp * ed * (target/(2.0*np.pi))**2
    h_for_target = k / max(target**2, 1e-300) if k > 0 else 0.0
    return dict(alpha=alpha, lambda_edot=lam_edot, ell_bath=ell_bath,
                target=target, k_for_target=k_for_target, h_for_target=h_for_target)

print(f"Grid {Nx}x{Ny}, dx={dx*1e6:.3f} um")

# ================================================================
# 3. ARRHENIUS-TAYLOR POTENTIAL
# ================================================================
class ATPotential:
    """Arrhenius barrier model + physical free-energy potential.

    The free energy Φ(ρ,T) is constructed from three physical branches:

      Φ_elastic  = ½ α μ(T) b² ρ
                   Taylor stored energy — linear in ρ, sets the energy scale.

      Φ_entropy  = C_ent · ρ · (ln(ρ/ρ₀) - 1)
                   Configurational entropy of dislocation arrangements.
                   Creates a logarithmic peak at low ρ that prevents ρ→0.
                   (Kröner/Wilkens; "ρ ln ρ" storage term in the literature.)

      Φ_ordering = -A_ord · exp(-(log₁₀ρ - x_ord)²/(2 w_ord²))
                   Read-Shockley / KWC ordering energy.  A Gaussian dip
                   centred at ρ_ord represents the energy gain when
                   dislocations self-organise into low-angle walls.
                   Temperature-dependent through μ(T) → A_ord(T).

    The chemical potential μ = dΦ/dρ drives the Cahn-Hilliard.
    The Arrhenius EXP-floor model is used separately for the flow stress.
    """
    def __init__(s, p):
        s.b = p['b']; s.p = p['pTaylor']; s.eta0 = p['eta0']
        s.barrier_model = str(p.get('arrhenius_barrier_model', 'exp_floor')).lower()
        s.G00 = p['expf_G00_eV']*eV_J; s.gT = p['expf_gT']
        s.Tref = p['expf_Tref']; s.sigc0 = p['expf_sigc0']
        s.sT = p['expf_sT']; s.a = p['expf_a']; s.n = p['expf_n']
        s.fl = p['expf_floor']; s.rmin = p['rho_min']
        s._Tc = None
        # potential params
        s.alpha_T = p['alpha_Taylor']
        s.C_ent_frac = p['C_ent_frac']
        s.rho_ref = p['rho_ref_ent']
        s.r_ord = p['r_ord']
        s.w_r = p['w_r_ord']
        s.A_ord_scale = p['A_ord_scale']

    def G0(s, T): return s.G00*np.exp(-s.gT*(np.asarray(T, dtype=float)-s.Tref)/s.Tref)
    def sigc(s, T): return s.sigc0*np.exp(-s.sT*(np.asarray(T, dtype=float)-s.Tref)/s.Tref)

    def barrier_name(s):
        m = str(getattr(s, 'barrier_model', 'exp_floor')).lower()
        if m in ['exp', 'expfloor', 'exp_floor', 'exponential', 'exponential_floor']:
            return 'exp_floor'
        if m in ['kaa', 'kocks', 'kocks_argon_ashby', 'bracket', 'kocks_floor']:
            return 'kaa'
        return 'exp_floor'

    def mu_shear(s, T):
        """Temperature-dependent shear modulus for Fe/superalloy scale [Pa].

        Vectorized.  Used consistently for stored dislocation energy E*(rho,T),
        Read-Shockley/KWC energy scaling, and diagnostic potential construction.
        """
        T_arr = np.asarray(T, dtype=float)
        mu_0 = float(P.get('Estar_mu0', 82.0e9))
        dmu_dT = float(P.get('Estar_dmu_dT', -3.7e7))
        mu_floor = float(P.get('Estar_mu_floor', 5.0e9))
        return np.maximum(mu_0 + dmu_dT*(T_arr - 300.0), mu_floor)

    def Estar_coeff(s, T):
        """A_E(T) [J/m] such that E*(rho,T)=A_E(T)*rho."""
        alpha_fac = float(P.get('alpha_Taylor', 0.30)) if P.get('Estar_use_alpha', False) else 1.0
        return 0.5 * alpha_fac * s.mu_shear(T) * s.b**2

    def barrier_G(s, sig, T):
        """Activation free energy G*(sigma,T) [J].

        Default EXP/EXP-floor branch:
            G = G0(T) * [ f + (1-f) exp(-a (sigma/sigc(T))^n) ]
        with f=expf_floor.  f=0 reproduces the uploaded median EXP fit.

        Legacy KAA/bracket branch, retained for diagnostics:
            G = G0(T) * max(1-(sigma/sigc(T))^a, f)^n.
        """
        sig = np.maximum(np.asarray(sig, dtype=float), 0.0)
        T = np.maximum(np.asarray(T, dtype=float), 1.0)
        sc = np.maximum(s.sigc(T), 1e-300)
        r = sig / sc
        f = float(np.clip(getattr(s, 'fl', 0.0), 0.0, 0.999999999))
        G0v = s.G0(T)
        if s.barrier_name() == 'kaa':
            ratio = np.clip(r, 0.0, 1.0-1e-15)
            bracket = np.maximum(1.0 - ratio**s.a, f)
            return G0v * bracket**s.n
        E = np.exp(np.clip(-s.a * np.maximum(r, 0.0)**s.n, -700.0, 50.0))
        return G0v * (f + (1.0 - f)*E)

    def barrier_variable_part(s, sig, T):
        """Stress-dependent part of G for dG/dsigma in EXP-floor branch [J]."""
        sig = np.maximum(np.asarray(sig, dtype=float), 0.0)
        T = np.maximum(np.asarray(T, dtype=float), 1.0)
        sc = np.maximum(s.sigc(T), 1e-300)
        r = sig / sc
        f = float(np.clip(getattr(s, 'fl', 0.0), 0.0, 0.999999999))
        if s.barrier_name() == 'kaa':
            ratio = np.clip(r, 0.0, 1.0-1e-15)
            bracket = np.maximum(1.0 - ratio**s.a, f)
            return s.G0(T) * bracket**s.n
        E = np.exp(np.clip(-s.a * np.maximum(r, 0.0)**s.n, -700.0, 50.0))
        return s.G0(T) * (1.0 - f) * E

    def S_over_kB(s, sig, T):
        """Activation entropy S*/kB at fixed stress for the EXP-floor branch."""
        sig = np.maximum(np.asarray(sig, dtype=float), 0.0)
        T = np.maximum(np.asarray(T, dtype=float), 1.0)
        if s.barrier_name() == 'kaa':
            dT = np.maximum(1e-3*T, 1e-3)
            Gp = s.barrier_G(sig, T + dT)
            Gm = s.barrier_G(sig, np.maximum(T - dT, 1.0))
            return -(Gp - Gm)/(2.0*dT) / kB_J
        sc = np.maximum(s.sigc(T), 1e-300)
        r = sig / sc
        f = float(np.clip(getattr(s, 'fl', 0.0), 0.0, 0.999999999))
        E = np.exp(np.clip(-s.a * np.maximum(r, 0.0)**s.n, -700.0, 50.0))
        G0v = s.G0(T)
        bracket = f + (1.0 - f)*E
        term = s.gT*bracket + (1.0 - f)*E*s.a*s.n*s.sT*np.maximum(r, 0.0)**s.n
        return G0v * term / (kB_J*s.Tref)

    def H_eV(s, sig, T):
        G = s.barrier_G(sig, T)
        S = s.S_over_kB(sig, T) * kB_J
        return (G + np.asarray(T, dtype=float)*S) / eV_J

    def vstar_m3(s, sig, T):
        """Activation volume -dG/dsigma [m^3].  Analytic for EXP-floor."""
        sig = np.maximum(np.asarray(sig, dtype=float), 0.0)
        T = np.maximum(np.asarray(T, dtype=float), 1.0)
        sc = np.maximum(s.sigc(T), 1e-300)
        r = sig / sc
        if s.barrier_name() == 'kaa':
            ratio = np.clip(r, 0.0, 1.0-1e-15)
            f = float(np.clip(s.fl, 0.0, 0.999999999))
            bracket = np.maximum(1.0 - ratio**s.a, f)
            active = (1.0 - ratio**s.a) > f
            G0v = s.G0(T)
            v = G0v * s.n * np.maximum(bracket, 1e-300)**(s.n-1.0) * s.a * np.maximum(ratio, 0.0)**(s.a-1.0) / sc
            return np.where(active, v, 0.0)
        Gvar = s.barrier_variable_part(sig, T)
        return Gvar * s.a*s.n * np.maximum(r, 0.0)**(s.n-1.0) / sc

    def vstar_b3(s, sig, T):
        return s.vstar_m3(sig, T) / max(s.b**3, 1e-300)

    def _collective_enabled(s):
        mode = str(P.get('collective_taylor_mode', 'multi_hit')).lower()
        return bool(P.get('use_collective_taylor', False)) and mode not in ['off', 'false', 'none', 'independent', 'single', 'single_hit']

    def _poisson_tail_ge_m(s, lam, m):
        """Poisson probability P[N >= m] for possibly non-integer m.

        For integer m this is 1 - exp(-lam) sum_{j=0}^{m-1} lam^j/j!.
        scipy.special.gammainc is the regularized lower incomplete gamma,
        which gives the same Poisson tail and remains stable for large lam/m.
        """
        lam = np.maximum(np.asarray(lam, dtype=float), 0.0)
        m = np.maximum(np.asarray(m, dtype=float), 1.0)
        out = gammainc(m, lam)
        return np.clip(np.where(np.isfinite(out), out, 0.0), 0.0, 1.0)

    def _collective_fields(s, sig, rho, T):
        """Fields entering the v26 collective Taylor depinning closure.

        The independent v25 rate is
            gdot_ind = eta0 (b/X)^p exp[-G(sig,T)/kT].

        v26c interprets (b/X)^p as the total effective isolated-site
        multiplicity.  A correlated domain contains n_c elementary sites, so
        only N_eff=(b/X)^p/n_c^q independent domains add linearly.  This
        domain-count closure regularizes the high-density prefactor without
        imposing a nanosecond simultaneous-hit bottleneck on the imposed-rate
        stress inversion.

        The Poisson multi-hit tail is still evaluated as a diagnostic/activity
        variable:
            Lambda = n_c h1 t_c,
            h_domain = P[N>=m; Lambda]/t_c.
        Setting collective_rate_closure='poisson_tail' recovers the stronger
        v26b behavior:
            gdot_coll = ((b/X)^p/n_c) h_domain.
        """
        sig, rho, T = np.broadcast_arrays(np.maximum(np.asarray(sig, dtype=float), 0.0),
                                          np.maximum(np.asarray(rho, dtype=float), s.rmin),
                                          np.maximum(np.asarray(T, dtype=float), 1.0))
        X = 1.0/np.sqrt(2.0*rho)
        G = s.barrier_G(sig, T)
        h1 = s.eta0 * np.exp(np.clip(-G/(kB_J*T), -700.0, 40.0))
        n_site = np.maximum((s.b/np.maximum(X, s.b*1e-12))**s.p, 0.0)
        gdot_ind = n_site * h1

        # Elastic triggering length: a stress perturbation of order mu*b/ell
        # matters over the distance at which it exceeds the remaining local
        # barrier margin.  sig is already the local activation stress used in G*.
        mu_loc = s.mu_shear(T)
        sigc_loc = np.maximum(s.sigc(T), 1e-300)
        tau0 = max(float(P.get('collective_tau0_MPa', 100.0)), 0.0)*1e6
        margin = np.maximum(sigc_loc - sig, 0.0) + max(tau0, 1.0)
        ell_el = float(P.get('collective_C_el', 1.0)) * mu_loc * s.b / margin
        ell_max = float(P.get('collective_ell_max_um', 0.10))*1e-6
        if not np.isfinite(ell_max) or ell_max <= 0.0:
            ell_max = float(P.get('L_phys', 1.0e-5))
        ell = np.minimum(ell_max, np.maximum(X, ell_el))
        nc = np.clip(ell/np.maximum(X, s.b*1e-12), 1.0, max(float(P.get('collective_nc_max', 20.0)), 1.0))

        # Optional smooth density switch if one wants to activate cooperativity
        # only above a specified forest density.  By default no separate density
        # gate is used; nc emerges from elastic length / forest spacing.
        rho_sw = P.get('collective_density_switch', None)
        if rho_sw is not None:
            try:
                rho_sw = float(rho_sw)
                if rho_sw > 0:
                    pw = max(float(P.get('collective_density_switch_power', 4.0)), 1e-9)
                    w = 1.0/(1.0 + (rho_sw/np.maximum(rho, s.rmin))**pw)
                    nc = 1.0 + w*(nc - 1.0)
            except Exception:
                pass

        eta_m = float(P.get('collective_eta_m', 0.25))
        m = 1.0 + eta_m*np.maximum(nc - 1.0, 0.0)
        m = np.clip(m, max(float(P.get('collective_m_min', 1.0)), 1.0),
                    max(float(P.get('collective_m_max', 8.0)), 1.0))
        if P.get('collective_m_round', False):
            m = np.maximum(np.rint(m), 1.0)

        tc_mode = str(P.get('collective_tc_mode', 'fixed')).lower()
        if tc_mode.startswith('el') or tc_mode in ['wave', 'velocity', 'length_over_v']:
            vch = max(float(P.get('collective_v_char', 2.0e3)), 1e-12)
            tc = ell / vch
        else:
            tc = np.full_like(rho, max(float(P.get('collective_tc', 1.0e-9)), 1e-300))
        tc = np.clip(tc, max(float(P.get('collective_tc_min', 1.0e-11)), 1e-300),
                     max(float(P.get('collective_tc_max', 1.0e-7)), 1e-300))

        lam = np.maximum(nc * h1 * tc, 0.0)
        pcomp = s._poisson_tail_ge_m(lam, m)
        h_domain = pcomp / np.maximum(tc, 1e-300)

        closure = str(P.get('collective_rate_closure', 'domain_count')).lower()
        if closure in ['poisson', 'poisson_tail', 'multi_hit_rate', 'v26b']:
            n_corr = n_site / np.maximum(nc, 1.0)
            gdot_raw = n_corr * h_domain
        else:
            # Safer production closure: correlated domains, not elementary
            # sites, add linearly.  This removes the pathological high-density
            # independent-site prefactor while keeping the elementary Arrhenius
            # hit rate as the local mobility scale.
            q = np.clip(float(P.get('collective_domain_power', 1.0)), 0.0, max(float(getattr(s, 'p', 4.0)), 4.0))
            # v27c: when the rho-state model is active, the independent-site
            # prefactor must be reduced to a correlated-domain count.  In this
            # 2D implementation the physically relevant count is closer to an
            # area/domain correction q≈2, not the full old site exponent p=4.
            if P.get('use_rho_state_partition', False) and P.get('collective_rhostate_enforce_domain_power', True):
                qmin = float(P.get('collective_rhostate_min_domain_power', getattr(s, 'p', 4.0)))
                q = max(q, qmin)
            n_corr = n_site / np.maximum(nc, 1.0)**q
            gdot_raw = n_corr * h1

        # Keep the collective closure from deleting essentially all plastic
        # mobility in an imposed-rate solve.  This is a numerical/closure
        # safeguard, not a stress cap: it preserves the old rate when nc=1 and
        # prevents runaway heat from a singular stress inversion when nc is large.
        min_sup = float(P.get('collective_min_suppression', 0.0))
        if P.get('use_rho_state_partition', False) and P.get('collective_rhostate_disable_min_suppression', True):
            min_sup = 0.0
        if min_sup > 0.0:
            gdot_raw = np.maximum(gdot_raw, min_sup * gdot_ind)
        gdot_coll = np.where(np.isfinite(gdot_raw), gdot_raw, 0.0)
        suppression = gdot_coll / np.maximum(gdot_ind, 1e-300)
        return dict(gdot_ind=gdot_ind, gdot_coll=gdot_coll, h1=h1, n_site=n_site,
                    X=X, ell=ell, nc=nc, m=m, tc=tc, Lambda=lam,
                    P_complete=pcomp, h_domain=h_domain, n_corr=n_corr,
                    suppression=suppression, margin=margin)

    def gdot_independent(s, sig, rho, T):
        """Original v25 isolated-event Arrhenius-Taylor rate."""
        sig = np.maximum(np.asarray(sig, dtype=float), 0.0)
        rho = np.maximum(np.asarray(rho, dtype=float), s.rmin)
        T = np.maximum(np.asarray(T, dtype=float), 1.0)
        X = 1.0/np.sqrt(2.0*rho)
        G = s.barrier_G(sig, T)
        pref = s.eta0 * (s.b/X)**s.p
        return pref * np.exp(np.clip(-G/(kB_J*T), -700.0, 40.0))

    def gdot(s, sig, rho, T):
        """Forward Arrhenius rate; v26 can use collective multi-hit depinning."""
        if s._collective_enabled():
            return s._collective_fields(sig, rho, T)['gdot_coll']
        return s.gdot_independent(sig, rho, T)

    def collective_diag(s, sig, rho, T):
        """Scalar diagnostics for the collective Taylor closure."""
        if not s._collective_enabled():
            return dict(collective_enabled=0.0)
        cf = s._collective_fields(sig, rho, T)
        def mmean(name):
            arr = np.asarray(cf[name], dtype=float)
            return float(np.nanmean(arr)) if arr.size else np.nan
        def mpct(name, q):
            arr = np.asarray(cf[name], dtype=float)
            return float(np.nanpercentile(arr, q)) if arr.size else np.nan
        return dict(
            collective_enabled=1.0,
            collective_nc_mean=mmean('nc'),
            collective_nc_p95=mpct('nc', 95),
            collective_m_mean=mmean('m'),
            collective_m_p95=mpct('m', 95),
            collective_ell_nm_mean=1e9*mmean('ell'),
            collective_ell_nm_p95=1e9*mpct('ell', 95),
            collective_tc_mean=mmean('tc'),
            collective_lambda_mean=mmean('Lambda'),
            collective_lambda_p95=mpct('Lambda', 95),
            collective_Pcomplete_mean=mmean('P_complete'),
            collective_suppression_mean=mmean('suppression'),
            collective_suppression_p05=mpct('suppression', 5),
            collective_gdot_ind_mean=mmean('gdot_ind'),
            collective_gdot_coll_mean=mmean('gdot_coll'),
            collective_margin_MPa_mean=1e-6*mmean('margin'),
        )

    def sigma_inv(s, rho, T, edot):
        """Inverse: find sigma such that gdot(sigma, rho, T) = edot."""
        rho = max(float(rho), s.rmin)
        def res(sig): return float(s.gdot(sig, rho, T)) - edot
        if res(0) >= 0: return 0.0
        sc = float(np.asarray(s.sigc(T)))
        hi = sc*max(float(P.get('expf_sigma_ratio_cap', 6.0)), 1.5)
        try:
            while res(hi) < 0 and hi < 2e11: hi *= 3
            return float(brentq(res, 0, hi, xtol=100))
        except Exception:
            return hi

    def sigma_arrhenius_vec(s, rho, T, edot):
        """Vectorized inverse Arrhenius Taylor stress σ_AT(ρ,T,edot).

        Default branch analytically inverts the uploaded EXP/EXP-floor barrier fit.
        Legacy KAA/bracket inversion remains available via arrhenius_barrier_model='kaa'.
        """
        rho = np.maximum(np.asarray(rho, dtype=float), s.rmin)
        T = np.maximum(np.asarray(T, dtype=float), 1.0)
        rho, T = np.broadcast_arrays(rho, T)
        edot = max(float(edot), 1e-300)

        # The collective/multi-hit closure has no useful closed-form inverse,
        # but it remains monotone in stress.  Use vectorized bisection for the
        # tabulated potential/diagnostic curves.
        if s._collective_enabled():
            sigma = np.zeros_like(rho, dtype=float)
            r0 = s.gdot(0.0, rho, T)
            active = r0 < edot
            if not np.any(active):
                return sigma
            cap = max(float(P.get('expf_sigma_ratio_cap', 6.0)), 1.0)
            lo = np.zeros_like(rho, dtype=float)
            hi = s.sigc(T) * cap
            hi = np.where(np.isfinite(hi) & (hi > 0.0), hi, 3.0e9)
            rhi = s.gdot(hi, rho, T)
            for _ in range(8):
                need = active & (rhi < edot) & (hi < 2.0e11)
                if not np.any(need):
                    break
                hi = np.where(need, hi*3.0, hi)
                rhi = s.gdot(hi, rho, T)
            for _ in range(36):
                mid = 0.5*(lo + hi)
                rm = s.gdot(mid, rho, T)
                go_hi = active & (rm >= edot)
                hi = np.where(go_hi, mid, hi)
                lo = np.where(active & (~go_hi), mid, lo)
            sigma = np.where(active, 0.5*(lo+hi), 0.0)
            return sigma

        pref = s.eta0 * (s.b*np.sqrt(2.0*rho))**s.p
        G0v = np.asarray(s.G0(T), dtype=float)
        sc = np.asarray(s.sigc(T), dtype=float)
        G0v, sc, _rho_bcast = np.broadcast_arrays(G0v, sc, rho)
        Greq = -kB_J*T*np.log(np.maximum(edot/np.maximum(pref, 1e-300), 1e-300))
        sigma = np.zeros_like(rho, dtype=float)
        active = Greq < G0v
        f = float(np.clip(getattr(s, 'fl', 0.0), 0.0, 0.999999999))
        if not np.any(active):
            return sigma

        # NOTE v26a bug fix: in the analytical independent-event inverse, T is
        # broadcast to the rho table, so G0v and sigc are arrays.  All active-set
        # algebra below must index those arrays with active; otherwise numpy tries
        # to combine the shorter active vector with the full rho table.
        Ga = G0v[active]
        sca = sc[active]
        Greqa = Greq[active]

        if s.barrier_name() == 'kaa':
            Gfloor = Ga * (f**s.n)
            frac = np.clip(Greqa / np.maximum(Ga, 1e-300), 0.0, 1.0)
            bracket = frac**(1.0/s.n)
            bracket = np.maximum(bracket, f)
            val = sca * np.maximum(1.0 - bracket, 0.0)**(1.0/s.a)
            if f > 0.0:
                val_floor = sca*np.maximum(1.0-f,0.0)**(1.0/s.a)
                val = np.where(Greqa <= Gfloor, val_floor, val)
            sigma[active] = val
            return sigma

        y = (Greqa/np.maximum(Ga, 1e-300) - f) / max(1.0 - f, 1e-300)
        cap = max(float(P.get('expf_sigma_ratio_cap', 6.0)), 1.0)
        r = np.full_like(y, cap, dtype=float)
        ok = y > 0.0
        r[ok] = (-np.log(np.clip(y[ok], 1e-300, 1.0))/max(s.a, 1e-300))**(1.0/s.n)
        r = np.clip(r, 0.0, cap)
        sigma[active] = sca * r
        return sigma

    def _cumtrapz0(s, y, x):
        y = np.asarray(y, dtype=float); x = np.asarray(x, dtype=float)
        out = np.zeros_like(y)
        if y.size > 1:
            out[1:] = np.cumsum(0.5*(y[1:]+y[:-1])*(x[1:]-x[:-1]))
        return out

    def _interp_log(s, rho, arr):
        rho_arr = np.asarray(rho, dtype=float)
        rr = np.maximum(rho_arr, s.rmin)
        if not hasattr(s, '_log_rho_tab'):
            # Fallback should only be used during construction failures.
            return np.zeros_like(rho_arr, dtype=float)
        vals = np.interp(np.log(rr), s._log_rho_tab, np.asarray(arr, dtype=float),
                         left=float(arr[0]), right=float(arr[-1]))
        return vals

    def build(s, T, edot):
        """Build tabulated potential Φ(ρ,T,edot) and its derivatives.

        v22 potential modes:
          thermo_stored/stored_energy production thermodynamic branch E*(rho,T)
          surrogate                   legacy alias for thermo_stored
          arrhenius_sigma             diagnostic Φ_AT = σ_AT(ρ,T,edot), units J/m^3
          arrhenius_sigma_plus_elastic diagnostic σ_AT + E*(rho,T)
          arrhenius_integral_b2       diagnostic b²∫σ_AT dρ lower-bound alternative
          arrhenius_work              diagnostic ∫σ_AT bℓ dρ work-conjugate potential
        """
        if s._Tc is not None and abs(T - s._Tc) < 2.0: return
        s._Tc = T
        s._edot_c = float(edot)

        # rho_c: density where zero-stress rate = edot (Taylor peak proxy / scale)
        G0v = s.G0(T)
        arg = np.clip(-G0v/(kB_J*T), -700, 40)
        min_rate = s.eta0 * np.exp(arg)
        if min_rate > 1e-300:
            s.rho_c = max(0.5*(edot/min_rate)**(2.0/s.p)/s.b**2, s.rmin*10)
        else:
            s.rho_c = 1e18
        s.sigma_c = float(s.sigma_arrhenius_vec(np.array([s.rho_c]), T, edot)[0])
        s.rho_peak_ind = float(s.rho_c)  # independent-site kinetic peak/crossover diagnostic

        # v27: do not use the independent-Taylor kinetic peak as the structural
        # density scale for CH/KWC organization.  If a structural density scale
        # has been initialized, use it for the thermodynamic ordering coordinate;
        # otherwise fall back to rho_c for legacy behavior.
        rho_ch = s.rho_c
        if P.get('use_rho_state_partition', False) and P.get('rho_state_use_structural_scale_for_ch', True):
            cand = P.get('_rho_state_ref_runtime', P.get('rho_state_ref_abs', None))
            try:
                cand = float(cand)
                if np.isfinite(cand) and cand > s.rmin:
                    rho_ch = cand
            except Exception:
                pass
        s.rho_ch = float(max(rho_ch, s.rmin))

        # Temperature-dependent coefficients
        mu_T = float(s.mu_shear(T))
        s.A1 = float(s.Estar_coeff(T))            # J/m  stored line-energy coefficient A_E(T)
        s.C_ent = s.C_ent_frac * s.A1             # J/m  (entropy/storage coefficient)
        s.A_ord_r = s.A_ord_scale * s.A1 * s.rho_ch * s.w_r  # J/m^3

        # Build table for fast CH/nucleation evaluation.
        # Use a broad range so interpolation is stable when rho moves away from rho_c.
        lo = np.log10(max(s.rmin, 1e10))
        hi = np.log10(min(P.get('rho_max', 5e18), max(5e18, s.rho_c*20)))
        s.rho_tab = np.logspace(lo, hi, 2400)
        s._log_rho_tab = np.log(s.rho_tab)
        r = s.rho_tab / max(getattr(s, 'rho_ch', s.rho_c), s.rmin)

        # Components
        s.Phi_el_tab = s.A1 * s.rho_tab
        s.Phi_ent_tab = s.C_ent * s.rho_tab * (np.log(np.maximum(s.rho_tab / s.rho_ref, 1e-30)) - 1.0)
        s.Phi_ord_tab = -s.A_ord_r * np.exp(-(r - s.r_ord)**2 / (2*s.w_r**2))
        s.Phi_low_tab = s._lowrho_phi(s.rho_tab)
        # sigma_arrhenius_vec gives the local/effective activation stress.  For
        # the Arrhenius-Taylor macro flow-stress curve used in the paper, convert
        # to applied stress by dividing by the Taylor stress concentration
        # phi_T = X/b, i.e. sigma_macro = sigma_local*b/X.  This is what gives
        # the increasing-then-decreasing Taylor peak instead of a monotonic
        # local barrier-stress curve.
        s.sigma_eff_tab = s.sigma_arrhenius_vec(s.rho_tab, T, edot)
        Xtab = 1.0/np.sqrt(2.0*np.maximum(s.rho_tab, s.rmin))
        if P.get('arrhenius_phi_use_taylor_concentration', True):
            conc = float(P.get('arrhenius_phi_concentration_prefactor', 1.0)) * s.b / np.maximum(Xtab, s.b*1e-12)
            s.sigma_at_tab = s.sigma_eff_tab * conc
            s.sigma_at_label = 'macro sigma_AT = sigma_local*b/X'
        else:
            s.sigma_at_tab = s.sigma_eff_tab.copy()
            s.sigma_at_label = 'local sigma_eff'
        s.Phi_arr_sigma_tab = float(P.get('arrhenius_phi_scale', 1.0)) * s.sigma_at_tab
        s.Phi_arr_int_tab = float(P.get('arrhenius_phi_scale', 1.0)) * s.b**2 * s._cumtrapz0(s.sigma_at_tab, s.rho_tab)

        # v18: work-conjugate Arrhenius-Taylor potential.
        # General form: Φ_AT^work = ∫ σ_AT(ρ) b ℓ(ρ) dρ.
        # ℓ=b recovers the old b²∫σ_AT dρ lower-bound diagnostic.
        # ℓ=X_peak preserves the spinodal onset at dσ_AT/dρ<0 while putting the
        # magnitude on the same order as E*=½μb²ρ.  ℓ=X(ρ) is the more local
        # swept-area/Orowan form but shifts the spinodal by the derivative of X.
        work_len_mode = str(P.get('arrhenius_work_length_mode', 'X_peak')).lower()
        i_peak = int(np.nanargmax(s.sigma_at_tab)) if s.sigma_at_tab.size else 0
        s.rho_at_peak = float(s.rho_tab[i_peak])
        s.X_at_peak = float(1.0 / np.sqrt(2.0 * max(s.rho_at_peak, s.rmin)))
        if work_len_mode in ['b', 'burgers']:
            ell_tab = np.full_like(s.rho_tab, s.b)
            ell_label = 'ell=b'
            s.arrhenius_work_length_mode = 'b'
        elif work_len_mode in ['x_peak', 'xpeak', 'xc', 'x_c', 'peak']:
            ell_tab = np.full_like(s.rho_tab, s.X_at_peak)
            ell_label = 'ell=X_peak'
            s.arrhenius_work_length_mode = 'x_peak'
        elif work_len_mode in ['x_local', 'x', 'spacing']:
            ell_tab = Xtab.copy()
            ell_label = 'ell=X(rho)'
            s.arrhenius_work_length_mode = 'x_local'
        else:
            print(f"WARNING: unknown arrhenius_work_length_mode={work_len_mode!r}; using X_peak")
            ell_tab = np.full_like(s.rho_tab, s.X_at_peak)
            ell_label = 'ell=X_peak'
            s.arrhenius_work_length_mode = 'x_peak'
        s.arrhenius_work_ell_tab = ell_tab
        s.arrhenius_work_ell_label = ell_label
        s.Phi_arr_work_raw_tab = s._cumtrapz0(s.sigma_at_tab * s.b * ell_tab, s.rho_tab)
        s.Phi_arr_work_tab = float(P.get('arrhenius_phi_scale', 1.0)) * s.Phi_arr_work_raw_tab
        s.mu_arr_work_tab = float(P.get('arrhenius_phi_scale', 1.0)) * s.sigma_at_tab * s.b * ell_tab
        s.Phi_pp_arr_work_tab = np.gradient(s.mu_arr_work_tab, s.rho_tab, edge_order=2)

        mode = str(P.get('potential_mode', 'thermo_stored')).lower()
        s.potential_mode = mode
        if mode in ['thermo_stored', 'stored_energy', 'line_energy', 'surrogate']:
            s.Phi_base_tab = s.Phi_el_tab.copy()
            s.Phi_base_label = 'thermodynamic stored-energy branch'
            s.potential_mode = 'thermo_stored' if mode != 'surrogate' else 'surrogate'
        elif mode == 'arrhenius_sigma':
            s.Phi_base_tab = s.Phi_arr_sigma_tab.copy()
            s.Phi_base_label = 'diagnostic sigma_AT as Phi'
        elif mode == 'arrhenius_sigma_plus_elastic':
            s.Phi_base_tab = s.Phi_arr_sigma_tab + s.Phi_el_tab
            s.Phi_base_label = 'diagnostic sigma_AT + E*'
        elif mode == 'arrhenius_integral_b2':
            s.Phi_base_tab = s.Phi_arr_int_tab.copy()
            s.Phi_base_label = 'diagnostic b^2 int sigma_AT drho'
        elif mode == 'arrhenius_work':
            s.Phi_base_tab = s.Phi_arr_work_tab.copy()
            s.Phi_base_label = f'diagnostic int sigma_AT*b*ell drho ({s.arrhenius_work_ell_label})'
        else:
            print(f"WARNING: unknown potential_mode={mode!r}; falling back to thermo_stored")
            s.potential_mode = 'thermo_stored'
            s.Phi_base_tab = s.Phi_el_tab.copy()
            s.Phi_base_label = 'thermodynamic stored-energy branch'

        s.Phi_tab = s.Phi_base_tab.copy()
        if P.get('use_potential_entropy', True):
            s.Phi_tab = s.Phi_tab + s.Phi_ent_tab
        if P.get('use_potential_ordering', True):
            s.Phi_tab = s.Phi_tab + s.Phi_ord_tab
        if P.get('use_potential_lowrho_penalty', True):
            s.Phi_tab = s.Phi_tab + s.Phi_low_tab

        # Numerical derivatives of the actual tabulated thermodynamic potential.
        # CH sees this selected landscape.  In production v22 this is the stored
        # energy + entropy + ordering branch; σ_AT is used separately as kinetics.
        s.mu_tab = np.gradient(s.Phi_tab, s.rho_tab, edge_order=2)
        s.Phi_pp_tab = np.gradient(s.mu_tab, s.rho_tab, edge_order=2)

        # Arrhenius-Taylor kinetic instability diagnostic: dσ_AT/dρ < 0.
        s.dsigma_at_drho_tab = np.gradient(s.sigma_at_tab, s.rho_tab, edge_order=2)
        s.d2sigma_at_drho2_tab = np.gradient(s.dsigma_at_drho_tab, s.rho_tab, edge_order=2)
        s.kinetic_spinodal_slack_tab = np.maximum(0.0, -s.dsigma_at_drho_tab)
        s.spinodal_slack_scale = max(float(np.nanmax(s.kinetic_spinodal_slack_tab)), 1e-300)
        kspin = np.isfinite(s.dsigma_at_drho_tab) & (s.dsigma_at_drho_tab < 0.0)
        if np.any(kspin):
            kidx = np.where(kspin)[0]
            s.kin_spin_rho_lo = float(s.rho_tab[kidx[0]])
            s.kin_spin_rho_hi = float(s.rho_tab[kidx[-1]])
        else:
            s.kin_spin_rho_lo = np.nan
            s.kin_spin_rho_hi = np.nan
        s.rho_at_sigma_peak = float(s.rho_tab[i_peak])
        s.sigma_at_peak = float(s.sigma_at_tab[i_peak])

        # Find thermodynamic spinodal bounds (where Φ'' < 0)
        spin = np.isfinite(s.Phi_pp_tab) & (s.Phi_pp_tab < 0)
        if np.any(spin):
            idx = np.where(spin)[0]
            s.rho_spin_lo = s.rho_tab[idx[0]]
            s.rho_spin_hi = s.rho_tab[idx[-1]]
        else:
            s.rho_spin_lo = s.rho_c
            s.rho_spin_hi = s.rho_c

        sigmax = float(np.nanmax(s.sigma_at_tab)) if s.sigma_at_tab.size else np.nan
        rhomax = float(s.rho_tab[int(np.nanargmax(s.sigma_at_tab))]) if s.sigma_at_tab.size else np.nan
        i_peak_print = int(np.nanargmax(s.sigma_at_tab)) if s.sigma_at_tab.size else 0
        E_star_tab = 0.5 * mu_T * s.b**2 * s.rho_tab
        ratio_peak = float(s.Phi_arr_work_tab[i_peak_print] / max(float(E_star_tab[i_peak_print]), 1e-300))
        mu_ratio_peak = float(s.mu_arr_work_tab[i_peak_print] / max(0.5 * mu_T * s.b**2, 1e-300))
        print(f"  AT: barrier={s.barrier_name()}, mode={s.potential_mode}, rho_c={s.rho_c:.2e}, mu(T)={mu_T/1e9:.0f}GPa, "
              f"A1={s.A1:.2e} J/m, C_ent={s.C_ent:.2e}, A_ord_r={s.A_ord_r:.2e} J/m³")
        print(f"      sigma_AT max={sigmax/1e6:.0f}MPa at rho={rhomax:.1e}; "
              f"{s.sigma_at_label}; ordering dip at r_ord={s.r_ord:.2f} (rho={s.r_ord*s.rho_c:.1e}), "
              f"thermo spinodal [{s.rho_spin_lo:.1e}, {s.rho_spin_hi:.1e}]")
        print(f"      AT kinetic peak={s.sigma_at_peak/1e6:.0f}MPa at rho={s.rho_at_sigma_peak:.1e}; "
              f"kinetic dσ/dρ<0 range [{s.kin_spin_rho_lo:.1e}, {s.kin_spin_rho_hi:.1e}]")
        print(f"      AT work diagnostic: {s.arrhenius_work_ell_label}, X_peak={s.X_at_peak:.3e} m, "
              f"Phi_work/E* at peak={ratio_peak:.3g}, "
              f"mu_work/mu_E* at peak={mu_ratio_peak:.3g}")

    def _Phi(s, rho):
        """Selected variational potential Φ(ρ) [J/m³] from the v14 table."""
        return s._interp_log(np.maximum(np.asarray(rho, dtype=float), s.rmin), s.Phi_tab)

    def _lowrho_phi(s, rho):
        """Soft low-density free-energy penalty [J/m^3].

        This is intentionally separate from rho_min.  rho_min is only a numerical
        guard (~1e8 m^-2), while this term penalizes unrealistically depleted
        cells under load without imposing a hard density clamp.
        """
        if (not P.get('use_lowrho_soft_penalty', True)) or (not P.get('use_potential_lowrho_penalty', True)):
            return np.zeros_like(np.asarray(rho, dtype=float))
        rho = np.maximum(np.asarray(rho, dtype=float), s.rmin)
        r = rho / max(s.rho_c, s.rmin)
        rsoft = max(float(P.get('rho_soft_floor_frac', 0.03)), 1e-9)
        strength = float(P.get('lowrho_mu_strength', 1.0))
        # Phi_floor = A1*rho_c*strength*rsoft*exp(-r/rsoft)
        # so mu_floor/A1 = -strength*exp(-r/rsoft).
        return s.A1 * s.rho_c * strength * rsoft * np.exp(-r/rsoft)

    def _lowrho_mu(s, rho):
        """Chemical-potential contribution from the soft low-rho penalty [J/m]."""
        if (not P.get('use_lowrho_soft_penalty', True)) or (not P.get('use_potential_lowrho_penalty', True)):
            return np.zeros_like(np.asarray(rho, dtype=float))
        rho = np.maximum(np.asarray(rho, dtype=float), s.rmin)
        r = rho / max(s.rho_c, s.rmin)
        rsoft = max(float(P.get('rho_soft_floor_frac', 0.03)), 1e-9)
        strength = float(P.get('lowrho_mu_strength', 1.0))
        return -s.A1 * strength * np.exp(-r/rsoft)

    def _mu(s, rho):
        """Chemical potential μ = dΦ/dρ [J/m] from the selected v14 table."""
        return s._interp_log(np.maximum(np.asarray(rho, dtype=float), s.rmin), s.mu_tab)

    def _Phi_pp(s, rho):
        """Thermodynamic curvature d²Φ/dρ² from the selected table."""
        return s._interp_log(np.maximum(np.asarray(rho, dtype=float), s.rmin), s.Phi_pp_tab)

    def sigma_at(s, rho):
        return s._interp_log(np.maximum(np.asarray(rho, dtype=float), s.rmin), s.sigma_at_tab)

    def dsigma_at_drho(s, rho):
        return s._interp_log(np.maximum(np.asarray(rho, dtype=float), s.rmin), s.dsigma_at_drho_tab)

    def kinetic_spinodal_slack(s, rho):
        return s._interp_log(np.maximum(np.asarray(rho, dtype=float), s.rmin), s.kinetic_spinodal_slack_tab)

    def kinetic_spinodal_mask(s, rho):
        return s.dsigma_at_drho(rho) < 0.0

    def kinetic_hazard_gate(s, rho):
        """Optional legacy/diagnostic Arrhenius negative-slope gate.

        v23 production behavior is diagnostic-only: this returns ones unless
        use_arrhenius_kinetic_instability is explicitly enabled with hard/soft
        mode.  The default nucleation rate is controlled by the thermodynamic
        barrier and local plastic activity, not a Boolean dσ_AT/dρ switch.
        """
        arr = np.asarray(rho, dtype=float)
        if not P.get('use_arrhenius_kinetic_instability', False):
            return np.ones_like(arr)
        mode = str(P.get('arrhenius_hazard_gate_mode', 'diagnostic')).lower()
        if mode in ['diagnostic', 'none', 'off']:
            return np.ones_like(arr)
        floor = float(np.clip(P.get('arrhenius_hazard_gate_floor', 0.0), 0.0, 1.0))
        if mode == 'soft':
            slack = s.kinetic_spinodal_slack(rho) / max(float(getattr(s, 'spinodal_slack_scale', 1.0)), 1e-300)
            return np.clip(floor + (1.0-floor)*slack, floor, 1.0)
        return np.where(s.kinetic_spinodal_mask(rho), 1.0, floor)

    def mu_dw(s, r):
        """Chemical potential in normalised units (r = ρ/ρ_c).
        Returns μ normalised by A1 so that |μ| ~ O(1)."""
        rho = np.maximum(r * s.rho_c, s.rmin)
        return s._mu(rho) / max(s.A1, 1e-30)

    def Phi_pp_dw(s, r):
        """Curvature in normalised units."""
        rho = np.maximum(r * s.rho_c, s.rmin)
        return s._Phi_pp(rho) * s.rho_c / max(s.A1, 1e-30)


# ================================================================
# 4. SLIP GEOMETRY
# ================================================================
nSlip = P['nSlip']
base_ang = np.deg2rad(np.array(P['slip_angles_deg']))
_P11_mean = max(np.mean(np.abs([0.5*np.sin(2*a) for a in base_ang])), 1e-6)
drive_sc = 1.0 / _P11_mean

def build_slip(psi):
    ang = base_ang[None,None,:] + psi[:,:,None]
    sv = np.stack([np.cos(ang), np.sin(ang)], axis=-1)
    nv = np.stack([np.cos(ang), -np.sin(ang)], axis=-1)
    Sch = np.zeros((Nx,Ny,nSlip,2,2))
    for s in range(nSlip):
        for i in range(2):
            for j in range(2):
                Sch[:,:,s,i,j] = 0.5*(sv[:,:,s,i]*nv[:,:,s,j]+nv[:,:,s,i]*sv[:,:,s,j])
    s11 = Sch[:,:,:,0,0]
    return ang, sv, nv, Sch, s11

print(f"Slip: {P['slip_angles_deg']}, drive_scale={drive_sc:.2f}")

# ================================================================
# 5. STRESS SOLVER
# ================================================================
mu_iso = (P['C11']-P['C12']+3*P['C44'])/5.0
lam_iso = (P['C11']+4*P['C12']-2*P['C44'])/5.0
E_ps = (P['C11']**2-P['C12']**2)/P['C11']
nu_ps = P['C12']/P['C11']

# Isotropic stiffness
C4 = np.zeros((2,2,2,2))
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                C4[i,j,k,l] = lam_iso*(i==j)*(k==l)+mu_iso*((i==k)*(j==l)+(i==l)*(j==k))

# Green operator
Gamma = np.zeros((2,2,2,2,Nx,Ny))
for i in range(2):
    ki = [KX,KY][i]
    for j in range(2):
        kj = [KX,KY][j]
        for k in range(2):
            kk = [KX,KY][k]
            for l in range(2):
                kl = [KX,KY][l]
                Gamma[i,j,k,l] = (
                    0.25*((i==k)*kj*kl+(j==k)*ki*kl+(i==l)*kj*kk+(j==l)*ki*kk)/(mu_iso*K2nz)
                    -(lam_iso+mu_iso)/(mu_iso*(lam_iso+2*mu_iso))*ki*kj*kk*kl/K2nz**2)
Gamma[:,:,:,:,0,0] = 0

def ms_solve(eps_p, eps_bar, nit=3):
    eps = np.zeros((Nx,Ny,2,2))
    for i in range(2):
        for j in range(2): eps[:,:,i,j] = eps_bar[i,j]
    for _ in range(nit):
        sig = np.einsum('ijkl,...kl->...ij', C4, eps-eps_p)
        sh = np.zeros((Nx,Ny,2,2), dtype=complex)
        deh = np.zeros_like(sh)
        for i in range(2):
            for j in range(2): sh[:,:,i,j] = np.fft.fft2(sig[:,:,i,j])
        for i in range(2):
            for j in range(2):
                acc = np.zeros((Nx,Ny), dtype=complex)
                for k in range(2):
                    for l in range(2): acc += Gamma[i,j,k,l]*sh[:,:,k,l]
                deh[:,:,i,j] = acc
        for i in range(2):
            for j in range(2):
                eps[:,:,i,j] -= np.real(np.fft.ifft2(deh[:,:,i,j]))
                eps[:,:,i,j] += eps_bar[i,j]-eps[:,:,i,j].mean()
        eps = 0.5*(eps+eps.transpose(0,1,3,2))
    return np.einsum('ijkl,...kl->...ij', C4, eps-eps_p), eps


# ================================================================
# 5b. v27 DISLOCATION-STATE HELPERS
# ================================================================
def _rho_mobile_field(rp_arr, rm_arr):
    return np.maximum(np.sum(rp_arr + rm_arr, axis=2), P['rho_min'])


def _rho_forest_total_field(rho_forest_arr=None):
    if (rho_forest_arr is None) or (not P.get('use_rho_state_partition', False)):
        return np.zeros((Nx, Ny), dtype=float)
    return np.maximum(np.sum(rho_forest_arr, axis=2), 0.0)


def _rho_wall_field(rho_wall_arr=None):
    if (rho_wall_arr is None) or (not P.get('use_rho_state_partition', False)):
        return np.zeros((Nx, Ny), dtype=float)
    return np.maximum(np.asarray(rho_wall_arr, dtype=float), 0.0)


def _rho_total_state(rp_arr, rm_arr, rho_forest_arr=None, rho_wall_arr=None):
    if not P.get('use_rho_state_partition', False):
        return _rho_mobile_field(rp_arr, rm_arr)
    return np.maximum(_rho_mobile_field(rp_arr, rm_arr)
                      + _rho_forest_total_field(rho_forest_arr)
                      + _rho_wall_field(rho_wall_arr), P['rho_min'])


def _rho_obstacle_for_slip(ss, rp_arr, rm_arr, rho_total_arr=None,
                           rho_forest_arr=None, rho_wall_arr=None,
                           rho_GB_arr=None, kappa_arr=None):
    """Obstacle density used in the Taylor/Arrhenius law for slip system ss.

    The glissile signed populations rp/rm are not assumed to be the same thing
    as the forest/junction density.  In the v27 reduced CDD closure, the Taylor
    spacing is controlled mainly by rho_forest plus organized wall/GND content,
    with a small residual contribution from mobile line density.  Setting
    use_rho_state_partition=False recovers the v26 behavior.
    """
    mobile_s = np.maximum(rp_arr[:, :, ss] + rm_arr[:, :, ss], 2*P['rho_min'])
    if not P.get('use_rho_state_partition', False):
        return mobile_s
    if rho_total_arr is None:
        rho_total_arr = _rho_total_state(rp_arr, rm_arr, rho_forest_arr, rho_wall_arr)
    mode = str(P.get('rho_state_obstacle_mode', 'forest_wall_gnd')).lower()
    if mode in ['mobile', 'mobile_total', 'legacy']:
        return mobile_s
    if mode in ['total', 'rho_total']:
        return np.maximum(rho_total_arr, 2*P['rho_min'])
    forest_s = np.zeros_like(mobile_s)
    forest_tot = np.zeros_like(mobile_s)
    if rho_forest_arr is not None:
        forest_s = np.maximum(rho_forest_arr[:, :, ss], 0.0)
        forest_tot = _rho_forest_total_field(rho_forest_arr)
    chi = float(np.clip(P.get('rho_state_forest_mix', 0.35), 0.0, 1.0))
    obs = (1.0 - chi)*forest_s + chi*forest_tot
    obs += float(P.get('rho_state_mobile_obstacle_weight', 0.10))*mobile_s
    obs += float(P.get('rho_state_wall_obstacle_weight', 1.0))*_rho_wall_field(rho_wall_arr)
    if rho_GB_arr is not None:
        obs += float(P.get('rho_state_gb_obstacle_weight', 1.0))*np.maximum(rho_GB_arr, 0.0)
    if kappa_arr is None:
        kappa_arr = np.sum(rp_arr-rm_arr, axis=2)
    obs += float(P.get('rho_state_gnd_obstacle_weight', 0.25))*np.abs(kappa_arr)
    return np.maximum(obs, 2*P['rho_min'])


def _rho_ch_scale():
    """Structural density scale used only to nondimensionalize rho in CH/KWC.

    This is not the independent-Taylor kinetic peak.  The independent peak is
    kept as rho_peak_ind/rho_c diagnostic.  For v27, the default structural scale
    is the initialized network density or an explicitly supplied material scale.
    """
    if P.get('use_rho_state_partition', False) and P.get('rho_state_use_structural_scale_for_ch', True):
        val = P.get('_rho_state_ref_runtime', None)
        if val is None:
            val = P.get('rho_state_ref_abs', None)
        try:
            val = float(val)
            if np.isfinite(val) and val > P['rho_min']:
                return val
        except Exception:
            pass
    return max(float(globals().get('rho_c', P.get('rho_min', 1e8))), P['rho_min'])


def _rho_structural_field(rho_forest_arr=None, rho_wall_arr=None):
    """Slow structural density that can organize into walls/cells.

    Mobile density is a carrier population and should not be the conserved
    phase-field variable that undergoes CH-like segregation.
    """
    return np.maximum(_rho_forest_total_field(rho_forest_arr) + _rho_wall_field(rho_wall_arr), 0.0)


def _rho_network_ref_scale():
    """Physical network reference scale for GB source/saturation terms.

    This intentionally avoids the independent Taylor kinetic peak rho_c, which
    is only a crossover diagnostic in the collective model.
    """
    for key in ['_rho_state_total_ref_runtime', '_rho_state_ref_total_runtime', '_rho_state_ref_runtime']:
        try:
            val = float(P.get(key, np.nan))
            if np.isfinite(val) and val > P['rho_min']:
                return val
        except Exception:
            pass
    return max(float(globals().get('rho_c', P.get('rho_min', 1e8))), P['rho_min'])


def _collective_activity_field(cf):
    if cf is None:
        return None
    mode = str(P.get('collective_activity_mode', 'suppression_nc')).lower()
    nc = np.asarray(cf.get('nc', 1.0), dtype=float)
    Pcomp = np.asarray(cf.get('P_complete', 0.0), dtype=float)
    sup = np.asarray(cf.get('suppression', 1.0), dtype=float)
    if mode in ['pcomplete', 'poisson', 'tail']:
        A = Pcomp
    elif mode in ['suppression', 'supp']:
        A = 1.0 - sup
    else:
        ncmax = max(float(P.get('collective_nc_max', 20.0)), 1.0)
        nc_part = np.clip((nc - 1.0)/max(ncmax - 1.0, 1e-12), 0.0, 1.0)
        A = (1.0 - sup) * nc_part
    A = np.maximum(A - float(P.get('collective_activity_floor', 0.0)), 0.0)
    pw = max(float(P.get('collective_activity_power', 1.0)), 1e-12)
    A = np.clip(A, 0.0, 1.0)**pw
    # v27c: collective activity is a coarse-grained correlation/activity field,
    # not a cell-by-cell Bernoulli noise source.  Smooth over an elastic
    # interaction scale supplied in microns; set to 0 for the raw v27 field.
    sm_um = float(P.get('collective_activity_smooth_um', 0.0))
    if sm_um > 0.0:
        sig_px = sm_um*1e-6/max(dx, 1e-30)
        if sig_px > 0.25:
            A = ndimage.gaussian_filter(A, sig_px, mode='wrap')
    return np.clip(A, 0.0, 1.0)




def _diffuse_activity_isotropic(A, dt, D):
    """Implicit spectral isotropic diffusion for scalar or per-slip A."""
    if D <= 0.0 or dt <= 0.0:
        return A
    A = np.asarray(A, dtype=float)
    denom = 1.0 + dt*D*(KX*KX + KY*KY)
    if A.ndim == 2:
        return np.real(np.fft.ifft2(np.fft.fft2(A)/denom))
    out = np.empty_like(A)
    for ss in range(A.shape[2]):
        out[:, :, ss] = np.real(np.fft.ifft2(np.fft.fft2(A[:, :, ss])/denom))
    return out


def _diffuse_activity_fixed_lab(A, dt, Dpar, Dperp):
    """Old v28 fixed lab-frame anisotropic diffusion control.

    This mode is deliberately labeled as a control because it can impose ideal
    straight bands independent of grains.
    """
    if (Dpar <= 0.0 and Dperp <= 0.0) or dt <= 0.0:
        return A
    try:
        theta = np.deg2rad(float(P.get('collective_activity_angle_deg', P.get('slip_angles_deg', [0.0])[0])))
    except Exception:
        theta = 0.0
    sx, sy = np.cos(theta), np.sin(theta)
    kpar = KX*sx + KY*sy
    kperp = -KX*sy + KY*sx
    denom = 1.0 + dt*(Dpar*kpar*kpar + Dperp*kperp*kperp)
    A = np.asarray(A, dtype=float)
    if A.ndim == 2:
        return np.real(np.fft.ifft2(np.fft.fft2(A)/denom))
    out = np.empty_like(A)
    for ss in range(A.shape[2]):
        out[:, :, ss] = np.real(np.fft.ifft2(np.fft.fft2(A[:, :, ss])/denom))
    return out


def _laplacian_crystallographic(A, theta):
    """Orientation-resolved anisotropic Laplacian components with periodic BCs.

    This is intentionally conservative/experimental.  It respects the local
    grain/slip direction through theta(x,y), but because the explicit operator
    can be costly at low rates, production defaults set Dpar=Dperp=0.
    """
    # second derivatives, periodic, grid spacing in meters
    dxm = float(dx)
    Axx = (np.roll(A, -1, axis=0) - 2*A + np.roll(A, 1, axis=0))/(dxm*dxm)
    Ayy = (np.roll(A, -1, axis=1) - 2*A + np.roll(A, 1, axis=1))/(dxm*dxm)
    Axy = (np.roll(np.roll(A, -1, axis=0), -1, axis=1)
           - np.roll(np.roll(A, -1, axis=0),  1, axis=1)
           - np.roll(np.roll(A,  1, axis=0), -1, axis=1)
           + np.roll(np.roll(A,  1, axis=0),  1, axis=1))/(4*dxm*dxm)
    sx, sy = np.cos(theta), np.sin(theta)
    d2par = sx*sx*Axx + 2*sx*sy*Axy + sy*sy*Ayy
    d2perp = sy*sy*Axx - 2*sx*sy*Axy + sx*sx*Ayy
    return d2par, d2perp


def _diffuse_activity_crystallographic(A, dt, psi_field, Dpar, Dperp):
    """Experimental per-slip local-orientation diffusion.

    Unlike fixed_lab_slip_control, the preferred direction is the local grain
    orientation plus the slip-system angle.  A hard substep cap prevents low-rate
    runs from spending all time in this optional control.
    """
    if A.ndim != 3 or (Dpar <= 0.0 and Dperp <= 0.0) or dt <= 0.0:
        return A
    dxm = float(dx)
    Dmax = max(Dpar, Dperp, 0.0)
    if Dmax <= 0.0:
        return A
    # explicit stability; cap substeps and reduce effective dt per substep if needed
    max_cfl = 0.12
    nsub = int(np.ceil(max(1.0, Dmax*dt/(max_cfl*dxm*dxm))))
    nsub_cap = int(P.get('collective_activity_crystallographic_max_substeps', 25))
    if nsub > nsub_cap:
        nsub = nsub_cap
    dts = dt / max(nsub, 1)
    out = A.copy()
    slips = list(P.get('slip_angles_deg', [0.0, 90.0]))
    psi0 = np.asarray(psi_field, dtype=float) if psi_field is not None else np.zeros_like(out[:, :, 0])
    for _ in range(nsub):
        for ss in range(out.shape[2]):
            th = np.deg2rad(psi0 + float(slips[ss % len(slips)]))
            d2p, d2q = _laplacian_crystallographic(out[:, :, ss], th)
            out[:, :, ss] = out[:, :, ss] + dts*(Dpar*d2p + Dperp*d2q)
        out = np.clip(out, 0.0, 1.0)
    return out


def _activity_memory_advance(Aold, source, dt, psi_field=None, per_slip=False):
    """Advance persistent collective activity for ASB localization.

    v29: the default modes do not impose a lab-frame band direction.  The old
    fixed anisotropic diffusion remains available only as an explicit control.
    """
    mode = str(P.get('collective_activity_memory_mode', 'crystallographic_local')).lower()
    if (not P.get('use_collective_activity_memory', False)) or mode in ('none', 'off', 'false'):
        return np.zeros_like(np.asarray(source, dtype=float))

    Aold = np.asarray(Aold, dtype=float)
    src = np.clip(np.asarray(source, dtype=float), 0.0, 1.0)
    # Allow switching between scalar and per-slip states without restart trouble.
    if Aold.shape != src.shape:
        Aold = np.zeros_like(src)

    tauA = max(float(P.get('collective_activity_tau', 2.0e-6)), max(dt, 1e-300))
    A = Aold + dt*(src - Aold)/tauA
    Dpar = max(float(P.get('collective_activity_D_parallel', 0.0)), 0.0)
    Dperp = max(float(P.get('collective_activity_D_perp', 0.0)), 0.0)

    if mode in ('local', 'scalar_local', 'crystallographic_local'):
        pass
    elif mode in ('isotropic', 'scalar_isotropic'):
        A = _diffuse_activity_isotropic(A, dt, Dpar if Dpar > 0 else Dperp)
    elif mode in ('fixed_lab_slip_control', 'fixed_slip_control', 'fixed'):
        A = _diffuse_activity_fixed_lab(A, dt, Dpar, Dperp)
    elif mode in ('crystallographic', 'grain_slip', 'crystal'):
        if A.ndim == 3:
            A = _diffuse_activity_crystallographic(A, dt, psi_field, Dpar, Dperp)
        else:
            A = _diffuse_activity_isotropic(A, dt, Dperp)
    else:
        # Unknown mode: safe fallback is local memory, not imposed bands.
        pass
    return np.clip(A, 0.0, 1.0)


def _activity_memory_scalar(A):
    A = np.asarray(A, dtype=float)
    if A.ndim == 3:
        return np.nanmean(A, axis=2)
    return A


def _activity_memory_for_slip(A, ss):
    A = np.asarray(A, dtype=float)
    if A.ndim == 3:
        return A[:, :, ss % A.shape[2]]
    return A

def _slip_grad_factor(gamma_field, ss):
    if gamma_field is None:
        return 1.0
    gg = grad_mag(gamma_field[:, :, ss])
    med = float(np.nanmedian(gg))
    scl = max(med, float(np.nanmean(gg))*0.25, 1e-30)
    return np.clip(gg/scl, 0.0, 5.0)

# ================================================================
# 6. MACRO BISECTION
# ================================================================
def macro_bisect(edot_tgt, tau_back, rp, rm, T_field, s11, Sch):
    """Find sigma_bar giving mean(eps_dot_11) = edot_tgt."""
    def eval_sb(sb):
        gd = np.zeros((Nx,Ny,nSlip))
        rho_tot_eval = _rho_total_state(rp, rm, globals().get('rho_forest', None), globals().get('rho_wall', None))
        kappa_eval = np.sum(rp-rm, axis=2)
        for s in range(nSlip):
            rs = _rho_obstacle_for_slip(s, rp, rm, rho_tot_eval, globals().get('rho_forest', None),
                                        globals().get('rho_wall', None), globals().get('rho_GB', None), kappa_eval)
            tau = s11[:,:,s]*sb - tau_back[:,:,s]
            seq = np.abs(tau)*drive_sc
            mag = ATpot.gdot(seq, rs, T_field)
            gd[:,:,s] = np.sign(tau)*mag
        # eps_dot_11
        ed11 = np.zeros((Nx,Ny))
        for s in range(nSlip):
            ed11 += gd[:,:,s]*Sch[:,:,s,0,0]
        return ed11.mean(), gd
    sL, sU = 0.0, 3e9
    rL, _ = eval_sb(sL)
    if rL >= edot_tgt:
        _, gd = eval_sb(0)
        return 0.0, gd
    rU, _ = eval_sb(sU)
    while rU < edot_tgt and sU < 5e10:
        sU *= 2; rU, _ = eval_sb(sU)
    for _ in range(25):
        sM = 0.5*(sL+sU)
        rM, _ = eval_sb(sM)
        if rM < edot_tgt: sL = sM
        else: sU = sM
    sb = 0.5*(sL+sU)
    _, gd = eval_sb(sb)
    return sb, gd


# ================================================================
# 7. ADVECTION
# ================================================================
def advect(f, vx, vy, dtl):
    vxp = np.maximum(vx,0); vxm = np.minimum(vx,0)
    vyp = np.maximum(vy,0); vym = np.minimum(vy,0)
    fx = (vxp*(f-np.roll(f,1,0))+vxm*(np.roll(f,-1,0)-f))/dx
    fy = (vyp*(f-np.roll(f,1,1))+vym*(np.roll(f,-1,1)-f))/dy
    return np.maximum(f-dtl*(fx+fy), P['rho_min'])


# ================================================================
# 8. COMPATIBILITY PENALTY
# ================================================================
def F_comp_derivs(kappa_tot, rho_GB, psi):
    """Returns dF/d|kappa|, dF/drho_GB, dF/d(|grad psi|)."""
    gp = grad_mag(psi)
    tgt_k = P['c_alpha']*gp/P['b']
    tgt_g = P['c_GB']*gp/P['b']
    res_a = np.abs(kappa_tot)-tgt_k
    res_g = rho_GB-tgt_g
    dFdk = P['A_alpha']*res_a
    dFdg = P['A_GB']*res_g
    dFdgp = -(P['A_alpha']*P['c_alpha']*res_a + P['A_GB']*P['c_GB']*res_g)/P['b']
    return dFdk, dFdg, dFdgp


# ================================================================
# 9. GRAIN INITIALISATION
# ================================================================
def init_grains():
    Ng = min(P['poly_n'], P['grain_max'])
    rng = np.random.default_rng(P['poly_seed'])
    seeds = np.column_stack((rng.uniform(0,Nx,Ng), rng.uniform(0,Ny,Ng)))
    Xg, Yg = np.arange(Nx)[:,None], np.arange(Ny)[None,:]
    lab = np.zeros((Nx,Ny), int); dmin = np.full((Nx,Ny), 1e30)
    for g,(sx,sy) in enumerate(seeds):
        dxp = np.abs(Xg-sx); dxp = np.minimum(dxp, Nx-dxp)
        dyp = np.abs(Yg-sy); dyp = np.minimum(dyp, Ny-dyp)
        d2 = dxp**2+dyp**2
        m = d2<dmin; lab[m]=g; dmin[m]=d2[m]
    sp = np.deg2rad(P['poly_spread_deg'])
    mm = np.deg2rad(P['poly_min_mis_deg'])
    pv_active = rng.uniform(-sp/2, sp/2, Ng)
    for i in range(1,Ng):
        for _ in range(200):
            if np.min(np.abs(pv_active[i]-pv_active[:i]))>=mm: break
            pv_active[i] = rng.uniform(-sp/2, sp/2)
    pv = np.zeros(P['grain_max'])
    pv[:Ng] = pv_active
    eta = np.zeros((Nx,Ny,P['grain_max']))
    for g in range(Ng): eta[:,:,g] = (lab==g).astype(float)
    # smooth interfaces
    for _ in range(3):
        for g in range(Ng):
            e = eta[:,:,g]
            nb = 0.25*(np.roll(e,1,0)+np.roll(e,-1,0)+np.roll(e,1,1)+np.roll(e,-1,1))
            eta[:,:,g] = 0.75*e+0.25*nb
        es = np.sum(eta[:,:,:Ng],2,keepdims=True)+1e-30
        eta[:,:,:Ng] /= es
    lab = np.argmax(eta[:,:,:Ng],2)
    psi = np.zeros((Nx,Ny))
    for g in range(Ng): psi += eta[:,:,g]*pv[g]
    gb = np.zeros((Nx,Ny))
    for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        gb = np.maximum(gb, (lab!=np.roll(lab,(di,dj),(0,1))).astype(float))
    print(f"  Grains: {Ng}, psi=[{np.rad2deg(pv[:Ng].min()):.1f},{np.rad2deg(pv[:Ng].max()):.1f}] deg")
    return lab, eta, psi, pv, gb, Ng


def angle_wrap(a):
    """Wrap angles to [-pi, pi] for orientation residuals."""
    return (a + np.pi) % (2*np.pi) - np.pi


def eta_purity_fields(eta, Ng, active_thresh=None):
    """Return top-two eta amplitudes and simple mixedness diagnostics.

    This avoids np.partition copies when grain_max is large: it scans active eta
    fields once, updating the largest and second-largest value in each cell.
    eta_entropy is a diagnostic only; the solver uses the pair support and purity
    mask below to distinguish two-grain interfaces from many-grain mixed patches.
    """
    if Ng <= 0:
        z = np.zeros((Nx, Ny))
        return z, z, z.astype(int), z, z.astype(int)
    if active_thresh is None:
        active_thresh = float(P.get('eta_active_thresh', 0.05))
    eta_max = np.zeros((Nx, Ny), dtype=float)
    eta_second = np.zeros((Nx, Ny), dtype=float)
    lab_from_eta = np.zeros((Nx, Ny), dtype=np.int32)
    eta_entropy = np.zeros((Nx, Ny), dtype=float)
    eta_nactive = np.zeros((Nx, Ny), dtype=np.int16)
    eps = 1e-30
    for g in range(Ng):
        eg = np.asarray(eta[:, :, g], dtype=float)
        better = eg > eta_max
        eta_second = np.where(better, eta_max, np.maximum(eta_second, eg))
        lab_from_eta = np.where(better, g, lab_from_eta)
        eta_max = np.where(better, eg, eta_max)
        eta_entropy -= np.where(eg > eps, eg*np.log(eg + eps), 0.0)
        eta_nactive += (eg > active_thresh)
    return eta_max, eta_second, lab_from_eta, eta_entropy, eta_nactive


def eta_pure_mask(eta, Ng, eta_min=None, second_frac_max=None):
    """Cells sufficiently dominated by one eta field to be treated as grain interior.

    This is used only for topology bookkeeping / hard-edge support, not for the
    variational AC update.  It prevents transient spinodal/mixed patches from
    being converted into many permanent grain IDs.
    """
    if eta_min is None:
        eta_min = float(P.get('component_relabel_eta_min', P.get('gb_hard_eta_min', 0.65)))
    if second_frac_max is None:
        second_frac_max = float(P.get('component_relabel_second_frac_max', P.get('gb_hard_second_frac_max', 0.35)))
    eta_max, eta_second, _, _, _ = eta_purity_fields(eta, Ng)
    return (eta_max >= eta_min) & (eta_second <= second_frac_max*np.maximum(eta_max, 1e-30))


def diffuse_gb_support(eta, lab, Ng):
    """Continuous GB support with v13 mixed-patch protection.

    A true diffuse two-grain interface should have two large eta amplitudes.
    The pair-support 4*eta_max*eta_second is therefore high for normal GBs but
    low for a many-field interpenetrating patch.  Hard label edges are included
    only where both sides are phase-pure, so checkerboard labels in a mixed
    region cannot make gb_mask=1 everywhere.
    """
    if Ng <= 0:
        return np.zeros((Nx, Ny))

    eta_max, eta_second, _, _, _ = eta_purity_fields(eta, Ng)
    if P.get('use_pairwise_gb_support', True):
        soft = np.clip(float(P.get('gb_pair_support_scale', 4.0))*eta_max*eta_second, 0.0, 1.0)
    else:
        e = eta[:, :, :Ng]
        ss = np.sum(e*e, axis=2)
        soft = np.clip(2.0*(1.0 - ss), 0.0, 1.0)

    hard = np.zeros((Nx, Ny))
    if P.get('use_purity_aware_hard_gb_edges', True):
        pure = eta_pure_mask(eta, Ng,
                             eta_min=float(P.get('gb_hard_eta_min', 0.65)),
                             second_frac_max=float(P.get('gb_hard_second_frac_max', 0.35)))
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            neigh_pure = np.roll(pure, (di, dj), (0, 1))
            edge = (lab != np.roll(lab, (di, dj), (0, 1))) & pure & neigh_pure
            hard = np.maximum(hard, edge.astype(float))
    else:
        for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            hard = np.maximum(hard, (lab != np.roll(lab, (di,dj), (0,1))).astype(float))

    support = np.maximum(soft, hard)
    # one cheap smoothing pass so Frank-Bilby relaxation acts over the diffuse width
    support = np.maximum(support, 0.25*(np.roll(support,1,0)+np.roll(support,-1,0)+
                                        np.roll(support,1,1)+np.roll(support,-1,1)))
    return np.clip(support, 0.0, 1.0)


class _TopoDSU:
    """Small union-find used only for periodic connected-component bookkeeping."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1]*n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        if a <= 0 or b <= 0:
            return
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _periodic_component_roots(mask, conn8=True):
    """Return a component-root image for a boolean mask on the periodic grid.

    ndimage.label is non-periodic, so this merges labels across opposite domain
    faces.  This avoids falsely splitting grains that simply wrap through the
    periodic boundary.
    """
    if not np.any(mask):
        return np.zeros_like(mask, dtype=np.int32), {}
    structure = np.ones((3, 3), dtype=np.int8) if conn8 else np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.int8)
    cc, ncc = ndimage.label(mask, structure=structure)
    if ncc <= 1:
        return cc.astype(np.int32), ({1: int(np.sum(mask))} if ncc == 1 else {})
    dsu = _TopoDSU(ncc + 1)

    # Merge across x-periodic and y-periodic faces.
    for j in range(Ny):
        dsu.union(int(cc[0, j]), int(cc[Nx-1, j]))
    for i in range(Nx):
        dsu.union(int(cc[i, 0]), int(cc[i, Ny-1]))

    if conn8:
        # Diagonal periodic adjacencies across the faces.
        for j in range(Ny):
            dsu.union(int(cc[0, j]), int(cc[Nx-1, (j-1) % Ny]))
            dsu.union(int(cc[0, j]), int(cc[Nx-1, (j+1) % Ny]))
        for i in range(Nx):
            dsu.union(int(cc[i, 0]), int(cc[(i-1) % Nx, Ny-1]))
            dsu.union(int(cc[i, 0]), int(cc[(i+1) % Nx, Ny-1]))

    root_img = np.zeros_like(cc, dtype=np.int32)
    roots = {}
    vals = np.unique(cc)
    for val in vals:
        if val == 0:
            continue
        root = dsu.find(int(val))
        root_img[cc == val] = root
        roots[root] = 0
    for root in roots:
        roots[root] = int(np.sum(root_img == root))
    return root_img, roots


def _periodic_dilate(mask, niter=1, conn8=True):
    """Periodic binary dilation by roll operations."""
    out = mask.astype(bool).copy()
    shifts = [(-1,0),(1,0),(0,-1),(0,1)]
    if conn8:
        shifts += [(-1,-1),(-1,1),(1,-1),(1,1)]
    for _ in range(max(int(niter), 0)):
        acc = out.copy()
        for di, dj in shifts:
            acc |= np.roll(out, (di, dj), axis=(0, 1))
        out = acc
    return out


def grain_topology_stats(lab, min_px=10):
    """Connected-component count for the hard argmax grain map."""
    topo_components = 0
    multi_labels = 0
    max_components = 0
    for gid in np.unique(lab):
        mask = (lab == gid)
        _, roots = _periodic_component_roots(mask, conn8=True)
        nbig = int(sum(sz >= min_px for sz in roots.values()))
        topo_components += nbig
        if nbig > 1:
            multi_labels += 1
        max_components = max(max_components, nbig)
    return dict(topo_components=int(topo_components),
                multi_component_labels=int(multi_labels),
                max_components_per_label=int(max_components))


def split_disconnected_grain_components(eta, psi_gv, Ng, lab, psi_lat=None, psi_plastic=None):
    """Promote disconnected components of a grain ID into unused eta fields.

    This is bookkeeping, not physics: it does not reset rho, rp/rm, temperature, or
    plastic strain.  It prevents one eta_i from representing two separated grains
    that can later merge artificially just because they share an order-parameter ID.

    The largest connected component keeps the original ID.  Additional components
    above component_relabel_min_px get fresh eta IDs when slots are available.
    Periodic boundary connectivity is respected.
    """
    if (not P.get('use_component_relabel', True)) or Ng >= P['grain_max']:
        return eta, psi_gv, Ng, lab, {'splits': 0, 'unassigned': 0, **grain_topology_stats(lab, P.get('component_relabel_min_px', 24))}

    min_px = int(P.get('component_relabel_min_px', 24))
    max_splits = int(P.get('component_relabel_max_splits_per_step', 8))
    dilate_px = int(P.get('component_relabel_dilate_px', 1))
    keep_orientation = bool(P.get('component_relabel_keep_orientation', True))
    jitter = np.deg2rad(float(P.get('component_relabel_jitter_deg', 0.0)))
    rng = np.random.default_rng(int(P.get('poly_seed', 42)) + 104729 + int(Ng))

    splits = 0
    unassigned = 0
    active_ids = list(range(Ng))

    if P.get('component_relabel_require_pure', True):
        pure_for_split = eta_pure_mask(eta, Ng,
            eta_min=float(P.get('component_relabel_eta_min', 0.65)),
            second_frac_max=float(P.get('component_relabel_second_frac_max', 0.35)))
    else:
        pure_for_split = np.ones((Nx, Ny), dtype=bool)

    # Work from the current hard argmax labels.  Sort by component size so the
    # largest component keeps the original ID and sizeable satellites are split.
    for gid in active_ids:
        if gid >= Ng or Ng >= P['grain_max'] or splits >= max_splits:
            break
        mask = (lab == gid) & pure_for_split
        if int(np.sum(mask)) < 2*min_px:
            continue
        root_img, roots = _periodic_component_roots(mask, conn8=True)
        big = [(root, sz) for root, sz in roots.items() if sz >= min_px]
        if len(big) <= 1:
            continue
        big.sort(key=lambda kv: kv[1], reverse=True)
        # Keep the largest component under the old ID.
        for root, sz in big[1:]:
            if Ng >= P['grain_max'] or splits >= max_splits:
                unassigned += 1
                continue
            comp = (root_img == root) & pure_for_split
            move = _periodic_dilate(comp, dilate_px, conn8=True) & (eta[:, :, gid] > 1e-10)
            if int(np.sum(comp)) < min_px or not np.any(move):
                continue
            new_gid = Ng
            eta[:, :, new_gid] = 0.0
            eta[:, :, new_gid] = np.where(move, eta[:, :, gid], eta[:, :, new_gid])
            eta[:, :, gid] = np.where(move, 0.0, eta[:, :, gid])

            if keep_orientation:
                psi_gv[new_gid] = psi_gv[gid]
            elif psi_lat is not None and psi_plastic is not None and np.any(comp):
                # Estimate the grain-owned orientation from the local lattice field.
                vals = angle_wrap(psi_lat[comp] - psi_plastic[comp])
                psi_gv[new_gid] = float(np.angle(np.mean(np.exp(1j*vals))))
            else:
                psi_gv[new_gid] = psi_gv[gid]
            if jitter > 0:
                psi_gv[new_gid] = angle_wrap(psi_gv[new_gid] + rng.uniform(-jitter, jitter))

            # v15 provenance: topology relabels are bookkeeping births.  If the
            # parent lineage was explicit hazard nucleation, this is a hazard
            # descendant; otherwise it is a spinodal/topology-origin grain.
            parent_origin = int(grain_origin_lineage[gid]) if ('grain_origin_lineage' in globals() and gid < len(grain_origin_lineage)) else ORIGIN_INITIAL
            lineage = ORIGIN_HAZARD if parent_origin == ORIGIN_HAZARD else ORIGIN_SPINODAL
            cx, cy = _component_centroid(comp)
            _record_grain_birth(new_gid, lineage, MECH_TOPOLOGY, parent=gid,
                                step=int(globals().get('current_step_for_provenance', -1)),
                                x=cx, y=cy, area_px=int(np.sum(comp)),
                                theta_deg=np.rad2deg(float(psi_gv[new_gid])),
                                theta_max_deg=np.nan, R_um=np.nan, barrier_eV=np.nan)

            Ng += 1
            splits += 1

    # Preserve partition of unity.  This is not a physical relaxation step.
    es = np.sum(eta[:, :, :Ng], axis=2, keepdims=True) + 1e-30
    eta[:, :, :Ng] /= es
    lab = np.argmax(eta[:, :, :Ng], axis=2)
    stats = grain_topology_stats(lab, min_px)
    stats.update(dict(splits=int(splits), unassigned=int(unassigned)))
    return eta, psi_gv, Ng, lab, stats



def reconstruct_psi_lat(eta, psi_gv, psi_plastic, Ng):
    """Grain-slaved lattice orientation plus a plastic residual.

    This prevents migrating grains from leaving behind ghost orientations.
    The grain-owned part follows eta_i psi_i; only psi_plastic is evolved by
    spin/compatibility kinetics.
    """
    if Ng <= 0:
        base = np.zeros((Nx, Ny))
    else:
        weights = eta[:, :, :Ng]
        esum = np.sum(weights, axis=2) + 1e-30
        base = np.sum(weights * psi_gv[:Ng][None,None,:], axis=2) / esum
    psi = base + psi_plastic
    return np.clip(psi, -np.deg2rad(P['psi_max_deg']), np.deg2rad(P['psi_max_deg']))


def reset_gb_fields_to_current_topology(rho_GB, eta, lab, Ng):
    """Erase stale boundary density away from the current diffuse GB topology."""
    support = diffuse_gb_support(eta, lab, Ng)
    return rho_GB * np.where(support > P.get('gb_support_floor', 0.02), 1.0, 0.0), support


# ================================================================
# 9b. v15 DRX PROVENANCE ACCOUNTING (defined before initialisation)
# ================================================================
ORIGIN_UNASSIGNED = -1
ORIGIN_INITIAL = 0
ORIGIN_SPINODAL = 1
ORIGIN_HAZARD = 2

MECH_UNASSIGNED = -1
MECH_INITIAL = 0
MECH_TOPOLOGY = 1
MECH_HAZARD = 2

def _record_grain_birth(gid, origin, mechanism, parent=-1, step=-1, x=np.nan, y=np.nan,
                        area_px=0, theta_deg=np.nan, theta_max_deg=np.nan,
                        R_um=np.nan, barrier_eV=np.nan):
    """Record grain-field provenance for diagnostics only."""
    if not P.get('track_grain_provenance', True):
        return
    if 'grain_origin_lineage' not in globals():
        return
    if gid < 0 or gid >= len(grain_origin_lineage):
        return
    global _grain_step_topology_births, _grain_step_hazard_births
    grain_origin_lineage[gid] = int(origin)
    grain_birth_mechanism[gid] = int(mechanism)
    grain_parent[gid] = int(parent) if parent is not None else -1
    grain_birth_step[gid] = int(step)
    grain_birth_x[gid] = float(x)
    grain_birth_y[gid] = float(y)
    grain_birth_area_px[gid] = int(area_px)
    grain_birth_theta_deg[gid] = float(theta_deg)
    grain_birth_theta_max_deg[gid] = float(theta_max_deg)
    grain_birth_R_um[gid] = float(R_um)
    grain_birth_barrier_eV[gid] = float(barrier_eV)
    if int(mechanism) == MECH_TOPOLOGY:
        _grain_step_topology_births += 1
    elif int(mechanism) == MECH_HAZARD:
        _grain_step_hazard_births += 1

def _provenance_counts(Ng):
    """Counts of allocated grain fields by lineage and birth mechanism."""
    if (not P.get('track_grain_provenance', True)) or ('grain_origin_lineage' not in globals()):
        return dict(
            grain_initial_lineage=Ng, grain_spinodal_lineage=0, grain_hazard_lineage=0,
            grain_initial_births=Ng, grain_topology_births=0, grain_hazard_births=0,
            grain_topology_step_births=0, grain_hazard_step_births=0)
    oo = grain_origin_lineage[:Ng]
    mm = grain_birth_mechanism[:Ng]
    return dict(
        grain_initial_lineage=int(np.sum(oo == ORIGIN_INITIAL)),
        grain_spinodal_lineage=int(np.sum(oo == ORIGIN_SPINODAL)),
        grain_hazard_lineage=int(np.sum(oo == ORIGIN_HAZARD)),
        grain_initial_births=int(np.sum(mm == MECH_INITIAL)),
        grain_topology_births=int(np.sum(mm == MECH_TOPOLOGY)),
        grain_hazard_births=int(np.sum(mm == MECH_HAZARD)),
        grain_topology_step_births=int(globals().get('_grain_step_topology_births', 0)),
        grain_hazard_step_births=int(globals().get('_grain_step_hazard_births', 0)),
    )

def _component_centroid(mask):
    if not np.any(mask):
        return np.nan, np.nan
    ii, jj = np.where(mask)
    return float(np.mean(ii)), float(np.mean(jj))



# ================================================================
# 9d. v25 restart/checkpoint helpers
# ================================================================
def _rng_state_to_json(rng):
    try:
        return json.dumps(rng.bit_generator.state)
    except Exception:
        return ''


def _rng_state_from_json(rng, state_json):
    if state_json is None:
        return rng
    try:
        if isinstance(state_json, np.ndarray):
            state_json = state_json.item()
        if isinstance(state_json, bytes):
            state_json = state_json.decode('utf-8')
        if str(state_json).strip():
            rng.bit_generator.state = json.loads(str(state_json))
    except Exception as exc:
        print(f"WARNING: could not restore nucleation RNG state: {exc}")
    return rng


def _circular_mean_angle(vals, default=0.0):
    vv = np.asarray(vals, dtype=float)
    good = np.isfinite(vv)
    if not np.any(good):
        return float(default)
    z = np.mean(np.exp(1j*vv[good]))
    if not np.isfinite(z):
        return float(default)
    return float(np.angle(z))


def _eta_from_labels(lab_arr, Ng_load, sigma_px=0.75):
    """Approximate diffuse eta fields from a hard label map.

    This is only used when loading old diagnostic fields_*.npz files that did
    not save eta.  Exact v25 checkpoints store eta and do not call this path.
    """
    lab_i = np.asarray(lab_arr, dtype=np.int32)
    eta_new = np.zeros((Nx, Ny, P['grain_max']), dtype=float)
    Ng_load = int(min(max(Ng_load, 1), P['grain_max']))
    sig = float(max(sigma_px, 0.0))
    for g in range(Ng_load):
        e = (lab_i == g).astype(float)
        if sig > 0.0:
            e = ndimage.gaussian_filter(e, sig, mode='wrap')
        eta_new[:, :, g] = e
    es = np.sum(eta_new[:, :, :Ng_load], axis=2)
    # If labels were not contiguous, make the largest label at least assigned.
    bad = es <= 1e-300
    if np.any(bad):
        eta_new[bad, lab_i[bad] % Ng_load] = 1.0
        es = np.sum(eta_new[:, :, :Ng_load], axis=2)
    eta_new[:, :, :Ng_load] /= np.maximum(es[:, :, None], 1e-300)
    return eta_new


def _resize_grain_array(arr, dtype=None, fill=0):
    aa = np.asarray(arr)
    if dtype is None:
        dtype = aa.dtype
    out_arr = np.full(P['grain_max'], fill, dtype=dtype)
    ncopy = min(P['grain_max'], aa.size)
    out_arr[:ncopy] = aa.ravel()[:ncopy]
    return out_arr


def _load_restart_state(path, eta, psi_gv, Ng, lab, psi_plastic, rp, rm, rho, T,
                        rho_GB, gamma_slip, eps_p, E_tot, H_nuc, E_nuc,
                        grain_origin_lineage, grain_birth_mechanism, grain_parent,
                        grain_birth_step, grain_birth_x, grain_birth_y,
                        grain_birth_area_px, grain_birth_theta_deg,
                        grain_birth_theta_max_deg, grain_birth_R_um,
                        grain_birth_barrier_eV):
    """Load an exact v25 checkpoint or approximate state from old fields_*.npz."""
    if path is None or str(path).strip() == '':
        return (False, eta, psi_gv, Ng, lab, psi_plastic, rp, rm, rho, T,
                rho_GB, gamma_slip, eps_p, E_tot, H_nuc, E_nuc,
                grain_origin_lineage, grain_birth_mechanism, grain_parent,
                grain_birth_step, grain_birth_x, grain_birth_y,
                grain_birth_area_px, grain_birth_theta_deg,
                grain_birth_theta_max_deg, grain_birth_R_um,
                grain_birth_barrier_eV)
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"restart_file not found: {path}")
    z = np.load(path, allow_pickle=True)
    files = set(z.files)
    exact = ('eta' in files and 'psi_gv' in files)
    print(f"\n=== Loading restart state: {path} ===")
    print(f"  restart type: {'exact v25 checkpoint' if exact else 'approximate diagnostic NPZ reconstruction'}")
    if 'rho' in files and z['rho'].shape != (Nx, Ny):
        raise ValueError(f"restart rho shape {z['rho'].shape} does not match current grid {(Nx, Ny)}")
    if 'rp' in files and z['rp'].shape[:2] != (Nx, Ny):
        raise ValueError(f"restart rp shape {z['rp'].shape} does not match current grid {(Nx, Ny)}")

    if 'rp' in files and 'rm' in files:
        rp_load = np.asarray(z['rp'], dtype=float)
        rm_load = np.asarray(z['rm'], dtype=float)
        if rp_load.shape[2] != nSlip:
            raise ValueError(f"restart nSlip={rp_load.shape[2]} but current nSlip={nSlip}")
        rp = np.clip(rp_load, P['rho_min'], P['rho_max'])
        rm = np.clip(rm_load, P['rho_min'], P['rho_max'])
        rho = np.maximum(np.sum(rp+rm, axis=2), P['rho_min'])
    elif 'rho' in files:
        rho = np.clip(np.asarray(z['rho'], dtype=float), P['rho_min'], P['rho_max'])
        rp = np.zeros((Nx, Ny, nSlip)); rm = np.zeros_like(rp)
        for ss in range(nSlip):
            rp[:, :, ss] = 0.5*rho/nSlip
            rm[:, :, ss] = 0.5*rho/nSlip
    else:
        raise ValueError("restart file must contain either rp/rm or rho")

    if exact:
        eta_load = np.asarray(z['eta'], dtype=float)
        Ng = int(z['Ng']) if 'Ng' in files else int(eta_load.shape[2])
        Ng = int(min(max(Ng, 1), P['grain_max']))
        eta = np.zeros((Nx, Ny, P['grain_max']), dtype=float)
        eta[:, :, :Ng] = eta_load[:, :, :Ng]
        psi_gv = np.zeros(P['grain_max'], dtype=float)
        psi_tmp = np.asarray(z['psi_gv'], dtype=float).ravel()
        psi_gv[:min(Ng, psi_tmp.size)] = psi_tmp[:min(Ng, psi_tmp.size)]
        lab = np.argmax(eta[:, :, :Ng], axis=2)
    else:
        if 'lab' not in files:
            raise ValueError("diagnostic restart file needs lab when eta is absent")
        lab = np.asarray(z['lab'], dtype=np.int32)
        Ng = int(np.nanmax(lab)) + 1
        Ng = int(min(max(Ng, 1), P['grain_max']))
        eta = _eta_from_labels(lab, Ng, sigma_px=P.get('restart_lab_to_eta_smooth_sigma', 0.75))
        psi_plastic_tmp = np.asarray(z['psi_plastic'], dtype=float) if 'psi_plastic' in files else np.zeros((Nx, Ny))
        psi_lat_tmp = np.asarray(z['psi_lat'], dtype=float) if 'psi_lat' in files else psi_plastic_tmp.copy()
        psi_gv = np.zeros(P['grain_max'], dtype=float)
        for g in range(Ng):
            m = lab == g
            psi_gv[g] = _circular_mean_angle(psi_lat_tmp[m] - psi_plastic_tmp[m], default=0.0)
        lab = np.argmax(eta[:, :, :Ng], axis=2)

    if 'psi_plastic' in files:
        psi_plastic = np.asarray(z['psi_plastic'], dtype=float)
    else:
        psi_plastic = np.zeros((Nx, Ny), dtype=float)
    T = np.asarray(z['T'], dtype=float) if 'T' in files else np.full((Nx, Ny), P['T0'])
    scale = float(P.get('restart_temperature_perturb_scale', 1.0))
    offset = float(P.get('restart_temperature_offset_K', 0.0))
    if scale != 1.0:
        Tm = float(np.nanmean(T))
        T = Tm + scale*(T - Tm)
    if offset != 0.0:
        T = T + offset
    T = np.where(np.isfinite(T), T, P['T0'])
    rho_GB = np.asarray(z['rho_GB'], dtype=float) if 'rho_GB' in files else np.zeros((Nx, Ny), dtype=float)

    if (not P.get('restart_reset_plastic_strain', False)) and 'gamma_slip' in files:
        gamma_slip = np.asarray(z['gamma_slip'], dtype=float)
    else:
        gamma_slip = np.zeros((Nx, Ny, nSlip), dtype=float)
    if (not P.get('restart_reset_plastic_strain', False)) and 'eps_p' in files:
        eps_p = np.asarray(z['eps_p'], dtype=float)
    else:
        eps_p = np.zeros((Nx, Ny, 2, 2), dtype=float)
    if (not P.get('restart_reset_clock', True)) and 'E_tot' in files:
        E_tot = np.asarray(z['E_tot'], dtype=float)
    else:
        E_tot = np.zeros((2, 2), dtype=float)
    if P.get('restart_E11', None) is not None:
        E_tot[0,0] = float(P.get('restart_E11'))

    if P.get('restart_reset_hazard', False):
        H_nuc = np.zeros((Nx, Ny), dtype=float)
        E_nuc = -np.log(np.maximum(_rng_nuc.random((Nx, Ny)), 1e-300))
    else:
        H_nuc = np.asarray(z['H_nuc'], dtype=float) if 'H_nuc' in files else np.zeros((Nx, Ny), dtype=float)
        E_nuc = np.asarray(z['E_nuc'], dtype=float) if 'E_nuc' in files else -np.log(np.maximum(_rng_nuc.random((Nx, Ny)), 1e-300))

    for name, arr, dtype, fill in [
        ('grain_origin_lineage', grain_origin_lineage, np.int16, ORIGIN_UNASSIGNED),
        ('grain_birth_mechanism', grain_birth_mechanism, np.int16, MECH_UNASSIGNED),
        ('grain_parent', grain_parent, np.int32, -1),
        ('grain_birth_step', grain_birth_step, np.int32, -1),
        ('grain_birth_x', grain_birth_x, float, np.nan),
        ('grain_birth_y', grain_birth_y, float, np.nan),
        ('grain_birth_area_px', grain_birth_area_px, np.int32, 0),
        ('grain_birth_theta_deg', grain_birth_theta_deg, float, np.nan),
        ('grain_birth_theta_max_deg', grain_birth_theta_max_deg, float, np.nan),
        ('grain_birth_R_um', grain_birth_R_um, float, np.nan),
        ('grain_birth_barrier_eV', grain_birth_barrier_eV, float, np.nan),
    ]:
        if name in files:
            val = _resize_grain_array(z[name], dtype=dtype, fill=fill)
            arr[...] = val
    if not np.any(grain_birth_mechanism[:Ng] != MECH_UNASSIGNED):
        for g in range(Ng):
            _record_grain_birth(g, ORIGIN_INITIAL, MECH_INITIAL, parent=-1, step=0,
                                x=np.nan, y=np.nan, area_px=int(np.sum(lab == g)),
                                theta_deg=float(np.rad2deg(psi_gv[g])),
                                theta_max_deg=np.nan, R_um=np.nan, barrier_eV=np.nan)

    if 'rng_nuc_state_json' in files:
        _rng_state_from_json(_rng_nuc, z['rng_nuc_state_json'])

    print(f"  loaded Ng={Ng}, rho={np.nanmean(rho):.2e}, T={np.nanmean(T):.1f}-{np.nanmax(T):.1f} K")
    if not exact:
        print("  WARNING: eta/psi_gv were reconstructed from hard labels; use only for short branch tests.")
    return (True, eta, psi_gv, Ng, lab, psi_plastic, rp, rm, rho, T,
            rho_GB, gamma_slip, eps_p, E_tot, H_nuc, E_nuc,
            grain_origin_lineage, grain_birth_mechanism, grain_parent,
            grain_birth_step, grain_birth_x, grain_birth_y,
            grain_birth_area_px, grain_birth_theta_deg,
            grain_birth_theta_max_deg, grain_birth_R_um,
            grain_birth_barrier_eV)

# ================================================================
# 10. BUILD + INITIALISE
# ================================================================
print("\n=== Variational DRX v34: candidate-nucleus DRX + GB-transmission/process-zone ASB ===")
ATpot = ATPotential(P)
ATpot.build(P['T0'], P['edot_app'])
rho_c = ATpot.rho_c

lab, eta, psi_lat, psi_gv, gb_mask, Ng = init_grains()
psi_plastic = np.zeros((Nx, Ny))
gb_mask = diffuse_gb_support(eta, lab, Ng)
psi_lat = reconstruct_psi_lat(eta, psi_gv, psi_plastic, Ng)

# initial rho: default remains relative to rho_c for calibrated DRX runs,
# but high-rate/low-T ASB tests can make rho_c exceed any physical density.
r0 = float(P.get('rho0_rhoc_frac', 0.90))
rho0_mode = str(P.get('rho0_mode', 'relative_to_rho_c')).lower()
rho0_abs = P.get('rho0_abs', None)
if rho0_abs is not None:
    try:
        rho0 = float(rho0_abs)
        rho0_mode = 'absolute'
    except Exception:
        rho0 = r0 * rho_c
elif rho0_mode in ['absolute', 'abs', 'fixed']:
    rho0 = float(P.get('rho0_abs', r0*rho_c) or r0*rho_c)
elif rho0_mode in ['km', 'km_equilibrium', 'equilibrium']:
    k2_init = _km_k2_from_T(np.array([P['T0']]))[0]
    rho0 = (float(P.get('KM_k1', 5.0e8))/max(k2_init, 1e-30))**2
else:
    rho0 = r0 * rho_c
if P.get('rho0_cap_to_rho_max', True):
    rho0_cap = float(P.get('rho0_max_frac_rho_max', 0.75))*float(P.get('rho_max', rho0))
    if np.isfinite(rho0_cap) and rho0 > rho0_cap:
        print(f"  rho0 requested {rho0:.2e} exceeds physical density bound {rho0_cap:.2e}; using bound.")
        rho0 = rho0_cap
rho0 = float(np.clip(rho0, P['rho_min'], P.get('rho_max', rho0)))
print(f"  rho0={rho0:.2e} (mode={rho0_mode}, r0={r0:.2f}, rho_c={rho_c:.2e})")

rng_init = np.random.default_rng(99)
rp = np.zeros((Nx,Ny,nSlip))
rm = np.zeros_like(rp)
for s in range(nSlip):
    for g in range(Ng):
        m = lab==g
        r0g = rho0*(1+0.03*rng_init.standard_normal())
        r0g = max(r0g, P['rho_min'])
        rp[m,s] = 0.5*r0g/nSlip
        rm[m,s] = 0.5*r0g/nSlip
rho = np.maximum(np.sum(rp+rm, axis=2), P['rho_min'])

T = np.full((Nx,Ny), P['T0'])
rho_GB = np.zeros((Nx,Ny))
gamma_slip = np.zeros((Nx,Ny,nSlip))
eps_p = np.zeros((Nx,Ny,2,2))
E_tot = np.zeros((2,2))

# v11 cumulative hazard memory for finite-amplitude nucleation.  H_nuc is a
# local integrated hazard; E_nuc is the exponential random threshold for the
# next event in each patch.  These fields are reset locally after nucleation or
# GB sweep, but otherwise retain exposure history.
_rng_nuc = np.random.default_rng(int(P.get('nuc_rng_seed', 271828)))
H_nuc = np.zeros((Nx, Ny), dtype=float)
E_nuc = -np.log(np.maximum(_rng_nuc.random((Nx, Ny)), 1e-300))

# v34 candidate-nucleus state.  Candidate states are deliberately separate from
# eta/grain labels so the grain count represents persistent nuclei only.
nuc_cand_active = np.zeros((Nx, Ny), dtype=bool)
nuc_cand_age = np.zeros((Nx, Ny), dtype=np.int16)
nuc_cand_best_barrier = np.full((Nx, Ny), np.inf, dtype=float)
nuc_cand_birth_step = np.full((Nx, Ny), -1, dtype=np.int32)

# v15 grain provenance arrays.  These are diagnostics only; field evolution
# still follows CH/AC/hazard/topology kinetics.
grain_origin_lineage = np.full(P['grain_max'], ORIGIN_UNASSIGNED, dtype=np.int16)
grain_birth_mechanism = np.full(P['grain_max'], MECH_UNASSIGNED, dtype=np.int16)
grain_parent = np.full(P['grain_max'], -1, dtype=np.int32)
grain_birth_step = np.full(P['grain_max'], -1, dtype=np.int32)
grain_birth_x = np.full(P['grain_max'], np.nan, dtype=float)
grain_birth_y = np.full(P['grain_max'], np.nan, dtype=float)
grain_birth_area_px = np.zeros(P['grain_max'], dtype=np.int32)
grain_birth_theta_deg = np.full(P['grain_max'], np.nan, dtype=float)
grain_birth_theta_max_deg = np.full(P['grain_max'], np.nan, dtype=float)
grain_birth_R_um = np.full(P['grain_max'], np.nan, dtype=float)
grain_birth_barrier_eV = np.full(P['grain_max'], np.nan, dtype=float)
_grain_step_topology_births = 0
_grain_step_hazard_births = 0
current_step_for_provenance = 0
for _gid0 in range(Ng):
    _record_grain_birth(_gid0, ORIGIN_INITIAL, MECH_INITIAL, parent=-1, step=0,
                        x=np.nan, y=np.nan, area_px=int(np.sum(lab == _gid0)),
                        theta_deg=float(np.rad2deg(psi_gv[_gid0])),
                        theta_max_deg=np.nan, R_um=np.nan, barrier_eV=np.nan)
# Do not count initial grains as "step births".
_grain_step_topology_births = 0
_grain_step_hazard_births = 0

# v25 restart/branch state load.  This happens after default allocation so any
# missing arrays in an old diagnostic NPZ can safely fall back to zeros.
(_restart_loaded, eta, psi_gv, Ng, lab, psi_plastic, rp, rm, rho, T,
 rho_GB, gamma_slip, eps_p, E_tot, H_nuc, E_nuc,
 grain_origin_lineage, grain_birth_mechanism, grain_parent,
 grain_birth_step, grain_birth_x, grain_birth_y, grain_birth_area_px,
 grain_birth_theta_deg, grain_birth_theta_max_deg, grain_birth_R_um,
 grain_birth_barrier_eV) = _load_restart_state(
    P.get('restart_file', None), eta, psi_gv, Ng, lab, psi_plastic, rp, rm, rho, T,
    rho_GB, gamma_slip, eps_p, E_tot, H_nuc, E_nuc,
    grain_origin_lineage, grain_birth_mechanism, grain_parent,
    grain_birth_step, grain_birth_x, grain_birth_y, grain_birth_area_px,
    grain_birth_theta_deg, grain_birth_theta_max_deg, grain_birth_R_um,
    grain_birth_barrier_eV)
gb_mask = diffuse_gb_support(eta, lab, Ng)
psi_lat = reconstruct_psi_lat(eta, psi_gv, psi_plastic, Ng)

# v27 dislocation-state partition.  Existing rp/rm are interpreted as the
# glissile/mobile signed population.  If no v27 restart fields are present, split
# the current total density into mobile and forest/wall components so that the
# total physical density is preserved.
rho_forest = np.zeros((Nx, Ny, nSlip), dtype=float)
rho_wall = np.zeros((Nx, Ny), dtype=float)
if P.get('use_rho_state_partition', False):
    rho_current = np.maximum(np.sum(rp+rm, axis=2), P['rho_min'])
    loaded_state = False
    if _restart_loaded and P.get('rho_state_load_from_restart', True):
        try:
            ztmp = np.load(Path(P.get('restart_file')), allow_pickle=True)
            if 'rho_forest' in ztmp.files:
                rf_load = np.asarray(ztmp['rho_forest'], dtype=float)
                if rf_load.shape == rho_forest.shape:
                    rho_forest = np.clip(rf_load, 0.0, P['rho_max'])
                    loaded_state = True
            if 'rho_wall' in ztmp.files:
                rw_load = np.asarray(ztmp['rho_wall'], dtype=float)
                if rw_load.shape == rho_wall.shape:
                    rho_wall = np.clip(rw_load, 0.0, P['rho_max'])
                    loaded_state = True
        except Exception as exc:
            print(f"  v27 rho-state restart fields not loaded: {exc}")
    if not loaded_state:
        fm = float(np.clip(P.get('rho_state_mobile_fraction', 0.65), 0.0, 1.0))
        fw = float(np.clip(P.get('rho_state_wall_fraction', 0.0), 0.0, 1.0-fm))
        ff = float(np.clip(P.get('rho_state_forest_fraction', 1.0-fm-fw), 0.0, 1.0-fm-fw))
        # Renormalize if the supplied fractions do not exactly sum to unity.
        totf = max(fm + ff + fw, 1e-30)
        fm, ff, fw = fm/totf, ff/totf, fw/totf
        scale_m = fm
        for ss in range(nSlip):
            rp[:, :, ss] *= scale_m
            rm[:, :, ss] *= scale_m
            rho_forest[:, :, ss] = ff * rho_current / max(nSlip, 1)
        rho_wall = fw * rho_current * np.clip(gb_mask + 0.05, 0.0, 1.0) if fw > 0 else np.zeros_like(rho_current)
    rho = _rho_total_state(rp, rm, rho_forest, rho_wall)
    rho_struct_init = _rho_structural_field(rho_forest, rho_wall)
    rho_total_init = rho.copy()
    ref_mode = str(P.get('rho_state_ref_mode', 'initial_structural')).lower()
    if P.get('rho_state_ref_abs', None) is not None:
        rho_ref_state = float(P.get('rho_state_ref_abs'))
    elif ref_mode in ['km', 'km_equilibrium', 'equilibrium']:
        k2_init_ref = _km_k2_from_T(np.array([P['T0']]))[0]
        rho_ref_state = (float(P.get('KM_k1', 5.0e8))/max(k2_init_ref, 1e-30))**2
    elif ref_mode in ['initial_structural', 'structural', 'forest_wall']:
        rho_ref_state = float(np.nanmean(np.maximum(rho_struct_init, P['rho_min'])))
    else:
        rho_ref_state = float(np.nanmean(rho_total_init))
    rho_ref_state = float(np.clip(rho_ref_state, P['rho_min'], P.get('rho_max', rho_ref_state)))
    P['_rho_state_ref_runtime'] = rho_ref_state
    P['_rho_state_total_ref_runtime'] = float(np.clip(np.nanmean(rho_total_init), P['rho_min'], P.get('rho_max', np.nanmean(rho_total_init))))
    P['_rho_state_struct_ref_runtime'] = float(np.clip(np.nanmean(np.maximum(rho_struct_init, P['rho_min'])), P['rho_min'], P.get('rho_max', np.nanmean(np.maximum(rho_struct_init, P['rho_min'])))))
    # Rebuild the potential so rho_ch/order terms use the structural density scale.
    ATpot.build(P['T0'], P['edot_app'])
    rho_c = ATpot.rho_c
    print(f"  v27 rho state: <rho_m>={np.nanmean(_rho_mobile_field(rp,rm)):.2e}, "
          f"<rho_f>={np.nanmean(_rho_forest_total_field(rho_forest)):.2e}, "
          f"<rho_w>={np.nanmean(_rho_wall_field(rho_wall)):.2e}, "
          f"rho_state_ref={rho_ref_state:.2e}, rho_peak_ind={rho_c:.2e}")
else:
    rho = np.maximum(np.sum(rp+rm, axis=2), P['rho_min'])
    P['_rho_state_ref_runtime'] = float(rho_c)

# KM equilibrium diagnostic
kB_eV = 8.617333262145e-5
k2T = float(_km_k2_from_T(P['T0']))
rho_eq = (P['KM_k1']/k2T)**2
print(f"  KM: k2(T0)={k2T:.2f}, rho_eq={rho_eq:.2e} (ratio={rho_eq/rho_c:.2f}); local_T_recovery={P.get('KM_recovery_local_T', True)}")
print(f"  sigma at rho0: {ATpot.sigma_inv(rho0, P['T0'], P['edot_app'])/1e6:.0f} MPa")

hist = {k:[] for k in ['t','rho_mean','rho_max','rho_std','sigma','T_mean',
    'T_max','eps','n_grains','psi_max','rho_GB_max','F_total','F_grad','F_comp',
    'gb_area_frac','highrho_on_gb_frac','corr_r_gb','corr_r_gradpsi','corr_r_kappa',
    'km_net_mean','ch_delta_abs_mean','ac_eta_delta_mean','rhoGB_delta_mean',
    'gb_hp_src_mean','gb_hp_sink_mean','gb_hp_xi_mean','gb_hp_xi_source_mean','gb_hp_xi_trans_mean','gb_hp_xi_sink_mean',
    'grain_initial_lineage','grain_spinodal_lineage','grain_hazard_lineage',
    'grain_topology_births','grain_hazard_births','heat_dT_mech_step','heat_dT_local_max_step','k2_eff_mean','k2_eff_max']}



# ================================================================
# 10a. v6 COUPLING / ABLATION HELPERS
# ================================================================
def _smoothstep01(x):
    """C1 smooth step and derivative on [0,1]."""
    xx = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    h = xx*xx*(3.0 - 2.0*xx)
    dh = 6.0*xx*(1.0 - xx)
    return h, dh


def _rho_eta_fields(r, rho, kappa_tot, gb_mask):
    """Return v6 density/KWC coupling fields.

    H_r is the smooth high-rho thermodynamic factor.  precursor is an agnostic
    structural field assembled from |kappa|/rho, |grad r|, and existing GB support.
    It is deliberately diagnostic/variational: it says where a boundary *would*
    reduce the density/orientation free energy, but it does not directly create
    a grain label.
    """
    rlo = float(P.get('rho_eta_r_lo', 0.90))
    rhi = float(P.get('rho_eta_r_hi', 1.45))
    denom = max(rhi - rlo, 1e-9)
    if P.get('rho_eta_use_relative_r', False):
        # Use a local-relative density coordinate so the rho-eta coupling remains
        # active after transient global density shifts.  Median is treated as a
        # slowly varying reference for the local derivative.
        r_ref = max(float(np.nanmedian(r)), P['rho_min']/max(rho_c, P['rho_min']), 1e-12)
        rH = r / r_ref
        Hr, dHdx = _smoothstep01((rH - rlo)/denom)
        dHdr = dHdx / (denom * r_ref)
    else:
        Hr, dHdx = _smoothstep01((r - rlo)/denom)
        dHdr = dHdx / denom

    kfrac = np.abs(kappa_tot) / np.maximum(rho, P['rho_min'])
    kscaled = np.clip(kfrac / max(float(P.get('nuc_min_kappa_frac', 0.04)), 1e-12), 0.0, 1.0)

    if P.get('rho_eta_use_grad_r', True):
        gr = grad_mag(r)
        q = float(np.nanquantile(gr, 0.95)) if np.any(np.isfinite(gr)) else 0.0
        gscaled = np.clip(gr / max(q, 1e-30), 0.0, 1.0)
    else:
        gscaled = np.zeros_like(r)

    precursor = (float(P.get('rho_eta_kappa_weight', 1.0))*kscaled +
                 float(P.get('rho_eta_grad_weight', 0.35))*gscaled +
                 float(P.get('rho_eta_gb_weight', 0.50))*np.clip(gb_mask, 0.0, 1.0))
    precursor = np.clip(precursor, float(P.get('rho_eta_precursor_floor', 0.02)), 1.0)
    drive = Hr * precursor
    return Hr, dHdr, precursor, drive


def _ch_step_variable_mobility(r, mu_ch, gb_mask, precursor, activity=None):
    """Semi-implicit CH step with optional GB mobility barrier.

    The constant-mobility spectral update is retained when the barrier is off.
    With the barrier on, only the bulk/stabilized flux uses the spatially varying
    M(x); the stabilizing fourth-order term uses the mean M for robustness.
    """
    r_mean_before = float(np.nanmean(r))
    Cs = float(P['CH_Cs'])
    base_scale = float(P.get('ch_base_scale', 1.0))
    use_plastic_M = bool(P.get('use_ch_plasticity_mobility', False))
    use_gb_M = bool(P.get('use_ch_mobility_barrier', True))

    if (not use_gb_M) and (not use_plastic_M) and abs(base_scale - 1.0) < 1e-12:
        r_hat = np.fft.fft2(r)
        mu_hat = np.fft.fft2(mu_ch)
        Mdt = P['dt']*P['M_ch']
        num = (1 + Mdt*Cs*K2)*r_hat - Mdt*K2*mu_hat
        den = 1 + Mdt*(Cs*K2 + P['kappa_r']*K4)
        den[0,0] = 1.0
        out_r = np.real(np.fft.ifft2(num/den))
    else:
        Mfield = P['M_ch'] * base_scale * np.ones_like(r)
        if use_gb_M:
            barrier = float(np.clip(P.get('ch_mobility_gb_barrier', 0.90), 0.0, 0.999))
            floor = float(np.clip(P.get('ch_mobility_floor_frac', 0.03), 1e-6, 1.0))
            block = np.clip(gb_mask, 0.0, 1.0)
            if P.get('rho_eta_mobility_on_precursor', False):
                block = np.maximum(block, np.clip(precursor, 0.0, 1.0))
            Mfield *= np.maximum(floor, 1.0 - barrier*block)
        if use_plastic_M and activity is not None:
            act = np.asarray(activity, dtype=float)
            act_ref = max(float(np.nanmean(np.abs(act))), float(P.get('edot_app', 1.0))*1e-12, 1e-30)
            rel = np.maximum(np.abs(act) / act_ref, 0.0)
            pwr = float(P.get('ch_plasticity_power', 0.5))
            pfloor = float(np.clip(P.get('ch_plasticity_floor_frac', 0.10), 1e-6, 1.0))
            pcap = float(max(P.get('ch_plasticity_cap_frac', 2.0), pfloor))
            Mfield *= np.clip(rel**pwr, pfloor, pcap)
        Mbar = max(float(np.nanmean(Mfield)), 1e-30)
        mu_exp = mu_ch - Cs*r
        div_flux = ddx(Mfield*ddx(mu_exp)) + ddy(Mfield*ddy(mu_exp))
        r_hat = np.fft.fft2(r)
        rhs = r_hat + P['dt']*np.fft.fft2(div_flux)
        den = 1.0 + P['dt']*Mbar*(Cs*K2 + P['kappa_r']*K4)
        den[0,0] = 1.0
        out_r = np.real(np.fft.ifft2(rhs/den))
    out_r += r_mean_before - float(np.nanmean(out_r))
    if P.get('use_ch_increment_limiter', True):
        lim = float(P.get('ch_max_frac_step', 0.05))
        if lim > 0:
            dr = out_r - r
            mx = float(np.nanmax(np.abs(dr))) if np.size(dr) else 0.0
            if mx > lim:
                out_r = r + dr*(lim/mx)
                out_r += r_mean_before - float(np.nanmean(out_r))
    return out_r


# ================================================================
# 10b. v8 ARRHENIUS HALL-PETCH GB SOURCE/SINK HELPERS
# ================================================================
def _grain_size_field_from_labels(lab, Ng):
    """Equivalent grain diameter field [m] from current hard labels.

    The source/sink stress concentration uses the current grain scale so that
    shrinking/growing grains immediately alter the Hall-Petch channel.  The
    equivalent diameter is 2*sqrt(A/pi); this is only a local geometric scale,
    not a new grain-growth law.
    """
    lab_i = np.asarray(lab, dtype=int)
    counts = np.bincount(np.clip(lab_i.ravel(), 0, max(Ng-1, 0)), minlength=max(Ng, 1)).astype(float)
    areas = np.maximum(counts*dx*dy, dx*dy)
    d_eq = 2.0*np.sqrt(areas/np.pi)
    if Ng <= 0:
        return np.full_like(lab_i, dx, dtype=float)
    return d_eq[np.clip(lab_i, 0, Ng-1)]


def _gb_hp_xi_fields(rho, gb_mask, lab, Ng):
    """Split GB stress-concentration factors for source, transmission, and storage.

    v20 used one xi_HP ~ sqrt(d_g/X_rho) for every GB process.  That pileup form
    is retained only for pileup-assisted slip transmission.  GB source nucleation
    instead uses a GB-step/triple-junction concentration that does not increase
    with total rho; optionally it is screened at high rho to mimic shielding/back
    stress.  The weak residual-storage/sink branch uses little or no xi.
    """
    active = gb_mask > P.get('gb_hp_min_gb_support', 0.25)
    d_g = _grain_size_field_from_labels(lab, Ng)
    rho_safe = np.maximum(rho, P['rho_min'])
    X_pileup = 1.0/np.sqrt(np.maximum(2.0*rho_safe, 1e-300))

    # Source: no positive rho^(1/4) amplification.  It represents local GB-step
    # or triple-junction concentration, not a pileup length ratio.
    xi_src = np.full_like(rho_safe, float(P.get('gb_hp_source_xi_prefactor', 8.0)), dtype=float)
    if P.get('gb_hp_source_use_backstress_screen', True):
        rho_screen = max(float(P.get('gb_hp_source_rho_screen_frac', 1.0))*max(rho_c, P['rho_min']), P['rho_min'])
        screen = 1.0/np.sqrt(1.0 + rho_safe/rho_screen)
        xi_src = xi_src*screen
    else:
        screen = np.ones_like(rho_safe)
    xi_src = np.clip(xi_src, float(P.get('gb_hp_source_xi_floor', P.get('gb_hp_xi_floor', 1.0))),
                     float(P.get('gb_hp_source_xi_cap', P.get('gb_hp_xi_cap', 20.0))))

    # Transmission: pileup-assisted GB crossing/transmission.  This is the only
    # branch where the sqrt(d_g/X) Hall-Petch concentration is used by default.
    xi_tr = float(P.get('gb_hp_trans_xi_prefactor', 1.0))*np.sqrt(np.maximum(d_g, dx)/np.maximum(X_pileup, P['b']))
    xi_tr = np.clip(xi_tr, float(P.get('gb_hp_trans_xi_floor', P.get('gb_hp_xi_floor', 1.0))),
                    float(P.get('gb_hp_trans_xi_cap', P.get('gb_hp_xi_cap', 80.0))))

    # Weak residual storage: do not erase the barrier by pileup amplification.
    xi_sink = np.full_like(rho_safe, float(P.get('gb_hp_sink_xi_prefactor', 1.0)), dtype=float)
    xi_sink = np.clip(xi_sink, float(P.get('gb_hp_sink_xi_floor', 1.0)),
                      float(P.get('gb_hp_sink_xi_cap', 5.0)))

    floor = float(P.get('gb_hp_xi_floor', 1.0))
    xi_src = np.where(active, xi_src, floor)
    xi_tr = np.where(active, xi_tr, floor)
    xi_sink = np.where(active, xi_sink, 1.0)
    return dict(source=xi_src, trans=xi_tr, sink=xi_sink, screen=screen, X_pileup=X_pileup)


def _gb_hp_rate_given_xi(seq, xi, T_field, gb_mask, extra_barrier_eV=None):
    """Arrhenius GB event rate [1/s] for a supplied xi field.

    v30: extra_barrier_eV adds a crystallographic/residual-Burgers
    transmission barrier without changing the HP stress concentration.
    """
    if not P.get('use_gb_hp_source_sink', True):
        return np.zeros_like(seq, dtype=float)
    seq_eff = np.maximum(seq, 0.0) * xi
    Tloc = np.maximum(np.asarray(T_field, dtype=float), 1.0)
    G = float(P.get('gb_hp_A_mult', 7.0)) * ATpot.barrier_G(seq_eff, Tloc)
    if extra_barrier_eV is not None:
        G = G + np.asarray(extra_barrier_eV, dtype=float)*eV_J
    rate = ATpot.eta0 * np.exp(np.clip(-G/(kB_J*Tloc), -700.0, 40.0))
    rate = np.minimum(rate, float(P.get('gb_hp_rate_cap', 5.0e6)))
    rate = np.where(gb_mask > P.get('gb_hp_min_gb_support', 0.25), rate, 0.0)
    return rate


def _gb_hp_rate(seq, rho, T_field, gb_mask, lab, Ng):
    """Backward-compatible transmission-rate helper."""
    xis = _gb_hp_xi_fields(rho, gb_mask, lab, Ng)
    rate = _gb_hp_rate_given_xi(seq, xis['trans'], T_field, gb_mask)
    return rate, xis['trans']





def _gb_slip_transmission_fields(lab, psi_field, gb_mask, T_field=None):
    """Crystallographic GB slip-transmission compatibility fields.

    For each cell/slip system, inspect neighboring grains across the diffuse GB.
    The best outgoing slip system in the neighboring grain is chosen by minimizing
    an Arrhenius barrier composed of (i) grain misorientation and (ii) normalized
    residual Burgers vector b_res/b left in the boundary.  In 2-D this uses the
    slip-line unit vectors; it is a reduced proxy for a full Luster-Morris m' and
    residual-Burgers analysis.
    """
    shp = (Nx, Ny, nSlip)
    if (not P.get('use_gb_slip_transmission_barrier', True)) or nSlip <= 0:
        one = np.ones(shp, dtype=float)
        zero = np.zeros(shp, dtype=float)
        return dict(factor=one, barrier_eV=zero, mis_deg=zero, mprime=one,
                    bres=zero, rot_drive=np.zeros((Nx, Ny), dtype=float))

    psi = np.asarray(psi_field, dtype=float)
    lab_i = np.asarray(lab, dtype=int)

    # v31: use hard grain-owned orientations for GB transmission barriers.
    # The diffuse KWC field interpolates across a boundary and therefore can
    # underestimate the actual misorientation seen by an incoming dislocation.
    # For transmission, use the grain labels and psi_gv, optionally adding the
    # local plastic residual orientation.
    if P.get('gb_trans_use_hard_grain_orientation', True):
        try:
            gv = globals().get('psi_gv', None)
            if gv is not None:
                psi_hard = np.asarray(gv, dtype=float)[np.clip(lab_i, 0, len(gv)-1)]
                if P.get('gb_trans_include_plastic_orientation', True):
                    pp = globals().get('psi_plastic', None)
                    if pp is not None:
                        psi_hard = psi_hard + np.asarray(pp, dtype=float)
                psi = angle_wrap(psi_hard)
        except Exception:
            pass

    base = np.deg2rad(np.asarray(P.get('slip_angles_deg', [0.0]), dtype=float))
    if base.size < nSlip:
        base = np.resize(base, nSlip)
    gb_core = np.clip(np.asarray(gb_mask, dtype=float), 0.0, 1.0)
    active_core = gb_core > float(P.get('gb_hp_min_gb_support', 0.25))

    # Initialize as transparent away from GBs.
    best_bar = np.zeros(shp, dtype=float)
    best_mis = np.zeros(shp, dtype=float)
    best_mp = np.ones(shp, dtype=float)
    best_bres = np.zeros(shp, dtype=float)
    have = np.zeros(shp, dtype=bool)
    rot_sum = np.zeros((Nx, Ny), dtype=float)
    rot_w = np.zeros((Nx, Ny), dtype=float)

    Gmis = float(P.get('gb_trans_misorientation_barrier_eV', 0.25))
    Gres = float(P.get('gb_trans_residual_barrier_eV', 0.55))
    pwr = max(float(P.get('gb_trans_barrier_power', 2.0)), 1e-12)
    mis_ref = max(np.deg2rad(float(P.get('gb_trans_mis_ref_deg', 30.0))), 1e-12)
    bres_ref = max(float(P.get('gb_trans_bres_ref', 1.0)), 1e-12)
    use_worst = bool(P.get('gb_trans_use_neighbor_worst', True))

    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
        lab_n = np.roll(lab_i, (di, dj), (0, 1))
        psi_n = np.roll(psi, (di, dj), (0, 1))
        is_gb_face = (lab_n != lab_i) & active_core
        if not np.any(is_gb_face):
            continue
        dpsi = np.abs(angle_wrap(psi - psi_n))
        mis_term = (np.sin(0.5*np.minimum(dpsi, np.pi)) / max(np.sin(0.5*mis_ref), 1e-12))**pwr
        # Rotation drive: residual GB content tends to reduce misorientation.
        rot_sum += np.where(is_gb_face, angle_wrap(psi_n - psi), 0.0)
        rot_w += is_gb_face.astype(float)
        for ss in range(nSlip):
            th_in = psi + base[ss]
            # Choose the best outgoing slip system/sign in the neighbor.
            local_best_bar = None
            local_best_mp = None
            local_best_bres = None
            outgoing_mode = str(P.get('gb_trans_outgoing_mode', 'same_index')).lower()
            if outgoing_mode in ('same', 'same_index', 'same_slip'):
                out_list = [ss]
            else:
                out_list = range(nSlip)
            for tt in out_list:
                th_out = psi_n + base[tt % nSlip]
                c = np.abs(np.cos(angle_wrap(th_in - th_out)))
                mp = np.clip(c*c, 0.0, 1.0)  # 2-D proxy for |b.b'||n.n'|
                bres = np.sqrt(np.maximum(0.0, 2.0*(1.0 - c)))  # |b - sign b'| / b
                bar = Gmis*mis_term + Gres*(bres/bres_ref)**pwr
                if local_best_bar is None:
                    local_best_bar = bar; local_best_mp = mp; local_best_bres = bres
                else:
                    better = bar < local_best_bar
                    local_best_bar = np.where(better, bar, local_best_bar)
                    local_best_mp = np.where(better, mp, local_best_mp)
                    local_best_bres = np.where(better, bres, local_best_bres)
            if use_worst:
                # If a cell borders several grains, the hardest adjacent crossing
                # should not be silently bypassed by one easy neighbor.
                replace = is_gb_face & ((~have[:, :, ss]) | (local_best_bar > best_bar[:, :, ss]))
            else:
                replace = is_gb_face & ((~have[:, :, ss]) | (local_best_bar < best_bar[:, :, ss]))
            best_bar[:, :, ss] = np.where(replace, local_best_bar, best_bar[:, :, ss])
            best_mp[:, :, ss] = np.where(replace, local_best_mp, best_mp[:, :, ss])
            best_bres[:, :, ss] = np.where(replace, local_best_bres, best_bres[:, :, ss])
            best_mis[:, :, ss] = np.where(replace, np.rad2deg(dpsi), best_mis[:, :, ss])
            have[:, :, ss] |= (is_gb_face & replace)

    # Convert barrier to a multiplicative Arrhenius transmission factor at GBs.
    Tloc = np.maximum(np.asarray(T_field if T_field is not None else P.get('T0', 1100.0), dtype=float), 1.0)
    if Tloc.shape != (Nx, Ny):
        Tloc = np.full((Nx, Ny), float(np.nanmean(Tloc)))
    kBT_eV = 8.617333262145e-5*Tloc
    fac = np.exp(np.clip(-best_bar/np.maximum(kBT_eV[:, :, None], 1e-12), -700.0, 0.0))
    fac = np.clip(fac, float(P.get('gb_trans_min_factor', 1.0e-4)), 1.0)
    gbw = np.clip(gb_core, 0.0, 1.0)[:, :, None]
    coupling = np.clip(float(P.get('gb_trans_gdot_coupling', 1.0)), 0.0, 1.0)
    fac_eff = 1.0 - coupling*gbw*(1.0 - fac)
    fac_eff = np.where(have, fac_eff, 1.0)

    rot_drive = np.where(rot_w > 0.0, rot_sum/np.maximum(rot_w, 1e-30), 0.0)
    return dict(factor=fac_eff, barrier_eV=best_bar, mis_deg=best_mis,
                mprime=best_mp, bres=best_bres, rot_drive=rot_drive)


def _smooth_gb_field(field, gb_weight, length_um=None, passes=1):
    """Smooth a GB-local field to represent cooperative boundary rotation.

    Local rotation of one boundary segment is mechanically constrained by
    neighboring segments.  This helper performs weighted smoothing only over
    the diffuse GB support so the rotation drive is coherent along connected
    GB structure rather than a cell-local independent torque.
    """
    f = np.asarray(field, dtype=float)
    w = np.clip(np.asarray(gb_weight, dtype=float), 0.0, 1.0)
    if length_um is None or float(length_um) <= 0.0:
        return f*w
    sig_px = max((float(length_um)*1e-6)/max(dx, 1e-30), 0.0)
    out = f*w
    den = w.copy()
    for _ in range(max(int(passes), 1)):
        num = ndimage.gaussian_filter(out, sig_px, mode='wrap')
        den_s = ndimage.gaussian_filter(den, sig_px, mode='wrap')
        out = np.where(den_s > 1e-12, num/np.maximum(den_s, 1e-30), 0.0)*w
        den = w
    return out


def _net_signed_gb_fraction(rp_arr, rm_arr, rho_GB_arr, gb_mask_arr):
    """Estimate the non-cancelling signed Burgers fraction at a GB.

    Scalar rho_GB stores total residual content.  Opposite signs can cancel, so
    only the net signed content should drive lattice rotation.  This proxy uses
    the local signed mobile/GND imbalance relative to total GB residual content.
    """
    kappa = np.sum(np.asarray(rp_arr, dtype=float) - np.asarray(rm_arr, dtype=float), axis=2)
    denom = np.maximum(np.abs(kappa) + np.maximum(rho_GB_arr, 0.0), P.get('rho_min', 1e8))
    frac = np.clip(np.abs(kappa)/denom, 0.0, 1.0)
    return frac*np.clip(gb_mask_arr, 0.0, 1.0)


def _conservative_process_zone_filter(field, sigma_um=None, preserve_mean=True):
    """Spread heat/plastic work over a finite process-zone width.

    The finite-loading budget constrains the domain-average mechanical work.
    This filter is a conservative spatial regularization of the partition
    function, representing the fact that a slip-transfer/avalanche/accommodation
    event has finite width and cannot deposit all work into a single grid cell.
    """
    f = np.asarray(field, dtype=float)
    if sigma_um is None:
        sigma_um = float(P.get('heat_process_zone_sigma_um', 0.0))
    sig_px = (float(sigma_um)*1.0e-6)/max(dx, 1e-30)
    sig_px = max(sig_px, float(P.get('heat_process_zone_min_sigma_px', 0.0)))
    if (not np.isfinite(sig_px)) or sig_px <= 0.0:
        return f, dict(active=0.0, sigma_px=0.0, raw_max=float(np.nanmax(f)), smooth_max=float(np.nanmax(f)))
    raw_mean = float(np.nanmean(f))
    raw_max = float(np.nanmax(f))
    sm = ndimage.gaussian_filter(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), sig_px, mode='wrap')
    if preserve_mean:
        sm_mean = float(np.nanmean(sm))
        if np.isfinite(raw_mean) and np.isfinite(sm_mean) and sm_mean > 1e-300:
            sm *= raw_mean/sm_mean
    return sm, dict(active=1.0, sigma_px=float(sig_px), raw_max=raw_max,
                    smooth_max=float(np.nanmax(sm)), raw_mean=raw_mean,
                    smooth_mean=float(np.nanmean(sm)))


def _gb_blocked_heat_partition(plastic_power, gb_trans_fields, gb_mask_arr):
    """Partition unresolved blocked GB work away from immediate heat.

    Poor slip transmission implies part of the local plastic work represents
    residual Burgers storage/backstress or GB sliding/accommodation rather than
    Taylor-Quinney heat in the same cell.  This returns a heat-partitioned
    power field and diagnostic fields.  The operation is conservative only in
    the later finite-loading normalization sense; here it changes the spatial
    partition function and records the non-heat fraction.
    """
    pp = np.asarray(plastic_power, dtype=float)
    if (not P.get('use_gb_blocked_work_partition', True)) or gb_trans_fields is None:
        z = np.zeros_like(pp)
        return pp, dict(active=0.0, blocked_frac_mean=0.0, blocked_power_mean=0.0,
                        blocked_power_max=0.0, stored_power_mean=0.0, heat_scale_min=1.0), z
    try:
        fac = np.asarray(gb_trans_fields.get('factor'), dtype=float)
        if fac.ndim == 3:
            # strongest blocked slip system controls the unresolved incompatibility.
            fscalar = np.nanmin(fac, axis=2)
        else:
            fscalar = fac
    except Exception:
        fscalar = np.ones_like(pp)
    gbw = np.clip(np.asarray(gb_mask_arr, dtype=float), 0.0, 1.0)**max(float(P.get('gb_blocked_work_support_power', 2.0)), 1e-12)
    pwr = max(float(P.get('gb_blocked_work_factor_power', 1.0)), 1e-12)
    blocked_frac = gbw*np.clip(1.0 - fscalar, 0.0, 1.0)**pwr
    chi_h = float(np.clip(P.get('gb_blocked_work_heat_fraction', 0.25), 0.0, 1.0))
    chi_s = float(np.clip(P.get('gb_blocked_work_store_fraction', 0.50), 0.0, 1.0))
    heat_scale = 1.0 - blocked_frac*(1.0 - chi_h)
    pp_heat = pp*heat_scale
    blocked_power = pp*np.clip(blocked_frac*(1.0 - chi_h), 0.0, 1.0)
    stored_power = blocked_power*chi_s
    diag = dict(active=1.0,
                blocked_frac_mean=float(np.nanmean(blocked_frac)),
                blocked_frac_max=float(np.nanmax(blocked_frac)),
                blocked_power_mean=float(np.nanmean(blocked_power)),
                blocked_power_max=float(np.nanmax(blocked_power)),
                stored_power_mean=float(np.nanmean(stored_power)),
                stored_power_max=float(np.nanmax(stored_power)),
                heat_scale_min=float(np.nanmin(heat_scale)),
                heat_scale_mean=float(np.nanmean(heat_scale)))
    return pp_heat, diag, stored_power


def apply_gb_comoving_gnd_projection(rp, rm, rho, gb_mask_old, gb_mask_new, H_nuc=None, E_nuc=None):
    """Remove stale GB-bound signed GND when a GB leaves a cell.

    This is not a density sink.  It preserves total mobile density on each slip
    system by transferring majority sign into minority sign.  The operation is
    weighted by the decrease in diffuse GB support between the previous and
    current topology.  Therefore it acts on GND that was associated with a
    moving GB, but it does not broadly erase interior lattice GND generated by
    plastic incompatibility.

    Physical interpretation: GB dislocation content is attached to the moving
    boundary.  If the boundary sweeps away, the old cell should not retain the
    full signed Frank-Bilby content as a free lattice nucleation seed.  Any
    residual lattice GND can still be generated later by transport and the
    compatibility feedback.
    """
    if not P.get('use_gb_comoving_gnd', True):
        return rp, rm, rho, H_nuc, E_nuc, dict(depart_mean=0.0, relax_mean=0.0)

    pwr = max(float(P.get('gb_comoving_support_power', 4.0)), 1.0)
    old = np.clip(gb_mask_old, 0.0, 1.0)**pwr
    new = np.clip(gb_mask_new, 0.0, 1.0)**pwr
    depart = np.clip(old - new, 0.0, 1.0)
    if not np.any(depart > 1e-12):
        return rp, rm, rho, H_nuc, E_nuc, dict(depart_mean=0.0, relax_mean=0.0)

    tau = max(float(P.get('gb_comoving_relax_tau', 2.5e-7)), P['dt'])
    relax = np.clip((P['dt']/tau)*depart, 0.0, float(P.get('gb_comoving_max_frac_step', 0.35)))
    relax_abs_acc = 0.0

    for ss in range(nSlip):
        rps = rp[:, :, ss]
        rms = rm[:, :, ss]
        kss = rps - rms
        # dks is the desired change in (rho+ - rho-), directed toward zero.
        dks = -relax * kss
        pos = dks > 0
        amt = np.zeros_like(dks)
        # Positive dks: transfer rm -> rp.
        amt[pos] = np.minimum(0.5*dks[pos], np.maximum(rms[pos]-P['rho_min'], 0.0))
        rps += amt; rms -= amt
        neg = dks < 0
        amt2 = np.zeros_like(dks)
        # Negative dks: transfer rp -> rm.
        amt2[neg] = np.minimum(-0.5*dks[neg], np.maximum(rps[neg]-P['rho_min'], 0.0))
        rps -= amt2; rms += amt2
        rp[:, :, ss] = rps
        rm[:, :, ss] = rms
        relax_abs_acc += float(np.nanmean(np.abs(dks))) / max(nSlip, 1)

    rho = np.maximum(np.sum(rp + rm, axis=2), P['rho_min'])

    # The local metastable object has changed because the boundary left.  Do not
    # let old GB-bound exposure carry over as a nucleation clock in the lattice.
    if H_nuc is not None:
        reset_strength = float(P.get('gb_comoving_hazard_reset_strength', 4.0))
        H_nuc = H_nuc * np.exp(-reset_strength*depart)
        if E_nuc is not None:
            # Where the departure is strong, redraw part of the exponential
            # threshold.  This is a smooth, stochastic reset rather than a hard
            # gate; it prevents stale GB exposure from causing delayed nuclei.
            p_redraw = np.clip(1.0 - np.exp(-reset_strength*depart), 0.0, 1.0)
            rr = _rng_nuc.random(E_nuc.shape)
            newE = -np.log(np.maximum(_rng_nuc.random(E_nuc.shape), 1e-300))
            E_nuc = np.where(rr < p_redraw, newE, E_nuc)

    return rp, rm, rho, H_nuc, E_nuc, dict(depart_mean=float(np.nanmean(depart)), relax_mean=relax_abs_acc)

def apply_gb_hp_source_sink(rp, rm, rho, rho_GB, gb_mask, lab, Ng, sig_use, Sch, T_field, psi_lat=None):
    """Apply v9 Arrhenius Hall-Petch GB source + conservative transmission.

    v8 treated the GB channel as a strong source/sink.  That can over-delete
    mobile density and inject nearly single-signed GND content.  v9 keeps the
    useful part (GBs are high-barrier Arrhenius sources with HP stress
    concentration), but replaces the strong sink by a Birnbaum/Robertson-like
    slip-transmission closure:

      * source: mostly neutral mobile pair creation at real GB cores;
      * transmission: conservative redistribution of rho+/rho- among active
        slip systems, preserving total mobile Burgers content in the cell;
      * residual: a small fraction of transmitted content is stored in rho_GB,
        representing residual Burgers vector left in the boundary;
      * weak sink: only a small, capacity-limited mobile -> rho_GB transfer.

    Signed GND should primarily be created by the existing variational
    compatibility feedback dF/dkappa, not by imposing a strongly signed source.
    """
    if not P.get('use_gb_hp_source_sink', True):
        return rp, rm, rho, rho_GB, dict(src_mean=0.0, sink_mean=0.0, rate_mean=0.0, xi_mean=np.nan, xi_source_mean=np.nan, xi_trans_mean=np.nan, xi_sink_mean=np.nan, xi_screen_mean=np.nan)

    tau_abs = np.zeros((Nx, Ny, nSlip))
    tau_signed = np.zeros_like(tau_abs)
    for ss in range(nSlip):
        tau = np.zeros((Nx, Ny))
        for ii in range(2):
            for jj in range(2):
                tau += sig_use[:, :, ii, jj]*Sch[:, :, ss, ii, jj]
        tau_signed[:, :, ss] = tau
        tau_abs[:, :, ss] = np.abs(tau)*drive_sc

    seq_max = np.max(tau_abs, axis=2)
    xi_fields = _gb_hp_xi_fields(rho, gb_mask, lab, Ng)
    gb_trans = _gb_slip_transmission_fields(lab, psi_lat if psi_lat is not None else np.zeros_like(rho), gb_mask, T_field)
    # Use the easiest active slip-system barrier for the scalar GB transmission hazard.
    gb_bar_min = np.nanmin(gb_trans['barrier_eV'], axis=2)
    rate_src = _gb_hp_rate_given_xi(seq_max, xi_fields['source'], T_field, gb_mask)
    rate_tr = _gb_hp_rate_given_xi(seq_max, xi_fields['trans'], T_field, gb_mask, extra_barrier_eV=gb_bar_min)
    rate_sink = _gb_hp_rate_given_xi(seq_max, xi_fields['sink'], T_field, gb_mask)
    p_evt_src = 1.0 - np.exp(-rate_src*P['dt'])
    p_evt_tr = 1.0 - np.exp(-rate_tr*P['dt'])
    p_evt_sink = 1.0 - np.exp(-rate_sink*P['dt'])

    # Localize HP source/transmission to the true GB core, not the diffuse tails.
    support_power = float(P.get('gb_hp_support_power', 4.0))
    hp_support = np.clip(gb_mask, 0.0, 1.0)**support_power
    hp_support = np.where(gb_mask > P.get('gb_hp_min_gb_support', 0.25), hp_support, 0.0)

    # Stress weights for choosing transmitted/activated slip systems.
    w = tau_abs + 1e-300
    wsum = np.sum(w, axis=2) + 1e-300
    wnorm = w / wsum[:, :, None]

    # ---- Source: GBs replenish mobile carriers, but do not impose GND sign. ----
    # v27c: this source is a GB carrier reservoir, not the independent Taylor
    # kinetic-peak density.  Scaling by rho_c at high rate artificially injects
    # orders of magnitude too much density and destabilizes the microstructure.
    src_scale_mode = str(P.get('gb_hp_source_density_scale', 'initial_total')).lower()
    if src_scale_mode in ['kinetic', 'rho_c', 'legacy']:
        rho_src_scale = max(rho_c, P['rho_min'])
    else:
        rho_src_scale = max(_rho_network_ref_scale(), P['rho_min'])
    rho_src_target = float(P.get('gb_hp_source_sat_frac', 0.95))*rho_src_scale
    source_room = np.clip((rho_src_target - rho)/max(rho_src_target - P['rho_min'], 1.0), 0.0, 1.0)
    drho_src = float(P.get('gb_hp_source_strength', 0.08))*rho_src_scale*p_evt_src*hp_support*source_room

    signed_bias = float(np.clip(P.get('gb_hp_source_signed_bias', 0.0), 0.0, 0.20))
    for ss in range(nSlip):
        add = drho_src*wnorm[:, :, ss]
        # Default signed_bias=0 gives exactly neutral pair creation.  A small bias
        # can be used later if a specific GB source mechanism requires it.
        pos = tau_signed[:, :, ss] >= 0.0
        frac_p = np.where(pos, 0.5 + signed_bias, 0.5 - signed_bias)
        frac_m = 1.0 - frac_p
        rp[:, :, ss] += frac_p*add
        rm[:, :, ss] += frac_m*add

    rho = np.maximum(np.sum(rp+rm, axis=2), P['rho_min'])

    # ---- Transmission: preserve mobile Burgers content; redistribute slip systems. ----
    # This is the simplest continuum surrogate for slip transmission across a GB:
    # mobile density is not deleted, but is reprojected toward the locally favored
    # outgoing slip-system distribution.  Sign-resolved totals are preserved.
    trans = float(P.get('gb_hp_transmission_strength', 0.60))*p_evt_tr*hp_support
    trans = np.clip(trans, 0.0, 1.0)

    rp_tot = np.sum(rp, axis=2)
    rm_tot = np.sum(rm, axis=2)
    transfer_measure = np.zeros((Nx, Ny))
    for ss in range(nSlip):
        target_p = rp_tot*wnorm[:, :, ss]
        target_m = rm_tot*wnorm[:, :, ss]
        dp = trans*(target_p - rp[:, :, ss])
        dm = trans*(target_m - rm[:, :, ss])
        transfer_measure += np.abs(dp) + np.abs(dm)
        rp[:, :, ss] += dp
        rm[:, :, ss] += dm

    # Guard tiny numerical undershoots while preserving sign-resolved totals as much
    # as possible.  This is a numerical bound, not a physical sink.
    rp = np.maximum(rp, P['rho_min'])
    rm = np.maximum(rm, P['rho_min'])
    rho = np.maximum(np.sum(rp+rm, axis=2), P['rho_min'])

    # Residual Burgers vector content: transmitted dislocations generally do not
    # match perfectly across a GB.  Store a small scalar residual in rho_GB; the
    # existing F_comp and signed-GND feedback convert that incompatibility into
    # rotation/GND structure rather than deleting mobile density.
    # v30: incompatible transmission leaves more residual Burgers content in the GB.
    bres_mean = np.nanmean(gb_trans.get('bres', np.zeros((Nx, Ny, nSlip))), axis=2)
    residual_multiplier = 1.0 + float(P.get('gb_trans_store_residual_scale', 1.0))*np.clip(bres_mean, 0.0, 2.0)
    residual = float(P.get('gb_hp_transmission_residual_fraction', 0.06))*transfer_measure*hp_support*residual_multiplier
    rho_GB = np.clip(rho_GB + residual, 0.0, P['rho_max'])

    # ---- Very weak capacity-limited sink: mobile content can be incorporated into
    # GB residual structure only when the Frank-Bilby target has capacity. ----
    if psi_lat is not None:
        rho_GB_target = np.clip(P.get('frank_bilby_coeff', 1.0)*gb_mask*grad_mag(psi_lat)/P['b'], 0.0, P['rho_max'])
        cap = np.clip((rho_GB_target - rho_GB)/np.maximum(rho_GB_target, P['rho_min']), 0.0, 1.0)
    else:
        cap = 0.0
    sink_floor = max(float(P.get('gb_hp_sink_floor_rho', P.get('sweep_wake_rho', 1e14))), P['rho_min'])
    removable = np.maximum(rho - sink_floor, 0.0)
    drho_sink = float(P.get('gb_hp_weak_sink_strength', 0.015))*p_evt_sink*hp_support*cap*removable
    scale = np.where(rho > P['rho_min'], np.maximum(rho - drho_sink, P['rho_min'])/np.maximum(rho, P['rho_min']), 1.0)
    for ss in range(nSlip):
        rp[:, :, ss] *= scale
        rm[:, :, ss] *= scale
    rho = np.maximum(np.sum(rp+rm, axis=2), P['rho_min'])
    rho_GB = np.clip(rho_GB + float(P.get('gb_hp_sink_to_rhoGB_fraction', 1.0))*drho_sink, 0.0, P['rho_max'])

    active = hp_support > 0.0
    diag = dict(
        src_mean=float(np.nanmean(drho_src)),
        sink_mean=float(np.nanmean(drho_sink)),
        rate_mean=float(np.nanmean(rate_tr[active])) if np.any(active) else 0.0,
        source_rate_mean=float(np.nanmean(rate_src[active])) if np.any(active) else 0.0,
        trans_rate_mean=float(np.nanmean(rate_tr[active])) if np.any(active) else 0.0,
        sink_rate_mean=float(np.nanmean(rate_sink[active])) if np.any(active) else 0.0,
        xi_mean=float(np.nanmean(xi_fields['trans'][active])) if np.any(active) else np.nan,
        xi_source_mean=float(np.nanmean(xi_fields['source'][active])) if np.any(active) else np.nan,
        xi_trans_mean=float(np.nanmean(xi_fields['trans'][active])) if np.any(active) else np.nan,
        xi_sink_mean=float(np.nanmean(xi_fields['sink'][active])) if np.any(active) else np.nan,
        xi_screen_mean=float(np.nanmean(xi_fields['screen'][active])) if np.any(active) else np.nan,
        gb_trans_mis_deg_mean=float(np.nanmean(gb_trans['mis_deg'][active])) if np.any(active) else np.nan,
        gb_trans_mis_deg_max=float(np.nanmax(gb_trans['mis_deg'][active])) if np.any(active) else np.nan,
        gb_trans_mprime_mean=float(np.nanmean(gb_trans['mprime'][active])) if np.any(active) else np.nan,
        gb_trans_mprime_min=float(np.nanmin(gb_trans['mprime'][active])) if np.any(active) else np.nan,
        gb_trans_bres_mean=float(np.nanmean(gb_trans['bres'][active])) if np.any(active) else np.nan,
        gb_trans_bres_max=float(np.nanmax(gb_trans['bres'][active])) if np.any(active) else np.nan,
        gb_trans_barrier_eV_mean=float(np.nanmean(gb_trans['barrier_eV'][active])) if np.any(active) else np.nan,
        gb_trans_barrier_eV_max=float(np.nanmax(gb_trans['barrier_eV'][active])) if np.any(active) else np.nan,
        gb_trans_factor_mean=float(np.nanmean(gb_trans['factor'][active])) if np.any(active) else np.nan,
        gb_trans_factor_min=float(np.nanmin(gb_trans['factor'][active])) if np.any(active) else np.nan,
    )
    return rp, rm, rho, rho_GB, diag


# ================================================================
# 10c. v11 GND-BOUNDED CUMULATIVE-HAZARD NUCLEATION HELPERS
# ================================================================
def _nuc_low_rho_from_potential():
    """Low-density product state for a nucleus from the current potential.

    This replaces the old hard reset to sweep_wake_rho as the thermodynamic
    reference state.  If the tabulated potential has no resolved low-density
    minimum, fall back to the old wake density as a numerical lower bound.
    """
    try:
        rt = np.asarray(ATpot.rho_tab, dtype=float)
        ph = np.asarray(ATpot.Phi_tab, dtype=float)
        m = np.isfinite(rt) & np.isfinite(ph) & (rt >= P['rho_min']) & (rt <= max(_rho_ch_scale(), P['rho_min'])*1.05)
        if np.any(m):
            rr = rt[m]; pp = ph[m]
            return float(np.clip(rr[int(np.argmin(pp))], max(P['rho_min'], 1e-30), P['rho_max']))
    except Exception:
        pass
    return float(np.clip(P.get('sweep_wake_rho', 1e14), P['rho_min'], P['rho_max']))


def _rs_gamma(theta, T_field):
    """Read-Shockley/KWC boundary energy [J/m^2] for misorientation theta.

    gamma_HA(T) scales with the temperature-dependent shear modulus so this is
    the same physical temperature dependence used by the stitched potential.
    """
    th = np.abs(np.asarray(theta, dtype=float))
    Tloc = np.asarray(T_field, dtype=float)
    thm = max(np.deg2rad(float(P.get('nuc_rs_theta_m_deg', 15.0))), 1e-12)
    mu_ref = max(ATpot.mu_shear(float(P.get('T0', 1300.0))), 1.0)
    # gamma_GB is interpreted as the high-angle/saturated GB energy at T0.
    gamma_HA = float(P.get('nuc_gamma_GB', 0.5))*np.maximum(ATpot.mu_shear(np.nanmean(Tloc))/mu_ref, 0.05)
    x = np.clip(th/thm, 1e-12, 1.0)
    gamma = gamma_HA*x*(1.0 - np.log(x))
    gamma = np.where(th >= thm, gamma_HA, gamma)
    gamma = np.where(th <= 1e-15, 0.0, gamma)
    return gamma


def _hazard_site_factor(rho, kappa_tot, gb_mask, r_field):
    """Continuous site multiplicity for the local nucleation hazard.

    This is intentionally not a binary gate.  GB cores, residual GND structure,
    and density-gradient structure raise the prefactor, but all patches retain a
    small floor so bulk spinodal/metastable nucleation remains possible.
    """
    floor = float(np.clip(P.get('nuc_hazard_site_floor', 0.02), 0.0, 1.0))
    kfrac = np.abs(kappa_tot)/np.maximum(rho, P['rho_min'])
    # Smooth saturating measures avoid hard thresholds.
    k0 = max(float(P.get('nuc_min_kappa_frac', 0.04)), 1e-12)
    kfac = kfrac/(kfrac + k0)
    gr = grad_mag(r_field)
    q = float(np.nanquantile(gr, 0.95)) if np.any(np.isfinite(gr)) else 0.0
    gfac = gr/(gr + max(q, 1e-30)) if q > 0 else np.zeros_like(gr)
    site = (floor
            + float(P.get('nuc_site_gb_weight', 1.0))*np.clip(gb_mask, 0.0, 1.0)
            + float(P.get('nuc_site_kappa_weight', 0.7))*kfac
            + float(P.get('nuc_site_grad_r_weight', 0.3))*gfac)
    if P.get('nuc_require_organized_structure', False):
        # v27c: a finite-amplitude eta/grain insertion is only allowed where
        # collective deformation has created an organized wall/GND structure.
        # Otherwise low-angle rearrangements remain continuous recovery/CH/AC.
        wall = globals().get('rho_wall', None)
        if wall is None:
            wall_frac = np.zeros_like(rho, dtype=float)
        else:
            wall_frac = np.maximum(np.asarray(wall, dtype=float), 0.0)/np.maximum(rho, P['rho_min'])
        wf0 = max(float(P.get('nuc_wall_fraction_scale', 0.03)), 1e-12)
        kf0 = max(float(P.get('nuc_gnd_fraction_scale', 0.05)), 1e-12)
        maturity = wall_frac/(wall_frac + wf0) + kfrac/(kfrac + kf0)
        maturity = np.clip(maturity, 0.0, 1.0)
        site *= maturity
    return np.clip(site, floor, 1.0 + float(P.get('nuc_site_gb_weight', 1.0))
                   + float(P.get('nuc_site_kappa_weight', 0.7))
                   + float(P.get('nuc_site_grad_r_weight', 0.3)))

def _hazard_activity_prefactor(gdot_field=None, rp_field=None, rm_field=None):
    """Local kinetic attempt/activity prefactor for finite-amplitude hazard nucleation.

    The thermodynamic nucleation barrier remains ΔG*_th.  Arrhenius kinetics enter
    through the local plastic event activity: if no slip is occurring, there are
    very few renewal attempts to realize a finite-amplitude nucleus.  This is not
    a dσ_AT/dρ gate; the negative-slope Arrhenius-Taylor condition is diagnostic
    unless explicitly enabled as a legacy option.
    """
    if not P.get('use_plastic_activity_hazard_prefactor', True):
        return np.ones((Nx, Ny), dtype=float)
    floor = float(np.clip(P.get('hazard_activity_floor', 1.0e-4), 0.0, 1.0))
    cap = max(float(P.get('hazard_activity_cap', 50.0)), floor)
    power = max(float(P.get('hazard_activity_power', 1.0)), 0.0)
    mode = str(P.get('hazard_activity_prefactor_mode', 'gdot')).lower()
    if gdot_field is None:
        return np.full((Nx, Ny), floor, dtype=float)
    gd = np.asarray(gdot_field, dtype=float)
    if mode in ['orowan_flux', 'flux'] and rp_field is not None and rm_field is not None:
        # rho_m * v = |gdot|/b, so this is equivalent to slip activity per
        # Burgers length.  Normalize back to a strain-rate-like quantity below.
        activity = np.sum(np.abs(gd), axis=2)
    else:
        activity = np.sum(np.abs(gd), axis=2)
    ref = P.get('hazard_activity_ref', None)
    if ref is None:
        ref = max(float(P.get('edot_app', 1.0)), 1e-30)
    else:
        ref = max(float(ref), 1e-30)
    ratio = np.maximum(activity, 0.0) / ref
    if power != 1.0:
        ratio = ratio**power
    return np.clip(floor + (1.0 - floor)*ratio, floor, cap)


def _nuc_residual_gnd_budget(kappa_tot, rho_GB, psi_lat):
    """Residual Burgers/GND content available to feed a new boundary [m^-2].

    Existing orientation gradients already consume part of the local Burgers
    budget through the Frank-Bilby relation.  Only the residual can support a
    new trial misorientation.  This makes the candidate misorientation bounded
    by the local GND content instead of imposed by nuc_mis_deg.
    """
    gp = grad_mag(psi_lat)
    rho_budget = np.abs(kappa_tot) + float(P.get('nuc_rhoGB_feed_weight', 0.5))*np.maximum(rho_GB, 0.0)
    consumed = gp/max(P['b'], 1e-30)
    return np.maximum(rho_budget - consumed, 0.0)


def _nuc_barrier_fields(rho, kappa_tot, rho_GB, gb_mask, psi_lat, T_field, activity_factor=None):
    """Compute local finite-amplitude nucleation barrier and hazard fields.

    The best trial orientation is selected from a small quadrature set bounded by
    the local residual GND/Frank-Bilby budget.  The returned barrier is an event
    energy [J], obtained from a 2-D classical barrier per unit depth multiplied
    by an activation thickness of order b.
    """
    r_field = rho/max(_rho_ch_scale(), P['rho_min'])
    rho_low = _nuc_low_rho_from_potential()

    # Thermodynamic driving force density [J/m^3].
    dPhi = ATpot._Phi(np.maximum(rho, P['rho_min'])) - ATpot._Phi(rho_low)
    gp = grad_mag(psi_lat)
    ra = np.abs(kappa_tot) - P['c_alpha']*gp/P['b']
    rg = rho_GB - P['c_GB']*gp/P['b']
    comp_density = 0.5*(P['A_alpha']*ra**2 + P['A_GB']*rg**2)

    # Local GND budget -> maximum misorientation for a reference feed/capture radius.
    Rmin = max(int(P.get('nuc_min_radius_cells', 2))*dx, dx)
    Rmax = max(float(P.get('nuc_max_radius_um', 0.35))*1e-6, Rmin)
    # Use the phase-field-resolvable radius as the conservative capture area.  The
    # actual inserted radius is later chosen from R*=gamma/Deltaf and clipped by
    # the same numerical bounds.
    Rfeed = Rmin
    rho_res = _nuc_residual_gnd_budget(kappa_tot, rho_GB, psi_lat)
    lam_max = float(P.get('nuc_gnd_feed_efficiency', 0.5))*rho_res*Rfeed/2.0
    theta_max = 2.0*np.arctan(0.5*P['b']*lam_max)

    # If the user passes JSON, lists arrive as normal lists; if not, this default
    # excludes zero so a zero-misorientation density-cleaning event is not counted
    # as a new grain.
    fracs = np.asarray(P.get('nuc_theta_candidates_frac', [-1.0,-0.5,-0.25,0.25,0.5,1.0]), dtype=float)
    if fracs.size == 0:
        fracs = np.asarray([-1.0, 1.0])

    best_barrier = np.full_like(rho, np.inf, dtype=float)
    best_dG_depth = np.full_like(rho, np.inf, dtype=float)
    best_dFdens = np.zeros_like(rho, dtype=float)
    best_theta = np.zeros_like(rho, dtype=float)
    best_R = np.full_like(rho, Rmin, dtype=float)
    best_gamma = np.zeros_like(rho, dtype=float)

    spinodal = ATpot.kinetic_spinodal_mask(rho)
    thickness = max(float(P.get('nuc_barrier_thickness_b', 1.0))*P['b'], 1e-12*P['b'])

    for ff in fracs:
        theta = ff*theta_max
        abt = np.abs(theta)
        # A patch with no GND budget cannot create a distinct order-parameter
        # grain.  Very low-angle changes are better interpreted as continuous
        # recovery/subgrain rotation, so keep them in the spinodal/AC pathway
        # rather than allocating a new eta field.
        theta_field_min = np.deg2rad(float(P.get('nuc_min_field_mis_deg', 1.0)))
        ok_theta = abt > theta_field_min
        gamma = _rs_gamma(abt, T_field)
        # Compatibility relief scales with the fraction of the available residual
        # Burgers budget used by the trial boundary.  F_comp can be numerically
        # large because it is a coarse compatibility penalty, so cap its local
        # contribution to O(stored-energy relief) rather than letting it dominate
        # the barrier and generate zero-barrier nuclei.
        relief_frac = np.zeros_like(abt, dtype=float)
        np.divide(abt*abt, theta_max*theta_max, out=relief_frac, where=theta_max > 1e-30)
        relief_frac = np.clip(relief_frac, 0.0, 1.0)
        comp_cap = float(P.get('nuc_comp_relief_cap_factor', 2.0))*(np.maximum(np.abs(dPhi), 0.0) + max(ATpot.A1*_rho_ch_scale(), 1.0))
        comp_relief_density = np.minimum(np.maximum(comp_density, 0.0)*relief_frac, comp_cap)
        df = dPhi + float(P.get('nuc_comp_relief_factor', 0.5))*comp_relief_density
        df = np.where(ok_theta, df, -np.inf)
        positive = df > 1e-30
        dG_depth = np.where(positive, np.pi*gamma*gamma/np.maximum(df, 1e-300), np.inf)  # J/m
        Rstar = np.where(positive, gamma/np.maximum(df, 1e-300), Rmax)
        Rstar = np.clip(Rstar, Rmin, Rmax)
            # v23: Arrhenius-Taylor instability is diagnostic only by default;
        # do not reduce the thermodynamic barrier with Phi_AT.  The kinetics enter
        # the hazard through the local plastic-activity prefactor below.
        dG_depth_eff = dG_depth
        barrier = dG_depth_eff*thickness
        better = barrier < best_barrier
        best_barrier = np.where(better, barrier, best_barrier)
        best_dG_depth = np.where(better, dG_depth_eff, best_dG_depth)
        best_dFdens = np.where(better, df, best_dFdens)
        best_theta = np.where(better, theta, best_theta)
        best_R = np.where(better, Rstar, best_R)
        best_gamma = np.where(better, gamma, best_gamma)

    site = _hazard_site_factor(rho, kappa_tot, gb_mask, r_field)
    # Each possible embryo samples a finite patch, not an independent atomic site
    # per grid cell.  Weight by cell area / embryo area so the total hazard does
    # not grow spuriously under grid refinement.
    patch_weight = np.clip((dx*dy)/(np.pi*np.maximum(best_R, dx)**2), 1e-6, 1.0)
    # v23: production hazard uses thermodynamic barrier times local kinetic
    # attempt/activity.  The Arrhenius negative-slope gate remains optional and
    # diagnostic; by default it is identically one.
    gate_AT = ATpot.kinetic_hazard_gate(rho)
    if activity_factor is None:
        activity_factor = np.ones_like(rho, dtype=float)
    activity_factor = np.asarray(activity_factor, dtype=float)
    rate = (float(P.get('nuc_attempt_freq', 1.0e6))*site*patch_weight*activity_factor*gate_AT*
            np.exp(np.clip(-best_barrier/(kB_J*np.maximum(T_field, 1.0)), -700.0, 40.0)))
    rate = np.where(np.isfinite(best_barrier), rate, 0.0)
    rate = np.minimum(rate, float(P.get('nuc_hazard_rate_cap', 1.0e7)))

    return dict(rate=rate, barrier=best_barrier, dG_depth=best_dG_depth, dF_density=best_dFdens,
                theta=best_theta, theta_max=theta_max, R=best_R, gamma=best_gamma,
                rho_low=rho_low, spinodal=spinodal, site=site, rho_res=rho_res,
                gate_AT=gate_AT, activity_factor=activity_factor)


def _draw_exp_threshold(shape):
    return -np.log(np.maximum(_rng_nuc.random(shape), 1e-300))


# ================================================================
# 10b. v15 DRX PROVENANCE ACCOUNTING
# ================================================================
ORIGIN_UNASSIGNED = -1
ORIGIN_INITIAL = 0
ORIGIN_SPINODAL = 1
ORIGIN_HAZARD = 2

MECH_UNASSIGNED = -1
MECH_INITIAL = 0
MECH_TOPOLOGY = 1
MECH_HAZARD = 2

def _prov_name(code_):
    return {ORIGIN_INITIAL: 'initial', ORIGIN_SPINODAL: 'spinodal', ORIGIN_HAZARD: 'hazard'}.get(int(code_), 'unassigned')

def _mech_name(code_):
    return {MECH_INITIAL: 'initial', MECH_TOPOLOGY: 'topology', MECH_HAZARD: 'hazard'}.get(int(code_), 'unassigned')

def _record_grain_birth(gid, origin, mechanism, parent=-1, step=-1, x=np.nan, y=np.nan,
                        area_px=0, theta_deg=np.nan, theta_max_deg=np.nan,
                        R_um=np.nan, barrier_eV=np.nan):
    """Record the bookkeeping/provenance of a newly allocated eta field.

    This is diagnostic accounting only.  It does not affect the evolution.
    """
    if not P.get('track_grain_provenance', True):
        return
    if 'grain_origin_lineage' not in globals():
        return
    if gid < 0 or gid >= len(grain_origin_lineage):
        return

    global _grain_step_topology_births, _grain_step_hazard_births
    grain_origin_lineage[gid] = int(origin)
    grain_birth_mechanism[gid] = int(mechanism)
    grain_parent[gid] = int(parent) if parent is not None else -1
    grain_birth_step[gid] = int(step)
    grain_birth_x[gid] = float(x)
    grain_birth_y[gid] = float(y)
    grain_birth_area_px[gid] = int(area_px)
    grain_birth_theta_deg[gid] = float(theta_deg)
    grain_birth_theta_max_deg[gid] = float(theta_max_deg)
    grain_birth_R_um[gid] = float(R_um)
    grain_birth_barrier_eV[gid] = float(barrier_eV)

    if int(mechanism) == MECH_TOPOLOGY:
        _grain_step_topology_births += 1
    elif int(mechanism) == MECH_HAZARD:
        _grain_step_hazard_births += 1

def _provenance_counts(Ng):
    """Counts of currently allocated grain fields by lineage and birth mechanism."""
    if (not P.get('track_grain_provenance', True)) or ('grain_origin_lineage' not in globals()):
        return dict(
            grain_initial_lineage=Ng, grain_spinodal_lineage=0, grain_hazard_lineage=0,
            grain_initial_births=Ng, grain_topology_births=0, grain_hazard_births=0,
            grain_topology_step_births=0, grain_hazard_step_births=0)
    oo = grain_origin_lineage[:Ng]
    mm = grain_birth_mechanism[:Ng]
    return dict(
        grain_initial_lineage=int(np.sum(oo == ORIGIN_INITIAL)),
        grain_spinodal_lineage=int(np.sum(oo == ORIGIN_SPINODAL)),
        grain_hazard_lineage=int(np.sum(oo == ORIGIN_HAZARD)),
        grain_initial_births=int(np.sum(mm == MECH_INITIAL)),
        grain_topology_births=int(np.sum(mm == MECH_TOPOLOGY)),
        grain_hazard_births=int(np.sum(mm == MECH_HAZARD)),
        grain_topology_step_births=int(globals().get('_grain_step_topology_births', 0)),
        grain_hazard_step_births=int(globals().get('_grain_step_hazard_births', 0)),
    )

def _component_centroid(mask):
    if not np.any(mask):
        return np.nan, np.nan
    ii, jj = np.where(mask)
    return float(np.mean(ii)), float(np.mean(jj))


def apply_hazard_nucleation(eta, psi_gv, Ng, lab, psi_lat, psi_plastic,
                            rp, rm, rho, rho_GB, gb_mask, kappa_tot, T_field,
                            H_nuc, E_nuc, activity_factor=None):
    global rho_forest, rho_wall, nuc_cand_active, nuc_cand_age, nuc_cand_best_barrier, nuc_cand_birth_step
    """Advance cumulative nucleation hazard and insert at most one embryo.

    The hazard is evaluated for every local patch.  No rho/kappa/gradpsi candidate
    threshold is used; unfavorable regions simply have exponentially small rates.
    A successful event allocates a new eta field, sets its orientation from the
    GND-bounded best trial misorientation, and transfers the removed mobile
    density into a boundary shell as rho_GB instead of deleting it.
    """
    if P.get('disable_nucleation', False) or (not P.get('use_hazard_nucleation', True)) or Ng >= P['grain_max']-1:
        return eta, psi_gv, Ng, lab, psi_lat, psi_plastic, rp, rm, rho, rho_GB, gb_mask, H_nuc, E_nuc, dict(cand=0, best_dF=np.nan, best_score=np.nan, event=0)

    fields = _nuc_barrier_fields(rho, kappa_tot, rho_GB, gb_mask, psi_lat, T_field, activity_factor=activity_factor)
    dt_h = P['dt']*max(int(P.get('nuc_interval', 20)), 1)
    H_nuc = H_nuc + fields['rate']*dt_h
    excess = H_nuc - E_nuc
    possible = np.isfinite(fields['barrier']) & (fields['rate'] > 0.0)
    triggered = possible & (excess >= 0.0)

    finite_barriers = fields['barrier'][possible]
    best_barrier = float(np.nanmin(finite_barriers)) if finite_barriers.size else np.nan
    best_score = float(np.nanmax(fields['rate'][possible])) if np.any(possible) else 0.0
    diag = dict(cand=int(np.sum(possible)), best_dF=best_barrier, best_score=best_score,
                event=0, hazard_max=float(np.nanmax(fields['rate'])) if np.any(possible) else 0.0,
                Hmax=float(np.nanmax(H_nuc)), theta_best_deg=np.nan, theta_max_deg=np.nan,
                R_best_um=np.nan, barrier_best_eV=(best_barrier/eV_J if np.isfinite(best_barrier) else np.nan),
                rho_low=float(fields['rho_low']), spinodal_frac=float(np.mean(fields['spinodal'])),
                activity_factor_mean=float(np.nanmean(fields.get('activity_factor', 1.0))),
                activity_factor_max=float(np.nanmax(fields.get('activity_factor', 1.0))))

    # v34: cumulative-hazard trigger starts/ages a candidate nucleus.  The
    # candidate must survive several hazard evaluations before it is promoted to
    # a permanent eta/grain field.  This changes the interpretation of grain IDs:
    # they are persistent DRX nuclei, not instantaneous hazard spikes.
    if P.get('use_nuc_candidate_incubation', True):
        max_bar_eV = float(P.get('nuc_candidate_max_barrier_eV', np.inf))
        min_rate = float(P.get('nuc_candidate_min_rate', 0.0))
        min_dF = float(P.get('nuc_candidate_min_dF_Jm3', 0.0))
        viable = (possible & np.isfinite(fields['barrier']) &
                  (fields['barrier']/eV_J <= max_bar_eV) &
                  (fields['rate'] >= min_rate) &
                  (fields.get('dF_density', np.zeros_like(rho)) >= min_dF))
        new_cand = triggered & viable & (~nuc_cand_active)
        decay_evals = max(int(P.get('nuc_candidate_decay_evals', 2)), 0)
        keep_active = nuc_cand_active & viable
        if decay_evals <= 0:
            nuc_cand_active = keep_active | new_cand
            nuc_cand_age = np.where(nuc_cand_active, nuc_cand_age + 1, 0)
        else:
            age = nuc_cand_age.astype(np.int32)
            age = np.where(keep_active | new_cand, age + 1, age - 1)
            age = np.clip(age, 0, 32767)
            nuc_cand_active = (age > 0) & (keep_active | new_cand | nuc_cand_active)
            nuc_cand_age = age.astype(np.int16)
        nuc_cand_best_barrier = np.where(new_cand, fields['barrier'], nuc_cand_best_barrier)
        nuc_cand_best_barrier = np.where(nuc_cand_active,
                                         np.minimum(nuc_cand_best_barrier, fields['barrier']),
                                         np.inf)
        nuc_cand_birth_step = np.where(new_cand,
                                       int(globals().get('current_step_for_provenance', -1)),
                                       nuc_cand_birth_step)
        hold = max(int(P.get('nuc_candidate_hold_evals', 6)), 1)
        promotable = nuc_cand_active & viable & (nuc_cand_age >= hold)
        diag.update(candidate_active=int(np.sum(nuc_cand_active)),
                    candidate_new=int(np.sum(new_cand)),
                    candidate_promotable=int(np.sum(promotable)),
                    candidate_age_max=int(np.nanmax(nuc_cand_age)) if nuc_cand_age.size else 0,
                    candidate_barrier_min_eV=float(np.nanmin(nuc_cand_best_barrier[nuc_cand_active])/eV_J) if np.any(nuc_cand_active) else np.nan)
        if P.get('nuc_candidate_diagnostic_only', False) or (not np.any(promotable)):
            return eta, psi_gv, Ng, lab, psi_lat, psi_plastic, rp, rm, rho, rho_GB, gb_mask, H_nuc, E_nuc, diag
        selmode = str(P.get('nuc_candidate_promote_select', 'oldest')).lower()
        if selmode == 'min_barrier':
            select_field = np.where(promotable, -fields['barrier'], -np.inf)
        elif selmode == 'max_excess':
            select_field = np.where(promotable, excess, -np.inf)
        else:
            select_field = np.where(promotable, nuc_cand_age.astype(float), -np.inf)
        triggered = promotable
    else:
        if not np.any(triggered):
            return eta, psi_gv, Ng, lab, psi_lat, psi_plastic, rp, rm, rho, rho_GB, gb_mask, H_nuc, E_nuc, diag
        if str(P.get('nuc_event_select', 'max_excess')) == 'min_barrier':
            select_field = np.where(triggered, -fields['barrier'], -np.inf)
        else:
            select_field = np.where(triggered, excess, -np.inf)
    ix = np.unravel_index(int(np.nanargmax(select_field)), select_field.shape)

    Rm = float(fields['R'][ix])
    Re = max(int(np.round(Rm/dx)), int(P.get('nuc_min_radius_cells', 2)))
    theta = float(fields['theta'][ix])
    theta_max = float(fields['theta_max'][ix])
    rho_low = float(fields['rho_low'])

    IX, IY = np.indices((Nx, Ny))
    d2 = (np.minimum(np.abs(IX-ix[0]), Nx-np.abs(IX-ix[0]))**2 +
          np.minimum(np.abs(IY-ix[1]), Ny-np.abs(IY-ix[1]))**2)
    em = d2 <= Re**2
    shell = (d2 <= (Re + 2)**2) & (~em)
    if np.sum(em) <= 2:
        # Numerical fallback: reset hazard locally and do not force a bad embryo.
        H_nuc[ix] = 0.0; E_nuc[ix] = _draw_exp_threshold(())
        return eta, psi_gv, Ng, lab, psi_lat, psi_plastic, rp, rm, rho, rho_GB, gb_mask, H_nuc, E_nuc, diag

    gid = Ng
    parent_psi = float(np.angle(np.mean(np.exp(1j*psi_lat[em]))))
    psi_new = float(angle_wrap(parent_psi + theta))
    psi_new = float(np.clip(psi_new, -np.deg2rad(P['psi_max_deg']), np.deg2rad(P['psi_max_deg'])))
    psi_gv[gid] = psi_new

    # Smooth order-parameter embryo.  This is the only discrete operation: it is
    # the finite-amplitude realization of the accumulated hazard, not a threshold
    # candidate insertion.
    eta[:, :, gid] = 0.0
    eta[:, :, gid] = np.where(em, 1.0, 0.0)
    e = eta[:, :, gid]
    nb = 0.25*(np.roll(e,1,0)+np.roll(e,-1,0)+np.roll(e,1,1)+np.roll(e,-1,1))
    eta[:, :, gid] = 0.7*e + 0.3*nb
    for ii in range(Ng):
        eta[:, :, ii] = np.where(em, eta[:, :, ii]*0.05, eta[:, :, ii])
    es = np.sum(eta[:, :, :gid+1], axis=2, keepdims=True) + 1e-30
    eta[:, :, :gid+1] /= es

    # Density product state comes from the low-density well of Phi.  Excess mobile
    # density is moved into the new boundary shell as residual GB content.
    old_rho = rho.copy()
    excess_rho = np.maximum(old_rho[em] - rho_low, 0.0)
    excess_integral = float(np.sum(excess_rho)*dx*dy)*float(P.get('nuc_excess_to_rhoGB_fraction', 1.0))
    for ss in range(nSlip):
        rp[:, :, ss] = np.where(em, rho_low/(2*nSlip), rp[:, :, ss])
        rm[:, :, ss] = np.where(em, rho_low/(2*nSlip), rm[:, :, ss])
        if P.get('use_rho_state_partition', False):
            rho_forest[:, :, ss] = np.where(em, 0.0, rho_forest[:, :, ss])
    if P.get('use_rho_state_partition', False):
        rho_wall = np.where(em, 0.0, rho_wall)
    if np.any(shell) and excess_integral > 0.0:
        add_shell = excess_integral/(float(np.sum(shell))*dx*dy)
        rho_GB = np.where(shell, np.minimum(rho_GB + add_shell, P['rho_max']), rho_GB)
    psi_plastic = np.where(em, 0.0, psi_plastic)
    rho = _rho_total_state(rp, rm, rho_forest, rho_wall)

    # v15 provenance: this is a true finite-amplitude cumulative-hazard nucleus.
    parent_gid = int(lab[ix]) if np.ndim(lab) == 2 else -1
    _record_grain_birth(gid, ORIGIN_HAZARD, MECH_HAZARD, parent=parent_gid,
                        step=int(globals().get('current_step_for_provenance', -1)),
                        x=float(ix[0]), y=float(ix[1]), area_px=int(np.sum(em)),
                        theta_deg=float(np.rad2deg(theta)),
                        theta_max_deg=float(np.rad2deg(theta_max)),
                        R_um=float(Rm*1e6),
                        barrier_eV=float(fields['barrier'][ix]/eV_J))
    Ng += 1
    lab = np.argmax(eta[:, :, :Ng], axis=2)
    gb_mask = diffuse_gb_support(eta, lab, Ng)
    psi_lat = reconstruct_psi_lat(eta, psi_gv, psi_plastic, Ng)

    # Reset cumulative hazard in the transformed neighborhood and draw new
    # exponential thresholds there.  Other regions keep their hazard memory.
    rf = float(P.get('nuc_reset_hazard_radius_factor', 1.5))
    reset = d2 <= max(int(np.ceil(rf*Re)), Re+1)**2
    H_nuc = np.where(reset, 0.0, H_nuc)
    E_nuc = np.where(reset, _draw_exp_threshold(H_nuc.shape), E_nuc)
    if P.get('use_nuc_candidate_incubation', True):
        nuc_cand_active = np.where(reset, False, nuc_cand_active)
        nuc_cand_age = np.where(reset, 0, nuc_cand_age)
        nuc_cand_best_barrier = np.where(reset, np.inf, nuc_cand_best_barrier)
        nuc_cand_birth_step = np.where(reset, -1, nuc_cand_birth_step)

    diag.update(event=1, theta_best_deg=float(np.rad2deg(theta)), theta_max_deg=float(np.rad2deg(theta_max)),
                R_best_um=float(Rm*1e6), barrier_best_eV=float(fields['barrier'][ix]/eV_J),
                best_dF=float(fields['barrier'][ix]), best_score=float(fields['rate'][ix]))
    print(f"  hazard nucleus grain {gid} at {ix}: ΔG*={diag['barrier_best_eV']:.2f} eV, "
          f"theta={diag['theta_best_deg']:.2f}°/{diag['theta_max_deg']:.2f}°, R={diag['R_best_um']:.3f} µm")
    return eta, psi_gv, Ng, lab, psi_lat, psi_plastic, rp, rm, rho, rho_GB, gb_mask, H_nuc, E_nuc, diag

# ================================================================
# 10c. DIAGNOSTIC HELPERS
# ================================================================
def _safe_corr(a, b, mask=None):
    """Robust Pearson correlation for field diagnostics."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        aa = aa[m]; bb = bb[m]
    else:
        aa = aa.ravel(); bb = bb.ravel()
    good = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[good]; bb = bb[good]
    if aa.size < 3:
        return np.nan
    sa = aa.std(); sb = bb.std()
    if sa < 1e-300 or sb < 1e-300:
        return np.nan
    return float(np.mean((aa-aa.mean())*(bb-bb.mean()))/(sa*sb))


def _top_mask(field, frac=0.10):
    """Mask for the top frac of a field."""
    f = np.asarray(field, dtype=float)
    frac = float(np.clip(frac, 1.0/f.size, 0.95))
    q = np.nanquantile(f, 1.0-frac)
    return f >= q


def _masked_mean(field, mask, default=np.nan):
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return float(default)
    return float(np.nanmean(np.asarray(field, dtype=float)[m]))


def _masked_sum_frac(field, mask):
    f = np.maximum(np.asarray(field, dtype=float), 0.0)
    den = float(np.nansum(f))
    if den <= 1e-300:
        return np.nan
    return float(np.nansum(f[np.asarray(mask, dtype=bool)]) / den)


def _angle_diff_mod_pi(a, b):
    return float(np.abs(np.arctan2(np.sin(a-b), np.cos(a-b+0.0)) if False else ((a-b+np.pi/2) % np.pi - np.pi/2)))


def _asb_band_metrics(T, rho, heat_diag=None, km_diag=None, psi_lat=None):
    """Scalar diagnostics for hot-band/ASB branch screening.

    These metrics are intentionally observational.  They do not affect the
    solver.  A convincing ASB branch should show growing temperature/plastic
    power localization, negative T-logrho correlation, and hot-band recovery
    exceeding storage.
    """
    if heat_diag is None: heat_diag = {}
    if km_diag is None: km_diag = {}
    T = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)
    qdot = np.asarray(heat_diag.get('_qdot_field', np.zeros_like(T)), dtype=float)
    gabs = np.asarray(km_diag.get('_gdot_abs_field', np.zeros_like(T)), dtype=float)
    storage = np.asarray(km_diag.get('_storage_rate_field', np.zeros_like(T)), dtype=float)
    anni = np.asarray(km_diag.get('_anni_rate_field', np.zeros_like(T)), dtype=float)
    diffrec = np.asarray(km_diag.get('_diffrec_rate_field', np.zeros_like(T)), dtype=float)
    hot_frac = float(np.clip(P.get('asb_diag_hot_frac', 0.05), 1.0/T.size, 0.50))
    cold_frac = float(np.clip(P.get('asb_diag_cold_frac', 0.50), hot_frac, 0.95))
    hot = _top_mask(T, hot_frac)
    cold = T <= np.nanquantile(T, cold_frac)
    qtop = _top_mask(qdot, hot_frac) if np.nanmax(qdot) > np.nanmin(qdot) else hot
    gtop = _top_mask(gabs, hot_frac) if np.nanmax(gabs) > np.nanmin(gabs) else hot
    net = storage - anni - diffrec

    # Orientation/anisotropy of the temperature mode from the structure tensor.
    dTx = ddx(T); dTy = ddy(T)
    w = dTx*dTx + dTy*dTy
    if float(np.nansum(w)) > 1e-300:
        Jxx = float(np.nansum(w*dTx*dTx) / np.nansum(w))
        Jyy = float(np.nansum(w*dTy*dTy) / np.nansum(w))
        Jxy = float(np.nansum(w*dTx*dTy) / np.nansum(w))
        tr = Jxx + Jyy
        disc = np.sqrt(max((Jxx-Jyy)**2 + 4.0*Jxy*Jxy, 0.0))
        lam1 = 0.5*(tr + disc); lam2 = 0.5*(tr - disc)
        theta_grad = 0.5*np.arctan2(2.0*Jxy, Jxx-Jyy)
        theta_band = float(((theta_grad + 0.5*np.pi + 0.5*np.pi) % np.pi) - 0.5*np.pi)
        anis = float((lam1-lam2)/max(lam1+lam2, 1e-300))
    else:
        theta_band = np.nan; anis = np.nan

    slip_mean_offset = _circular_mean_angle(psi_lat.ravel(), default=0.0) if psi_lat is not None else 0.0
    if np.isfinite(theta_band):
        diffs = []
        for aa in base_ang:
            d = ((theta_band - (float(aa)+slip_mean_offset) + 0.5*np.pi) % np.pi) - 0.5*np.pi
            diffs.append(abs(d))
        align_deg = float(np.rad2deg(min(diffs))) if diffs else np.nan
    else:
        align_deg = np.nan

    return dict(
        asb_T_std=float(np.nanstd(T)),
        asb_T_range=float(np.nanmax(T)-np.nanmin(T)),
        asb_T_hot_minus_mean=float(_masked_mean(T, hot) - np.nanmean(T)),
        asb_rho_hot_mean=_masked_mean(rho, hot),
        asb_rho_cold_mean=_masked_mean(rho, cold),
        asb_rho_hot_over_cold=float(_masked_mean(rho, hot)/max(_masked_mean(rho, cold), P['rho_min'])),
        asb_corr_T_rho=_safe_corr(T, rho),
        asb_corr_T_logrho=_safe_corr(T, np.log10(np.maximum(rho, P['rho_min']))),
        asb_corr_T_qdot=_safe_corr(T, qdot),
        asb_corr_T_gdot=_safe_corr(T, gabs),
        asb_qdot_top5_frac=_masked_sum_frac(qdot, qtop),
        asb_gdot_top5_frac=_masked_sum_frac(gabs, gtop),
        asb_storage_hot_mean=_masked_mean(storage, hot, 0.0),
        asb_anni_hot_mean=_masked_mean(anni, hot, 0.0),
        asb_diffrec_hot_mean=_masked_mean(diffrec, hot, 0.0),
        asb_net_rhodot_hot_mean=_masked_mean(net, hot, 0.0),
        asb_recovery_over_storage_hot=float((_masked_mean(anni+diffrec, hot, 0.0)+1.0)/(_masked_mean(storage, hot, 0.0)+1.0)),
        asb_band_anisotropy_T=anis,
        asb_band_angle_deg=float(np.rad2deg(theta_band)) if np.isfinite(theta_band) else np.nan,
        asb_band_alignment_to_slip_deg=align_deg,
    )


def _energy_audit(r_f, rho, eta, psi_lat, kappa_tot, rho_GB, gb_mask, Ng):
    """Return variational energy components and coupling-residual metrics.

    Units are per unit out-of-plane depth for the 2-D cell.  These numbers are
    mainly intended for trends: if rho changes but F_bulk/F_grad/F_comp do not
    respond, the intended variational channel is disconnected.
    """
    dA = dx*dy
    F_bulk = float(np.nansum(ATpot._Phi(np.maximum(rho, P['rho_min']))) * dA)
    F_r_grad = float(0.5*P['kappa_r']*np.nansum(ddx(r_f)**2 + ddy(r_f)**2) * dA)
    F_eta_grad = 0.0
    F_eta_barrier = 0.0
    if Ng > 0:
        e = eta[:, :, :Ng]
        for gi in range(Ng):
            F_eta_grad += float(0.5*P['kappa_eta']*np.nansum(ddx(e[:,:,gi])**2 + ddy(e[:,:,gi])**2) * dA)
        sum_e2 = np.sum(e*e, axis=2)
        sum_e4 = np.sum(e**4, axis=2)
        # pair barrier, counted once; derivative scale is approximate but diagnostic trend is robust
        F_eta_barrier = float(0.5*P['W_eta']*np.nansum(np.maximum(sum_e2*sum_e2 - sum_e4, 0.0)) * dA)
    gp = grad_mag(psi_lat)
    ra = np.abs(kappa_tot) - P['c_alpha']*gp/P['b']
    rg = rho_GB - P['c_GB']*gp/P['b']
    F_comp_alpha = float(0.5*P['A_alpha']*np.nansum(ra**2)*dA)
    F_comp_GB = float(0.5*P['A_GB']*np.nansum(rg**2)*dA)
    F_total = F_bulk + F_r_grad + F_eta_grad + F_eta_barrier + F_comp_alpha + F_comp_GB
    fb_target = np.clip(P.get('frank_bilby_coeff', 1.0)*gb_mask*gp/P['b'], 0.0, P['rho_max'])
    return dict(
        F_bulk=F_bulk, F_r_grad=F_r_grad, F_eta_grad=F_eta_grad,
        F_eta_barrier=F_eta_barrier, F_comp_alpha=F_comp_alpha,
        F_comp_GB=F_comp_GB, F_total_full=F_total,
        comp_alpha_rms=float(np.sqrt(np.nanmean(ra**2))),
        comp_GB_rms=float(np.sqrt(np.nanmean(rg**2))),
        FB_target_mean=float(np.nanmean(fb_target)),
        FB_target_max=float(np.nanmax(fb_target)),
        rhoGB_over_FB_mean=float(np.nanmean(rho_GB / np.maximum(fb_target, P['rho_min']))),
    )


def _diagnostic_row(n, t, rho, rho_c, eta, lab, Ng, psi_lat, psi_plastic, kappa_tot,
                    rp, rm, rho_GB, gb_mask, mu_ch, sigma_bar, T, E_tot, nflip,
                    km_store_mean, km_anni_mean, ch_delta_abs_mean,
                    ch_delta_std, ac_eta_delta_mean, gnd_transfer_mean,
                    rhoGB_delta_mean, nuc_diag, gb_hp_diag=None, topology_diag=None, heat_diag=None, km_diag=None):
    if gb_hp_diag is None:
        gb_hp_diag = {}
    if topology_diag is None:
        topology_diag = {}
    if heat_diag is None:
        heat_diag = {}
    if km_diag is None:
        km_diag = {}
    rho_scale_diag = _rho_ch_scale()
    r_f = rho/max(rho_scale_diag, P['rho_min'])
    rho_mobile_diag = _rho_mobile_field(rp, rm)
    rho_forest_diag = _rho_forest_total_field(globals().get('rho_forest', None))
    rho_wall_diag = _rho_wall_field(globals().get('rho_wall', None))
    gp = grad_mag(psi_lat)
    kfrac = np.abs(kappa_tot)/np.maximum(rho, P['rho_min'])
    top = _top_mask(rho, P.get('diag_top_frac', 0.10))
    low = _top_mask(-rho, P.get('diag_top_frac', 0.10))
    gb = gb_mask > P.get('diag_gb_thresh', 0.20)
    high_on_gb = float(np.mean(gb[top])) if np.any(top) else np.nan
    gb_on_high = float(np.mean(top[gb])) if np.any(gb) else np.nan
    gb_area = float(np.mean(gb))
    eta_max_diag, eta_second_diag, _, eta_entropy_diag, eta_nactive_diag = eta_purity_fields(eta, Ng)
    mixed_eta = eta_entropy_diag > float(P.get('eta_mixed_warning_entropy', 1.0))
    eterms = _energy_audit(r_f, rho, eta, psi_lat, kappa_tot, rho_GB, gb_mask, Ng)
    # v22 thermodynamic/kinetic diagnostics.
    mu_T_field = ATpot.mu_shear(T)
    A_E_diag = ATpot.Estar_coeff(T) if P.get('use_temperature_dependent_Estar', True) else 0.5*mu_iso*P['b']**2
    Estar_diag = A_E_diag * rho
    dsig_AT = ATpot.dsigma_at_drho(rho)
    gate_AT_diag = ATpot.kinetic_hazard_gate(rho)
    prov = _provenance_counts(Ng)
    qdot_mech = float(heat_diag.get('qdot_mean', P['taylor_quinney'] * sigma_bar * P['edot_app']))
    dT_mech_step = float(heat_diag.get('dT_mean_step', P['dt'] * qdot_mech / max(P['cp_rho_vol'], 1.0)))
    row = dict(
        step=int(n), t_us=float(t*1e6), eps_pct=float(E_tot[0,0]*100),
        sigma_MPa=float(sigma_bar/1e6), T_mean=float(np.nanmean(T)), T_max=float(np.nanmax(T)),
        rho_c=float(rho_c),
        rho_peak_ind=float(getattr(ATpot, 'rho_peak_ind', rho_c)),
        rho_ch_ref=float(rho_scale_diag),
        rho_mean=float(np.nanmean(rho)), rho_max=float(np.nanmax(rho)),
        rho_min=float(np.nanmin(rho)), rho_std=float(np.nanstd(rho)),
        rho_mobile_mean=float(np.nanmean(rho_mobile_diag)),
        rho_forest_mean=float(np.nanmean(rho_forest_diag)),
        rho_wall_mean=float(np.nanmean(rho_wall_diag)),
        rho_mobile_frac_mean=float(np.nanmean(rho_mobile_diag/np.maximum(rho, P['rho_min']))),
        rho_forest_frac_mean=float(np.nanmean(rho_forest_diag/np.maximum(rho, P['rho_min']))),
        rho_wall_frac_mean=float(np.nanmean(rho_wall_diag/np.maximum(rho, P['rho_min']))),
        r_mean=float(np.nanmean(r_f)), r_max=float(np.nanmax(r_f)), r_min=float(np.nanmin(r_f)),
        r_p10=float(np.nanquantile(r_f, 0.10)), r_p50=float(np.nanquantile(r_f, 0.50)),
        r_p90=float(np.nanquantile(r_f, 0.90)),
        rho_top10_mean=_masked_mean(rho, top), rho_low10_mean=_masked_mean(rho, low),
        mu_mean=float(np.nanmean(mu_ch)), mu_std=float(np.nanstd(mu_ch)),
        mu_top10_mean=_masked_mean(mu_ch, top), mu_low10_mean=_masked_mean(mu_ch, low),
        mu_T_mean_GPa=float(np.nanmean(mu_T_field)/1e9),
        Estar_mean_Jm3=float(np.nanmean(Estar_diag)),
        Estar_max_Jm3=float(np.nanmax(Estar_diag)),
        potential_mode=str(P.get('potential_mode', 'thermo_stored')),
        AT_spinodal_frac=float(np.nanmean(dsig_AT < 0.0)),
        AT_hazard_gate_mean=float(np.nanmean(gate_AT_diag)),
        AT_dsigma_min=float(np.nanmin(dsig_AT)),
        AT_dsigma_mean=float(np.nanmean(dsig_AT)),
        sigma_AT_peak_MPa=float(getattr(ATpot, 'sigma_at_peak', np.nan)/1e6),
        rho_AT_peak=float(getattr(ATpot, 'rho_at_sigma_peak', np.nan)),
        gb_area_frac=gb_area, highrho_on_gb_frac=high_on_gb, gb_on_highrho_frac=gb_on_high,
        eta_max_mean=float(np.nanmean(eta_max_diag)), eta_second_mean=float(np.nanmean(eta_second_diag)),
        eta_entropy_mean=float(np.nanmean(eta_entropy_diag)), eta_entropy_max=float(np.nanmax(eta_entropy_diag)),
        eta_nactive_mean=float(np.nanmean(eta_nactive_diag)), eta_mixed_area_frac=float(np.mean(mixed_eta)),
        rho_gb_mean=_masked_mean(rho, gb), rho_offgb_mean=_masked_mean(rho, ~gb),
        mu_gb_mean=_masked_mean(mu_ch, gb), mu_offgb_mean=_masked_mean(mu_ch, ~gb),
        psi_max_deg=float(np.rad2deg(np.nanmax(np.abs(psi_lat)))),
        psi_std_deg=float(np.rad2deg(np.nanstd(psi_lat))),
        psi_plastic_max_deg=float(np.rad2deg(np.nanmax(np.abs(psi_plastic)))),
        gradpsi_mean=float(np.nanmean(gp)), gradpsi_max=float(np.nanmax(gp)),
        kappa_signed_mean=float(np.nanmean(kappa_tot)),
        kappa_signed_std=float(np.nanstd(kappa_tot)),
        kappa_pos_area_frac=float(np.mean(kappa_tot > 0)),
        kappa_neg_area_frac=float(np.mean(kappa_tot < 0)),
        kappa_abs_mean=float(np.nanmean(np.abs(kappa_tot))),
        kappa_abs_max=float(np.nanmax(np.abs(kappa_tot))),
        kappa_frac_mean=float(np.nanmean(kfrac)), kappa_frac_max=float(np.nanmax(kfrac)),
        corr_signed_kappa_r=_safe_corr(kappa_tot, r_f),
        corr_signed_kappa_gb=_safe_corr(kappa_tot, gb_mask),
        rhoGB_mean=float(np.nanmean(rho_GB)), rhoGB_max=float(np.nanmax(rho_GB)),
        corr_r_gb=_safe_corr(r_f, gb_mask),
        corr_r_gradpsi=_safe_corr(r_f, gp),
        corr_r_kappa=_safe_corr(r_f, np.abs(kappa_tot)),
        corr_mu_gb=_safe_corr(mu_ch, gb_mask),
        km_store_mean=float(km_store_mean), km_anni_mean=float(km_anni_mean),
        km_net_mean=float(km_store_mean-km_anni_mean-float(km_diag.get('diffrec_mean', 0.0))),
        ch_delta_abs_mean=float(ch_delta_abs_mean), ch_delta_std=float(ch_delta_std),
        ac_eta_delta_mean=float(ac_eta_delta_mean),
        gnd_transfer_mean=float(gnd_transfer_mean), rhoGB_delta_mean=float(rhoGB_delta_mean),
        gb_hp_src_mean=float(gb_hp_diag.get('src_mean', 0.0)),
        gb_hp_sink_mean=float(gb_hp_diag.get('sink_mean', 0.0)),
        gb_hp_rate_mean=float(gb_hp_diag.get('rate_mean', 0.0)),
        gb_hp_xi_mean=float(gb_hp_diag.get('xi_mean', np.nan)),
        gb_hp_xi_source_mean=float(gb_hp_diag.get('xi_source_mean', np.nan)),
        gb_hp_xi_trans_mean=float(gb_hp_diag.get('xi_trans_mean', gb_hp_diag.get('xi_mean', np.nan))),
        gb_hp_xi_sink_mean=float(gb_hp_diag.get('xi_sink_mean', np.nan)),
        gb_hp_xi_screen_mean=float(gb_hp_diag.get('xi_screen_mean', np.nan)),
        gb_trans_mis_deg_mean=float(gb_hp_diag.get('gb_trans_mis_deg_mean', np.nan)),
        gb_trans_mis_deg_max=float(gb_hp_diag.get('gb_trans_mis_deg_max', np.nan)),
        gb_trans_mprime_mean=float(gb_hp_diag.get('gb_trans_mprime_mean', np.nan)),
        gb_trans_mprime_min=float(gb_hp_diag.get('gb_trans_mprime_min', np.nan)),
        gb_trans_bres_mean=float(gb_hp_diag.get('gb_trans_bres_mean', np.nan)),
        gb_trans_bres_max=float(gb_hp_diag.get('gb_trans_bres_max', np.nan)),
        gb_trans_barrier_eV_mean=float(gb_hp_diag.get('gb_trans_barrier_eV_mean', np.nan)),
        gb_trans_barrier_eV_max=float(gb_hp_diag.get('gb_trans_barrier_eV_max', np.nan)),
        gb_trans_factor_mean=float(gb_hp_diag.get('gb_trans_factor_mean', np.nan)),
        gb_trans_factor_min=float(gb_hp_diag.get('gb_trans_factor_min', np.nan)),
        n_grains=int(Ng),
        topo_components=int(topology_diag.get('topo_components', Ng)),
        multi_component_labels=int(topology_diag.get('multi_component_labels', 0)),
        max_components_per_label=int(topology_diag.get('max_components_per_label', 1)),
        topology_splits=int(topology_diag.get('splits', 0)),
        topology_unassigned=int(topology_diag.get('unassigned', 0)),
        grain_initial_lineage=prov['grain_initial_lineage'],
        grain_spinodal_lineage=prov['grain_spinodal_lineage'],
        grain_hazard_lineage=prov['grain_hazard_lineage'],
        grain_initial_births=prov['grain_initial_births'],
        grain_topology_births=prov['grain_topology_births'],
        grain_hazard_births=prov['grain_hazard_births'],
        grain_topology_step_births=prov['grain_topology_step_births'],
        grain_hazard_step_births=prov['grain_hazard_step_births'],
        nuc_activity_factor_mean=float(nuc_diag.get('activity_factor_mean', np.nan)),
        nuc_activity_factor_max=float(nuc_diag.get('activity_factor_max', np.nan)),
        nuc_candidate_active=int(nuc_diag.get('candidate_active', 0)),
        nuc_candidate_new=int(nuc_diag.get('candidate_new', 0)),
        nuc_candidate_promotable=int(nuc_diag.get('candidate_promotable', 0)),
        nuc_candidate_age_max=int(nuc_diag.get('candidate_age_max', 0)),
        nuc_candidate_barrier_min_eV=float(nuc_diag.get('candidate_barrier_min_eV', np.nan)),
        heat_qdot_MWm3=float(qdot_mech/1e6),
        heat_qdot_local_max_MWm3=float(heat_diag.get('qdot_max', np.nan)/1e6),
        heat_qdot_local_std_MWm3=float(heat_diag.get('qdot_std', np.nan)/1e6),
        heat_dT_mech_step=float(dT_mech_step),
        heat_dT_local_max_step=float(heat_diag.get('dT_max_step', np.nan)),
        thermal_dt_active=float(heat_diag.get('thermal_dt_active', np.nan)),
        thermal_dt=float(heat_diag.get('thermal_dt', np.nan)),
        thermal_dt_base=float(heat_diag.get('thermal_dt_base', np.nan)),
        thermal_dt_sub_equiv=float(heat_diag.get('thermal_dt_sub_equiv', np.nan)),
        thermal_dt_dT_allow=float(heat_diag.get('thermal_dt_dT_allow', np.nan)),
        thermal_dt_dT_macro_pred=float(heat_diag.get('thermal_dt_dT_macro_pred', np.nan)),
        heat_mode=str(heat_diag.get('mode', 'global_sigma_edot')),
        k2_eff_mean=float(km_diag.get('k2_eff_mean', np.nan)),
        k2_eff_max=float(km_diag.get('k2_eff_max', np.nan)),
        k2_eff_min=float(km_diag.get('k2_eff_min', np.nan)),
        km_diffrec_mean=float(km_diag.get('diffrec_mean', 0.0)),
        diffrec_D_mean=float(km_diag.get('diffrec_D_mean', np.nan)),
        diffrec_D_max=float(km_diag.get('diffrec_D_max', np.nan)),
        diffrec_arg_mean=float(km_diag.get('diffrec_arg_mean', np.nan)),
        diffrec_arg_max=float(km_diag.get('diffrec_arg_max', np.nan)),
        Pplastic_mean_Jm3s=float(km_diag.get('Pplastic_mean', np.nan)),
        Pplastic_max_Jm3s=float(km_diag.get('Pplastic_max', np.nan)),
        Pstore_allowed_mean_Jm3s=float(km_diag.get('Pstore_allowed_mean', np.nan)),
        Pstore_KM_mean_Jm3s=float(km_diag.get('Pstore_KM_mean', np.nan)),
        storage_cap_active_frac=float(km_diag.get('storage_cap_active_frac', np.nan)),
        storage_violation_max_Jm3s=float(km_diag.get('storage_violation_max', np.nan)),
        v_orowan_mean=float(km_diag.get('v_orowan_mean', np.nan)),
        v_orowan_p95=float(km_diag.get('v_orowan_p95', np.nan)),
        v_orowan_max=float(km_diag.get('v_orowan_max', np.nan)),
        v_adv_mean=float(km_diag.get('v_adv_mean', np.nan)),
        v_adv_max=float(km_diag.get('v_adv_max', np.nan)),
        v_cfl_active_frac=float(km_diag.get('v_cfl_active_frac', np.nan)),
        gdot_abs_mean=float(km_diag.get('gdot_abs_mean', np.nan)),
        gdot_abs_max=float(km_diag.get('gdot_abs_max', np.nan)),
        collective_enabled=float(km_diag.get('collective_enabled', 0.0)),
        collective_nc_mean=float(km_diag.get('collective_nc_mean', np.nan)),
        collective_nc_p95=float(km_diag.get('collective_nc_p95', np.nan)),
        collective_m_mean=float(km_diag.get('collective_m_mean', np.nan)),
        collective_m_p95=float(km_diag.get('collective_m_p95', np.nan)),
        collective_ell_nm_mean=float(km_diag.get('collective_ell_nm_mean', np.nan)),
        collective_ell_nm_p95=float(km_diag.get('collective_ell_nm_p95', np.nan)),
        collective_tc_mean=float(km_diag.get('collective_tc_mean', np.nan)),
        collective_lambda_mean=float(km_diag.get('collective_lambda_mean', np.nan)),
        collective_lambda_p95=float(km_diag.get('collective_lambda_p95', np.nan)),
        collective_Pcomplete_mean=float(km_diag.get('collective_Pcomplete_mean', np.nan)),
        collective_suppression_mean=float(km_diag.get('collective_suppression_mean', np.nan)),
        collective_suppression_p05=float(km_diag.get('collective_suppression_p05', np.nan)),
        collective_gdot_ind_mean=float(km_diag.get('collective_gdot_ind_mean', np.nan)),
        collective_gdot_coll_mean=float(km_diag.get('collective_gdot_coll_mean', np.nan)),
        collective_margin_MPa_mean=float(km_diag.get('collective_margin_MPa_mean', np.nan)),
        collective_activity_mean=float(km_diag.get('collective_activity_mean', np.nan)),
        collective_activity_max=float(km_diag.get('collective_activity_max', np.nan)),
        collective_wall_src_mean=float(km_diag.get('collective_wall_src_mean', np.nan)),
        collective_lock_src_mean=float(km_diag.get('collective_lock_src_mean', np.nan)),
        collective_activity_memory_mean=float(km_diag.get('collective_activity_memory_mean', np.nan)),
        collective_activity_memory_max=float(km_diag.get('collective_activity_memory_max', np.nan)),
        finite_loading=float(km_diag.get('finite_loading', np.nan)),
        finite_work_scale=float(km_diag.get('finite_work_scale', np.nan)),
        finite_rate_allow=float(km_diag.get('finite_rate_allow', np.nan)),
        finite_heat_macro_power=float(km_diag.get('finite_heat_macro_power', np.nan)),
        finite_heat_raw_power_mean=float(km_diag.get('finite_heat_raw_power_mean', np.nan)),
        heat_process_zone_active=float(heat_diag.get('process_zone_active', np.nan)),
        heat_process_zone_sigma_px=float(heat_diag.get('process_zone_sigma_px', np.nan)),
        heat_process_zone_raw_qdot_max_MWm3=float(heat_diag.get('process_zone_raw_qdot_max', np.nan)/1e6),
        heat_process_zone_smooth_qdot_max_MWm3=float(heat_diag.get('process_zone_smooth_qdot_max', np.nan)/1e6),
        gb_blocked_active=float(heat_diag.get('gb_blocked_active', np.nan)),
        gb_blocked_frac_mean=float(heat_diag.get('gb_blocked_frac_mean', np.nan)),
        gb_blocked_frac_max=float(heat_diag.get('gb_blocked_frac_max', np.nan)),
        gb_blocked_power_mean_MWm3=float(heat_diag.get('gb_blocked_power_mean', np.nan)/1e6),
        gb_blocked_power_max_MWm3=float(heat_diag.get('gb_blocked_power_max', np.nan)/1e6),
        gb_blocked_stored_power_mean_MWm3=float(heat_diag.get('gb_blocked_stored_power_mean', np.nan)/1e6),
        gb_blocked_heat_scale_min=float(heat_diag.get('gb_blocked_heat_scale_min', np.nan)),
        gb_blocked_stored_rhoGB_mean=float(heat_diag.get('gb_blocked_stored_rhoGB_mean', np.nan)),
        edot_p11_mean=float(km_diag.get('edot_p11_mean', np.nan)),
        stress_dot_MPa_s=float(km_diag.get('stress_dot_MPa_s', np.nan)),
        sigma_bar_old_MPa=float(km_diag.get('sigma_bar_old_MPa', np.nan)),
        slip_dt_active=float(km_diag.get('slip_dt_active', np.nan)),
        dt_after_slip=float(km_diag.get('dt_after_slip', np.nan)),
        nflip=int(nflip),
        nuc_candidates=int(nuc_diag.get('cand', 0)),
        nuc_best_dF=float(nuc_diag.get('best_dF', np.nan)),
        nuc_best_score=float(nuc_diag.get('best_score', np.nan)),
        nuc_event=int(nuc_diag.get('event', 0)),
        nuc_hazard_max=float(nuc_diag.get('hazard_max', np.nan)),
        nuc_Hmax=float(nuc_diag.get('Hmax', np.nan)),
        nuc_barrier_best_eV=float(nuc_diag.get('barrier_best_eV', np.nan)),
        nuc_theta_best_deg=float(nuc_diag.get('theta_best_deg', np.nan)),
        nuc_theta_max_deg=float(nuc_diag.get('theta_max_deg', np.nan)),
        nuc_R_best_um=float(nuc_diag.get('R_best_um', np.nan)),
        nuc_spinodal_frac=float(nuc_diag.get('spinodal_frac', np.nan)),
    )
    row.update(_asb_band_metrics(T, rho, heat_diag=heat_diag, km_diag=km_diag, psi_lat=psi_lat))
    # Per-slip signed populations; fixed field names keep the CSV schema stable.
    for ss in range(min(nSlip, 4)):
        ks = rp[:,:,ss] - rm[:,:,ss]
        row[f'kappa_s{ss}_mean'] = float(np.nanmean(ks))
        row[f'kappa_s{ss}_abs_mean'] = float(np.nanmean(np.abs(ks)))
        row[f'kappa_s{ss}_frac_mean'] = float(np.nanmean(np.abs(ks)/np.maximum(rho, P['rho_min'])))
    row.update(eterms)
    return row

# ================================================================
# 11. MAIN LOOP
# ================================================================
# v28b base timestep selection.  For strain-rate sweeps, a fixed dt gives wildly
# different total strain at different rates.  Default to a fixed strain increment
# per step unless dt_base is explicitly supplied.
if P.get('dt_base', None) is not None:
    _dt_runtime = float(P.get('dt_base'))
    _dt_mode_runtime = 'explicit_dt_base'
else:
    _mode = str(P.get('dt_base_mode', 'strain_increment')).lower()
    if _mode in ('strain_increment', 'strain', 'depsilon', 'deps'):
        _deps = max(float(P.get('dt_strain_step', 1.0e-3)), 1.0e-300)
        _ed = max(abs(float(P.get('edot_app', 0.0))), 1.0e-300)
        _dt_runtime = _deps / _ed
        _dt_mode_runtime = f'strain_increment(deps={_deps:.3g})'
    else:
        _dt_runtime = float(P.get('dt', 1.0))
        _dt_mode_runtime = 'legacy_fixed_dt'
_dt_min_base = float(P.get('dt_base_min', 0.0) or 0.0)
if _dt_min_base > 0:
    _dt_runtime = max(_dt_runtime, _dt_min_base)
_dt_max_base = P.get('dt_base_max', None)
if _dt_max_base is not None:
    _dt_runtime = min(_dt_runtime, float(_dt_max_base))
P['_dt_base_runtime'] = float(_dt_runtime)
P['_dt_mode_runtime'] = _dt_mode_runtime
P['dt'] = P['_dt_base_runtime']
print(f"\nRunning {P['nSteps']} steps, dt_base={P['_dt_base_runtime']:.3e}, "
      f"edot={P['edot_app']:.0e}, dt_mode={P['_dt_mode_runtime']}, "
      f"deps/step={P['_dt_base_runtime']*abs(float(P.get('edot_app',0.0))):.3e}")
print("="*72)
print(f"Mode switches: CH={P.get('use_ch_step', True)}  freeze_KWC={P.get('freeze_kwc_eta', False)}  "
      f"freeze_GND={not P.get('use_signed_gnd_feedback', True)}  mobility_barrier={P.get('use_ch_mobility_barrier', True)}  "
      f"plasticity_M={P.get('use_ch_plasticity_mobility', False)}  rho_eta={P.get('use_rho_eta_coupling', True)}")
print(f"v25 thermodynamic potential: mode={P.get('potential_mode','thermo_stored')}  arr_scale={P.get('arrhenius_phi_scale',1.0):.3g}  "
      f"work_len={P.get('arrhenius_work_length_mode','X_peak')}  "
      f"Taylor_conc={P.get('arrhenius_phi_use_taylor_concentration', True)}  "
      f"entropy={P.get('use_potential_entropy', True)}  ordering={P.get('use_potential_ordering', True)}")
print(f"v25 barrier: model={ATpot.barrier_name()}  G00={P['expf_G00_eV']:.3g}eV  "
      f"sigc0={P['expf_sigc0']/1e6:.0f}MPa  a={P['expf_a']:.3g}  n={P['expf_n']:.3g}  "
      f"floor={P['expf_floor']:.2g}  sigma_ratio_cap={P.get('expf_sigma_ratio_cap',6.0):.2g}")
print(f"v27c collective Taylor: enabled={P.get('use_collective_taylor', False)}  "
      f"mode={P.get('collective_taylor_mode','multi_hit')}  closure={P.get('collective_rate_closure','domain_count')}  "
      f"tc={P.get('collective_tc',1e-9):.1e}s  eta_m={P.get('collective_eta_m',0.25):.2g}  "
      f"m_max={P.get('collective_m_max',8.0):.2g}  domain_power={P.get('collective_domain_power',1.0):.2g}  "
      f"min_sup={P.get('collective_min_suppression',0.0):.2g}  ell_max={P.get('collective_ell_max_um',0.10):.2g}um  "
      f"tau0={P.get('collective_tau0_MPa',100.0):.2g}MPa")
print(f"Stored energy: mu(T)={P.get('use_temperature_dependent_Estar', True)}  "
      f"Estar_alpha={P.get('Estar_use_alpha', False)}  "
      f"storage_cap={P.get('use_storage_energy_cap', True)}  "
      f"stored_fraction={P.get('stored_work_fraction', None) if P.get('stored_work_fraction', None) is not None else max(0.0, 1.0-float(P.get('taylor_quinney',0.9))):.3g}")
print(f"v27c rho-state partition: enabled={P.get('use_rho_state_partition', False)}  "
      f"store_to_forest={P.get('rho_state_store_to_forest', True)}  "
      f"obstacle={P.get('rho_state_obstacle_mode','forest_wall_gnd')}  "
      f"rho_ch_ref={_rho_ch_scale():.2e}  rho_peak_ind={rho_c:.2e}  "
      f"wall_conv={P.get('collective_wall_conversion',0.0):.2g}  "
      f"heat_A={P.get('collective_heat_partition_weight',0.0):.2g}")
print(f"Arrhenius kinetic gate: enabled={P.get('use_arrhenius_kinetic_instability', False)}  "
      f"mode={P.get('arrhenius_hazard_gate_mode','diagnostic')}  floor={P.get('arrhenius_hazard_gate_floor',1.0):.2g}")
print(f"Hazard activity prefactor: enabled={P.get('use_plastic_activity_hazard_prefactor', True)}  "
      f"mode={P.get('hazard_activity_prefactor_mode','gdot')}  floor={P.get('hazard_activity_floor',1.0e-4):.1e}  "
      f"cap={P.get('hazard_activity_cap',50.0):.3g}  power={P.get('hazard_activity_power',1.0):.2g}")
print(f"Heat update: source={'local' if P.get('use_local_heat_source', True) else 'global'}  "
      f"mode={P.get('heat_update_mode')}  k={P.get('k_thermal')} W/m/K  "
      f"bath={P.get('T_bath_coupling')} W/m^3/K")
print(f"v32 loading: finite_elastic={P.get('use_finite_elastic_loading', False)}  "
      f"Eeff={'E_ps' if P.get('finite_loading_Eeff', None) is None else P.get('finite_loading_Eeff')}  "
      f"activity_memory={P.get('use_collective_activity_memory', False)}  "
      f"mode={P.get('collective_activity_memory_mode', 'crystallographic_local')}  "
      f"tauA={P.get('collective_activity_tau', None)}s  "
      f"rate_w={P.get('collective_activity_rate_weight', None)}  "
      f"Dpar/Dperp={P.get('collective_activity_D_parallel', None)}/{P.get('collective_activity_D_perp', None)} m2/s")
print(f"Thermal controls: adaptive_dt={P.get('use_adaptive_thermal_dt', False)}  "
      f"max_dT_step={P.get('thermal_dt_max_dT_step', None)}K  "
      f"log_change={P.get('thermal_dt_log_change', None)}  "
      f"validity_stop={P.get('use_thermal_validity_stop', False)}  "
      f"Tmax_limit={P.get('thermal_validity_Tmax_K', None)}K  "
      f"Tmean_limit={P.get('thermal_validity_Tmean_K', None)}K")
print(f"Mechanical validity: stop={P.get('use_mechanical_validity_stop', False)}  "
      f"mode={P.get('mechanical_validity_mode', 'fit_or_ideal')}  "
      f"fit_fraction={P.get('mechanical_validity_fit_fraction', 1.0)}  "
      f"ideal_mu_frac={P.get('mechanical_validity_ideal_mu_frac', 0.12)}")
if P.get('asb_print_thermal_scales', True):
    _tsc = _thermal_asb_scale_report()
    print(f"ASB thermal scales: alpha={_tsc['alpha']:.2e} m2/s  "
          f"lambda_edot≈{_tsc['lambda_edot']*1e6:.1f} um  "
          f"ell_bath≈{_tsc['ell_bath']*1e6 if np.isfinite(_tsc['ell_bath']) else np.inf:.1f} um  "
          f"L={P['L_phys']*1e6:.1f} um")
    print(f"                   for target λ={_tsc['target']*1e6:.1f} um: "
          f"k≈{_tsc['k_for_target']:.2g} W/m/K at current edot, "
          f"bath≈{_tsc['h_for_target']:.2g} W/m3/K at current k")
print(f"Diffusive recovery: enabled={P.get('use_lattice_diffusive_recovery', True)}  "
      f"K={P.get('diffrec_K'):.2e}  D0={P.get('diffrec_D0_m2_s'):.1e} m^2/s  "
      f"Q={P.get('diffrec_Q_eV'):.2f} eV  alloy_factor={P.get('diffrec_alloy_D_factor'):.2g}  "
      f"max_frac={P.get('diffrec_max_frac_step'):.2g}")
print(f"v9 controls: ch_base_scale={P.get('ch_base_scale', float('nan'))}  "
      f"lowrho_penalty={P.get('use_lowrho_penalty', P.get('lowrho_penalty', False))} "
      f"rho_min={P.get('rho_min', float('nan')):.1e}  "
      f"L_ac={P.get('L_ac', float('nan')):.2e}  "
      f"plastic_spin_weight={P.get('plastic_spin_weight', float('nan'))}")

if P.get("use_temperature_dependent_gb_mobility", False):
    try:
        _gb_fac_1300 = _gb_mobility_factor_from_T(np.array([1300.0]))
        _gb_fac_1500 = _gb_mobility_factor_from_T(np.array([1500.0]))
        _gb_fac_1800 = _gb_mobility_factor_from_T(np.array([1800.0]))
        _gb_fac_msg = (
            f"factor(1300/1500/1800K)="
            f"{float(np.ravel(_gb_fac_1300)[0]):.3g}/"
            f"{float(np.ravel(_gb_fac_1500)[0]):.3g}/"
            f"{float(np.ravel(_gb_fac_1800)[0]):.3g}"
        )
    except Exception as _e:
        _gb_fac_msg = f"factor diagnostic failed: {_e}"
    print(f"GB mobility active check: T-dependent=True "
          f"Q={P.get('gb_mobility_Q_eV', None)}eV "
          f"Tref={P.get('gb_mobility_Tref', None) or P.get('T0', 1300.0)}K "
          f"{_gb_fac_msg} "
          f"local_T={P.get('gb_mobility_use_local_T', True)} "
          f"rhoGB_relax={P.get('gb_mobility_apply_to_rhoGB_relax', True)}")
else:
    print("GB mobility active check: T-dependent=False")

print(f"GB-HP source/transmission: enabled={P.get('use_gb_hp_source_sink', True)}  A_mult={P.get('gb_hp_A_mult', 7.0):.2g}  "
      f"xi_source={P.get('gb_hp_source_xi_prefactor', 8.0):.2g} screened={P.get('gb_hp_source_use_backstress_screen', True)}  "
      f"xi_trans_cap={P.get('gb_hp_trans_xi_cap', P.get('gb_hp_xi_cap', 80.0)):.1f}  "
      f"source_strength={P.get('gb_hp_source_strength', 0.08):.2g}  "
      f"trans_strength={P.get('gb_hp_transmission_strength', 0.60):.2g}  weak_sink={P.get('gb_hp_weak_sink_strength', 0.015):.2g}")
print(f"v32 GB slip transmission: enabled={P.get('use_gb_slip_transmission_barrier', True)}  "
      f"Gmis={P.get('gb_trans_misorientation_barrier_eV', 0.25):.2g}eV  "
      f"Gres={P.get('gb_trans_residual_barrier_eV', 0.55):.2g}eV  "
      f"out={P.get('gb_trans_outgoing_mode', 'same_index')}  "
      f"poly_spread={P.get('poly_spread_deg', np.nan):.1f}deg min_mis={P.get('poly_min_mis_deg', np.nan):.1f}deg  "
      f"res_rot={P.get('use_gb_residual_rotation', True)} rate={P.get('gb_residual_rotation_rate', 0.0):.2g}/s")
print(f"Topology relabel: enabled={P.get('use_component_relabel', True)}  interval={P.get('component_relabel_interval',25)}  "
      f"min_px={P.get('component_relabel_min_px',24)}  max_splits={P.get('component_relabel_max_splits_per_step',8)}  "
      f"pure={P.get('component_relabel_require_pure', True)}  grain_max={P.get('grain_max')}")
print(f"v13 GB support: pairwise={P.get('use_pairwise_gb_support', True)}  hard_pure={P.get('use_purity_aware_hard_gb_edges', True)}  "
      f"eta_min={P.get('gb_hard_eta_min',0.65):.2f}  second_frac={P.get('gb_hard_second_frac_max',0.35):.2f}")
print(f"v11 hazard nucleation: enabled={P.get('use_hazard_nucleation', True) and not P.get('disable_nucleation', False)}  "
      f"attempt={P.get('nuc_attempt_freq',1e12):.1e}/s  gamma0={P.get('nuc_gamma_GB',0.5):.2f} J/m^2  "
      f"GND_feed={P.get('nuc_gnd_feed_efficiency',0.5):.2f}  theta_m={P.get('nuc_rs_theta_m_deg',15.0):.1f} deg")
print(f"Output: save_interval={P.get('save_interval')}  plot_interval={P.get('plot_interval')}  "
      f"main_png={P.get('save_main_panels', True)} signed_png={P.get('save_signed_panels', False)}  "
      f"plot_dpi={P.get('plot_dpi')} max_png={P.get('max_saved_png_frames')}")

sigma_bar = ATpot.sigma_inv(rho0, P['T0'], P['edot_app']) if P.get('finite_loading_init_from_inverse', True) else 0.0
out = Path(os.environ.get('DRX_OUTDIR', './output')); out.mkdir(exist_ok=True, parents=True)

_png_saved_count = 0
def _safe_savefig(fig, fname, dpi=None):
    """Save a figure without aborting a long run if disk space is exhausted.

    Returns True if the file was written.  On OSError, closes the figure, prints
    a warning, and optionally disables further PNG plotting.
    """
    global _png_saved_count
    try:
        fig.savefig(fname, dpi=(P.get('plot_dpi', 100) if dpi is None else dpi))
        _png_saved_count += 1
        plt.close(fig)
        return True
    except OSError as exc:
        plt.close(fig)
        print(f"WARNING: could not save figure {fname}: {exc}")
        if P.get('disable_plots_on_save_error', True):
            P['save_main_panels'] = False
            P['save_signed_panels'] = False
            print("WARNING: disabling further PNG plot saves; NPZ/CSV output will continue if possible.")
        return False


def _save_restart_checkpoint(step_local):
    """Save exact continuation checkpoint for branch/sweep workflows."""
    if not P.get('write_restart_npz', True):
        return None
    prefix = str(P.get('restart_prefix', 'drx_v25_restart'))
    fname = out / f"{prefix}_{int(step_local):06d}.npz"
    try:
        np.savez_compressed(
            fname,
            rho=rho, rp=rp, rm=rm,
            rho_forest=globals().get('rho_forest', np.zeros((Nx, Ny, nSlip))),
            rho_wall=globals().get('rho_wall', np.zeros((Nx, Ny))),
            rho_mobile=_rho_mobile_field(rp, rm),
            collective_activity_memory=globals().get('collective_activity_memory', np.zeros((Nx, Ny))),
            eta=eta[:, :, :Ng], psi_gv=psi_gv[:Ng], Ng=np.array(Ng, dtype=np.int32),
            lab=lab, psi_lat=psi_lat, psi_plastic=psi_plastic, T=T, rho_GB=rho_GB,
            gamma_slip=gamma_slip, eps_p=eps_p, E_tot=E_tot,
            H_nuc=H_nuc, E_nuc=E_nuc, kappa_tot=np.sum(rp-rm, axis=2),
            rho_c=np.array(rho_c), rho_peak_ind=np.array(getattr(ATpot, 'rho_peak_ind', rho_c)), rho_ch_ref=np.array(_rho_ch_scale()), sigma_bar=np.array(sigma_bar), step=np.array(step_local, dtype=np.int32),
            grain_origin_lineage=grain_origin_lineage[:Ng],
            grain_birth_mechanism=grain_birth_mechanism[:Ng],
            grain_parent=grain_parent[:Ng],
            grain_birth_step=grain_birth_step[:Ng],
            grain_birth_x=grain_birth_x[:Ng],
            grain_birth_y=grain_birth_y[:Ng],
            grain_birth_area_px=grain_birth_area_px[:Ng],
            grain_birth_theta_deg=grain_birth_theta_deg[:Ng],
            grain_birth_theta_max_deg=grain_birth_theta_max_deg[:Ng],
            grain_birth_R_um=grain_birth_R_um[:Ng],
            grain_birth_barrier_eV=grain_birth_barrier_eV[:Ng],
            rng_nuc_state_json=np.array(_rng_state_to_json(_rng_nuc)),
            P_json=np.array(json.dumps(P, default=str)),
        )
        return fname
    except OSError as exc:
        print(f"WARNING: could not save restart checkpoint {fname}: {exc}")
        return None

_diag_csv_fh = None
_diag_csv_writer = None
_last_nuc_diag = {'cand': 0, 'best_dF': np.nan, 'best_score': np.nan}
_last_topology_diag = {'splits': 0, 'unassigned': 0, 'topo_components': int(Ng), 'multi_component_labels': 0, 'max_components_per_label': 1}
if P.get('write_diag_csv', True):
    _diag_csv_path = out / P.get('diag_csv_name', 'drx_v8_gb_hp_source_sink_diagnostics.csv')
    _diag_csv_fh = open(_diag_csv_path, 'w', newline='')
    print(f"Diagnostics CSV: {_diag_csv_path}")
tw0 = _wtime.time()
sim_time = 0.0
_last_dt_report_bucket = None
_stop_run = False
_mode0 = str(P.get('collective_activity_memory_mode', 'crystallographic_local')).lower()
if _mode0 in ('crystallographic_local', 'crystallographic', 'grain_slip', 'crystal'):
    collective_activity_memory = np.zeros((Nx, Ny, nSlip), dtype=float)
else:
    collective_activity_memory = np.zeros((Nx, Ny), dtype=float)

for n in range(P['nSteps']):
    t = sim_time
    P['dt'] = P['_dt_base_runtime']
    current_step_for_provenance = int(n)
    _grain_step_topology_births = 0
    _grain_step_hazard_births = 0
    km_store_mean = 0.0
    km_anni_mean = 0.0
    ch_delta_abs_mean = 0.0
    ch_delta_std = 0.0
    ac_eta_delta_mean = 0.0
    gnd_transfer_mean = 0.0
    rhoGB_delta_mean = 0.0
    gb_hp_diag = {'src_mean': 0.0, 'sink_mean': 0.0, 'rate_mean': 0.0, 'xi_mean': np.nan}
    heat_diag = {'qdot_mean': 0.0, 'qdot_max': 0.0, 'qdot_std': 0.0, 'dT_mean_step': 0.0, 'dT_max_step': 0.0, 'mode': 'uncomputed'}
    thermal_dt_diag = {'active': 0, 'dt_base': float(P.get('_dt_base_runtime', P.get('dt', 1.0))),
                       'dt': float(P.get('dt', 1.0)), 'dT_allow': np.nan,
                       'dT_macro_pred': np.nan, 'n_sub_equiv': 1.0, 'QeV': np.nan}
    finite_loading = bool(P.get('use_finite_elastic_loading', False))
    km_diag = {'k2_eff_mean': np.nan, 'k2_eff_max': np.nan, 'k2_eff_min': np.nan,
               'diffrec_mean': 0.0, 'diffrec_D_mean': np.nan, 'diffrec_arg_mean': np.nan, 'diffrec_arg_max': np.nan,
               'Pplastic_mean': np.nan, 'Pplastic_max': np.nan, 'Pstore_allowed_mean': np.nan,
               'Pstore_KM_mean': np.nan, 'storage_cap_active_frac': np.nan,
               'storage_violation_max': np.nan, 'v_orowan_mean': np.nan, 'v_orowan_p95': np.nan,
               'v_orowan_max': np.nan, 'v_adv_mean': np.nan, 'v_adv_max': np.nan,
               'v_cfl_active_frac': np.nan, 'gdot_abs_mean': np.nan, 'gdot_abs_max': np.nan}

    # --- Temperature-dependent potential refresh and grain-slaved orientation ---
    if P.get('update_potential_with_temperature', True) and (n % max(int(P.get('potential_update_interval', 50)), 1) == 0):
        ATpot.build(finite_clipped_T_mean(T) if 'T' in globals() else P['T0'], P['edot_app'])
        rho_c = ATpot.rho_c
    if P.get('use_grain_slaved_orientation', True):
        psi_lat = reconstruct_psi_lat(eta, psi_gv, psi_plastic, Ng)

    # --- Slip geometry ---
    ang, sv, nv, Sch, s11 = build_slip(psi_lat)
    _gb_trans_fields_step = None

    # --- Backstress (Taylor type from signed content) ---
    tau_bk = np.zeros((Nx,Ny,nSlip))
    for s in range(nSlip):
        ks = rp[:,:,s]-rm[:,:,s]
        rs = np.maximum(rp[:,:,s]+rm[:,:,s], 2*P['rho_min'])
        delta = 1.0/np.sqrt(np.maximum(rs, P['rho_min']))
        tau_bk[:,:,s] = mu_iso*P['b']*ks*delta*0.1

    # --- Macro loading law ---
    # Legacy mode solves an algebraic stress that enforces <edot_p>=edot_app.
    # v28 finite-loading mode instead keeps sigma_bar as a dynamic variable and
    # updates it later from sigma_dot=E_eff*(edot_app-<edot_p>).
    if finite_loading:
        sb_new = sigma_bar
        gdot = np.zeros((Nx, Ny, nSlip))
    else:
        sb_new, gdot = macro_bisect(P['edot_app'], tau_bk, rp, rm, T, s11, Sch)
        alpha_sb = 0.3 if n > 0 else 1.0
        sigma_bar = (1-alpha_sb)*sigma_bar + alpha_sb*sb_new

        # v26e: stop if the imposed-rate solve requires a stress outside the
        # constitutive/solid-strength validity range.
        _bad_sig, _bad_sig_reason = _mechanical_validity_exceeded(sigma_bar, T)
        if _bad_sig:
            print(f"MECHANICAL VALIDITY STOP at step {n}: {_bad_sig_reason}; "
                  f"{P.get('mechanical_validity_stop_reason', '')}")
            if P.get('mechanical_validity_save_restart', True):
                _save_restart_checkpoint(n)
            _stop_run = True
            break

    # v26d/e: thermo-Arrhenius adaptive mechanics step. This resolves, rather
    # than caps, rapid heating/softening. All explicit updates below use P['dt'].
    dt_eff, thermal_dt_diag = _adaptive_thermal_dt(sigma_bar, T)
    P['dt'] = dt_eff
    if P.get('thermal_dt_print_changes', True) and thermal_dt_diag.get('active', 0):
        bucket = int(np.ceil(np.log10(max(thermal_dt_diag.get('n_sub_equiv', 1.0), 1.0))))
        if bucket != _last_dt_report_bucket or (n % max(int(P.get('diag_interval', 50)), 1) == 0):
            print(f"  [{n}] adaptive dt: {P['dt']:.2e}s (base {P['_dt_base_runtime']:.2e}s, "
                  f"sub≈{thermal_dt_diag.get('n_sub_equiv', 1.0):.1f}, "
                  f"ΔTbase≈{thermal_dt_diag.get('dT_macro_pred', np.nan):.2g}K, "
                  f"allow≈{thermal_dt_diag.get('dT_allow', np.nan):.2g}K)")
            _last_dt_report_bucket = bucket

    # --- Heterogeneous stress (MS) ---
    ep_mean = np.array([[eps_p[:,:,0,0].mean(), eps_p[:,:,0,1].mean()],
                         [eps_p[:,:,0,1].mean(), eps_p[:,:,1,1].mean()]])
    e11 = sigma_bar/max(E_ps,1)
    e22 = -nu_ps*e11
    ebar = np.array([[e11+ep_mean[0,0], ep_mean[0,1]],[ep_mean[0,1], e22+ep_mean[1,1]]])
    sig_f, eps_f = ms_solve(eps_p, ebar, P['ms_iters'])

    ah = P['hs_alpha']
    sig_u = np.zeros_like(sig_f); sig_u[:,:,0,0] = sigma_bar
    sig_use = ah*sig_f + (1-ah)*sig_u

    # recompute gamma_dot with heterogeneous stress
    tau_resolved = np.zeros((Nx, Ny, nSlip))
    tau_effective = np.zeros((Nx, Ny, nSlip))
    _gcf = float(P.get('local_gdot_cap_factor', 10.0))
    gdot_cap = (_gcf * abs(P['edot_app']) / max(_P11_mean, 1e-6)) if _gcf > 0.0 else np.inf
    rho_total_for_gdot = _rho_total_state(rp, rm, rho_forest, rho_wall)
    kappa_for_gdot = np.sum(rp-rm, axis=2)
    for s in range(nSlip):
        rs = _rho_obstacle_for_slip(s, rp, rm, rho_total_for_gdot, rho_forest, rho_wall, rho_GB, kappa_for_gdot)
        tau = np.zeros((Nx,Ny))
        for i in range(2):
            for j in range(2): tau += sig_use[:,:,i,j]*Sch[:,:,s,i,j]
        tnet = tau - tau_bk[:,:,s]
        tau_resolved[:,:,s] = tau
        tau_effective[:,:,s] = tnet
        mag = ATpot.gdot(np.abs(tnet)*drive_sc, rs, T)
        # v29: persistent activity is allowed to modify the local correlated-slip
        # susceptibility, so heat still comes from tau*gdot rather than an
        # independent activity-weighted heat source.  Default mode is
        # per-slip/local, not fixed lab-frame band diffusion.
        if P.get('use_collective_activity_memory', False):
            try:
                Aprev_s = _activity_memory_for_slip(collective_activity_memory, s)
                aw = float(P.get('collective_activity_rate_weight', 0.0))
                ap = max(float(P.get('collective_activity_rate_power', 1.0)), 1e-12)
                if aw != 0.0:
                    mag = mag * (1.0 + aw*np.clip(Aprev_s, 0.0, 1.0)**ap)
            except Exception:
                pass
        # v30: GB slip-transmission barrier.  This suppresses local slip-rate
        # only at GB cores when the outgoing slip system would leave a large
        # residual Burgers vector or cross a high-misorientation boundary.
        if P.get('use_gb_slip_transmission_barrier', True):
            try:
                if _gb_trans_fields_step is None:
                    _gb_trans_fields_step = _gb_slip_transmission_fields(lab, psi_lat, gb_mask, T)
                mag = mag * _gb_trans_fields_step['factor'][:, :, s]
            except Exception:
                pass
        if np.isfinite(gdot_cap):
            mag = np.minimum(mag, gdot_cap)  # optional numerical safety for legacy rate-control runs
        gdot[:,:,s] = np.sign(tnet)*mag

    # Local plastic strain rate implied by the heterogeneous stress field.
    ed11_loc = np.zeros((Nx, Ny))
    for ss in range(nSlip):
        ed11_loc += gdot[:, :, ss] * Sch[:, :, ss, 0, 0]
    ed11_mean = float(np.nanmean(ed11_loc))
    edot_target = float(P.get('edot_app', 0.0))

    # v31 finite-loading plastic-strain budget.  During one explicit step the
    # plastic strain increment cannot exceed the imposed increment plus the
    # elastic strain that can be released from the current macroscopic stress.
    # This is a work/compatibility constraint, not an ASB gate.
    finite_work_scale = 1.0
    finite_rate_allow = np.inf
    if finite_loading and P.get('use_finite_loading_work_budget', True) and P.get('finite_loading_scale_gdot_to_budget', True):
        Eeff_tmp = P.get('finite_loading_Eeff', None)
        Eeff_tmp = float(Eeff_tmp) if Eeff_tmp is not None else float(E_ps)
        elastic_release_rate = max(float(sigma_bar), 0.0)/max(Eeff_tmp*P['dt'], 1e-300)
        safety = float(np.clip(P.get('finite_loading_work_budget_safety', 0.95), 0.0, 1.0))
        finite_rate_allow = max(float(edot_target), 0.0) + safety*elastic_release_rate
        if np.isfinite(ed11_mean) and ed11_mean > finite_rate_allow > 0.0:
            finite_work_scale = finite_rate_allow/max(ed11_mean, 1e-300)
            gdot *= finite_work_scale
            ed11_loc *= finite_work_scale
            tau_resolved *= 1.0
            tau_effective *= 1.0
            ed11_mean = float(np.nanmean(ed11_loc))

    # Optional local slip-increment timestep criterion.  This is a resolution
    # condition on explicit updates, not a cap on the constitutive rate.
    slip_dt_active = 0.0
    if P.get('use_local_slip_increment_dt', False):
        gdmax = float(np.nanmax(np.abs(gdot))) if np.size(gdot) else 0.0
        dgmax = max(float(P.get('local_slip_max_increment', 2.0e-2)), 1e-300)
        if np.isfinite(gdmax) and gdmax*P['dt'] > dgmax and gdmax > 0.0:
            P['dt'] = max(dgmax/gdmax, float(P.get('thermal_dt_min', 1e-11)))
            slip_dt_active = 1.0

    stress_dot = 0.0
    sigma_bar_old = sigma_bar
    if finite_loading:
        Eeff = P.get('finite_loading_Eeff', None)
        Eeff = float(Eeff) if Eeff is not None else float(E_ps)
        damp = float(P.get('finite_loading_damping', 1.0))
        stress_dot = Eeff * (edot_target - ed11_mean)
        sigma_bar = sigma_bar + damp * stress_dot * P['dt']
        if P.get('finite_loading_nonnegative_stress', True):
            sigma_bar = max(float(sigma_bar), 0.0)
        _bad_sig, _bad_sig_reason = _mechanical_validity_exceeded(sigma_bar, T)
        if _bad_sig:
            print(f"MECHANICAL VALIDITY STOP at step {n}: {_bad_sig_reason}; "
                  f"{P.get('mechanical_validity_stop_reason', '')}")
            if P.get('mechanical_validity_save_restart', True):
                _save_restart_checkpoint(n)
            _stop_run = True
            break
    else:
        # Under exact strain-rate control, the heterogeneous stress re-solve can
        # change the volume-averaged plastic strain rate after macro_bisect().
        if P.get('enforce_macro_rate_after_ms', True):
            if np.isfinite(ed11_mean) and abs(ed11_mean) > 1.0e-300 and ed11_mean * edot_target > 0.0:
                gdot *= (edot_target / ed11_mean)
                ed11_loc *= (edot_target / ed11_mean)
                ed11_mean = float(np.nanmean(ed11_loc))

    km_diag.update(finite_loading=float(finite_loading),
                   edot_p11_mean=ed11_mean,
                   stress_dot_MPa_s=float(stress_dot/1e6),
                   sigma_bar_old_MPa=float(sigma_bar_old/1e6),
                   slip_dt_active=slip_dt_active,
                   dt_after_slip=float(P['dt']))

    # --- v26 collective Taylor diagnostics (observational; no evolution side effects) ---
    collective_diag = {'collective_enabled': 1.0 if ATpot._collective_enabled() else 0.0}
    if P.get('collective_diag', True) and ATpot._collective_enabled():
        _coll_rows = []
        rho_total_for_coll = _rho_total_state(rp, rm, rho_forest, rho_wall)
        kappa_for_coll = np.sum(rp-rm, axis=2)
        for ss in range(nSlip):
            rs = _rho_obstacle_for_slip(ss, rp, rm, rho_total_for_coll, rho_forest, rho_wall, rho_GB, kappa_for_coll)
            seq = np.abs(tau_effective[:,:,ss]) * drive_sc
            _coll_rows.append(ATpot.collective_diag(seq, rs, T))
        if _coll_rows:
            keys = sorted(set().union(*[d.keys() for d in _coll_rows]))
            collective_diag = {}
            for kk in keys:
                vals = [d.get(kk, np.nan) for d in _coll_rows]
                try:
                    collective_diag[kk] = float(np.nanmean(vals))
                except Exception:
                    collective_diag[kk] = np.nan

    # --- Plastic strain update ---
    for s in range(nSlip):
        gamma_slip[:,:,s] += gdot[:,:,s]*P['dt']
        for i in range(2):
            for j in range(2):
                eps_p[:,:,i,j] += gdot[:,:,s]*Sch[:,:,s,i,j]*P['dt']
    E_tot[0,0] += P['edot_app']*P['dt']

    # --- Kocks-Mecking (EXPLICIT KINETICS) ---
    # KM storage/recovery. v26 slip rates may already include collective multi-hit depinning.
    # rho_eq = (k1/k2)^2 sits inside the spinodal, so CH can separate.
    if P.get('KM_recovery_local_T', True):
        k2_eff = _km_k2_from_T(T)
    else:
        k2_eff = k2T  # legacy uniform recovery rate evaluated at T0
    km_diag = {
        'k2_eff_mean': float(np.nanmean(k2_eff)),
        'k2_eff_max': float(np.nanmax(k2_eff)),
        'k2_eff_min': float(np.nanmin(k2_eff)),
    }
    km_diag.update(collective_diag)

    # v22 storage power accounting.  Positive density storage is capped by the
    # stored fraction of local plastic mechanical power, while recovery remains
    # unconstrained by this cap.
    A_E_field_KM = np.maximum(ATpot.Estar_coeff(T), P.get('storage_cap_tiny', 1e-300))
    chi_store = P.get('stored_work_fraction', None)
    if chi_store is None:
        chi_store = max(0.0, 1.0 - float(P.get('taylor_quinney', 0.9)))
    chi_store = float(np.clip(chi_store, 0.0, 1.0))
    Pplastic_total = np.zeros((Nx, Ny))
    Pstore_allowed_total = np.zeros((Nx, Ny))
    Pstore_KM_total = np.zeros((Nx, Ny))
    storage_violation_field = np.zeros((Nx, Ny))
    KM_storage_rate_total = np.zeros((Nx, Ny))
    KM_anni_rate_total = np.zeros((Nx, Ny))
    diffrec_rate_field = np.zeros((Nx, Ny))
    storage_cap_active_count = 0.0
    storage_cap_count = 0

    collective_activity_total = np.zeros((Nx, Ny))
    collective_activity_by_slip = np.zeros((Nx, Ny, nSlip))
    wall_src_total = np.zeros((Nx, Ny))
    forest_lock_total = np.zeros((Nx, Ny))
    rho_mobile_pre_km = _rho_mobile_field(rp, rm)
    rho_total_pre_km = _rho_total_state(rp, rm, rho_forest, rho_wall)
    kappa_pre_km = np.sum(rp-rm, axis=2)

    for s in range(nSlip):
        rps = rp[:,:,s].copy(); rms = rm[:,:,s].copy()
        rfs = rho_forest[:, :, s].copy() if P.get('use_rho_state_partition', False) else np.zeros((Nx, Ny))
        mobile_s = np.maximum(rps+rms, 2*P['rho_min'])
        ag = np.abs(gdot[:,:,s])

        # Forest/obstacle density for KM storage and Taylor barriers.  In v27 this
        # is not simply the mobile density: organized wall/GND/GB content can also
        # constrain motion.
        rf = _rho_obstacle_for_slip(s, rp, rm, rho_total_pre_km, rho_forest, rho_wall, rho_GB, kappa_pre_km)

        A_coll = np.zeros((Nx, Ny))
        if P.get('use_collective_organization', True) and ATpot._collective_enabled():
            try:
                seq_s = np.abs(tau_effective[:, :, s]) * drive_sc
                cf_s = ATpot._collective_fields(seq_s, rf, T)
                A_tmp = _collective_activity_field(cf_s)
                if A_tmp is not None:
                    A_coll = np.asarray(A_tmp, dtype=float)
            except Exception:
                A_coll = np.zeros_like(rf)
        collective_activity_total += A_coll / max(nSlip, 1)
        collective_activity_by_slip[:, :, s] = A_coll

        # storage (proportional to forest obstacle spacing); collective activity
        # increases locking/storage but still respects the plastic-work storage cap.
        stor_raw = P['KM_k1']*np.sqrt(rf)*ag
        if P.get('use_collective_organization', True):
            stor_raw *= (1.0 + float(P.get('collective_storage_boost', 0.0))*A_coll)
        tau_store = tau_effective[:,:,s]
        if P.get('storage_power_use_absolute', True):
            P_slip = np.abs(tau_store) * np.abs(gdot[:,:,s])
        else:
            P_slip = np.maximum(tau_store * gdot[:,:,s], 0.0)
        Pplastic_total += P_slip
        P_allowed = chi_store * P_slip
        Pstore_allowed_total += P_allowed
        if P.get('use_storage_energy_cap', True) and P.get('storage_cap_apply_to_KM', True):
            stor_cap = P_allowed / A_E_field_KM
            active_cap = np.isfinite(stor_raw) & np.isfinite(stor_cap) & (stor_raw > stor_cap)
            storage_cap_active_count += float(np.sum(active_cap))
            storage_cap_count += int(stor_raw.size)
            stor = np.minimum(stor_raw, stor_cap)
            storage_violation_field = np.maximum(storage_violation_field, A_E_field_KM*np.maximum(stor_raw - stor_cap, 0.0))
        else:
            stor = stor_raw
        Pstore_KM_total += A_E_field_KM * np.maximum(stor, 0.0)

        # Collective extra locking transfers mobile line into forest/junction content.
        lock_extra = np.zeros_like(stor)
        if P.get('use_rho_state_partition', False) and P.get('use_collective_organization', True):
            lock_extra = float(P.get('collective_mobile_to_forest', 0.0))*A_coll*mobile_s*ag
            lock_extra = np.minimum(lock_extra, float(P.get('collective_wall_max_frac_step', 0.20))*mobile_s/max(P['dt'], 1e-300))
            forest_lock_total += lock_extra

        # dynamic recovery (per-population).  Mobile content recovers fastest;
        # forest/junction content recovers more slowly; wall content relaxes below.
        anni_p = float(P.get('rho_state_mobile_recovery_factor', 1.0))*k2_eff*rps*ag
        anni_m = float(P.get('rho_state_mobile_recovery_factor', 1.0))*k2_eff*rms*ag
        anni_f = float(P.get('rho_state_forest_recovery_factor', 0.35))*k2_eff*rfs*ag if P.get('use_rho_state_partition', False) else np.zeros_like(rfs)

        # Forest -> wall organization.  Requires collective activity and is enhanced
        # by GND and slip-gradient structure, following the CDD patterning picture.
        wall_src = np.zeros_like(stor)
        if P.get('use_rho_state_partition', False) and P.get('use_collective_organization', True):
            gnd_norm = np.abs(kappa_pre_km)/np.maximum(rho_total_pre_km, P['rho_min'])
            grad_fac = _slip_grad_factor(gamma_slip, s)
            geom = 1.0 + float(P.get('collective_wall_gnd_weight', 1.0))*gnd_norm \
                       + float(P.get('collective_wall_grad_gamma_weight', 0.5))*grad_fac
            wall_src = float(P.get('collective_wall_conversion', 0.20))*A_coll*rfs*ag*geom
            wall_src = np.minimum(wall_src, float(P.get('collective_wall_max_frac_step', 0.20))*np.maximum(rfs, 0.0)/max(P['dt'], 1e-300))
            wall_src_total += wall_src

        KM_storage_rate_total += np.maximum(stor + lock_extra, 0.0)
        KM_anni_rate_total += np.maximum(anni_p + anni_m + anni_f, 0.0)
        km_store_mean += float(np.nanmean(P['dt']*(stor + lock_extra)))/max(nSlip,1)
        km_anni_mean += float(np.nanmean(P['dt']*(anni_p+anni_m+anni_f)))/max(nSlip,1)

        if P.get('use_rho_state_partition', False) and P.get('rho_state_store_to_forest', True):
            # Stored density enters the forest/junction reservoir rather than the
            # mobile carrier pool.  This prevents KM storage from automatically
            # becoming glissile density.
            rp[:,:,s] = np.clip(rps - P['dt']*anni_p - 0.5*P['dt']*lock_extra, P['rho_min'], P['rho_max'])
            rm[:,:,s] = np.clip(rms - P['dt']*anni_m - 0.5*P['dt']*lock_extra, P['rho_min'], P['rho_max'])
            rho_forest[:, :, s] = np.clip(rfs + P['dt']*(stor + lock_extra - anni_f - wall_src), 0.0, P['rho_max'])
        else:
            rp[:,:,s] = np.clip(rps + P['dt']*(0.5*stor - anni_p), P['rho_min'], P['rho_max'])
            rm[:,:,s] = np.clip(rms + P['dt']*(0.5*stor - anni_m), P['rho_min'], P['rho_max'])

    if P.get('use_rho_state_partition', False):
        # Organized wall density relaxes thermally/kinetically but more slowly than
        # mobile content.  This is a state conversion, not a nucleation trigger.
        A_act = np.sum(np.abs(gdot), axis=2)
        tau_wall = max(float(P.get('collective_wall_relax_tau', 2.0e-6)), P['dt'])
        wall_rec = float(P.get('rho_state_wall_recovery_factor', 0.05))*k2_eff*rho_wall*A_act + rho_wall/tau_wall
        wall_rec = np.minimum(wall_rec, float(P.get('collective_wall_max_frac_step', 0.20))*np.maximum(rho_wall, 0.0)/max(P['dt'], 1e-300))
        rho_wall = np.clip(rho_wall + P['dt']*(wall_src_total - wall_rec), 0.0, P['rho_max'])

    km_diag.update(
        Pplastic_mean=float(np.nanmean(Pplastic_total)),
        Pplastic_max=float(np.nanmax(Pplastic_total)),
        Pstore_allowed_mean=float(np.nanmean(Pstore_allowed_total)),
        Pstore_KM_mean=float(np.nanmean(Pstore_KM_total)),
        storage_cap_active_frac=float(storage_cap_active_count/max(storage_cap_count, 1)),
        storage_violation_max=float(np.nanmax(storage_violation_field)),
        collective_activity_mean=float(np.nanmean(collective_activity_total)),
        collective_activity_max=float(np.nanmax(collective_activity_total)),
        collective_wall_src_mean=float(np.nanmean(P['dt']*wall_src_total)),
        collective_lock_src_mean=float(np.nanmean(P['dt']*forest_lock_total)),
    )

    # v20: lattice-diffusion-assisted local recovery added to the traditional
    # KM storage model.  It is a neutral sink: total recoverable density is
    # reduced proportionally across slip/sign populations, so it does not create
    # artificial signed GND.
    rho_after_km = _rho_total_state(rp, rm, rho_forest, rho_wall)
    diffrec_mean = 0.0
    if P.get('use_lattice_diffusive_recovery', True):
        rec_rate, D_L, rec_arg = _lattice_diffusive_recovery_rate(rho_after_km, T)
        drho_rec = P['dt'] * rec_rate
        min_total = 2.0 * nSlip * P['rho_min']
        recoverable = np.maximum(rho_after_km - min_total, 0.0)
        max_frac = float(np.clip(P.get('diffrec_max_frac_step', 0.05), 0.0, 1.0))
        drho_rec = np.minimum(drho_rec, max_frac * recoverable)
        drho_rec = np.minimum(drho_rec, recoverable)
        if P.get('use_rho_state_partition', False):
            mobile_tot = _rho_mobile_field(rp, rm)
            forest_tot = _rho_forest_total_field(rho_forest)
            wall_tot = _rho_wall_field(rho_wall)
            weights = (float(P.get('rho_state_mobile_recovery_factor', 1.0))*mobile_tot
                       + float(P.get('rho_state_forest_recovery_factor', 0.35))*forest_tot
                       + float(P.get('rho_state_wall_recovery_factor', 0.05))*wall_tot)
            weights = np.maximum(weights, 1e-300)
            dmob = drho_rec * float(P.get('rho_state_mobile_recovery_factor', 1.0))*mobile_tot/weights
            dfor = drho_rec * float(P.get('rho_state_forest_recovery_factor', 0.35))*forest_tot/weights
            dwal = drho_rec * float(P.get('rho_state_wall_recovery_factor', 0.05))*wall_tot/weights
            scale_mob = np.where(mobile_tot > min_total, np.maximum(mobile_tot - dmob, min_total)/np.maximum(mobile_tot, min_total), 1.0)
            scale_for = np.where(forest_tot > 0.0, np.maximum(forest_tot - dfor, 0.0)/np.maximum(forest_tot, 1e-300), 1.0)
            scale_wal = np.where(wall_tot > 0.0, np.maximum(wall_tot - dwal, 0.0)/np.maximum(wall_tot, 1e-300), 1.0)
            for ss in range(nSlip):
                rp[:,:,ss] = np.clip(rp[:,:,ss] * scale_mob, P['rho_min'], P['rho_max'])
                rm[:,:,ss] = np.clip(rm[:,:,ss] * scale_mob, P['rho_min'], P['rho_max'])
                rho_forest[:,:,ss] = np.clip(rho_forest[:,:,ss] * scale_for, 0.0, P['rho_max'])
            rho_wall = np.clip(rho_wall * scale_wal, 0.0, P['rho_max'])
        else:
            scale_rec = np.where(rho_after_km > min_total,
                                 np.maximum(rho_after_km - drho_rec, min_total) / np.maximum(rho_after_km, min_total),
                                 1.0)
            for ss in range(nSlip):
                rp[:,:,ss] = np.clip(rp[:,:,ss] * scale_rec, P['rho_min'], P['rho_max'])
                rm[:,:,ss] = np.clip(rm[:,:,ss] * scale_rec, P['rho_min'], P['rho_max'])
        diffrec_rate_field = drho_rec / max(P['dt'], 1e-300)
        diffrec_mean = float(np.nanmean(drho_rec))
        km_diag.update(
            diffrec_mean=diffrec_mean,
            diffrec_D_mean=float(np.nanmean(D_L)),
            diffrec_D_max=float(np.nanmax(D_L)),
            diffrec_arg_mean=float(np.nanmean(rec_arg)),
            diffrec_arg_max=float(np.nanmax(rec_arg)),
        )

    rho = _rho_total_state(rp, rm, rho_forest, rho_wall)

    # --- Orowan advection (EXPLICIT KINETICS) ---
    # v22 interpretation: gdot_s = rho_m,s b v_s, so v_s=|gdot_s|/(b rho_m,s).
    v_orowan_vals = []
    v_adv_vals = []
    v_cfl_active_vals = []
    gdot_abs_vals = []
    if P['use_advection']:
        vcap = P['v_cfl_frac']*dx/P['dt']
        for s in range(nSlip):
            rs = np.maximum(rp[:,:,s]+rm[:,:,s], P['rho_mobile_min'])
            v_orowan = np.abs(gdot[:,:,s]) / np.maximum(P['b']*rs, 1e-300)
            vmag = np.minimum(v_orowan, vcap)
            v_orowan_vals.append(v_orowan.ravel())
            v_adv_vals.append(vmag.ravel())
            v_cfl_active_vals.append((v_orowan > vcap).ravel())
            gdot_abs_vals.append(np.abs(gdot[:,:,s]).ravel())
            vx = vmag*sv[:,:,s,0]; vy = vmag*sv[:,:,s,1]
            rp[:,:,s] = advect(rp[:,:,s], +vx, +vy, P['dt'])
            rm[:,:,s] = advect(rm[:,:,s], -vx, -vy, P['dt'])
        rho = _rho_total_state(rp, rm, rho_forest, rho_wall)
    else:
        for s in range(nSlip):
            rs = np.maximum(rp[:,:,s]+rm[:,:,s], P['rho_mobile_min'])
            v_orowan_vals.append((np.abs(gdot[:,:,s]) / np.maximum(P['b']*rs, 1e-300)).ravel())
            gdot_abs_vals.append(np.abs(gdot[:,:,s]).ravel())
    if v_orowan_vals:
        vv = np.concatenate(v_orowan_vals)
        gg = np.concatenate(gdot_abs_vals)
        km_diag.update(v_orowan_mean=float(np.nanmean(vv)),
                       v_orowan_p95=float(np.nanpercentile(vv, 95)),
                       v_orowan_max=float(np.nanmax(vv)),
                       gdot_abs_mean=float(np.nanmean(gg)),
                       gdot_abs_max=float(np.nanmax(gg)))
        if v_adv_vals:
            va = np.concatenate(v_adv_vals)
            vcfl = np.concatenate(v_cfl_active_vals)
            km_diag.update(v_adv_mean=float(np.nanmean(va)),
                           v_adv_max=float(np.nanmax(va)),
                           v_cfl_active_frac=float(np.nanmean(vcfl)))
    # v25 field-level budget diagnostics for hot-band/ASB branch screening.
    km_diag.update(_gdot_abs_field=np.sum(np.abs(gdot), axis=2),
                   _storage_rate_field=KM_storage_rate_total,
                   _anni_rate_field=KM_anni_rate_total,
                   _diffrec_rate_field=diffrec_rate_field)

    # --- CAHN-HILLIARD (VARIATIONAL: dF/dr) ---
    rho_before_ch = rho.copy()
    rho_scale_ch = _rho_ch_scale()
    if P.get('use_rho_state_partition', False) and str(P.get('rho_state_ch_density_mode', 'structural')).lower() in ['structural', 'forest_wall', 'slow']:
        rho_ch_field = np.maximum(_rho_structural_field(rho_forest, rho_wall), P['rho_min'])
    else:
        rho_ch_field = rho
    r = rho_ch_field/max(rho_scale_ch, P['rho_min'])
    # Current structural fields used by both CH and KWC; updated again after eta.
    kappa_tot_for_ch = np.sum(rp-rm, axis=2)
    gb_mask_for_ch = diffuse_gb_support(eta, lab, Ng)
    Hr_eta, dHdr_eta, rho_eta_precursor, rho_eta_drive = _rho_eta_fields(r, rho, kappa_tot_for_ch, gb_mask_for_ch)
    mu_ch = ATpot.mu_dw(r)
    if P.get('use_rho_eta_coupling', True):
        # f_re = -A * gb_support * H(r) * precursor.  This term makes high-rho
        # regions energetically prefer current/incipient KWC support, while the
        # conjugate AC term below lets high-rho/GND bands create diffuse support.
        mu_ch = mu_ch - float(P.get('rho_eta_mu_strength', 0.0))*gb_mask_for_ch*rho_eta_precursor*dHdr_eta
    if P.get('use_ch_step', True):
        ch_activity = np.sum(np.abs(gdot), axis=2)
        r_new = _ch_step_variable_mobility(r, mu_ch, gb_mask_for_ch, rho_eta_precursor, activity=ch_activity)
        r_new = np.maximum(r_new, P['rho_min']/max(rho_scale_ch, P['rho_min']))
        rho_new_ch = np.maximum(r_new * max(rho_scale_ch, P['rho_min']), P['rho_min'])
        if P.get('use_rho_state_partition', False) and str(P.get('rho_state_ch_redistribute_mode', 'structural_only')).lower() in ['structural_only', 'structural', 'slow']:
            # v27c: CH-like variational segregation acts on the slow structural
            # density (forest/walls), not the mobile carrier population.  Scaling
            # rp/rm by the CH field recreates the old high-density site-prefactor
            # softening and drives the stress to zero.
            rho_struct_before = np.maximum(_rho_structural_field(rho_forest, rho_wall), P['rho_min'])
            scale = rho_new_ch / rho_struct_before
            struct_mass_before = float(np.sum(rho_struct_before))
            for s in range(nSlip):
                rho_forest[:,:,s] *= scale
            rho_wall *= scale
            rho_struct_after = np.maximum(_rho_structural_field(rho_forest, rho_wall), P['rho_min'])
            mass_ratio = struct_mass_before / max(float(rho_struct_after.sum()), 1.0)
            for s in range(nSlip):
                rho_forest[:,:,s] *= mass_ratio
            rho_wall *= mass_ratio
        else:
            scale = rho_new_ch / np.maximum(rho, P['rho_min'])
            rho_total_before = rho.sum()
            for s in range(nSlip):
                rp[:,:,s] *= scale
                rm[:,:,s] *= scale
                if P.get('use_rho_state_partition', False):
                    rho_forest[:,:,s] *= scale
            if P.get('use_rho_state_partition', False):
                rho_wall *= scale
            rho_raw_after_scale = _rho_total_state(rp, rm, rho_forest, rho_wall)
            mass_ratio = rho_total_before / max(float(rho_raw_after_scale.sum()), 1.0)
            for s in range(nSlip):
                rp[:,:,s] *= mass_ratio
                rm[:,:,s] *= mass_ratio
                if P.get('use_rho_state_partition', False):
                    rho_forest[:,:,s] *= mass_ratio
            if P.get('use_rho_state_partition', False):
                rho_wall *= mass_ratio
        rho = _rho_total_state(rp, rm, rho_forest, rho_wall)
    ch_delta = rho - rho_before_ch
    ch_delta_abs_mean = float(np.nanmean(np.abs(ch_delta)))
    ch_delta_std = float(np.nanstd(ch_delta))

    # --- ALLEN-CAHN (VARIATIONAL: dF/deta_i) ---
    eta_before_ac = eta[:, :, :Ng].copy()
    A_E_field = ATpot.Estar_coeff(T) if P.get('use_temperature_dependent_Estar', True) else 0.5*mu_iso*P['b']**2
    Estar = A_E_field * rho  # stored dislocation energy density E*(rho,T)
    sum_eta_sq = np.sum(eta[:,:,:Ng]**2, axis=2)
    # Recompute rho-eta drive after the CH update.  This is the piece that lets
    # high-rho/GND density bands lower the energy by becoming KWC support rather
    # than remaining CH stripes that ignore grain topology.
    if P.get('use_rho_state_partition', False) and str(P.get('rho_state_ch_density_mode', 'structural')).lower() in ['structural', 'forest_wall', 'slow']:
        r_ac = np.maximum(_rho_structural_field(rho_forest, rho_wall), P['rho_min'])/max(_rho_ch_scale(), P['rho_min'])
    else:
        r_ac = rho/max(_rho_ch_scale(), P['rho_min'])
    kappa_for_ac = np.sum(rp-rm, axis=2)
    gb_for_ac = diffuse_gb_support(eta, lab, Ng)
    Hr_ac, dHdr_ac, rho_eta_precursor_ac, rho_eta_drive_ac = _rho_eta_fields(r_ac, rho, kappa_for_ac, gb_for_ac)
    if not P.get('freeze_kwc_eta', False):
        L_ac_eff = P['L_ac'] * _gb_mobility_factor_from_T(T)
        for i in range(Ng):
            if np.max(eta[:,:,i]) < 1e-6: continue
            # functional derivative
            lap_ei = lap(eta[:,:,i])
            other_sq = sum_eta_sq - eta[:,:,i]**2
            Ei = Estar  # no copy needed; read-only in this loop
            E_ref = np.mean(Ei[eta[:,:,i]>0.3]) if np.any(eta[:,:,i]>0.3) else Estar.mean()
            dFdei = -P['kappa_eta']*lap_ei + P['W_eta']*2*eta[:,:,i]*other_sq + 2*eta[:,:,i]*(Ei-E_ref)
            if P.get('use_rho_eta_coupling', True):
                # f_re,AC = -A * D(r,kappa,grad r,GB) * sum_i eta_i(1-eta_i).
                # This is a variational precursor term: it promotes diffuse KWC
                # support only where high density coincides with GND/grad-r/GB
                # structure.  Existing W_eta and kappa_eta then sharpen it.
                dFdei += -float(P.get('rho_eta_ac_strength', 0.0))*rho_eta_drive_ac*(1.0 - 2.0*eta[:,:,i])
            deta_i = -P['dt'] * L_ac_eff * dFdei
            if P.get('use_ac_increment_limiter', True):
                lim = float(P.get('ac_max_abs_step', 0.02))
                deta_i = np.clip(deta_i, -lim, lim)
            eta[:,:,i] = np.clip(eta[:,:,i] + deta_i, 0, 1)
        # Remove numerical tails only; this is a roundoff cleanup, not phase selection.
        tail_zero = float(P.get('eta_tail_zero', 0.0))
        if tail_zero > 0.0:
            eta[:, :, :Ng] = np.where(eta[:, :, :Ng] < tail_zero, 0.0, eta[:, :, :Ng])
        # partition of unity
        es = np.sum(eta[:,:,:Ng],2,keepdims=True)+1e-30
        eta[:,:,:Ng] /= es

    ac_eta_delta_mean = float(np.nanmean(np.abs(eta[:, :, :Ng] - eta_before_ac))) if Ng > 0 else 0.0

    # detect label changes → sweep cleaning
    new_lab = np.argmax(eta[:,:,:Ng],2)
    flip = (new_lab!=lab) & (not P.get('freeze_kwc_eta', False))
    nflip = int(np.sum(flip))
    if nflip > 0:
        # Migrated GB wake: reset mobile density and plastic orientation in swept cells.
        for s in range(nSlip):
            rp[:,:,s] = np.where(flip, P['sweep_wake_rho']/(2*nSlip), rp[:,:,s])
            rm[:,:,s] = np.where(flip, P['sweep_wake_rho']/(2*nSlip), rm[:,:,s])
        if P.get('use_rho_state_partition', False):
            for ss in range(nSlip):
                rho_forest[:,:,ss] = np.where(flip, 0.0, rho_forest[:,:,ss])
            rho_wall = np.where(flip, 0.0, rho_wall)
        psi_plastic = np.where(flip, 0.0, psi_plastic)
        rho = _rho_total_state(rp, rm, rho_forest, rho_wall)
        # A GB sweep changes the local metastable object, so reset cumulative
        # nucleation exposure in swept cells while preserving exposure elsewhere.
        if P.get('use_hazard_nucleation', True):
            H_nuc = np.where(flip, 0.0, H_nuc)
            E_nuc = np.where(flip, _draw_exp_threshold(H_nuc.shape), E_nuc)
        lab = new_lab

    # Topology bookkeeping: split disconnected hard-label components into fresh
    # eta fields.  This occurs after migration/sweep accounting so relabeling is
    # not mistaken for a physical swept-front event.
    if (P.get('use_component_relabel', True) and (n % max(int(P.get('component_relabel_interval', 25)), 1) == 0)
            and (not P.get('freeze_kwc_eta', False))
            and (float(E_tot[0,0]) >= float(P.get('component_relabel_min_strain', 0.0)))):
        eta, psi_gv, Ng, lab, _last_topology_diag = split_disconnected_grain_components(
            eta, psi_gv, Ng, lab, psi_lat=psi_lat, psi_plastic=psi_plastic)
        if _last_topology_diag.get('splits', 0) > 0:
            print(f"  [{n}] topology split: +{_last_topology_diag['splits']} fields "
                  f"(Ng={Ng}, topo={_last_topology_diag.get('topo_components', Ng)}, "
                  f"multi-labels={_last_topology_diag.get('multi_component_labels', 0)})")
    elif P.get('use_component_relabel', True) and (n % max(int(P.get('diag_interval', 25)), 1) == 0):
        _last_topology_diag = {'splits': 0, 'unassigned': 0, **grain_topology_stats(lab, P.get('component_relabel_min_px', 24))}
    # Rebuild the GB support every step from current eta/lab so stale GB fields vanish.
    # In freeze_rhoGB ablations, update only the support mask; leave the stored
    # boundary density untouched so the test is a true frozen-KWC/GB comparison.
    gb_mask_old_for_gnd = gb_mask.copy()
    if P.get('freeze_rhoGB', False):
        gb_mask = diffuse_gb_support(eta, lab, Ng)
    else:
        rho_GB, gb_mask = reset_gb_fields_to_current_topology(rho_GB, eta, lab, Ng)
    if P.get('use_grain_slaved_orientation', True):
        psi_lat = reconstruct_psi_lat(eta, psi_gv, psi_plastic, Ng)

    # --- ORIENTATION (VARIATIONAL + plastic spin) ---
    kappa_tot = np.sum(rp-rm, axis=2)
    Lp12 = np.zeros((Nx,Ny)); Lp21 = np.zeros_like(Lp12)
    for s in range(nSlip):
        Lp12 += gdot[:,:,s]*sv[:,:,s,0]*nv[:,:,s,1]
        Lp21 += gdot[:,:,s]*sv[:,:,s,1]*nv[:,:,s,0]
    Wp12 = 0.5*(Lp12-Lp21)

    dFdk, dFdg, dFdgp = F_comp_derivs(kappa_tot, rho_GB, psi_lat)
    gp = grad_mag(psi_lat)
    gp_s = np.maximum(gp, 1e-12)
    nx_ = ddx(psi_lat)/gp_s; ny_ = ddy(psi_lat)/gp_s
    dFdpsi = -(ddx(dFdgp*nx_)+ddy(dFdgp*ny_))

    # Evolve only the plastic residual; the grain-owned orientation is reconstructed
    # from eta_i psi_i below.  This closes the orientation/phase-field bookkeeping.
    if P.get('freeze_orientation', False):
        dpsi_pl = np.zeros_like(psi_plastic)
    else:
        dpsi_rate = P.get('M_psi_plastic', P['M_psi'])*(-dFdpsi) + P.get('plastic_spin_weight', 0.1)*Wp12
        if P.get('use_gb_residual_rotation', True) and P.get('use_gb_slip_transmission_barrier', True):
            try:
                if _gb_trans_fields_step is None:
                    _gb_trans_fields_step = _gb_slip_transmission_fields(lab, psi_lat, gb_mask, T)
                rho_scale_rot = P.get('gb_residual_rotation_rho_scale', None)
                rho_scale_rot = float(rho_scale_rot) if rho_scale_rot is not None else max(_rho_ch_scale(), P['rho_min'])
                gb_res_w = np.clip(rho_GB/max(rho_scale_rot, P['rho_min']), 0.0, 1.0) * np.clip(gb_mask, 0.0, 1.0)
                if P.get('gb_residual_rotation_requires_net_signed', True):
                    net_frac = _net_signed_gb_fraction(rp, rm, rho_GB, gb_mask)
                    net_pow = max(float(P.get('gb_residual_rotation_net_power', 1.0)), 1e-12)
                    gb_res_w = gb_res_w * np.clip(net_frac, 0.0, 1.0)**net_pow
                rot_raw = gb_res_w*_gb_trans_fields_step.get('rot_drive', 0.0)
                rot_coop = _smooth_gb_field(rot_raw, gb_mask,
                                            length_um=P.get('gb_residual_rotation_smooth_um', 0.0),
                                            passes=P.get('gb_residual_rotation_smooth_passes', 1))
                # Elastic compatibility opposes isolated local rotation; only a
                # cooperatively smoothed rotation drive survives.
                penalty = float(P.get('gb_residual_rotation_elastic_penalty', 0.0))
                if penalty != 0.0:
                    rot_coop = rot_coop - penalty*(psi_plastic - _smooth_gb_field(psi_plastic, gb_mask,
                                                                                  length_um=P.get('gb_residual_rotation_smooth_um', 0.0),
                                                                                  passes=P.get('gb_residual_rotation_smooth_passes', 1)))
                dpsi_rate += float(P.get('gb_residual_rotation_rate', 0.0))*rot_coop
            except Exception:
                pass
        dpsi_lim_deg = float(P.get('gb_residual_rotation_cap_deg_step', 0.15))
        dpsi_pl = np.clip(dpsi_rate*P['dt'], -np.deg2rad(dpsi_lim_deg), np.deg2rad(dpsi_lim_deg))
    psi_plastic = np.clip(psi_plastic + dpsi_pl,
                          -np.deg2rad(P.get('psi_plastic_max_deg', 12.0)),
                          np.deg2rad(P.get('psi_plastic_max_deg', 12.0)))
    psi_lat = reconstruct_psi_lat(eta, psi_gv, psi_plastic, Ng)

    # --- SIGNED-GND FEEDBACK (VARIATIONAL: dF/dkappa drives rho+ <-> rho-) ---
    if P.get('use_signed_gnd_feedback', True):
        # dF/d(kappa_signed) = dF/d|kappa| * sign(kappa).  If kappa is nearly
        # zero, use the local plastic spin as the sign seed so a target wall can form.
        sgn_k = np.sign(kappa_tot)
        sgn_k = np.where(np.abs(kappa_tot) < 1e-6*np.maximum(rho, P['rho_min']), np.sign(Wp12), sgn_k)
        mu_k = dFdk * sgn_k
        norm = max(P['A_alpha']*max(rho_c, P['rho_min']), 1e-30)
        drive = -P.get('signed_gnd_feedback_rate', 0.0)*P['dt']*np.clip(mu_k/norm, -1.0, 1.0)
        # Distribute the signed increment to slip systems whose line direction can
        # support the local wall normal.  This is a minimal Nye-projection closure.
        weights = []
        wsum = np.zeros((Nx,Ny))
        for ss in range(nSlip):
            w = np.abs(sv[:,:,ss,0]*nx_ + sv[:,:,ss,1]*ny_) + P.get('signed_gnd_slip_weight_floor', 0.05)
            weights.append(w); wsum += w
        for ss in range(nSlip):
            rps = rp[:,:,ss]; rms = rm[:,:,ss]
            rs = np.maximum(rps+rms, 2*P['rho_min'])
            dks = drive * (weights[ss]/np.maximum(wsum, 1e-30)) * rs
            lim = P.get('signed_gnd_max_frac_step', 0.02)*rs
            dks = np.clip(dks, -lim, lim)
            gnd_transfer_mean += float(np.nanmean(np.abs(dks)))/max(nSlip,1)
            # dks = delta(rp-rm); transfer between signs preserves total rho.
            pos = dks > 0
            amt = np.zeros_like(dks)
            amt[pos] = np.minimum(0.5*dks[pos], np.maximum(rms[pos]-P['rho_min'], 0.0))
            rps += amt; rms -= amt
            neg = dks < 0
            amt2 = np.zeros_like(dks)
            amt2[neg] = np.minimum(-0.5*dks[neg], np.maximum(rps[neg]-P['rho_min'], 0.0))
            rps -= amt2; rms += amt2
            rp[:,:,ss] = rps; rm[:,:,ss] = rms
        rho = _rho_total_state(rp, rm, rho_forest, rho_wall)
        kappa_tot = np.sum(rp-rm, axis=2)

    # --- BOUNDARY DISLOCATION (Frank-Bilby relaxation + mobile absorption) ---
    if P.get('freeze_rhoGB', False):
        rhoGB_delta_mean = 0.0
    elif P.get('use_frank_bilby_rhoGB', True):
        rho_GB_before = rho_GB.copy()
        gb_mask = diffuse_gb_support(eta, lab, Ng)
        rho_GB_target = (P.get('frank_bilby_coeff', 1.0)*gb_mask*grad_mag(psi_lat)/P['b'])
        rho_GB_target = np.clip(rho_GB_target, 0.0, P['rho_max'])
        if P.get('use_temperature_dependent_gb_mobility', False) and P.get('gb_mobility_apply_to_rhoGB_relax', True):
            gb_mob_fac = _gb_mobility_factor_from_T(T)
            tau_rel = np.maximum(P.get('rhoGB_relax_tau', 2e-6) / np.maximum(gb_mob_fac, 1e-300), P['dt'])
            tau_dec = np.maximum(P.get('rhoGB_decay_tau', 1e-6) / np.maximum(gb_mob_fac, 1e-300), P['dt'])
        else:
            tau_rel = max(P.get('rhoGB_relax_tau', 2e-6), P['dt'])
            tau_dec = max(P.get('rhoGB_decay_tau', 1e-6), P['dt'])
        tau_loc = np.where(rho_GB_target >= rho_GB, tau_rel, tau_dec)
        delta_GB = (P['dt']/tau_loc)*(rho_GB_target - rho_GB)
        if P.get('rhoGB_absorb_mobile', True):
            inc = np.maximum(delta_GB, 0.0)*P.get('rhoGB_absorb_fraction', 0.75)
            removable = np.maximum(rho - P['sweep_wake_rho'], 0.0)
            remove = np.minimum(inc, removable)
            scale_abs = np.where(rho > P['rho_min'], np.maximum(rho-remove, P['rho_min'])/np.maximum(rho, P['rho_min']), 1.0)
            for ss in range(nSlip):
                rp[:,:,ss] *= scale_abs
                rm[:,:,ss] *= scale_abs
            rho = _rho_total_state(rp, rm, rho_forest, rho_wall)
        rho_GB = np.clip(rho_GB + delta_GB, 0.0, P['rho_max'])
        # Clear stale boundary content away from current diffuse interfaces.
        rho_GB = np.where(gb_mask > P.get('gb_support_floor', 0.02), rho_GB, 0.0)
        rhoGB_delta_mean = float(np.nanmean(rho_GB - rho_GB_before))
    else:
        rho_GB = np.maximum(rho_GB - P['dt']*1e3*dFdg, 0)
        rho_GB += P['dt']*0.01*np.sum(np.abs(gdot),2)*gb_mask

    # --- v12 COMOVING GB-GND PROJECTION ---
    # Neutralize signed GND left behind only where GB support has departed.
    # This preserves total mobile density and prevents old initialized GB content
    # from becoming an artificial interior nucleation source.
    rp, rm, rho, H_nuc, E_nuc, gb_comoving_diag = apply_gb_comoving_gnd_projection(
        rp, rm, rho, gb_mask_old_for_gnd, gb_mask, H_nuc=H_nuc, E_nuc=E_nuc)
    rho = _rho_total_state(rp, rm, rho_forest, rho_wall)
    kappa_tot = np.sum(rp-rm, axis=2)

    # --- v9 ARRHENIUS HALL-PETCH GB SOURCE/TRANSMISSION ---
    if P.get('use_gb_hp_source_sink', True):
        rp, rm, rho, rho_GB, gb_hp_diag = apply_gb_hp_source_sink(
            rp, rm, rho, rho_GB, gb_mask, lab, Ng, sig_use, Sch, T, psi_lat=psi_lat)
        rho = _rho_total_state(rp, rm, rho_forest, rho_wall)
        kappa_tot = np.sum(rp-rm, axis=2)

    # v28 persistent collective activity memory.  The source combines the
    # instantaneous multi-hit/domain-count activity with the actual local slip
    # activity.  It is used below as a spatial susceptibility for local plastic
    # work, not as an independent heat source.
    if P.get('use_collective_activity_memory', False):
        mem_mode = str(P.get('collective_activity_memory_mode', 'crystallographic_local')).lower()
        per_slip_mem = mem_mode in ('crystallographic_local', 'crystallographic', 'grain_slip', 'crystal')
        A_src = np.clip(collective_activity_by_slip if per_slip_mem else collective_activity_total, 0.0, 1.0)
        if P.get('collective_activity_memory_use_gdot', True):
            gref = P.get('collective_activity_memory_gdot_ref', None)
            gref = abs(float(P.get('edot_app', 1.0))) if gref is None else abs(float(gref))
            if per_slip_mem:
                gnorm = np.clip(np.abs(gdot)/max(gref/max(nSlip,1), 1e-300), 0.0, 10.0)
            else:
                gabs = np.sum(np.abs(gdot), axis=2)
                gnorm = np.clip(gabs/max(gref, 1e-300), 0.0, 10.0)
            A_src = np.clip(A_src * (gnorm/(1.0 + gnorm)), 0.0, 1.0)
        A_src *= float(P.get('collective_activity_memory_source_weight', 1.0))
        collective_activity_memory = _activity_memory_advance(collective_activity_memory, A_src, P['dt'], psi_lat, per_slip=per_slip_mem)
        A_mem_scalar = _activity_memory_scalar(collective_activity_memory)
        km_diag.update(collective_activity_memory_mean=float(np.nanmean(A_mem_scalar)),
                       collective_activity_memory_max=float(np.nanmax(A_mem_scalar)),
                       collective_activity_memory_mode=str(mem_mode))
    else:
        collective_activity_memory = np.zeros_like(collective_activity_total)

    # --- TEMPERATURE (EXPLICIT) ---
    # v17: heat locally from plastic dissipation rather than globally from σ_bar*edot_app.
    # Plastic power per volume is sum_s τ_s γdot_s.  By default use the effective
    # thermodynamic driving stress τ - τ_backstress; set local_heat_stress='resolved'
    # to use the resolved shear stress itself.
    if P.get('use_local_heat_source', True):
        if str(P.get('local_heat_stress', 'effective')).lower().startswith('res'):
            tau_heat = tau_resolved
            heat_mode = 'local_resolved_tau_gdot'
        else:
            tau_heat = tau_effective
            heat_mode = 'local_effective_tau_gdot'
        plastic_power = np.sum(tau_heat * gdot, axis=2)
        if P.get('local_heat_nonnegative', True):
            plastic_power = np.maximum(plastic_power, float(P.get('local_heat_floor_qdot', 0.0)))
        # v32: if poor GB transmission blocks slip, do not use that unresolved
        # incompatibility as a same-cell heat source.  Use it as a reduced
        # heat partition and optionally store part as residual GB content.
        gb_block_diag = dict(active=0.0, blocked_frac_mean=0.0, blocked_power_mean=0.0,
                             blocked_power_max=0.0, stored_power_mean=0.0, heat_scale_min=1.0)
        try:
            if P.get('use_gb_blocked_work_partition', True):
                if _gb_trans_fields_step is None:
                    _gb_trans_fields_step = _gb_slip_transmission_fields(lab, psi_lat, gb_mask, T)
                plastic_power, gb_block_diag, _gb_stored_power = _gb_blocked_heat_partition(plastic_power, _gb_trans_fields_step, gb_mask)
                if P.get('gb_blocked_work_store_to_rhoGB', True):
                    _Aline = max(float(getattr(ATpot, 'A1', 1.0e-9)), 1.0e-30)
                    _drho_store = P['dt']*_gb_stored_power/_Aline
                    _max_add = float(P.get('gb_blocked_work_rhoGB_max_frac_step', 0.05))*max(float(np.nanmean(rho_GB)) + _rho_network_ref_scale(), P['rho_min'])
                    _drho_store = np.clip(_drho_store, 0.0, _max_add)
                    rho_GB = np.clip(rho_GB + _drho_store*np.clip(gb_mask,0.0,1.0), 0.0, P['rho_max'])
                    gb_block_diag['stored_rhoGB_mean'] = float(np.nanmean(_drho_store*np.clip(gb_mask,0.0,1.0)))
                heat_mode = heat_mode + '_gb_blocked_partition'
        except Exception:
            pass
        if P.get('use_rho_state_partition', False) and P.get('use_collective_organization', True):
            try:
                Aheat = np.asarray(_activity_memory_scalar(collective_activity_memory) if P.get('collective_heat_use_activity_memory', True) else collective_activity_total, dtype=float)
                hw = float(P.get('collective_heat_partition_weight', 0.0))
                hpw = max(float(P.get('collective_heat_partition_power', 1.0)), 1e-12)
                plastic_power = plastic_power * (1.0 + hw*np.clip(Aheat, 0.0, 1.0)**hpw)
                heat_mode = heat_mode + '_collective_activity_weighted'
            except Exception:
                pass

        if P.get('use_energy_conserving_heat', True) and (not finite_loading):
            # Strain-rate controlled macroscopic work budget.  The raw local
            # tau*gdot field is used only as a spatial partition function; its
            # mean is normalized so <qdot>=beta*sigma_bar*edot_app.
            macro_power = max(float(sigma_bar) * float(P.get('edot_app', 0.0)), 0.0)
            tiny = float(P.get('heat_partition_tiny', 1.0e-300))
            pp_mean = float(np.nanmean(plastic_power))
            if np.isfinite(pp_mean) and pp_mean > tiny and macro_power > 0.0:
                qdot_field = P['taylor_quinney'] * macro_power * (plastic_power / pp_mean)
                heat_mode = heat_mode + '_partitioned_macro_work'
            else:
                qdot_field = np.full_like(T, P['taylor_quinney'] * macro_power)
                heat_mode = 'global_sigma_edot_fallback'
        elif finite_loading and P.get('use_finite_loading_work_budget', True):
            # v31 finite-loading work budget.  Heat is partitioned according to
            # local tau*gdot, but its volume average cannot exceed the external
            # mechanical power plus any released macroscopic elastic energy.
            Eeff_h = P.get('finite_loading_Eeff', None)
            Eeff_h = float(Eeff_h) if Eeff_h is not None else float(E_ps)
            dt_h = max(float(P.get('dt', 0.0)), 1e-300)
            U_old = 0.5*float(sigma_bar_old)**2/max(Eeff_h, 1.0)
            U_new = 0.5*float(sigma_bar)**2/max(Eeff_h, 1.0)
            ext_power = max(float(sigma_bar_old)*float(P.get('edot_app', 0.0)), 0.0)
            elastic_release_power = max((U_old - U_new)/dt_h, 0.0) if P.get('finite_loading_allow_elastic_unload', True) else 0.0
            macro_power = max(ext_power + elastic_release_power, 0.0)
            macro_power *= float(np.clip(P.get('finite_loading_work_budget_safety', 0.95), 0.0, 1.0))
            tiny = float(P.get('heat_partition_tiny', 1.0e-300))
            pp_mean = float(np.nanmean(plastic_power))
            if np.isfinite(pp_mean) and pp_mean > tiny and macro_power > 0.0:
                qdot_field = P['taylor_quinney'] * macro_power * (plastic_power / pp_mean)
                heat_mode = heat_mode + '_finite_work_budget'
            else:
                qdot_field = np.full_like(T, P['taylor_quinney'] * macro_power)
                heat_mode = 'finite_work_budget_fallback'
            km_diag.update(finite_heat_macro_power=float(macro_power),
                           finite_heat_raw_power_mean=float(pp_mean))
        else:
            qdot_field = P['taylor_quinney'] * plastic_power
    else:
        heat_mode = 'global_sigma_edot'
        qdot_field = np.full_like(T, P['taylor_quinney']*sigma_bar*P['edot_app'])
    # v32: finite process-zone regularization of the heat partition.  Preserve
    # the domain-average qdot set by the work budget, but prevent pixel-scale
    # thermal singularities at blocked GB/junction cells.
    heat_pz_diag = dict(active=0.0, sigma_px=0.0, raw_max=float(np.nanmax(qdot_field)), smooth_max=float(np.nanmax(qdot_field)))
    if P.get('use_heat_process_zone_kernel', True):
        try:
            qdot_field, heat_pz_diag = _conservative_process_zone_filter(
                qdot_field,
                sigma_um=P.get('heat_process_zone_sigma_um', 0.0),
                preserve_mean=bool(P.get('heat_process_zone_preserve_mean', True)))
            heat_mode = heat_mode + '_process_zone'
        except Exception:
            pass
    dT_heat = P['dt'] * qdot_field / max(P['cp_rho_vol'], 1.0)
    heat_diag = {
        'qdot_mean': float(np.nanmean(qdot_field)),
        'qdot_max': float(np.nanmax(qdot_field)),
        'qdot_std': float(np.nanstd(qdot_field)),
        'dT_mean_step': float(np.nanmean(dT_heat)),
        'dT_max_step': float(np.nanmax(dT_heat)),
        'mode': heat_mode,
        'process_zone_active': float(heat_pz_diag.get('active', 0.0)),
        'process_zone_sigma_px': float(heat_pz_diag.get('sigma_px', 0.0)),
        'process_zone_raw_qdot_max': float(heat_pz_diag.get('raw_max', np.nan)),
        'process_zone_smooth_qdot_max': float(heat_pz_diag.get('smooth_max', np.nan)),
        'gb_blocked_active': float(gb_block_diag.get('active', 0.0)) if 'gb_block_diag' in locals() else 0.0,
        'gb_blocked_frac_mean': float(gb_block_diag.get('blocked_frac_mean', 0.0)) if 'gb_block_diag' in locals() else 0.0,
        'gb_blocked_frac_max': float(gb_block_diag.get('blocked_frac_max', 0.0)) if 'gb_block_diag' in locals() else 0.0,
        'gb_blocked_power_mean': float(gb_block_diag.get('blocked_power_mean', 0.0)) if 'gb_block_diag' in locals() else 0.0,
        'gb_blocked_power_max': float(gb_block_diag.get('blocked_power_max', 0.0)) if 'gb_block_diag' in locals() else 0.0,
        'gb_blocked_stored_power_mean': float(gb_block_diag.get('stored_power_mean', 0.0)) if 'gb_block_diag' in locals() else 0.0,
        'gb_blocked_heat_scale_min': float(gb_block_diag.get('heat_scale_min', 1.0)) if 'gb_block_diag' in locals() else 1.0,
        'gb_blocked_stored_rhoGB_mean': float(gb_block_diag.get('stored_rhoGB_mean', 0.0)) if 'gb_block_diag' in locals() else 0.0,
        '_qdot_field': qdot_field,
        'thermal_dt_active': float(thermal_dt_diag.get('active', 0)),
        'thermal_dt': float(thermal_dt_diag.get('dt', P.get('dt', np.nan))),
        'thermal_dt_base': float(thermal_dt_diag.get('dt_base', P.get('_dt_base_runtime', np.nan))),
        'thermal_dt_sub_equiv': float(thermal_dt_diag.get('n_sub_equiv', 1.0)),
        'thermal_dt_dT_allow': float(thermal_dt_diag.get('dT_allow', np.nan)),
        'thermal_dt_dT_macro_pred': float(thermal_dt_diag.get('dT_macro_pred', np.nan)),
    }
    T = update_temperature_field(T, qdot_field)

    _bad_T, _bad_T_reason = _thermal_validity_exceeded(T)
    if _bad_T:
        print(f"THERMAL VALIDITY STOP at step {n}: {_bad_T_reason}; "
              f"{P.get('thermal_validity_stop_reason', '')}")
        if P.get('thermal_validity_save_restart', True):
            _save_restart_checkpoint(n)
        _stop_run = True

    # --- v11 GND-bounded cumulative-hazard nucleation ---
    if ((not P.get('disable_nucleation', False)) and P.get('use_hazard_nucleation', True)
            and n > 0 and n % max(int(P.get('nuc_interval', 20)), 1) == 0
            and E_tot[0,0] >= P.get('nuc_min_strain', 0.0)):
        hazard_activity_factor = _hazard_activity_prefactor(gdot, rp, rm)
        (eta, psi_gv, Ng, lab, psi_lat, psi_plastic,
         rp, rm, rho, rho_GB, gb_mask, H_nuc, E_nuc,
         _last_nuc_diag) = apply_hazard_nucleation(
            eta, psi_gv, Ng, lab, psi_lat, psi_plastic,
            rp, rm, rho, rho_GB, gb_mask, kappa_tot, T, H_nuc, E_nuc,
            activity_factor=hazard_activity_factor)
        kappa_tot = np.sum(rp-rm, axis=2)
    elif n % max(int(P.get('nuc_interval', 20)), 1) == 0:
        _last_nuc_diag = {'cand': 0, 'best_dF': np.nan, 'best_score': np.nan,
                          'event': 0, 'hazard_max': 0.0, 'Hmax': float(np.nanmax(H_nuc)),
                          'barrier_best_eV': np.nan, 'theta_best_deg': np.nan,
                          'theta_max_deg': np.nan, 'R_best_um': np.nan,
                          'activity_factor_mean': np.nan, 'activity_factor_max': np.nan}

    # --- DIAGNOSTICS ---
    if n%P['diag_interval']==0 or n==P['nSteps']-1:
        mr=float(rho.mean()); xr=float(rho.max()); sr=float(rho.std())
        pm=float(np.rad2deg(np.abs(psi_lat).max()))
        el=_wtime.time()-tw0
        r_f = rho/max(_rho_ch_scale(),P['rho_min'])
        mu_ch_diag = ATpot.mu_dw(r_f)
        diag_row = _diagnostic_row(
            n, t, rho, rho_c, eta, lab, Ng, psi_lat, psi_plastic, kappa_tot,
            rp, rm, rho_GB, gb_mask, mu_ch_diag, sigma_bar, T, E_tot, nflip,
            km_store_mean, km_anni_mean, ch_delta_abs_mean, ch_delta_std,
            ac_eta_delta_mean, gnd_transfer_mean, rhoGB_delta_mean, _last_nuc_diag, gb_hp_diag, _last_topology_diag, heat_diag, km_diag)
        if _diag_csv_fh is not None:
            if _diag_csv_writer is None:
                _diag_csv_writer = csv.DictWriter(_diag_csv_fh, fieldnames=list(diag_row.keys()))
                _diag_csv_writer.writeheader()
            _diag_csv_writer.writerow(diag_row)
            _diag_csv_fh.flush()
        print(f"  {n:5d}  t={t*1e6:7.1f}us  eps={E_tot[0,0]*100:5.1f}%  "
              f"<r>={r_f.mean():.2f}  rho={mr:.2e}±{sr:.1e}  "
              f"sig={sigma_bar/1e6:.0f}MPa  T={T.max():.0f}K  "
              f"psi={pm:.2f}°  k/r={diag_row['kappa_frac_mean']:.3f} "
              f"rhoGB={float(rho_GB.max()):.1e} gr={Ng}  flp={nflip}  [{el:.0f}s]")
        if P.get('diag_print_extended', True):
            print(f"        provenance: fields initial/spinodal/hazard="
                  f"{diag_row['grain_initial_lineage']}/{diag_row['grain_spinodal_lineage']}/{diag_row['grain_hazard_lineage']}  "
                  f"births topology/hazard={diag_row['grain_topology_births']}/{diag_row['grain_hazard_births']}  "
                  f"step topology/hazard={diag_row['grain_topology_step_births']}/{diag_row['grain_hazard_step_births']}  "
                  f"heat ΔTmean/max={diag_row['heat_dT_mech_step']:.2e}/{diag_row['heat_dT_local_max_step']:.2e}K  "
                  f"dt={diag_row.get('thermal_dt', P.get('dt', np.nan)):.1e}s  "
                  f"k2={diag_row['k2_eff_mean']:.2g}-{diag_row['k2_eff_max']:.2g}  "
                  f"Drec={diag_row['diffrec_D_mean']:.1e} m2/s")
            print(f"        budgets: KM Δρ={diag_row['km_net_mean']:.2e} "
                  f"(store {diag_row['km_store_mean']:.2e}, anni {diag_row['km_anni_mean']:.2e}, "
                  f"diffrec {diag_row['km_diffrec_mean']:.2e})  "
                  f"CH |Δρ|={diag_row['ch_delta_abs_mean']:.2e}  AC |Δη|={diag_row['ac_eta_delta_mean']:.2e}  "
                  f"GND |Δκ|={diag_row['gnd_transfer_mean']:.2e}  GB Δρ={diag_row['rhoGB_delta_mean']:.2e}  "
                  f"GBmove dep={gb_comoving_diag.get('depart_mean',0.0):.2e} relax={gb_comoving_diag.get('relax_mean',0.0):.2e}  "
                  f"GBHP src={diag_row['gb_hp_src_mean']:.2e} sink={diag_row['gb_hp_sink_mean']:.2e}  "
                  f"xi(src/trans/sink)={diag_row.get('gb_hp_xi_source_mean', np.nan):.2g}/"
                  f"{diag_row.get('gb_hp_xi_trans_mean', np.nan):.2g}/"
                  f"{diag_row.get('gb_hp_xi_sink_mean', np.nan):.2g}  "
                  f"GBtr m'={diag_row.get('gb_trans_mprime_mean', np.nan):.2g} "
                  f"bres={diag_row.get('gb_trans_bres_mean', np.nan):.2g} "
                  f"Gtr={diag_row.get('gb_trans_barrier_eV_mean', np.nan):.2g}eV "
                  f"fac={diag_row.get('gb_trans_factor_mean', np.nan):.2g}")

            if (P.get("use_temperature_dependent_gb_mobility", False)
                    and P.get("gb_mobility_print_diag", P.get("gb_mobility_diag", False))):
                try:
                    _gb_fac_diag = _gb_mobility_factor_from_T(T)
                    print(f"        GBmob: factor mean/max={np.nanmean(_gb_fac_diag):.3g}/{np.nanmax(_gb_fac_diag):.3g} "
                          f"Tmean/max={np.nanmean(T):.1f}/{np.nanmax(T):.1f}K")
                except Exception as _e:
                    print(f"        GBmob: factor diagnostic failed: {_e}")
            print(f"        ASB: Tstd={diag_row['asb_T_std']:.2e}K  "
                  f"T-rho={diag_row['asb_corr_T_logrho']:.2f}  "
                  f"rho_hot/cold={diag_row['asb_rho_hot_over_cold']:.2f}  "
                  f"netρdot_hot={diag_row['asb_net_rhodot_hot_mean']:.2e}  "
                  f"qdot_top={diag_row['asb_qdot_top5_frac']:.2f}  "
                  f"band/slip={diag_row['asb_band_alignment_to_slip_deg']:.1f}deg")
            print(f"        coupling: gbArea={100*diag_row['gb_area_frac']:.1f}%  "
                  f"topρ_on_GB={100*diag_row['highrho_on_gb_frac']:.1f}%  "
                  f"GB_in_topρ={100*diag_row['gb_on_highrho_frac']:.1f}%  "
                  f"corr(r,GB)={diag_row['corr_r_gb']:.2f}  corr(r,|∇ψ|)={diag_row['corr_r_gradpsi']:.2f}  "
                  f"corr(r,|κ|)={diag_row['corr_r_kappa']:.2f}  corr(κ,r)={diag_row['corr_signed_kappa_r']:.2f}")
            print(f"        nucleation: event={diag_row['nuc_event']}  candidates={diag_row['nuc_candidates']}  "
                  f"hmax={diag_row['nuc_hazard_max']:.2e}/s  Hmax={diag_row['nuc_Hmax']:.2f}  "
                  f"G*best={diag_row['nuc_barrier_best_eV']:.2f}eV  "
                  f"theta={diag_row['nuc_theta_best_deg']:.2f}/{diag_row['nuc_theta_max_deg']:.2f}deg  "
                  f"R={diag_row['nuc_R_best_um']:.3f}um  "
                  f"cand_active={diag_row.get('nuc_candidate_active',0)} ageMax={diag_row.get('nuc_candidate_age_max',0)}")
            print(f"        energies: Fbulk={diag_row['F_bulk']:.2e} Frgrad={diag_row['F_r_grad']:.2e} "
                  f"Feta={diag_row['F_eta_grad']+diag_row['F_eta_barrier']:.2e} "
                  f"Fcompα={diag_row['F_comp_alpha']:.2e} FcompGB={diag_row['F_comp_GB']:.2e}  "
                  f"FBtarget/max={diag_row['FB_target_max']:.2e}")
        hist['t'].append(t); hist['rho_mean'].append(mr); hist['rho_max'].append(xr)
        hist['rho_std'].append(sr); hist['sigma'].append(sigma_bar)
        hist['T_mean'].append(float(T.mean())); hist['T_max'].append(float(T.max()))
        hist['eps'].append(E_tot[0,0]); hist['n_grains'].append(Ng)
        hist['psi_max'].append(pm); hist['rho_GB_max'].append(float(rho_GB.max()))
        hist['gb_area_frac'].append(diag_row['gb_area_frac'])
        hist['highrho_on_gb_frac'].append(diag_row['highrho_on_gb_frac'])
        hist['corr_r_gb'].append(diag_row['corr_r_gb'])
        hist['corr_r_gradpsi'].append(diag_row['corr_r_gradpsi'])
        hist['corr_r_kappa'].append(diag_row['corr_r_kappa'])
        hist['km_net_mean'].append(diag_row['km_net_mean'])
        hist['ch_delta_abs_mean'].append(diag_row['ch_delta_abs_mean'])
        hist['ac_eta_delta_mean'].append(diag_row['ac_eta_delta_mean'])
        hist['rhoGB_delta_mean'].append(diag_row['rhoGB_delta_mean'])
        hist['gb_hp_src_mean'].append(diag_row['gb_hp_src_mean'])
        hist['gb_hp_sink_mean'].append(diag_row['gb_hp_sink_mean'])
        hist['gb_hp_xi_mean'].append(diag_row['gb_hp_xi_mean'])
        hist['gb_hp_xi_source_mean'].append(diag_row.get('gb_hp_xi_source_mean', np.nan))
        hist['gb_hp_xi_trans_mean'].append(diag_row.get('gb_hp_xi_trans_mean', np.nan))
        hist['gb_hp_xi_sink_mean'].append(diag_row.get('gb_hp_xi_sink_mean', np.nan))
        hist['grain_initial_lineage'].append(diag_row['grain_initial_lineage'])
        hist['grain_spinodal_lineage'].append(diag_row['grain_spinodal_lineage'])
        hist['grain_hazard_lineage'].append(diag_row['grain_hazard_lineage'])
        hist['grain_topology_births'].append(diag_row['grain_topology_births'])
        hist['grain_hazard_births'].append(diag_row['grain_hazard_births'])
        hist['heat_dT_mech_step'].append(diag_row['heat_dT_mech_step'])
        hist['F_total'].append(diag_row['F_total_full']); hist['F_grad'].append(diag_row['F_r_grad'])
        hist['F_comp'].append(diag_row['F_comp_alpha']+diag_row['F_comp_GB'])

    # --- SAVE FIELDS / FRAMES ---
    _save_due = (n % max(int(P.get('save_interval', 1)), 1) == 0 or n == P['nSteps']-1)
    _plot_due = (P.get('save_main_panels', True)
                 and (n % max(int(P.get('plot_interval', P.get('save_interval', 1))), 1) == 0 or n == P['nSteps']-1)
                 and (_png_saved_count < int(P.get('max_saved_png_frames', 10**9))))
    if _save_due or _plot_due:
        r_f = rho/max(_rho_ch_scale(),P['rho_min'])
        if _plot_due:
            fig,ax=plt.subplots(3,4,figsize=(18,13))

            im=ax[0,0].pcolormesh(np.log10(np.maximum(rho,P['rho_min'])).T,cmap='viridis')
            ax[0,0].set_title(f'log10(rho) step {n}'); plt.colorbar(im,ax=ax[0,0])

            im=ax[0,1].pcolormesh(r_f.T,cmap='RdBu_r',vmin=0,vmax=max(r_f.max()*1.1, 3))
            ax[0,1].set_title(f'r=rho/rho_c'); plt.colorbar(im,ax=ax[0,1])

            im=ax[0,2].pcolormesh(T.T,cmap='hot'); ax[0,2].set_title(f'T (K)')
            plt.colorbar(im,ax=ax[0,2])

            im=ax[0,3].pcolormesh(lab.T,cmap='tab20')
            ax[0,3].set_title(f'grain_id (n={Ng})'); plt.colorbar(im,ax=ax[0,3])

            im=ax[1,0].pcolormesh(np.rad2deg(psi_lat).T,cmap='coolwarm')
            ax[1,0].set_title('psi_lat (deg)'); plt.colorbar(im,ax=ax[1,0])

            im=ax[1,1].pcolormesh(np.abs(kappa_tot).T,cmap='inferno')
            ax[1,1].set_title('|kappa| (GND)'); plt.colorbar(im,ax=ax[1,1])

            im=ax[1,2].pcolormesh(rho_GB.T,cmap='YlOrRd')
            ax[1,2].set_title(f'rho_GB max={rho_GB.max():.1e}'); plt.colorbar(im,ax=ax[1,2])

            im=ax[1,3].pcolormesh(ATpot.mu_dw(r_f).T,cmap='PuOr',vmin=-1.5,vmax=1.5)
            ax[1,3].set_title('mu_CH (double-well)'); plt.colorbar(im,ax=ax[1,3])

            if len(hist['t'])>1:
                tu=np.array(hist['t'])*1e6
                ax[2,0].plot(tu,hist['rho_mean'],'b-',label='<rho>')
                ax[2,0].plot(tu,hist['rho_max'],'r-',label='max')
                ax[2,0].axhline(rho_c,color='gray',ls='--',label='rho_c')
                ax[2,0].legend(fontsize=7); ax[2,0].set_title('density'); ax[2,0].set_xlabel('t(us)')

                ax[2,1].plot(tu,np.array(hist['sigma'])/1e6,'k-')
                ax[2,1].set_title('stress (MPa)'); ax[2,1].set_xlabel('t(us)')

                ax[2,2].plot(tu,hist['T_max'],'r-',label='max')
                ax[2,2].plot(tu,hist['T_mean'],'b-',label='mean')
                ax[2,2].legend(fontsize=7); ax[2,2].set_title('T (K)'); ax[2,2].set_xlabel('t(us)')

                ax[2,3].plot(tu,hist['rho_std'],'g-')
                ax[2,3].set_title('rho heterogeneity'); ax[2,3].set_xlabel('t(us)')
            else:
                for a in ax[2,:]: a.set_visible(False)

            plt.tight_layout()
            _safe_savefig(fig, out/f'drx_v8_{n:06d}.png')

            if P.get('save_signed_panels', True):
                kslip = rp - rm
                ktot = np.sum(kslip, axis=2)
                kfrac_plot = ktot / np.maximum(rho, P['rho_min'])
                r_tmp = rho/max(_rho_ch_scale(), P['rho_min'])
                Hr_tmp, _, prec_tmp, drive_tmp = _rho_eta_fields(r_tmp, rho, ktot, gb_mask)
                fig2, ax2 = plt.subplots(2,3,figsize=(14,8))
                im=ax2[0,0].pcolormesh(kfrac_plot.T,cmap='RdBu_r',vmin=-0.5,vmax=0.5)
                ax2[0,0].set_title('signed kappa/rho'); plt.colorbar(im,ax=ax2[0,0])
                for ss in range(min(nSlip,2)):
                    im=ax2[0,1+ss].pcolormesh((kslip[:,:,ss]/np.maximum(rho,P['rho_min'])).T,
                                               cmap='RdBu_r',vmin=-0.5,vmax=0.5)
                    ax2[0,1+ss].set_title(f'slip {ss}: (rho+ - rho-)/rho'); plt.colorbar(im,ax=ax2[0,1+ss])
                im=ax2[1,0].pcolormesh(np.abs(ktot).T,cmap='inferno')
                ax2[1,0].set_title('|signed kappa|'); plt.colorbar(im,ax=ax2[1,0])
                im=ax2[1,1].pcolormesh(prec_tmp.T,cmap='viridis',vmin=0,vmax=1)
                ax2[1,1].set_title('rho-eta precursor'); plt.colorbar(im,ax=ax2[1,1])
                im=ax2[1,2].pcolormesh(drive_tmp.T,cmap='viridis',vmin=0,vmax=1)
                ax2[1,2].set_title('rho-eta drive H(r)*precursor'); plt.colorbar(im,ax=ax2[1,2])
                plt.tight_layout()
                _safe_savefig(fig2, out/f'drx_v8_signed_{n:06d}.png')

        if _save_due and P.get('write_field_npz', True):
            np.savez_compressed(out/f'drx_v8_fields_{n:06d}.npz',
                                rho=rho, rp=rp, rm=rm, kappa_tot=np.sum(rp-rm,axis=2),
                                rho_GB=rho_GB, gb_mask=gb_mask, lab=lab,
                                eta_max=eta_purity_fields(eta, Ng)[0],
                                eta_second=eta_purity_fields(eta, Ng)[1],
                                eta_entropy=eta_purity_fields(eta, Ng)[3],
                                eta_nactive=eta_purity_fields(eta, Ng)[4],
                                psi_lat=psi_lat, psi_plastic=psi_plastic,
                                T=T, mu_ch=ATpot.mu_dw(rho/max(_rho_ch_scale(),P['rho_min'])),
                                gb_hp_src_mean=gb_hp_diag.get('src_mean', 0.0),
                                gb_hp_sink_mean=gb_hp_diag.get('sink_mean', 0.0),
                                gb_hp_rate_mean=gb_hp_diag.get('rate_mean', 0.0),
                                gb_hp_xi_mean=gb_hp_diag.get('xi_mean', np.nan),
                                gb_hp_xi_source_mean=gb_hp_diag.get('xi_source_mean', np.nan),
                                gb_hp_xi_trans_mean=gb_hp_diag.get('xi_trans_mean', gb_hp_diag.get('xi_mean', np.nan)),
                                gb_hp_xi_sink_mean=gb_hp_diag.get('xi_sink_mean', np.nan),
                                gb_hp_xi_screen_mean=gb_hp_diag.get('xi_screen_mean', np.nan),
                                H_nuc=H_nuc, E_nuc=E_nuc,
                                nuc_hazard_max=_last_nuc_diag.get('hazard_max', 0.0),
                                nuc_barrier_best_eV=_last_nuc_diag.get('barrier_best_eV', np.nan),
                                nuc_theta_best_deg=_last_nuc_diag.get('theta_best_deg', np.nan),
                                nuc_theta_max_deg=_last_nuc_diag.get('theta_max_deg', np.nan),
                                nuc_R_best_um=_last_nuc_diag.get('R_best_um', np.nan),
                                grain_origin_lineage=grain_origin_lineage[:Ng],
                                grain_birth_mechanism=grain_birth_mechanism[:Ng],
                                grain_parent=grain_parent[:Ng],
                                grain_birth_step=grain_birth_step[:Ng],
                                grain_birth_x=grain_birth_x[:Ng],
                                grain_birth_y=grain_birth_y[:Ng],
                                grain_birth_area_px=grain_birth_area_px[:Ng],
                                grain_birth_theta_deg=grain_birth_theta_deg[:Ng],
                                grain_birth_theta_max_deg=grain_birth_theta_max_deg[:Ng],
                                grain_birth_R_um=grain_birth_R_um[:Ng],
                                grain_birth_barrier_eV=grain_birth_barrier_eV[:Ng])

        if P.get('write_restart_npz', True) and (n % max(int(P.get('restart_interval', P['save_interval'])), 1) == 0 or n == P['nSteps']-1):
            chk = _save_restart_checkpoint(n)
            if chk is not None and P.get('diag_print_extended', True):
                print(f"        restart checkpoint: {chk}")

    sim_time += P['dt']
    if _stop_run:
        break

if _diag_csv_fh is not None:
    _diag_csv_fh.close()

# ================================================================
# 12. SUMMARY
# ================================================================
print(f"\n=== Done ({_wtime.time()-tw0:.0f}s) ===")
print(f"rho={rho.mean():.2e}, sig={sigma_bar/1e6:.0f}MPa, "
      f"eps={E_tot[0,0]*100:.1f}%, T={T.max():.0f}K, grains={Ng}")
_prov_final = _provenance_counts(Ng)
print(f"provenance lineages initial/spinodal/hazard="
      f"{_prov_final['grain_initial_lineage']}/{_prov_final['grain_spinodal_lineage']}/{_prov_final['grain_hazard_lineage']}; "
      f"birth mechanisms initial/topology/hazard="
      f"{_prov_final['grain_initial_births']}/{_prov_final['grain_topology_births']}/{_prov_final['grain_hazard_births']}")

if len(hist['t'])>2:
    fig,ax=plt.subplots(2,3,figsize=(15,9))
    tu=np.array(hist['t'])*1e6
    ax[0,0].plot(tu,hist['rho_mean'],'b-',lw=2,label='<rho>')
    ax[0,0].plot(tu,hist['rho_max'],'r-',label='max')
    ax[0,0].axhline(rho_c,color='gray',ls='--',label=f'rho_c={rho_c:.1e}')
    ax[0,0].legend(); ax[0,0].set_title('Density'); ax[0,0].set_xlabel('t(us)')

    ax[0,1].plot(tu,np.array(hist['sigma'])/1e6,'k-',lw=2)
    ax[0,1].set_title('Flow stress (MPa)'); ax[0,1].set_xlabel('t(us)')

    ax[0,2].plot(tu,hist['rho_std'],'g-',lw=2)
    ax[0,2].set_title('rho heterogeneity'); ax[0,2].set_xlabel('t(us)')

    ax[1,0].plot(tu,hist['psi_max'],'c-',lw=2)
    ax[1,0].set_title('max |psi| (deg)'); ax[1,0].set_xlabel('t(us)')

    ax[1,1].plot(tu,hist['n_grains'],'m-',lw=2)
    ax[1,1].set_title('Active grains'); ax[1,1].set_xlabel('t(us)')

    ax[1,2].plot(tu,hist['T_max'],'r-',lw=2,label='max')
    ax[1,2].plot(tu,hist['T_mean'],'b-',label='mean')
    ax[1,2].legend(); ax[1,2].set_title('Temperature'); ax[1,2].set_xlabel('t(us)')

    plt.tight_layout()
    _safe_savefig(plt.gcf(), out/'drx_v25_summary.png', dpi=120)
    print(f"Summary: {out/'drx_v25_summary.png'}")


# Save diagnostics/audit summary
if len(hist['t'])>2:
    fig,ax=plt.subplots(2,3,figsize=(16,9))
    tu=np.array(hist['t'])*1e6
    ax[0,0].plot(tu, 100*np.array(hist['gb_area_frac']), 'k-', label='GB area')
    ax[0,0].plot(tu, 100*np.array(hist['highrho_on_gb_frac']), 'r-', label='top rho on GB')
    ax[0,0].set_title('GB / high-rho overlap'); ax[0,0].set_xlabel('t(us)'); ax[0,0].set_ylabel('%')
    ax[0,0].legend(fontsize=7)

    ax[0,1].plot(tu, hist['corr_r_gb'], label='corr(r,GB)')
    ax[0,1].plot(tu, hist['corr_r_gradpsi'], label='corr(r,|grad psi|)')
    ax[0,1].plot(tu, hist['corr_r_kappa'], label='corr(r,|kappa|)')
    ax[0,1].axhline(0,color='gray',lw=0.5); ax[0,1].set_ylim(-1.05,1.05)
    ax[0,1].set_title('Do density structures respect GB/GND fields?')
    ax[0,1].set_xlabel('t(us)'); ax[0,1].legend(fontsize=7)

    ax[0,2].semilogy(tu, np.abs(hist['km_net_mean'])+1, label='|KM net|')
    ax[0,2].semilogy(tu, np.array(hist['ch_delta_abs_mean'])+1, label='CH |Δrho|')
    ax[0,2].semilogy(tu, np.array(hist['rhoGB_delta_mean'])+1, label='GB Δrho')
    ax[0,2].semilogy(tu, np.array(hist['gb_hp_src_mean'])+1, label='GBHP src')
    ax[0,2].semilogy(tu, np.array(hist['gb_hp_sink_mean'])+1, label='GBHP sink')
    ax[0,2].set_title('Update magnitudes per diagnostic step')
    ax[0,2].set_xlabel('t(us)'); ax[0,2].legend(fontsize=7)

    ax[1,0].semilogy(tu, np.maximum(hist['F_total'], 1e-300), label='F total')
    ax[1,0].semilogy(tu, np.maximum(hist['F_grad'], 1e-300), label='F r-grad')
    ax[1,0].semilogy(tu, np.maximum(hist['F_comp'], 1e-300), label='F comp')
    ax[1,0].set_title('Energy channels'); ax[1,0].set_xlabel('t(us)'); ax[1,0].legend(fontsize=7)

    ax[1,1].plot(tu, hist['rho_mean'], 'b-', label='<rho>')
    ax[1,1].plot(tu, hist['rho_max'], 'r-', label='max rho')
    ax[1,1].set_title('Density evolution'); ax[1,1].set_xlabel('t(us)'); ax[1,1].legend(fontsize=7)

    ax[1,2].plot(tu, hist['n_grains'], 'm-', label='grains')
    ax[1,2].plot(tu, hist['psi_max'], 'c-', label='max |psi| deg')
    ax[1,2].plot(tu, hist['gb_hp_xi_source_mean'], label='xi source')
    ax[1,2].plot(tu, hist['gb_hp_xi_trans_mean'], 'k--', label='xi trans')
    ax[1,2].plot(tu, hist['gb_hp_xi_sink_mean'], ':', label='xi sink')
    ax[1,2].set_title('Topology/orientation/GBHP'); ax[1,2].set_xlabel('t(us)'); ax[1,2].legend(fontsize=7)

    plt.tight_layout()
    _safe_savefig(plt.gcf(), out/'drx_v25_diagnostic_audit.png', dpi=120)
    print(f"Diagnostic audit: {out/'drx_v25_diagnostic_audit.png'}")

# Save potential diagnostic
fig,ax=plt.subplots(1,3,figsize=(16,5))
rho_plot = np.logspace(np.log10(max(P['rho_min'],1e10)), np.log10(min(P['rho_max'],5e18)), 500)
r_plot = rho_plot / ATpot.rho_c

# Panel 1: Φ(ρ) — selected v20 potential and component terms
Phi_total = ATpot._Phi(rho_plot)
Phi_base = ATpot._interp_log(rho_plot, ATpot.Phi_base_tab)
Phi_el = ATpot._interp_log(rho_plot, ATpot.Phi_el_tab)
Phi_arr = ATpot._interp_log(rho_plot, ATpot.Phi_arr_sigma_tab)
Phi_work = ATpot._interp_log(rho_plot, ATpot.Phi_arr_work_tab)
Phi_ent = ATpot._interp_log(rho_plot, ATpot.Phi_ent_tab)
Phi_ord = ATpot._interp_log(rho_plot, ATpot.Phi_ord_tab)
ax[0].semilogx(rho_plot, Phi_total, 'k-', lw=2, label=r'$\Phi_{total}$')
ax[0].semilogx(rho_plot, Phi_base, 'b--', lw=1, alpha=0.75, label=ATpot.Phi_base_label)
ax[0].semilogx(rho_plot, Phi_arr, 'c:', lw=1, alpha=0.75, label=r'$\sigma_{AT}$ branch')
ax[0].semilogx(rho_plot, Phi_work, 'm-.', lw=1, alpha=0.75, label=r'$\int\sigma_{AT} b\ell d\rho$')
if P.get('use_potential_entropy', True):
    ax[0].semilogx(rho_plot, Phi_ent, 'g--', lw=1, alpha=0.55, label=r'$\rho\ln\rho$')
if P.get('use_potential_ordering', True):
    ax[0].semilogx(rho_plot, Phi_ord, 'r--', lw=1, alpha=0.55, label='ordering dip')
ax[0].axvline(ATpot.r_ord*ATpot.rho_c, color='r', ls=':', alpha=0.5,
              label=f'rho_ord={ATpot.r_ord*ATpot.rho_c:.1e}')
ax[0].axvline(ATpot.rho_c, color='orange', ls=':', alpha=0.5, label=f'rho_c={ATpot.rho_c:.1e}')
ax[0].set_xlabel(r'$\rho$ (m$^{-2}$)'); ax[0].set_ylabel(r'$\Phi$ (J/m$^3$)')
ax[0].set_title(f'Potential mode: {ATpot.potential_mode}'); ax[0].legend(fontsize=7)

# Panel 2: μ = dΦ/dρ (chemical potential)
mu_plot = ATpot._mu(rho_plot)
ax[1].semilogx(rho_plot, mu_plot, 'b-', lw=2)
ax[1].axhline(ATpot.A1, color='gray', ls='--', alpha=0.5, label=f'A1={ATpot.A1:.2e}')
ax[1].axvline(ATpot.rho_c, color='orange', ls=':', alpha=0.5)
ax[1].set_xlabel(r'$\rho$ (m$^{-2}$)'); ax[1].set_ylabel(r'$\mu = d\Phi/d\rho$ (J/m)')
ax[1].set_title('Chemical potential'); ax[1].legend(fontsize=7)

# Panel 3: Φ''(ρ) — curvature (negative = spinodal)
pp_plot = ATpot._Phi_pp(rho_plot)
ax[2].semilogx(rho_plot, pp_plot, 'r-', lw=2)
ax[2].axhline(0, color='gray', ls='--')
ax[2].fill_between(rho_plot, 0, pp_plot, where=pp_plot<0, alpha=0.3, color='red', label='spinodal')
ax[2].axvline(ATpot.rho_c, color='orange', ls=':', alpha=0.5)
ax[2].set_xlabel(r'$\rho$ (m$^{-2}$)'); ax[2].set_ylabel(r"$\Phi''(\rho)$")
ax[2].set_title('Curvature (spinodal where < 0)'); ax[2].legend(fontsize=7)
ax[2].set_xscale('log')

plt.tight_layout()
_safe_savefig(plt.gcf(), out/'drx_v25_potential.png', dpi=120)
print(f"Potential: {out/'drx_v25_potential.png'}")
