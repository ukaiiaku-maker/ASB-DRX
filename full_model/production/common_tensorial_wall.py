"""One nonlinear/JVP operator for tensorial signed-wall organization.

The residual in :func:`wall_residual` is the sole constitutive implementation.
The nonlinear accepted step and finite-difference Jacobian-vector product call
that function directly.  Space is two dimensional; crystallographic vectors,
plastic distortion, and the Nye tensor retain three components.

Units
-----
Signed line reservoirs and junction extent are m^-2, slip/plastic distortion
and wall order are dimensionless, Nye is m^-1, velocity is m/s, stress is Pa,
and temperature is K.  Every reaction extent below is m^-2 s^-1.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np

try:
    from .tensorial_nye import (
        JunctionTopology, SlipSystem3D, nye_from_plastic_distortion,
        plastic_distortion_from_slip, rotated_system_fields,
        alignment_increment_from_slip,
    )
    from .nonlocal_elasticity import solve_periodic_eigenstrain
except ImportError:  # direct execution by the production driver
    from tensorial_nye import (
        JunctionTopology, SlipSystem3D, nye_from_plastic_distortion,
        plastic_distortion_from_slip, rotated_system_fields,
        alignment_increment_from_slip,
    )
    from nonlocal_elasticity import solve_periodic_eigenstrain

KB_J_K = 1.380649e-23
EV_J = 1.602176634e-19


@dataclass(frozen=True)
class CommonWallState:
    mobile_plus_m2: np.ndarray
    mobile_minus_m2: np.ndarray
    forest_plus_m2: np.ndarray
    forest_minus_m2: np.ndarray
    wall_plus_m2: np.ndarray
    wall_minus_m2: np.ndarray
    junction_m2: np.ndarray
    wall_order: np.ndarray
    slip: np.ndarray
    beta_p: np.ndarray
    alignment_m2: np.ndarray
    family_nye_m1: np.ndarray
    orientation_rad: np.ndarray
    temperature_K: np.ndarray

    def validate(self, systems, topologies, admissibility=True):
        family_shape = np.asarray(self.mobile_plus_m2).shape
        if len(family_shape) != 3 or family_shape[2] != len(systems):
            raise ValueError("signed reservoirs require grid x family layout")
        for name in ("mobile_minus_m2", "forest_plus_m2", "forest_minus_m2",
                     "wall_plus_m2", "wall_minus_m2", "slip"):
            if np.asarray(getattr(self, name)).shape != family_shape:
                raise ValueError(f"{name} has inconsistent layout")
        grid = family_shape[:2]
        if np.asarray(self.junction_m2).shape != grid+(len(topologies),):
            raise ValueError("junction reservoir has inconsistent layout")
        if np.asarray(self.wall_order).shape != grid:
            raise ValueError("wall order has inconsistent layout")
        if np.asarray(self.beta_p).shape != grid+(3, 3):
            raise ValueError("plastic distortion has inconsistent layout")
        if np.asarray(self.alignment_m2).shape != family_shape+(3,):
            raise ValueError("alignment has inconsistent layout")
        if np.asarray(self.family_nye_m1).shape != family_shape+(3, 3):
            raise ValueError("family Nye has inconsistent layout")
        if np.asarray(self.orientation_rad).shape != grid or np.asarray(
                self.temperature_K).shape != grid:
            raise ValueError("orientation/temperature has inconsistent layout")
        for item in fields(self):
            value = np.asarray(getattr(self, item.name))
            if np.any(~np.isfinite(value)):
                raise ValueError(f"{item.name} contains nonfinite values")
        if admissibility:
            for name in ("mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
                         "forest_minus_m2", "wall_plus_m2", "wall_minus_m2",
                         "junction_m2"):
                if np.any(np.asarray(getattr(self, name)) < 0.0):
                    raise ValueError(f"{name} must be nonnegative")
            if np.any(np.asarray(self.temperature_K) <= 0.0):
                raise ValueError("temperature must be positive")
            if np.any(np.asarray(self.wall_order) < 0.0) or np.any(
                    np.asarray(self.wall_order) > 1.0):
                raise ValueError("wall order must lie in [0,1]")
        return grid, family_shape


@dataclass(frozen=True)
class CommonWallParameters:
    spacing_m: float
    burgers_m: float = 2.48e-10
    rho_reference_m2: float = 5.0e14
    line_energy_J_m: float = 1.5e-9
    correlation_energy_J_m: float = 8.0e-11
    forest_energy_J_m: float = 1.0e-10
    junction_energy_J_m: float = 2.0e-10
    wall_order_amplitude_J_m3: float = 1.0e6
    wall_order_barrier_J_m3: float = 2.0e5
    wall_partition_J_m: float = 2.0e-10
    wall_target_m2: float = 4.0e14
    wall_density_center_ratio: float = 1.3
    wall_density_width_ratio: float = 0.3
    wall_order_gradient_J_m: float = 2.0e-7
    attempt_frequency_s: float = 1.0e7
    critical_stress_Pa: float = 1.5e9
    exp_a: float = 2.2
    exp_n: float = 2.5
    exp_floor: float = 0.05
    lock_barrier_eV: float = 0.85
    unlock_barrier_eV: float = 1.05
    wall_barrier_eV: float = 0.80
    annihilation_barrier_eV: float = 1.20
    junction_barrier_eV: float = 0.85
    order_barrier_eV: float = 0.70
    activation_entropy_kB: float = 0.0
    reaction_energy_scale_J_m: float = 1.0e-9
    maximum_fraction_per_step: float = 0.15
    orientation_spin_weight: float = 1.0
    c11_Pa: float = 228e9
    c12_Pa: float = 132e9
    c44_Pa: float = 116.5e9
    elastic_iterations: int = 3
    glide_barrier_eV: float = 1.90
    glide_speed_attempt_m_s: float = 2.0e3
    volumetric_heat_capacity_J_m3_K: float = 3.5e6
    thermal_diffusivity_m2_s: float = 0.0
    bath_rate_s: float = 0.0
    bath_temperature_K: float = 1100.0
    mobile_correlation_diffusivity_m2_s: float = 1.0e-12
    multiplication_coefficient: float = 10.0

    def __post_init__(self):
        positive = (
            self.spacing_m, self.burgers_m, self.rho_reference_m2,
            self.line_energy_J_m, self.wall_order_barrier_J_m3,
            self.wall_partition_J_m, self.wall_target_m2,
            self.wall_density_width_ratio, self.attempt_frequency_s,
            self.critical_stress_Pa, self.exp_n,
            self.reaction_energy_scale_J_m,
            self.c11_Pa, self.c44_Pa, self.glide_speed_attempt_m_s,
            self.volumetric_heat_capacity_J_m3_K,
        )
        if any(not math.isfinite(float(value)) or value <= 0.0 for value in positive):
            raise ValueError("common wall parameters must be finite and positive")
        if not 0.0 <= self.exp_floor <= 1.0 or not 0.0 < self.maximum_fraction_per_step <= 1.0:
            raise ValueError("bounded fractions are invalid")
        if self.mobile_correlation_diffusivity_m2_s < 0.0:
            raise ValueError("mobile correlation diffusivity cannot be negative")
        if self.multiplication_coefficient < 0.0:
            raise ValueError("multiplication coefficient cannot be negative")


@dataclass(frozen=True)
class CommonWallDriving:
    glide_speed_m_s: np.ndarray | None = None
    resolved_stress_Pa: np.ndarray | None = None
    mean_strain: np.ndarray | None = None
    fixed_eigenstrain: np.ndarray | None = None


@dataclass(frozen=True)
class CommonWallResidual:
    state_rate: CommonWallState
    channel_rates_m2_s: dict
    plastic_power_W_m3: np.ndarray
    free_energy_rate_W_m3: np.ndarray
    heat_rate_W_m3: np.ndarray


def _spectral_gradient(field, spacing):
    value = np.asarray(field)
    nx, ny = value.shape[:2]
    kx = 2*np.pi*np.fft.fftfreq(nx, d=spacing)
    ky = 2*np.pi*np.fft.fftfreq(ny, d=spacing)
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    trailing = (1,)*(value.ndim-2)
    spectrum = np.fft.fftn(value, axes=(0, 1))
    dx = np.fft.ifftn(1j*kx.reshape(kx.shape+trailing)*spectrum,
                      axes=(0, 1))
    dy = np.fft.ifftn(1j*ky.reshape(ky.shape+trailing)*spectrum,
                      axes=(0, 1))
    return np.real(dx), np.real(dy)


def _divergence(vector, spacing):
    dx, _ = _spectral_gradient(vector[..., 0], spacing)
    _, dy = _spectral_gradient(vector[..., 1], spacing)
    return np.real(dx+dy)


def exp_floor_rate(stress_pa, temperature_K, barrier_eV, parameters):
    stress = np.abs(np.asarray(stress_pa, dtype=float))
    temperature = np.asarray(temperature_K, dtype=float)
    enthalpy = float(barrier_eV)*EV_J*(
        parameters.exp_floor+(1.0-parameters.exp_floor)*np.exp(
            -parameters.exp_a*(stress/parameters.critical_stress_Pa)**parameters.exp_n))
    free = enthalpy-KB_J_K*temperature*parameters.activation_entropy_kB
    return parameters.attempt_frequency_s*np.exp(np.clip(
        -free/(KB_J_K*temperature), -700.0, 40.0))


def wall_free_energy_derivatives(state, parameters, topologies=()):
    reservoirs = (
        state.mobile_plus_m2+state.mobile_minus_m2
        +state.forest_plus_m2+state.forest_minus_m2
        +state.wall_plus_m2+state.wall_minus_m2)
    multiplicity = (np.asarray(
        [topology.product_line_multiplicity for topology in topologies])
        if topologies else np.ones(state.junction_m2.shape[2]))
    topology_energy = (np.asarray(
        [topology.delta_free_energy_J_m for topology in topologies])
        if topologies else np.zeros(state.junction_m2.shape[2]))
    total = (np.sum(reservoirs, axis=2)
             +np.sum(state.junction_m2*multiplicity, axis=2))
    wall = np.sum(state.wall_plus_m2+state.wall_minus_m2, axis=2)
    q = state.wall_order
    reference = parameters.rho_reference_m2
    safe = np.maximum(total, 1e-30*reference)
    base_mu = (parameters.line_energy_J_m
               +parameters.correlation_energy_J_m*(np.log(safe/reference)+1.0))
    ratio = total/reference
    offset = (ratio-parameters.wall_density_center_ratio)/parameters.wall_density_width_ratio
    exponential = np.exp(-.5*offset*offset)
    dip = -parameters.wall_order_amplitude_J_m3*exponential
    dip_derivative = (parameters.wall_order_amplitude_J_m3*exponential*offset
                      /(parameters.wall_density_width_ratio*reference))
    h = q*q*(3.0-2.0*q)
    dh = 6.0*q*(1.0-q)
    partition_offset = wall-h*parameters.wall_target_m2
    wall_extra_mu = parameters.wall_partition_J_m*partition_offset/parameters.wall_target_m2
    common_mu = base_mu+h*dip_derivative
    forest_mu = common_mu+parameters.forest_energy_J_m
    wall_mu = common_mu+wall_extra_mu
    junction_mu = (multiplicity*common_mu[..., None]
                   +multiplicity*parameters.junction_energy_J_m
                   +topology_energy)
    barrier_derivative = (2.0*parameters.wall_order_barrier_J_m3*q*(1.0-q)
                          *(1.0-2.0*q))
    dq = (dh*dip+barrier_derivative
          -parameters.wall_partition_J_m*partition_offset*dh)
    return {
        "total_density_m2": total, "wall_density_m2": wall,
        "mobile_mu_J_m": common_mu, "forest_mu_J_m": forest_mu,
        "wall_mu_J_m": wall_mu, "junction_mu_J_m": junction_mu,
        "wall_order_derivative_J_m3": dq,
    }


def wall_free_energy_density_J_m3(state, parameters, topologies=()):
    """Local part of the exact common free energy differentiated above."""
    chemical = wall_free_energy_derivatives(state, parameters, topologies)
    total = chemical["total_density_m2"]
    wall = chemical["wall_density_m2"]
    q = state.wall_order
    reference = parameters.rho_reference_m2
    safe = np.maximum(total, 1e-30*reference)
    density = (parameters.line_energy_J_m*total
               +parameters.correlation_energy_J_m*total*np.log(safe/reference))
    forest = parameters.forest_energy_J_m*np.sum(
        state.forest_plus_m2+state.forest_minus_m2, axis=2)
    multiplicity = (np.asarray(
        [topology.product_line_multiplicity for topology in topologies])
        if topologies else np.ones(state.junction_m2.shape[2]))
    topology_energy = (np.asarray(
        [topology.delta_free_energy_J_m for topology in topologies])
        if topologies else np.zeros(state.junction_m2.shape[2]))
    junction = np.sum(
        state.junction_m2*(multiplicity*parameters.junction_energy_J_m
                           +topology_energy), axis=2)
    ratio = total/reference
    dip = -parameters.wall_order_amplitude_J_m3*np.exp(
        -.5*((ratio-parameters.wall_density_center_ratio)
             /parameters.wall_density_width_ratio)**2)
    h = q*q*(3.0-2.0*q)
    barrier = parameters.wall_order_barrier_J_m3*q*q*(1.0-q)*(1.0-q)
    partition = (.5*parameters.wall_partition_J_m
                 *(wall-h*parameters.wall_target_m2)**2
                 /parameters.wall_target_m2)
    return density+forest+junction+h*dip+barrier+partition


def _biased_exchange(source, target, delta_mu_J_m, rate_s, parameters):
    return _biased_exchange_components(
        source, target, delta_mu_J_m, rate_s, parameters)[0]


def _biased_exchange_components(source, target, delta_mu_J_m, rate_s,
                                parameters):
    argument = np.clip(delta_mu_J_m/(2.0*parameters.reaction_energy_scale_J_m),
                       -40.0, 40.0)
    forward = rate_s*source*np.exp(-argument)
    reverse = rate_s*target*np.exp(argument)
    return forward-reverse, forward+reverse


def resolved_driving_fields(state, driving, systems, parameters):
    """Resolve nonlocal stress and glide speed used by the common residual."""
    family_shape = state.mobile_plus_m2.shape
    if driving.resolved_stress_Pa is None:
        if driving.mean_strain is None:
            raise ValueError("mean strain is required when stress is not prescribed")
        beta2 = state.beta_p[..., :2, :2]
        eigenstrain = .5*(beta2+np.swapaxes(beta2, -1, -2))
        if driving.fixed_eigenstrain is not None:
            eigenstrain = eigenstrain+np.asarray(driving.fixed_eigenstrain)
        stress_tensor, _ = solve_periodic_eigenstrain(
            eigenstrain, driving.mean_strain, parameters.spacing_m,
            parameters.c11_Pa, parameters.c12_Pa, parameters.c44_Pa,
            iterations=parameters.elastic_iterations)
        _, directions, normals = rotated_system_fields(
            systems, state.orientation_rad)
        schmid = .5*(
            np.einsum("...si,...sj->...sij", directions[..., :2], normals[..., :2])
            +np.einsum("...si,...sj->...sij", normals[..., :2], directions[..., :2]))
        stress = np.einsum("...ij,...sij->...s", stress_tensor, schmid)
    else:
        stress = np.asarray(driving.resolved_stress_Pa, dtype=float)
    if stress.shape != family_shape:
        raise ValueError("resolved stress requires grid x family layout")
    if driving.glide_speed_m_s is None:
        activation = exp_floor_rate(
            stress, state.temperature_K[..., None],
            parameters.glide_barrier_eV, parameters)/parameters.attempt_frequency_s
        speed = (parameters.glide_speed_attempt_m_s*activation
                 *np.tanh(stress/parameters.critical_stress_Pa))
    else:
        speed = np.asarray(driving.glide_speed_m_s, dtype=float)
    if speed.shape != family_shape:
        raise ValueError("glide speed requires grid x family layout")
    return speed, stress


def wall_residual(state: CommonWallState, driving: CommonWallDriving,
                  systems: tuple[SlipSystem3D, ...],
                  topologies: tuple[JunctionTopology, ...],
                  parameters: CommonWallParameters, *,
                  differentiation_state=False):
    """Evaluate the single authoritative nonlinear residual."""
    grid, family_shape = state.validate(
        systems, topologies, admissibility=not differentiation_state)
    speed, stress = resolved_driving_fields(
        state, driving, systems, parameters)
    _, directions, _ = rotated_system_fields(systems, state.orientation_rad)
    planar = directions[..., :2]
    flux_plus = state.mobile_plus_m2[..., None]*speed[..., None]*planar
    flux_minus = -state.mobile_minus_m2[..., None]*speed[..., None]*planar
    if parameters.mobile_correlation_diffusivity_m2_s:
        for family in range(len(systems)):
            gx, gy = _spectral_gradient(
                state.mobile_plus_m2[..., family], parameters.spacing_m)
            flux_plus[..., family, 0] -= parameters.mobile_correlation_diffusivity_m2_s*gx
            flux_plus[..., family, 1] -= parameters.mobile_correlation_diffusivity_m2_s*gy
            gx, gy = _spectral_gradient(
                state.mobile_minus_m2[..., family], parameters.spacing_m)
            flux_minus[..., family, 0] -= parameters.mobile_correlation_diffusivity_m2_s*gx
            flux_minus[..., family, 1] -= parameters.mobile_correlation_diffusivity_m2_s*gy
    mp_rate = -_divergence(flux_plus, parameters.spacing_m)
    mm_rate = -_divergence(flux_minus, parameters.spacing_m)
    slip_rate = (parameters.burgers_m*speed
                 *(state.mobile_plus_m2+state.mobile_minus_m2))
    fp_rate = np.zeros(family_shape); fm_rate = np.zeros(family_shape)
    wp_rate = np.zeros(family_shape); wm_rate = np.zeros(family_shape)
    junction_rate = np.zeros(grid+(len(topologies),))

    chemical = wall_free_energy_derivatives(state, parameters, topologies)
    temperature = state.temperature_K[..., None]
    k_lock = exp_floor_rate(stress, temperature, parameters.lock_barrier_eV, parameters)
    delta_lock = chemical["forest_mu_J_m"]-chemical["mobile_mu_J_m"]
    lock_p, lock_p_turnover = _biased_exchange_components(
        state.mobile_plus_m2, state.forest_plus_m2,
        delta_lock[..., None], k_lock, parameters)
    lock_m, lock_m_turnover = _biased_exchange_components(
        state.mobile_minus_m2, state.forest_minus_m2,
        delta_lock[..., None], k_lock, parameters)
    mp_rate -= lock_p; mm_rate -= lock_m
    fp_rate += lock_p; fm_rate += lock_m

    k_wall = exp_floor_rate(stress, temperature, parameters.wall_barrier_eV, parameters)
    delta_wall = chemical["wall_mu_J_m"]-chemical["forest_mu_J_m"]
    transfer_p, transfer_p_turnover = _biased_exchange_components(
        state.forest_plus_m2, state.wall_plus_m2,
        delta_wall[..., None], k_wall, parameters)
    transfer_m, transfer_m_turnover = _biased_exchange_components(
        state.forest_minus_m2, state.wall_minus_m2,
        delta_wall[..., None], k_wall, parameters)
    fp_rate -= transfer_p; fm_rate -= transfer_m
    wp_rate += transfer_p; wm_rate += transfer_m

    k_ann = exp_floor_rate(stress, temperature,
                           parameters.annihilation_barrier_eV, parameters)
    annihilation = (k_ann*state.mobile_plus_m2*state.mobile_minus_m2
                    /parameters.rho_reference_m2)
    mp_rate -= annihilation; mm_rate -= annihilation

    # Loop multiplication creates equal positive/negative content.  It raises
    # line length but exactly preserves vector Burgers inventory; its rate is
    # the conventional d(rho)/d(gamma) ~ sqrt(rho) form.
    family_total = (state.mobile_plus_m2+state.mobile_minus_m2
                    +state.forest_plus_m2+state.forest_minus_m2
                    +state.wall_plus_m2+state.wall_minus_m2)
    multiplication = (0.5*parameters.multiplication_coefficient
                      *np.sqrt(np.maximum(family_total, 0.0))*np.abs(slip_rate))
    mp_rate += multiplication; mm_rate += multiplication

    junction_extents = []
    junction_turnovers = []
    for index, topology in enumerate(topologies):
        first = topology.parent_a; second = topology.parent_b
        source_first = (state.forest_plus_m2[..., first]
                        if topology.sign_a > 0 else state.forest_minus_m2[..., first])
        source_second = (state.forest_plus_m2[..., second]
                         if topology.sign_b > 0 else state.forest_minus_m2[..., second])
        pair_stress = np.maximum(np.abs(stress[..., first]), np.abs(stress[..., second]))
        kf = exp_floor_rate(pair_stress, state.temperature_K,
                            parameters.junction_barrier_eV, parameters)
        source_pair = source_first*source_second/parameters.rho_reference_m2
        delta_mu = (chemical["junction_mu_J_m"][..., index]
                    -2.0*chemical["forest_mu_J_m"])
        extent, turnover = _biased_exchange_components(
            source_pair, state.junction_m2[..., index], delta_mu, kf,
            parameters)
        target_first = fp_rate if topology.sign_a > 0 else fm_rate
        target_second = fp_rate if topology.sign_b > 0 else fm_rate
        target_first[..., first] -= extent
        target_second[..., second] -= extent
        junction_rate[..., index] += extent
        junction_extents.append(extent)
        junction_turnovers.append(turnover)

    beta_rate = plastic_distortion_from_slip(
        slip_rate, systems, state.orientation_rad)
    family_nye_rate = np.stack([
        nye_from_plastic_distortion(
            plastic_distortion_from_slip(
                np.where(np.arange(len(systems))[None, None, :] == family,
                         slip_rate, 0.0), systems, state.orientation_rad),
            parameters.spacing_m)
        for family in range(len(systems))], axis=2)
    alignment_rate = alignment_increment_from_slip(
        slip_rate, systems, state.orientation_rad, parameters.spacing_m)
    orientation_rate = parameters.orientation_spin_weight*.5*(
        beta_rate[..., 1, 0]-beta_rate[..., 0, 1])

    k_order = exp_floor_rate(
        np.max(np.abs(stress), axis=2), state.temperature_K,
        parameters.order_barrier_eV, parameters)
    qx, qy = _spectral_gradient(state.wall_order,
                                parameters.spacing_m)
    lap_q = _divergence(np.stack((qx, qy), axis=-1), parameters.spacing_m)
    q_chemical_potential = (chemical["wall_order_derivative_J_m3"]
                            -parameters.wall_order_gradient_J_m*lap_q)
    q_rate = (-k_order*q_chemical_potential
              /max(parameters.wall_order_barrier_J_m3, 1.0))

    zero_beta = np.zeros_like(state.beta_p)
    zero_nye = np.zeros_like(state.family_nye_m1)
    tx, ty = _spectral_gradient(state.temperature_K, parameters.spacing_m)
    lap_temperature = _divergence(
        np.stack((tx, ty), axis=-1), parameters.spacing_m)
    state_rate = CommonWallState(
        mp_rate, mm_rate, fp_rate, fm_rate, wp_rate, wm_rate,
        junction_rate, np.real(q_rate), slip_rate, beta_rate,
        alignment_rate, family_nye_rate, orientation_rate,
        np.zeros_like(state.temperature_K))
    plastic_power = np.sum(stress*slip_rate, axis=2)
    # Exact directional derivative of the declared defect free-energy
    # functional.  The q term includes the variational gradient contribution.
    free_energy_rate = (
        chemical["mobile_mu_J_m"]*np.sum(mp_rate+mm_rate, axis=2)
        +chemical["forest_mu_J_m"]*np.sum(fp_rate+fm_rate, axis=2)
        +chemical["wall_mu_J_m"]*np.sum(wp_rate+wm_rate, axis=2)
        +np.sum(chemical["junction_mu_J_m"]*junction_rate, axis=2)
        +q_chemical_potential*q_rate)
    heat_rate = plastic_power-free_energy_rate
    temperature_rate = (
        heat_rate/parameters.volumetric_heat_capacity_J_m3_K
        +parameters.thermal_diffusivity_m2_s*lap_temperature
        -parameters.bath_rate_s*(state.temperature_K-parameters.bath_temperature_K))
    state_rate = CommonWallState(
        mp_rate, mm_rate, fp_rate, fm_rate, wp_rate, wm_rate,
        junction_rate, np.real(q_rate), slip_rate, beta_rate,
        alignment_rate, family_nye_rate, orientation_rate,
        np.real(temperature_rate))
    channels = {
        "lock_plus": lock_p, "lock_minus": lock_m,
        "wall_plus": transfer_p, "wall_minus": transfer_m,
        "lock_plus_turnover": lock_p_turnover,
        "lock_minus_turnover": lock_m_turnover,
        "wall_plus_turnover": transfer_p_turnover,
        "wall_minus_turnover": transfer_m_turnover,
        "annihilation_pairs": annihilation,
        "multiplication_pairs": multiplication,
        "junction": (np.stack(junction_extents, axis=2)
                     if junction_extents else np.zeros(grid+(0,))),
        "junction_turnover": (np.stack(junction_turnovers, axis=2)
                              if junction_turnovers else np.zeros(grid+(0,))),
        "junction_line_sink": (np.stack([
            (2.0-topology.product_line_multiplicity)*extent
            for topology, extent in zip(topologies, junction_extents)], axis=2)
            if junction_extents else np.zeros(grid+(0,))),
        "order": q_rate,
    }
    return CommonWallResidual(
        state_rate, channels, plastic_power, free_energy_rate, heat_rate)


def balance_ledger(state, residual, systems, topologies):
    """Return exact periodic reaction/line/Burgers and energy closures."""
    rate = residual.state_rate
    signed_rate = (rate.mobile_plus_m2-rate.mobile_minus_m2
                   +rate.forest_plus_m2-rate.forest_minus_m2
                   +rate.wall_plus_m2-rate.wall_minus_m2)
    burgers_rate = np.einsum(
        "...a,ai->...i", signed_rate,
        np.stack([system.burgers_vector_m for system in systems]))
    for index, topology in enumerate(topologies):
        burgers_rate += (rate.junction_m2[..., index, None]
                         *topology.product_burgers_m)
    reaction_burgers_residual = np.mean(burgers_rate, axis=(0, 1))
    burgers_scale = float(np.mean(np.sum(
        np.abs(signed_rate)[..., None]
        *np.abs(np.stack([system.burgers_vector_m for system in systems]))[None, None, ...],
        axis=(2, 3))))
    for index, topology in enumerate(topologies):
        burgers_scale += float(np.mean(
            np.abs(rate.junction_m2[..., index])
            *np.linalg.norm(topology.product_burgers_m)))
    line_rate = np.sum(
        rate.mobile_plus_m2+rate.mobile_minus_m2
        +rate.forest_plus_m2+rate.forest_minus_m2
        +rate.wall_plus_m2+rate.wall_minus_m2, axis=2)
    for index, topology in enumerate(topologies):
        line_rate += (topology.product_line_multiplicity
                      *rate.junction_m2[..., index])
    annihilation_sink = 2.0*np.sum(
        residual.channel_rates_m2_s["annihilation_pairs"], axis=2)
    multiplication_source = 2.0*np.sum(
        residual.channel_rates_m2_s["multiplication_pairs"], axis=2)
    junction_sink = np.sum(
        residual.channel_rates_m2_s["junction_line_sink"], axis=2)
    line_closure = float(np.mean(
        line_rate+annihilation_sink-multiplication_source+junction_sink))
    energy_closure = (residual.heat_rate_W_m3
                      -residual.plastic_power_W_m3
                      +residual.free_energy_rate_W_m3)
    energy_scale = max(
        abs(float(np.mean(residual.heat_rate_W_m3))),
        abs(float(np.mean(residual.plastic_power_W_m3))),
        abs(float(np.mean(residual.free_energy_rate_W_m3))), 1.0)
    return {
        "burgers_rate_residual_m_s": reaction_burgers_residual,
        "relative_burgers_rate_residual": float(
            np.linalg.norm(reaction_burgers_residual)/max(burgers_scale, 1e-300)),
        "line_rate_m2_s": float(np.mean(line_rate)),
        "annihilation_line_sink_m2_s": float(np.mean(annihilation_sink)),
        "multiplication_line_source_m2_s": float(np.mean(multiplication_source)),
        "junction_line_sink_m2_s": float(np.mean(junction_sink)),
        "line_balance_residual_m2_s": line_closure,
        "plastic_power_W_m3": float(np.mean(residual.plastic_power_W_m3)),
        "free_energy_rate_W_m3": float(np.mean(residual.free_energy_rate_W_m3)),
        "heat_rate_W_m3": float(np.mean(residual.heat_rate_W_m3)),
        "energy_balance_residual_W_m3": float(np.mean(energy_closure)),
        "relative_energy_balance_residual": float(
            abs(np.mean(energy_closure))/energy_scale),
    }


def _state_map(state, function):
    return CommonWallState(**{
        item.name: function(item.name, np.asarray(getattr(state, item.name)))
        for item in fields(state)})


def accepted_euler_step(state, driving, systems, topologies, parameters, dt_s):
    """Advance the common residual with one global positivity/bound limiter."""
    residual = wall_residual(state, driving, systems, topologies, parameters)
    rate = residual.state_rate
    scale = 1.0
    nonnegative = (
        "mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
        "forest_minus_m2", "wall_plus_m2", "wall_minus_m2", "junction_m2")
    for name in nonnegative:
        value = np.asarray(getattr(state, name)); derivative = np.asarray(getattr(rate, name))
        mask = derivative < 0.0
        if np.any(mask):
            scale = min(scale, float(np.min(
                parameters.maximum_fraction_per_step*value[mask]
                /np.maximum(-dt_s*derivative[mask], 1e-300))))
    q = state.wall_order; dq = rate.wall_order
    upper = dq > 0.0; lower = dq < 0.0
    if np.any(upper):
        scale = min(scale, float(np.min(
            parameters.maximum_fraction_per_step*(1.0-q[upper])
            /np.maximum(dt_s*dq[upper], 1e-300))))
    if np.any(lower):
        scale = min(scale, float(np.min(
            parameters.maximum_fraction_per_step*q[lower]
            /np.maximum(-dt_s*dq[lower], 1e-300))))
    scale = float(np.clip(scale, 0.0, 1.0))
    updated = _state_map(
        state, lambda name, value: value+dt_s*scale*np.asarray(getattr(rate, name)))
    updated.validate(systems, topologies)
    return updated, residual, scale


def jacobian_vector_product(state, direction, driving, systems, topologies,
                            parameters, relative_step=1.0e-6):
    """Centered JVP of the exact residual used by nonlinear production."""
    candidate_steps = []
    for item in fields(state):
        base = np.asarray(getattr(state, item.name))
        vector = np.asarray(getattr(direction, item.name))
        magnitude = float(np.max(np.abs(vector)))
        if magnitude > 0.0:
            candidate_steps.append(
                relative_step*max(float(np.max(np.abs(base))), 1.0)/magnitude)
    if not candidate_steps:
        return _state_map(state, lambda name, value: np.zeros_like(value))
    step = min(candidate_steps)
    plus = _state_map(state, lambda name, value: value+step*np.asarray(getattr(direction, name)))
    minus = _state_map(state, lambda name, value: value-step*np.asarray(getattr(direction, name)))
    rp = wall_residual(
        plus, driving, systems, topologies, parameters,
        differentiation_state=True).state_rate
    rm = wall_residual(
        minus, driving, systems, topologies, parameters,
        differentiation_state=True).state_rate
    return _state_map(rp, lambda name, value: (
        value-np.asarray(getattr(rm, name)))/(2.0*step))


def active_component_layout(state, parameters):
    """Declared normalized coordinates used by the projected common operator."""
    layout = []
    density_names = (
        "mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
        "forest_minus_m2", "wall_plus_m2", "wall_minus_m2")
    for name in density_names:
        for index in np.ndindex(np.asarray(getattr(state, name)).shape[2:]):
            layout.append((name, index, parameters.rho_reference_m2))
    for index in np.ndindex(np.asarray(state.junction_m2).shape[2:]):
        layout.append(("junction_m2", index, parameters.rho_reference_m2))
    layout.append(("wall_order", (), 1.0))
    for index in np.ndindex((3, 3)):
        layout.append(("beta_p", index, 1.0))
    layout.append(("orientation_rad", (), 1.0))
    layout.append(("temperature_K", (), parameters.bath_temperature_K))
    return tuple(layout)


def _component(field, index):
    return np.asarray(field)[(...,)+tuple(index)] if index else np.asarray(field)


def fourier_symbol(state, driving, systems, topologies, parameters,
                   mode_numbers, relative_step=1e-6):
    """Return the normalized Fourier symbol of :func:`wall_residual`.

    The symbol is assembled exclusively from JVPs of the nonlinear residual.
    Passive integrated slip and diagnostic family-Nye coordinates are omitted;
    beta, orientation, temperature, every reservoir, and wall order are active.
    """
    nx, ny = state.wall_order.shape
    mx, my = map(int, mode_numbers)
    x = np.arange(nx)[:, None]/nx
    y = np.arange(ny)[None, :]/ny
    phase = 2*np.pi*(mx*x+my*y)
    cosine, sine = np.cos(phase), np.sin(phase)
    norm_cos = float(np.mean(cosine*cosine))
    norm_sin = float(np.mean(sine*sine))
    zero_mode = mx == 0 and my == 0
    if norm_cos <= 0.0 or (not zero_mode and norm_sin <= 0.0):
        raise ValueError("invalid Fourier mode")
    layout = active_component_layout(state, parameters)
    symbol = np.zeros((len(layout), len(layout)), dtype=complex)
    zero = {name: np.zeros_like(value) for name, value in state.__dict__.items()}
    for column, (input_name, input_index, input_scale) in enumerate(layout):
        direction_arrays = {name: value.copy() for name, value in zero.items()}
        target = direction_arrays[input_name]
        if input_index:
            target[(...,)+input_index] = input_scale*cosine
        else:
            target[...] = input_scale*cosine
        direction = CommonWallState(**direction_arrays)
        response = jacobian_vector_product(
            state, direction, driving, systems, topologies, parameters,
            relative_step=relative_step)
        for row, (output_name, output_index, output_scale) in enumerate(layout):
            value = _component(getattr(response, output_name), output_index)/output_scale
            real_part = float(np.mean(value*cosine)/norm_cos)
            imag_part = (0.0 if zero_mode else
                         -float(np.mean(value*sine)/norm_sin))
            symbol[row, column] = real_part+1j*imag_part
    frequency = np.sqrt((mx/(nx*parameters.spacing_m))**2
                        +(my/(ny*parameters.spacing_m))**2)
    wavelength = None if frequency == 0.0 else 1.0/frequency
    eigenvalues, eigenvectors = np.linalg.eig(symbol)
    return {
        "matrix_s_inv": symbol,
        "eigenvalues_s_inv": eigenvalues,
        "right_eigenvectors": eigenvectors,
        "layout": layout,
        "mode_numbers": (mx, my),
        "wavelength_m": (None if wavelength is None else float(wavelength)),
    }
