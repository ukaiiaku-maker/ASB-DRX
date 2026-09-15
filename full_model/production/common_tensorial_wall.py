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
        alignment_increment_from_slip, bcc_four_family_systems,
    )
    from .nonlocal_elasticity import solve_periodic_eigenstrain
except ImportError:  # direct execution by the production driver
    from tensorial_nye import (
        JunctionTopology, SlipSystem3D, nye_from_plastic_distortion,
        plastic_distortion_from_slip, rotated_system_fields,
        alignment_increment_from_slip, bcc_four_family_systems,
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
    multi_hit_coordination: np.ndarray
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
        if (np.asarray(self.wall_order).shape != grid
                or np.asarray(self.multi_hit_coordination).shape != grid):
            raise ValueError("wall order/coordination has inconsistent layout")
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
            if np.any(np.asarray(self.multi_hit_coordination) < 0.0) or np.any(
                    np.asarray(self.multi_hit_coordination) > 1.0):
                raise ValueError("multi-hit coordination must lie in [0,1]")
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
    wall_absent_penalty_J_m3: float = 4.0e5
    wall_gate_form: str = "joint_rational"
    wall_gate_half_density_ratio: float = 0.20
    wall_gate_half_polarization: float = 0.25
    wall_gate_joint_half: float = 0.08
    polarization_density_epsilon_ratio: float = 1.0e-12
    wall_partition_J_m: float = 2.0e-10
    wall_target_m2: float = 4.0e14
    wall_density_center_ratio: float = 1.3
    wall_density_width_ratio: float = 0.3
    wall_order_gradient_J_m: float = 2.0e-7
    wall_order_enabled: bool = True
    extensive_wall_partition_enabled: bool = False
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
    bound_active_tolerance: float = 1.0e-12
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
    transport_scheme: str = "spectral"
    multiplication_coefficient: float = 10.0
    taylor_alpha: float = 0.30
    taylor_wall_weight: float = 2.0
    taylor_junction_weight: float = 1.0
    taylor_regularization_Pa: float = 2.0e7
    multi_hit_enabled: bool = False
    multi_hit_relaxation_s: float = 1.0e-5
    multi_hit_collision_scale: float = 1.0
    multi_hit_lock_log_factor: float = 0.0
    multi_hit_wall_log_factor: float = 1.0
    multi_hit_junction_log_factor: float = 1.0
    multi_hit_annihilation_log_factor: float = -0.5

    def __post_init__(self):
        positive = (
            self.spacing_m, self.burgers_m, self.rho_reference_m2,
            self.line_energy_J_m, self.wall_order_barrier_J_m3,
            self.wall_absent_penalty_J_m3,
            self.wall_partition_J_m, self.wall_target_m2,
            self.wall_density_width_ratio, self.attempt_frequency_s,
            self.critical_stress_Pa, self.exp_n,
            self.reaction_energy_scale_J_m,
            self.c11_Pa, self.c44_Pa, self.glide_speed_attempt_m_s,
            self.volumetric_heat_capacity_J_m3_K,
            self.wall_gate_half_density_ratio,
            self.wall_gate_half_polarization, self.wall_gate_joint_half,
            self.polarization_density_epsilon_ratio,
            self.taylor_regularization_Pa, self.multi_hit_relaxation_s,
            self.multi_hit_collision_scale,
        )
        if any(not math.isfinite(float(value)) or value <= 0.0 for value in positive):
            raise ValueError("common wall parameters must be finite and positive")
        if not 0.0 <= self.exp_floor <= 1.0 or not 0.0 < self.maximum_fraction_per_step <= 1.0:
            raise ValueError("bounded fractions are invalid")
        if not 0.0 < self.bound_active_tolerance < 0.5:
            raise ValueError("bound active tolerance must lie in (0,0.5)")
        if self.mobile_correlation_diffusivity_m2_s < 0.0:
            raise ValueError("mobile correlation diffusivity cannot be negative")
        if self.transport_scheme not in ("spectral", "upwind"):
            raise ValueError("transport scheme must be spectral or upwind")
        if self.multiplication_coefficient < 0.0:
            raise ValueError("multiplication coefficient cannot be negative")
        if self.taylor_alpha < 0.0 or self.taylor_wall_weight < 0.0 or self.taylor_junction_weight < 0.0:
            raise ValueError("Taylor weights cannot be negative")
        if self.wall_gate_form not in ("product_rational", "joint_rational"):
            raise ValueError("unknown wall polarization gate")
        if any(abs(value) > 5.0 for value in (
                self.multi_hit_lock_log_factor, self.multi_hit_wall_log_factor,
                self.multi_hit_junction_log_factor,
                self.multi_hit_annihilation_log_factor)):
            raise ValueError("multi-hit rate modifiers must remain bounded")


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


def _periodic_upwind_rate(density, velocity, spacing):
    """Conservative first-order finite-volume advection on a periodic grid."""
    rho = np.asarray(density, dtype=float)
    vel = np.asarray(velocity, dtype=float)
    rate = np.zeros_like(rho)
    for axis in (0, 1):
        right_rho = np.roll(rho, -1, axis=axis)
        face_velocity = 0.5*(vel[..., axis]+np.roll(vel[..., axis], -1, axis=axis))
        face_flux = (np.maximum(face_velocity, 0.0)*rho
                     +np.minimum(face_velocity, 0.0)*right_rho)
        rate -= (face_flux-np.roll(face_flux, 1, axis=axis))/spacing
    return rate


def exp_floor_rate(stress_pa, temperature_K, barrier_eV, parameters):
    stress = np.abs(np.asarray(stress_pa, dtype=float))
    temperature = np.asarray(temperature_K, dtype=float)
    enthalpy = float(barrier_eV)*EV_J*(
        parameters.exp_floor+(1.0-parameters.exp_floor)*np.exp(
            -parameters.exp_a*(stress/parameters.critical_stress_Pa)**parameters.exp_n))
    free = enthalpy-KB_J_K*temperature*parameters.activation_entropy_kB
    return parameters.attempt_frequency_s*np.exp(np.clip(
        -free/(KB_J_K*temperature), -700.0, 40.0))


def wall_polarization_invariants(state, systems, parameters):
    """Objective wall-content/polarization gate and exact density derivatives."""
    wall_plus = np.asarray(state.wall_plus_m2)
    wall_minus = np.asarray(state.wall_minus_m2)
    wall = np.sum(wall_plus+wall_minus, axis=2)
    signed = wall_plus-wall_minus
    burgers, directions, normals = rotated_system_fields(
        systems, state.orientation_rad)
    lines = np.cross(normals, directions)
    lines /= np.maximum(np.linalg.norm(lines, axis=-1)[..., None], 1e-300)
    basis = np.einsum("...ai,...aj->...aij", burgers, lines)
    alpha_wall = np.einsum("...a,...aij->...ij", signed, basis)
    norm = np.sqrt(np.sum(alpha_wall*alpha_wall, axis=(-2, -1)))
    dnorm_ds = np.einsum("...ij,...aij->...a", alpha_wall, basis)
    dnorm_ds /= np.maximum(norm[..., None], 1e-300)
    reference = parameters.rho_reference_m2
    epsilon = (parameters.polarization_density_epsilon_ratio
               *parameters.burgers_m*reference)
    denominator = parameters.burgers_m*wall+epsilon
    polarization = norm/denominator
    dpi_dnorm = 1.0/denominator
    dpi_dwall = -norm*parameters.burgers_m/(denominator*denominator)
    if parameters.wall_gate_form == "product_rational":
        half_rho = parameters.wall_gate_half_density_ratio*reference
        phi_gate = wall/(wall+half_rho)
        dphi_dwall = half_rho/(wall+half_rho)**2
        half_pi2 = parameters.wall_gate_half_polarization**2
        pi_gate = polarization**2/(polarization**2+half_pi2)
        dpi_gate = (2.0*polarization*half_pi2
                    /(polarization**2+half_pi2)**2)
        gate = phi_gate*pi_gate
        dgate_dnorm = phi_gate*dpi_gate*dpi_dnorm
        dgate_dwall = dphi_dwall*pi_gate+phi_gate*dpi_gate*dpi_dwall
    else:
        joint = wall/reference*polarization
        half2 = parameters.wall_gate_joint_half**2
        gate = joint*joint/(joint*joint+half2)
        dgate_djoint = 2.0*joint*half2/(joint*joint+half2)**2
        djoint_dnorm = wall/reference*dpi_dnorm
        djoint_dwall = polarization/reference+wall/reference*dpi_dwall
        dgate_dnorm = dgate_djoint*djoint_dnorm
        dgate_dwall = dgate_djoint*djoint_dwall
    dgate_plus = dgate_dwall[..., None]+dgate_dnorm[..., None]*dnorm_ds
    dgate_minus = dgate_dwall[..., None]-dgate_dnorm[..., None]*dnorm_ds
    return {
        "wall_density_m2": wall,
        "wall_alpha_m1": alpha_wall,
        "wall_alpha_norm_m1": norm,
        "wall_polarization": polarization,
        "wall_gate": gate,
        "wall_gate_derivative_plus_m2": dgate_plus,
        "wall_gate_derivative_minus_m2": dgate_minus,
    }


def wall_free_energy_derivatives(state, parameters, topologies=(), systems=None):
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
    if systems is None:
        systems = bcc_four_family_systems(parameters.burgers_m)
    invariants = wall_polarization_invariants(state, systems, parameters)
    wall = invariants["wall_density_m2"]
    gate = invariants["wall_gate"]
    q = state.wall_order
    reference = parameters.rho_reference_m2
    safe = np.maximum(total, 1e-30*reference)
    base_mu = (parameters.line_energy_J_m
               +parameters.correlation_energy_J_m*(np.log(safe/reference)+1.0))
    # Physical polarization initiates ordering without numerical q noise:
    # h'(0)=2 and h'(1)=0. The objective gate makes this source identically
    # zero for absent or unpolarized wall content.
    h = q*(2.0-q)
    dh = 2.0*(1.0-q)
    partition_offset = wall-gate*h*parameters.wall_target_m2
    partition_direct_mu = (parameters.wall_partition_J_m*partition_offset
                           /parameters.wall_target_m2)
    common_mu = base_mu
    forest_mu = common_mu+parameters.forest_energy_J_m
    denergy_dgate = (
        -parameters.wall_absent_penalty_J_m3*q*q
        -parameters.wall_order_amplitude_J_m3*h
        -parameters.wall_partition_J_m*partition_offset*h)
    wall_plus_mu = (common_mu[..., None]+partition_direct_mu[..., None]
                    +denergy_dgate[..., None]
                    *invariants["wall_gate_derivative_plus_m2"])
    wall_minus_mu = (common_mu[..., None]+partition_direct_mu[..., None]
                     +denergy_dgate[..., None]
                     *invariants["wall_gate_derivative_minus_m2"])
    junction_mu = (multiplicity*common_mu[..., None]
                   +multiplicity*parameters.junction_energy_J_m
                   +topology_energy)
    barrier_derivative = (2.0*parameters.wall_order_barrier_J_m3*q*(1.0-q)
                          *(1.0-2.0*q))
    dq = (barrier_derivative
          +2.0*parameters.wall_absent_penalty_J_m3*(1.0-gate)*q
          -parameters.wall_order_amplitude_J_m3*gate*dh
          -parameters.wall_partition_J_m*partition_offset*gate*dh)
    return {
        "total_density_m2": total, "wall_density_m2": wall,
        **invariants,
        "mobile_mu_J_m": common_mu, "forest_mu_J_m": forest_mu,
        "wall_plus_mu_J_m": wall_plus_mu,
        "wall_minus_mu_J_m": wall_minus_mu,
        "junction_mu_J_m": junction_mu,
        "wall_order_derivative_J_m3": dq,
    }


def wall_free_energy_density_J_m3(state, parameters, topologies=(), systems=None):
    """Local part of the exact common free energy differentiated above."""
    chemical = wall_free_energy_derivatives(
        state, parameters, topologies, systems)
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
    gate = chemical["wall_gate"]
    h = q*(2.0-q)
    barrier = parameters.wall_order_barrier_J_m3*q*q*(1.0-q)*(1.0-q)
    absent = parameters.wall_absent_penalty_J_m3*(1.0-gate)*q*q
    ordering = -parameters.wall_order_amplitude_J_m3*gate*h
    partition = (.5*parameters.wall_partition_J_m
                 *(wall-gate*h*parameters.wall_target_m2)**2
                 /parameters.wall_target_m2)
    return density+forest+junction+barrier+absent+ordering+partition


def _biased_exchange(source, target, delta_mu_J_m, rate_s, parameters):
    return _biased_exchange_components(
        source, target, delta_mu_J_m, rate_s, parameters)[0]


def _biased_exchange_components(source, target, delta_mu_J_m, rate_s,
                                parameters):
    # Attempt-bounded detailed-balance split.  The former symmetric
    # exponential split had the correct ratio but made one coefficient
    # unbounded as |Delta mu| grew, producing a Zeno timestep when a parent
    # reservoir was exhausted. Here k+/k-=exp(-Delta mu/E) exactly, while
    # both coefficients remain smoothly in [0,2*k_attempt].
    bias = np.tanh(
        delta_mu_J_m/(2.0*parameters.reaction_energy_scale_J_m))
    forward = rate_s*source*(1.0-bias)
    reverse = rate_s*target*(1.0+bias)
    return forward-reverse, forward+reverse


def taylor_resistance_Pa(state, systems, topologies, parameters):
    """Objective family resistance from forest, wall, and junction content."""
    obstacle = (state.forest_plus_m2+state.forest_minus_m2
                +parameters.taylor_wall_weight
                *(state.wall_plus_m2+state.wall_minus_m2))
    obstacle = obstacle.copy()
    for index, topology in enumerate(topologies):
        contribution = (parameters.taylor_junction_weight
                        *topology.product_line_multiplicity
                        *state.junction_m2[..., index])
        obstacle[..., topology.parent_a] += .5*contribution
        obstacle[..., topology.parent_b] += .5*contribution
    return (parameters.taylor_alpha*parameters.c44_Pa*parameters.burgers_m
            *np.sqrt(np.maximum(obstacle, 0.0)))


def resolved_driving_components(state, driving, systems, topologies, parameters):
    """Return raw/effective stress, Taylor resistance, and glide speed."""
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
        raw = np.einsum("...ij,...sij->...s", stress_tensor, schmid)
    else:
        raw = np.asarray(driving.resolved_stress_Pa, dtype=float)
    if raw.shape != family_shape:
        raise ValueError("resolved stress requires grid x family layout")
    resistance = taylor_resistance_Pa(state, systems, topologies, parameters)
    if parameters.taylor_alpha == 0.0:
        effective = raw.copy()
    else:
        smooth = np.sqrt(raw*raw+resistance*resistance
                         +parameters.taylor_regularization_Pa**2)
        effective = raw*(1.0-resistance/smooth)
    if driving.glide_speed_m_s is None:
        activation = exp_floor_rate(
            effective, state.temperature_K[..., None],
            parameters.glide_barrier_eV, parameters)/parameters.attempt_frequency_s
        speed = (parameters.glide_speed_attempt_m_s*activation
                 *np.tanh(effective/parameters.critical_stress_Pa))
    else:
        speed = np.asarray(driving.glide_speed_m_s, dtype=float)
    if speed.shape != family_shape:
        raise ValueError("glide speed requires grid x family layout")
    return {"speed_m_s": speed, "raw_stress_Pa": raw,
            "effective_stress_Pa": effective,
            "taylor_resistance_Pa": resistance}


def resolved_driving_fields(state, driving, systems, parameters):
    """Resolve nonlocal stress and glide speed used by the common residual."""
    components = resolved_driving_components(
        state, driving, systems, (), parameters)
    return components["speed_m_s"], components["effective_stress_Pa"]


def wall_residual(state: CommonWallState, driving: CommonWallDriving,
                  systems: tuple[SlipSystem3D, ...],
                  topologies: tuple[JunctionTopology, ...],
                  parameters: CommonWallParameters, *,
                  differentiation_state=False):
    """Evaluate the single authoritative nonlinear residual."""
    grid, family_shape = state.validate(
        systems, topologies, admissibility=not differentiation_state)
    drive = resolved_driving_components(
        state, driving, systems, topologies, parameters)
    speed = drive["speed_m_s"]
    stress = drive["effective_stress_Pa"]
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
    if parameters.transport_scheme == "upwind":
        mp_rate = _periodic_upwind_rate(
            state.mobile_plus_m2, speed[..., None]*planar,
            parameters.spacing_m)
        mm_rate = _periodic_upwind_rate(
            state.mobile_minus_m2, -speed[..., None]*planar,
            parameters.spacing_m)
    else:
        mp_rate = -_divergence(flux_plus, parameters.spacing_m)
        mm_rate = -_divergence(flux_minus, parameters.spacing_m)
    slip_rate = (parameters.burgers_m*speed
                 *(state.mobile_plus_m2+state.mobile_minus_m2))
    fp_rate = np.zeros(family_shape); fm_rate = np.zeros(family_shape)
    wp_rate = np.zeros(family_shape); wm_rate = np.zeros(family_shape)
    junction_rate = np.zeros(grid+(len(topologies),))

    chemical = wall_free_energy_derivatives(
        state, parameters, topologies, systems)
    temperature = state.temperature_K[..., None]
    coordination = state.multi_hit_coordination[..., None]

    def coordinated(base, log_factor):
        if not parameters.multi_hit_enabled:
            return base
        return base*np.exp(log_factor*coordination)

    k_lock = exp_floor_rate(stress, temperature, parameters.lock_barrier_eV, parameters)
    k_lock = coordinated(k_lock, parameters.multi_hit_lock_log_factor)
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
    k_wall = coordinated(k_wall, parameters.multi_hit_wall_log_factor)
    delta_wall_p = (chemical["wall_plus_mu_J_m"]
                    -chemical["forest_mu_J_m"][..., None])
    delta_wall_m = (chemical["wall_minus_mu_J_m"]
                    -chemical["forest_mu_J_m"][..., None])
    wall_target_plus = state.wall_plus_m2
    wall_target_minus = state.wall_minus_m2
    if parameters.extensive_wall_partition_enabled:
        # In V23 wall_order is a derived ordered/total fraction, not a state.
        # Capture/release acts only on the disordered (tangle) complement.
        wall_target_plus = wall_target_plus*(1.0-state.wall_order[..., None])
        wall_target_minus = wall_target_minus*(1.0-state.wall_order[..., None])
    transfer_p, transfer_p_turnover = _biased_exchange_components(
        state.forest_plus_m2, wall_target_plus,
        delta_wall_p, k_wall, parameters)
    transfer_m, transfer_m_turnover = _biased_exchange_components(
        state.forest_minus_m2, wall_target_minus,
        delta_wall_m, k_wall, parameters)
    fp_rate -= transfer_p; fm_rate -= transfer_m
    wp_rate += transfer_p; wm_rate += transfer_m

    k_ann = exp_floor_rate(stress, temperature,
                           parameters.annihilation_barrier_eV, parameters)
    k_ann = coordinated(k_ann, parameters.multi_hit_annihilation_log_factor)
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
        if parameters.multi_hit_enabled:
            kf = kf*np.exp(parameters.multi_hit_junction_log_factor
                           *state.multi_hit_coordination)
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

    junction_turnover_array = (np.stack(junction_turnovers, axis=2)
                               if junction_turnovers else np.zeros(grid+(0,)))
    collision_frequency = (parameters.multi_hit_collision_scale
                           *np.sum(junction_turnover_array, axis=2)
                           /parameters.rho_reference_m2)
    coordination_rate = (np.zeros(grid) if not parameters.multi_hit_enabled else
                         (1.0-state.multi_hit_coordination)*collision_frequency
                         -state.multi_hit_coordination
                         /parameters.multi_hit_relaxation_s)
    tolerance = parameters.bound_active_tolerance
    coordination_rate = np.where(
        (state.multi_hit_coordination <= tolerance)&(coordination_rate < 0.0),
        0.0, coordination_rate)
    coordination_rate = np.where(
        (state.multi_hit_coordination >= 1.0-tolerance)&(coordination_rate > 0.0),
        0.0, coordination_rate)

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
    raw_q_rate = (-k_order*q_chemical_potential
                  /max(parameters.wall_order_barrier_J_m3, 1.0))
    # Directional bound-degenerate Onsager mobility. It preserves the sign of
    # -delta F/dq and therefore dissipation, permits a physical polarized-wall
    # source at q=0, and makes an outward rate vanish in proportion to distance
    # from the applicable bound. No order pixel can create a Zeno timestep.
    q_rate = np.where(raw_q_rate >= 0.0,
                      (1.0-state.wall_order)*raw_q_rate,
                      state.wall_order*raw_q_rate)
    if not parameters.wall_order_enabled:
        q_rate = np.zeros(grid)

    zero_beta = np.zeros_like(state.beta_p)
    zero_nye = np.zeros_like(state.family_nye_m1)
    tx, ty = _spectral_gradient(state.temperature_K, parameters.spacing_m)
    lap_temperature = _divergence(
        np.stack((tx, ty), axis=-1), parameters.spacing_m)
    state_rate = CommonWallState(
        mp_rate, mm_rate, fp_rate, fm_rate, wp_rate, wm_rate,
        junction_rate, np.real(q_rate), coordination_rate, slip_rate, beta_rate,
        alignment_rate, family_nye_rate, orientation_rate,
        np.zeros_like(state.temperature_K))
    # Raw mechanical work includes the smooth Taylor-friction loss; the
    # effective drive controls kinetics and has the same sign, so both raw and
    # effective stress powers are nonnegative.
    plastic_power = np.sum(drive["raw_stress_Pa"]*slip_rate, axis=2)
    # Exact directional derivative of the declared defect free-energy
    # functional.  The q term includes the variational gradient contribution.
    free_energy_rate = (
        chemical["mobile_mu_J_m"]*np.sum(mp_rate+mm_rate, axis=2)
        +chemical["forest_mu_J_m"]*np.sum(fp_rate+fm_rate, axis=2)
        +np.sum(chemical["wall_plus_mu_J_m"]*wp_rate
                +chemical["wall_minus_mu_J_m"]*wm_rate, axis=2)
        +np.sum(chemical["junction_mu_J_m"]*junction_rate, axis=2)
        +q_chemical_potential*q_rate)
    heat_rate = plastic_power-free_energy_rate
    temperature_rate = (
        heat_rate/parameters.volumetric_heat_capacity_J_m3_K
        +parameters.thermal_diffusivity_m2_s*lap_temperature
        -parameters.bath_rate_s*(state.temperature_K-parameters.bath_temperature_K))
    state_rate = CommonWallState(
        mp_rate, mm_rate, fp_rate, fm_rate, wp_rate, wm_rate,
        junction_rate, np.real(q_rate), coordination_rate, slip_rate, beta_rate,
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
        "junction_turnover": junction_turnover_array,
        "junction_line_sink": (np.stack([
            (2.0-topology.product_line_multiplicity)*extent
            for topology, extent in zip(topologies, junction_extents)], axis=2)
            if junction_extents else np.zeros(grid+(0,))),
        "order": q_rate, "coordination": coordination_rate,
        "collision_frequency_s": collision_frequency,
        "raw_stress_Pa": drive["raw_stress_Pa"],
        "effective_stress_Pa": stress,
        "taylor_resistance_Pa": drive["taylor_resistance_Pa"],
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
        donor = value
        if (parameters.extensive_wall_partition_enabled
                and name in ("wall_plus_m2", "wall_minus_m2")):
            donor = value*(1.0-state.wall_order[..., None])
        mask = derivative < 0.0
        if np.any(mask):
            scale = min(scale, float(np.min(
                parameters.maximum_fraction_per_step*donor[mask]
                /np.maximum(-dt_s*derivative[mask], 1e-300))))
    for name in ("wall_order", "multi_hit_coordination"):
        value = np.asarray(getattr(state, name))
        derivative = np.asarray(getattr(rate, name))
        upper = derivative > 0.0; lower = derivative < 0.0
        if np.any(upper):
            scale = min(scale, float(np.min(
                parameters.maximum_fraction_per_step*(1.0-value[upper])
                /np.maximum(dt_s*derivative[upper], 1e-300))))
        if np.any(lower):
            scale = min(scale, float(np.min(
                parameters.maximum_fraction_per_step*value[lower]
                /np.maximum(-dt_s*derivative[lower], 1e-300))))
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


def accepted_step_jacobian_vector_product(
        state, direction, driving, systems, topologies, parameters, dt_s,
        relative_step=1.0e-6):
    """Centered JVP of the actual bounded accepted-step map.

    This differentiates through the global adaptive acceptance factor.  It is
    therefore the authoritative tangent for finite-time/non-normal analysis,
    including active-set effects omitted by ``I + dt*J_residual``.
    """
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
    plus = _state_map(
        state, lambda name, value: value+step*np.asarray(getattr(direction, name)))
    minus = _state_map(
        state, lambda name, value: value-step*np.asarray(getattr(direction, name)))
    mapped_plus = accepted_euler_step(
        plus, driving, systems, topologies, parameters, dt_s)[0]
    mapped_minus = accepted_euler_step(
        minus, driving, systems, topologies, parameters, dt_s)[0]
    return _state_map(mapped_plus, lambda name, value: (
        value-np.asarray(getattr(mapped_minus, name)))/(2.0*step))


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
    layout.append(("multi_hit_coordination", (), 1.0))
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


def accepted_step_fourier_symbol(
        state, driving, systems, topologies, parameters, mode_numbers, dt_s,
        relative_step=1e-6):
    """Fourier projection of the authoritative accepted-step tangent map."""
    nx, ny = state.wall_order.shape
    mx, my = map(int, mode_numbers)
    x = np.arange(nx)[:, None]/nx; y = np.arange(ny)[None, :]/ny
    phase = 2*np.pi*(mx*x+my*y)
    cosine, sine = np.cos(phase), np.sin(phase)
    norm_cos = float(np.mean(cosine*cosine))
    norm_sin = float(np.mean(sine*sine))
    zero_mode = mx == 0 and my == 0
    layout = active_component_layout(state, parameters)
    symbol = np.zeros((len(layout), len(layout)), dtype=complex)
    zero = {name: np.zeros_like(value) for name, value in state.__dict__.items()}
    for column, (input_name, input_index, input_scale) in enumerate(layout):
        arrays = {name: value.copy() for name, value in zero.items()}
        target = arrays[input_name]
        if input_index:
            target[(...,)+input_index] = input_scale*cosine
        else:
            target[...] = input_scale*cosine
        response = accepted_step_jacobian_vector_product(
            state, CommonWallState(**arrays), driving, systems, topologies,
            parameters, dt_s, relative_step)
        for row, (output_name, output_index, output_scale) in enumerate(layout):
            value = _component(getattr(response, output_name), output_index)/output_scale
            real_part = float(np.mean(value*cosine)/norm_cos)
            imag_part = (0.0 if zero_mode else
                         -float(np.mean(value*sine)/norm_sin))
            symbol[row, column] = real_part+1j*imag_part
    frequency = np.sqrt((mx/(nx*parameters.spacing_m))**2
                        +(my/(ny*parameters.spacing_m))**2)
    return {"matrix": symbol, "singular_values": np.linalg.svd(
                symbol, compute_uv=False), "layout": layout,
            "mode_numbers": (mx, my),
            "wavelength_m": None if frequency == 0.0 else float(1/frequency)}
