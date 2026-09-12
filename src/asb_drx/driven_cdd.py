"""Gate B1 driven, Gate-A-coupled signed continuum dislocation dynamics.

This module keeps positive and negative populations independent and transports
them through density-weighted fluxes.  The only driving stress is the current
Gate A material-point response plus signed elastic/correlation stresses.  No
negative signed-density potential or prescribed wavelength is present.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from .bertin_bcc import (
    BertinBCCParameters,
    BertinBCCState,
    evaluate_bertin_bcc,
    initial_orientation,
)
from .signed_transport import default_slip_geometry, kinematics_from_populations, nye_tensor_1d


@dataclass(frozen=True)
class DrivenCDDParameters:
    domain_m: float = 16.0e-6
    backstress_coefficient: float = 1.0
    diffusion_coefficient: float = 1.0
    elastic_kernel_scale: float = 0.0
    cfl_limit: float = 0.35
    max_coupled_strain_increment: float = 1.25e-4
    density_floor_m2: float = 1.0e8
    collective_scale: float = 0.0
    same_family_lock_rate_s_inv: float = 0.0
    unlock_rate_s_inv: float = 0.0
    cross_family_lock_rate_s_inv: float = 0.0
    enforce_common_resolved_stress: bool = True
    homogenize_gate_a_pair_sources: bool = True

    def __post_init__(self) -> None:
        for name in ("domain_m", "backstress_coefficient", "diffusion_coefficient", "cfl_limit", "max_coupled_strain_increment", "density_floor_m2"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("elastic_kernel_scale", "same_family_lock_rate_s_inv", "unlock_rate_s_inv", "cross_family_lock_rate_s_inv"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not self.cfl_limit <= 0.5:
            raise ValueError("cfl_limit must not exceed 0.5")
        if self.collective_scale != 0.0:
            raise ValueError("DD collective closure is disabled in Gate B1")


@dataclass(frozen=True)
class DrivenCDDState:
    plastic_deformation_gradient: np.ndarray
    initial_orientation: np.ndarray
    mobile_plus_m2: np.ndarray
    mobile_minus_m2: np.ndarray
    locked_plus_m2: np.ndarray
    locked_minus_m2: np.ndarray
    temperature_K: np.ndarray
    axial_true_strain: float = 0.0
    time_s: float = 0.0
    accepted_steps: int = 0
    accumulated_slip: np.ndarray | None = None

    def __post_init__(self) -> None:
        fp = np.asarray(self.plastic_deformation_gradient, dtype=float)
        orientation = np.asarray(self.initial_orientation, dtype=float)
        plus = np.asarray(self.mobile_plus_m2, dtype=float)
        minus = np.asarray(self.mobile_minus_m2, dtype=float)
        locked_plus = np.asarray(self.locked_plus_m2, dtype=float)
        locked_minus = np.asarray(self.locked_minus_m2, dtype=float)
        temperature = np.asarray(self.temperature_K, dtype=float)
        slip = (
            np.zeros_like(plus) if self.accumulated_slip is None
            else np.asarray(self.accumulated_slip, dtype=float)
        )
        if plus.ndim != 2 or plus.shape[0] != 4:
            raise ValueError("signed populations must have shape (4, n)")
        n = plus.shape[1]
        if any(item.shape != plus.shape for item in (minus, locked_plus, locked_minus)):
            raise ValueError("all population arrays must have shape (4, n)")
        if fp.shape != (n, 3, 3) or orientation.shape != (n, 3, 3) or temperature.shape != (n,):
            raise ValueError("spatial Gate A fields have inconsistent shapes")
        if slip.shape != plus.shape:
            raise ValueError("accumulated slip must have shape (4, n)")
        if any(np.any(~np.isfinite(item)) for item in (fp, orientation, plus, minus, locked_plus, locked_minus, temperature, slip)):
            raise ValueError("state fields must be finite")
        if any(np.any(item < 0.0) for item in (plus, minus, locked_plus, locked_minus)):
            raise ValueError("all line populations must be nonnegative")
        if np.any(temperature <= 0.0) or np.any(np.linalg.det(fp) <= 0.0):
            raise ValueError("temperature and det(Fp) must be positive")
        if self.time_s < 0.0 or self.accepted_steps < 0:
            raise ValueError("time and accepted steps must be nonnegative")
        for name, value in (
            ("plastic_deformation_gradient", fp), ("initial_orientation", orientation),
            ("mobile_plus_m2", plus), ("mobile_minus_m2", minus),
            ("locked_plus_m2", locked_plus), ("locked_minus_m2", locked_minus),
            ("temperature_K", temperature),
            ("accumulated_slip", slip),
        ):
            object.__setattr__(self, name, value.copy())

    @property
    def grid_points(self) -> int:
        return self.mobile_plus_m2.shape[1]

    @property
    def physical_grain_count(self) -> int:
        return 1


@dataclass(frozen=True)
class DrivenCDDResponse:
    resolved_shear_Pa: np.ndarray
    effective_shear_Pa: np.ndarray
    taylor_friction_Pa: np.ndarray
    patterning_active: np.ndarray
    mrssp_angles_rad: np.ndarray
    inactive_weight: np.ndarray
    gate_a_shear_rates_s_inv: np.ndarray
    actual_shear_rates_s_inv: np.ndarray
    velocity_plus_m_s: np.ndarray
    velocity_minus_m_s: np.ndarray
    self_consistent_stress_Pa: np.ndarray
    backstress_Pa: np.ndarray
    diffusion_stress_Pa: np.ndarray
    orientations: np.ndarray
    slip_dyads: np.ndarray
    transport_substeps: int


@dataclass(frozen=True)
class CDDLedger:
    mobile_before_m_inv: float
    transport_boundary_flux_m_inv: float
    pair_multiplication_m_inv: float
    pair_annihilation_m_inv: float
    annihilation_limited_m_inv: float
    same_family_lock_transfer_m_inv: float
    cross_family_lock_transfer_m_inv: float
    unlock_transfer_m_inv: float
    mobile_after_m_inv: float
    locked_before_m_inv: float
    locked_after_m_inv: float
    total_balance_residual_m_inv: float
    maximum_family_signed_residual_m_inv: float
    vector_burgers_residual: float


def initialize_driven_cdd(
    grid_points: int,
    parameters: DrivenCDDParameters,
    gate_a_parameters: BertinBCCParameters,
    *,
    loading_axis_hkl: tuple[int, int, int] = (4, 1, 9),
    temperature_K: float = 300.0,
    density_scale: float = 1.0,
    noise_amplitude: float = 0.0,
    total_noise_amplitude: float | None = None,
    signed_noise_amplitude: float | None = None,
    seed: int = 0,
    remove_mode: int | None = None,
    spectral_noise_modes: int | None = None,
) -> DrivenCDDState:
    if grid_points < 16 or grid_points % 2:
        raise ValueError("an even grid with at least 16 points is required")
    total_amplitude = noise_amplitude if total_noise_amplitude is None else total_noise_amplitude
    signed_amplitude = noise_amplitude if signed_noise_amplitude is None else signed_noise_amplitude
    if density_scale <= 0.0 or not 0.0 <= total_amplitude < 0.2 or not 0.0 <= signed_amplitude < 0.2:
        raise ValueError("invalid density scale or noise amplitude")
    base = np.asarray([4.8, 4.7, 4.9, 4.75])[:, None] * 1.0e14 * density_scale
    rng = np.random.default_rng(seed)
    if spectral_noise_modes is not None:
        if spectral_noise_modes < 1 or spectral_noise_modes >= grid_points // 2:
            raise ValueError("spectral noise band must lie below grid Nyquist")
        coordinate = 2.0 * np.pi * np.arange(grid_points) / grid_points
        total_noise = np.zeros((4, grid_points))
        signed_noise = np.zeros((4, grid_points))
        for field in (total_noise, signed_noise):
            cosine_coefficients = rng.normal(size=(4, spectral_noise_modes))
            sine_coefficients = rng.normal(size=(4, spectral_noise_modes))
            for mode in range(1, spectral_noise_modes + 1):
                if mode == remove_mode:
                    continue
                field += cosine_coefficients[:, mode - 1, None] * np.cos(mode * coordinate)[None, :]
                field += sine_coefficients[:, mode - 1, None] * np.sin(mode * coordinate)[None, :]
    else:
        total_noise = rng.normal(size=(4, grid_points))
        signed_noise = rng.normal(size=(4, grid_points))
    for field in (total_noise, signed_noise):
        field -= np.mean(field, axis=1, keepdims=True)
        if remove_mode is not None:
            transform = np.fft.fft(field, axis=1)
            transform[:, remove_mode] = 0.0
            transform[:, -remove_mode] = 0.0
            field[:] = np.fft.ifft(transform, axis=1).real
        field /= np.max(np.abs(field), axis=1, keepdims=True)
    total = base * (1.0 + total_amplitude * total_noise)
    kappa = base * signed_amplitude * signed_noise
    plus = 0.5 * (total + kappa)
    minus = 0.5 * (total - kappa)
    _, beta, _ = kinematics_from_populations(
        plus, minus, parameters.domain_m, gate_a_parameters.burgers_m
    )
    fp = np.asarray([expm(item) for item in beta])
    orientation = initial_orientation(
        loading_axis_hkl, perturbation_axis_lab=(1.0, 0.0, 0.0), perturbation_deg=1.0
    )
    orientations = np.repeat(orientation[None, :, :], grid_points, axis=0)
    zeros = np.zeros_like(plus)
    return DrivenCDDState(
        fp, orientations, plus, minus, zeros, zeros,
        np.full(grid_points, temperature_K),
    )


def _spectral_derivative(fields: np.ndarray, domain_m: float) -> np.ndarray:
    n = fields.shape[-1]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=domain_m / n)
    return np.fft.ifft(1j * k * np.fft.fft(fields, axis=-1), axis=-1).real


def _interaction_matrix() -> np.ndarray:
    directions, normals = default_slip_geometry()
    lines = np.cross(np.asarray([1.0, 0.0, 0.0]), normals)
    norms = np.linalg.norm(lines, axis=1)
    lines[norms > 0.0] /= norms[norms > 0.0, None]
    matrix = (directions @ directions.T) * (lines @ lines.T)
    matrix += 0.25 * np.eye(4)
    return matrix


def correlation_stresses_Pa(
    state: DrivenCDDState,
    parameters: DrivenCDDParameters,
    gate_a_parameters: BertinBCCParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mobile_total = state.mobile_plus_m2 + state.mobile_minus_m2
    kappa = state.mobile_plus_m2 - state.mobile_minus_m2
    forest = np.maximum(np.sum(mobile_total + state.locked_plus_m2 + state.locked_minus_m2, axis=0), parameters.density_floor_m2)
    mu = np.asarray([gate_a_parameters.elastic_constants_Pa(value)[2] for value in state.temperature_K])
    back = -parameters.backstress_coefficient * mu[None, :] * gate_a_parameters.burgers_m * _spectral_derivative(kappa, parameters.domain_m) / forest[None, :]
    diffusion = -parameters.diffusion_coefficient * mu[None, :] * gate_a_parameters.burgers_m * _spectral_derivative(mobile_total, parameters.domain_m) / np.maximum(mobile_total, parameters.density_floor_m2)
    n = state.grid_points
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=parameters.domain_m / n)
    kernel = np.zeros(n, dtype=complex)
    mask = k != 0.0
    kernel[mask] = -1j / k[mask]
    coupled = _interaction_matrix() @ np.fft.fft(kappa, axis=1)
    self_consistent = np.fft.ifft(coupled * kernel[None, :], axis=1).real
    self_consistent *= parameters.elastic_kernel_scale * mu[None, :] * gate_a_parameters.burgers_m
    return self_consistent, back, diffusion


def _gate_a_differential_mobility_m_Pa_s(
    response,
    family: int,
    total_density_m2: float,
    temperature_K: float,
    parameters: BertinBCCParameters,
) -> float:
    """Derivative of the selected Gate-A speed branch with respect to stress."""

    tau_eff = float(response.effective_shear_Pa[family])
    if tau_eff <= 0.0:
        return 0.0
    _, _, mu = parameters.elastic_constants_Pa(temperature_K)
    critical = parameters.taylor_alpha * mu * parameters.burgers_m * math.sqrt(total_density_m2)
    chi = float(response.mrssp_angles_rad[family])
    v0 = parameters.velocity_T_m_s + (3.0 / math.pi) * (
        math.pi / 6.0 - chi
    ) * (parameters.velocity_AT_m_s - parameters.velocity_T_m_s)
    power = v0 * (tau_eff / critical) ** parameters.velocity_exponent
    drag = parameters.drag_velocity_m_s * (
        1.0 - math.exp(-tau_eff / parameters.drag_stress_Pa)
    )
    if power <= drag:
        return parameters.velocity_exponent * power / tau_eff
    return (
        parameters.drag_velocity_m_s
        * math.exp(-tau_eff / parameters.drag_stress_Pa)
        / parameters.drag_stress_Pa
    )


def _inactive_weights(rates: np.ndarray, parameters: BertinBCCParameters) -> np.ndarray:
    total = float(np.sum(np.abs(rates)))
    activity = np.abs(rates) / max(total, np.finfo(float).tiny)
    argument = parameters.activity_sharpness * (
        activity - parameters.activity_threshold
    )
    result = np.empty_like(argument)
    for index, value in enumerate(argument):
        if value >= 40.0:
            result[index] = math.exp(-float(value))
        elif value <= -40.0:
            result[index] = 1.0
        else:
            result[index] = 1.0 / (1.0 + math.exp(float(value)))
    return result


def _gate_a_source_channels(
    density_m2: float,
    shear_rate_s_inv: float,
    chi_rad: float,
    total_shear_rate_s_inv: float,
    inactive_weight: float,
    temperature_K: float,
    dt_s: float,
    parameters: BertinBCCParameters,
) -> tuple[float, float]:
    """Return gross Gate-A generation and removal over one frozen step.

    Gate A advances ``y=sqrt(rho)`` exactly under ``2*ydot=a-c*y``.
    Integrating ``a*y`` on that trajectory separates the two nonnegative
    channels without changing the exact homogeneous density update.
    """

    alpha_p = math.radians(parameters.activation_angle_deg)
    k1 = parameters.generation_coefficient_m_inv * (
        1.0 + parameters.generation_AT_slope / math.cos(chi_rad - alpha_p)
    )
    source = k1 * abs(shear_rate_s_inv)
    k2 = 0.0
    if total_shear_rate_s_inv > 0.0:
        k2 = parameters.annihilation_coefficient * math.log(
            total_shear_rate_s_inv / parameters.annihilation_reference_rate_s_inv
        ) * math.log(
            temperature_K / parameters.annihilation_reference_temperature_K
        )
    sink = k2 * abs(shear_rate_s_inv) + (
        parameters.inactive_relaxation_scale
        * inactive_weight
        * parameters.inactive_relaxation_s_inv
    )
    root0 = math.sqrt(density_m2)
    if abs(sink) < 1.0e-30:
        root1 = root0 + 0.5 * source * dt_s
        return max(root1 * root1 - density_m2, 0.0), 0.0
    factor = math.exp(-0.5 * sink * dt_s)
    steady_root = source / sink
    root1 = root0 * factor + steady_root * (1.0 - factor)
    integral_root = steady_root * dt_s + (root0 - steady_root) * (
        2.0 / sink
    ) * (1.0 - factor)
    generated = max(source * integral_root, 0.0)
    delta = root1 * root1 - density_m2
    return generated, max(generated - delta, 0.0)


def evaluate_driven_cdd(
    state: DrivenCDDState,
    parameters: DrivenCDDParameters,
    gate_a_parameters: BertinBCCParameters,
) -> DrivenCDDResponse:
    n = state.grid_points
    resolved = np.zeros((4, n))
    effective = np.zeros((4, n))
    friction = np.zeros((4, n))
    patterning_active = np.zeros((4, n), dtype=bool)
    angles = np.zeros((4, n))
    inactive = np.zeros((4, n))
    base_rates = np.zeros((4, n))
    base_velocity = np.zeros((4, n))
    mobility_b = np.zeros((4, n))
    orientations = np.zeros((n, 3, 3))
    dyads = np.zeros((4, n, 3, 3))
    total_family = state.mobile_plus_m2 + state.mobile_minus_m2 + state.locked_plus_m2 + state.locked_minus_m2
    gate_a_total_family = np.maximum(total_family, parameters.density_floor_m2)
    for cell in range(n):
        local = BertinBCCState(
            state.plastic_deformation_gradient[cell], state.initial_orientation[cell],
            gate_a_total_family[:, cell], state.axial_true_strain, state.time_s, state.accepted_steps,
        )
        response = evaluate_bertin_bcc(local, float(state.temperature_K[cell]), gate_a_parameters)
        resolved[:, cell] = response.resolved_shear_Pa
        effective[:, cell] = response.effective_shear_Pa
        angles[:, cell] = response.mrssp_angles_rad
        base_rates[:, cell] = response.shear_rates_s_inv
        orientations[cell] = response.orientation
        dyads[:, cell] = response.slip_dyads_intermediate
        forest = float(np.sum(gate_a_total_family[:, cell]))
        _, _, mu = gate_a_parameters.elastic_constants_Pa(
            float(state.temperature_K[cell])
        )
        local_friction = (
            gate_a_parameters.taylor_alpha
            * mu * gate_a_parameters.burgers_m * math.sqrt(forest)
        )
        for family in range(4):
            mobile = state.mobile_plus_m2[family, cell] + state.mobile_minus_m2[family, cell]
            if mobile > parameters.density_floor_m2:
                base_velocity[family, cell] = response.shear_rates_s_inv[family] / (
                    gate_a_parameters.burgers_m * mobile
                )
            friction[family, cell] = local_friction
            overdrive = response.effective_shear_Pa[family] - local_friction
            patterning_active[family, cell] = overdrive > 0.0
            if overdrive > 0.0:
                mobility_b[family, cell] = _gate_a_differential_mobility_m_Pa_s(
                    response, family, forest, float(state.temperature_K[cell]),
                    gate_a_parameters,
                )
        inactive[:, cell] = response.inactive_weight

    if parameters.enforce_common_resolved_stress:
        # One-dimensional periodic equilibrium has a spatially common driving
        # traction. Gate A supplies that traction and every local MRSSP/density
        # still controls activation, speed, and rotation.
        common_resolved = np.mean(resolved, axis=1)
        for cell in range(n):
            forest = float(np.sum(gate_a_total_family[:, cell]))
            _, _, mu = gate_a_parameters.elastic_constants_Pa(
                float(state.temperature_K[cell])
            )
            local_friction = (
                gate_a_parameters.taylor_alpha * mu
                * gate_a_parameters.burgers_m * math.sqrt(forest)
            )
            for family in range(4):
                chi = angles[family, cell]
                activation = gate_a_parameters.activation_stress_Pa / math.cos(
                    chi - math.radians(gate_a_parameters.activation_angle_deg)
                )
                tau_eff = max(abs(common_resolved[family]) - activation, 0.0)
                v0 = gate_a_parameters.velocity_T_m_s + (3.0 / math.pi) * (
                    math.pi / 6.0 - chi
                ) * (
                    gate_a_parameters.velocity_AT_m_s
                    - gate_a_parameters.velocity_T_m_s
                )
                power = v0 * (tau_eff / local_friction) ** gate_a_parameters.velocity_exponent
                drag = gate_a_parameters.drag_velocity_m_s * (
                    1.0 - math.exp(-tau_eff / gate_a_parameters.drag_stress_Pa)
                )
                speed = min(power, drag)
                resolved[family, cell] = common_resolved[family]
                effective[family, cell] = tau_eff
                base_rates[family, cell] = (
                    gate_a_total_family[family, cell] * gate_a_parameters.burgers_m
                    * speed * np.sign(common_resolved[family])
                )
                mobile = (
                    state.mobile_plus_m2[family, cell]
                    + state.mobile_minus_m2[family, cell]
                )
                base_velocity[family, cell] = (
                    base_rates[family, cell]
                    / (gate_a_parameters.burgers_m * mobile)
                    if mobile > parameters.density_floor_m2 else 0.0
                )
                friction[family, cell] = local_friction
                overdrive = tau_eff - local_friction
                patterning_active[family, cell] = overdrive > 0.0
                if overdrive > 0.0 and tau_eff > 0.0:
                    if power <= drag:
                        mobility_b[family, cell] = (
                            gate_a_parameters.velocity_exponent * power / tau_eff
                        )
                    else:
                        mobility_b[family, cell] = (
                            gate_a_parameters.drag_velocity_m_s
                            * math.exp(-tau_eff / gate_a_parameters.drag_stress_Pa)
                            / gate_a_parameters.drag_stress_Pa
                        )
                else:
                    mobility_b[family, cell] = 0.0
            inactive[:, cell] = _inactive_weights(base_rates[:, cell], gate_a_parameters)
    self_consistent, back, diffusion = correlation_stresses_Pa(state, parameters, gate_a_parameters)
    mobile_total = state.mobile_plus_m2 + state.mobile_minus_m2
    kappa = state.mobile_plus_m2 - state.mobile_minus_m2
    sign = np.sign(resolved)
    polarization_friction = (
        sign * friction * kappa / np.maximum(mobile_total, parameters.density_floor_m2)
    )
    # Groma--Zaiser sign structure: mean/back stress reverses the velocity of
    # the negative population, whereas the polarization-friction and
    # diffusion stresses shift both signed velocities in the same direction.
    # This distinction creates the rho/kappa cross-coupling in the published
    # linear operator; folding every term into one common velocity removes it.
    common_velocity = base_velocity + mobility_b * (self_consistent + back)
    same_sign_velocity = mobility_b * (polarization_friction + diffusion)
    velocity_plus = common_velocity + same_sign_velocity
    velocity_minus = -common_velocity + same_sign_velocity
    actual_rates = gate_a_parameters.burgers_m * (
        state.mobile_plus_m2 * velocity_plus - state.mobile_minus_m2 * velocity_minus
    )
    return DrivenCDDResponse(
        resolved, effective, friction, patterning_active, angles, inactive,
        base_rates, actual_rates, velocity_plus, velocity_minus,
        self_consistent, back, diffusion, orientations, dyads, 1,
    )


def _upwind_step(density: np.ndarray, velocity: np.ndarray, dt_s: float, dx_m: float) -> np.ndarray:
    """Positive conservative MUSCL/SSP-RK2 density-weighted transport."""

    face_velocity = 0.5 * (velocity + np.roll(velocity, -1, axis=1))

    def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.where(a * b > 0.0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

    def rhs(field: np.ndarray) -> np.ndarray:
        slope = minmod(field - np.roll(field, 1, axis=1), np.roll(field, -1, axis=1) - field)
        left = field + 0.5 * slope
        right = np.roll(field, -1, axis=1) - 0.5 * np.roll(slope, -1, axis=1)
        upstream = np.where(face_velocity >= 0.0, left, right)
        flux = face_velocity * upstream
        return -(flux - np.roll(flux, 1, axis=1)) / dx_m

    first = density + dt_s * rhs(density)
    updated = 0.5 * density + 0.5 * (first + dt_s * rhs(first))
    tolerance = 2.0e-13 * max(float(np.max(density)), 1.0)
    if np.min(updated) < -tolerance:
        raise RuntimeError("density-weighted upwind flux violated nonnegativity")
    return np.maximum(updated, 0.0)


def _driven_cdd_single_step(
    state: DrivenCDDState,
    axial_rate_s_inv: float,
    strain_increment: float,
    parameters: DrivenCDDParameters,
    gate_a_parameters: BertinBCCParameters,
) -> tuple[DrivenCDDState, DrivenCDDResponse, CDDLedger]:
    if axial_rate_s_inv == 0.0 or strain_increment * axial_rate_s_inv <= 0.0:
        raise ValueError("nonzero rate and strain increment must have the same sign")
    dt_s = strain_increment / axial_rate_s_inv
    response = evaluate_driven_cdd(state, parameters, gate_a_parameters)
    dx = parameters.domain_m / state.grid_points
    vmax = max(float(np.max(np.abs(response.velocity_plus_m_s))), float(np.max(np.abs(response.velocity_minus_m_s))))
    substeps = max(1, int(math.ceil(vmax * dt_s / (parameters.cfl_limit * dx))))
    plus = state.mobile_plus_m2.copy()
    minus = state.mobile_minus_m2.copy()
    subdt = dt_s / substeps
    mobile_before = float(np.sum(plus + minus) * dx)
    locked_before = float(np.sum(state.locked_plus_m2 + state.locked_minus_m2) * dx)
    for _ in range(substeps):
        # Freeze Gate A mobility over the accepted load step; update the
        # density-weighted advective flux conservatively.
        plus = _upwind_step(plus, response.velocity_plus_m_s, subdt, dx)
        minus = _upwind_step(minus, response.velocity_minus_m_s, subdt, dx)

    generated_total = np.zeros_like(plus)
    removed_requested = np.zeros_like(plus)
    source_density = (
        state.mobile_plus_m2 + state.mobile_minus_m2
        + state.locked_plus_m2 + state.locked_minus_m2
    )
    for cell in range(state.grid_points):
        total_rate = float(np.sum(np.abs(response.gate_a_shear_rates_s_inv[:, cell])))
        for family in range(4):
            generated_total[family, cell], removed_requested[family, cell] = (
                _gate_a_source_channels(
                    source_density[family, cell],
                    response.gate_a_shear_rates_s_inv[family, cell],
                    response.mrssp_angles_rad[family, cell], total_rate,
                    response.inactive_weight[family, cell],
                    float(state.temperature_K[cell]), dt_s, gate_a_parameters,
                )
            )
    if parameters.homogenize_gate_a_pair_sources:
        # Gate A supplies the macroscopic family-reservoir evolution.  The
        # transport-scale stability derivation conserves the nonzero-k line
        # inventory, so distribute each family source uniformly while
        # preserving its exact spatial integral and homogeneous reduction.
        generated_total[:] = np.mean(generated_total, axis=1, keepdims=True)
        removed_requested[:] = np.mean(
            removed_requested, axis=1, keepdims=True
        )
    # Pair creation adds equal signs; annihilation is bounded by the minority
    # population and therefore cannot erase net Burgers content.
    plus += 0.5 * generated_total
    minus += 0.5 * generated_total
    removable = 2.0 * np.minimum(plus, minus)
    removed_actual = np.minimum(removed_requested, removable)
    plus -= 0.5 * removed_actual
    minus -= 0.5 * removed_actual

    locked_plus = state.locked_plus_m2.copy()
    locked_minus = state.locked_minus_m2.copy()
    same_fraction = -math.expm1(-parameters.same_family_lock_rate_s_inv * dt_s)
    unlock_fraction = -math.expm1(-parameters.unlock_rate_s_inv * dt_s)
    cross_fraction = -math.expm1(-parameters.cross_family_lock_rate_s_inv * dt_s)
    same_plus = same_fraction * plus
    same_minus = same_fraction * minus
    plus -= same_plus; minus -= same_minus
    locked_plus += same_plus; locked_minus += same_minus
    unlocked_plus = unlock_fraction * locked_plus
    unlocked_minus = unlock_fraction * locked_minus
    locked_plus -= unlocked_plus; locked_minus -= unlocked_minus
    plus += unlocked_plus; minus += unlocked_minus
    # Cross-family association is a sign-preserving transfer of equal fractions
    # from every family. It is ledgered separately and does not assert a BCC
    # junction product topology.
    cross_plus = cross_fraction * plus
    cross_minus = cross_fraction * minus
    plus -= cross_plus; minus -= cross_minus
    locked_plus += cross_plus; locked_minus += cross_minus

    fp = np.empty_like(state.plastic_deformation_gradient)
    for cell in range(state.grid_points):
        plastic_velocity_gradient = np.zeros((3, 3))
        for family in range(4):
            plastic_velocity_gradient += (
                response.actual_shear_rates_s_inv[family, cell]
                * response.slip_dyads[family, cell]
            )
        symmetric = 0.5 * (
            plastic_velocity_gradient + plastic_velocity_gradient.T
        )
        spin = 0.5 * (
            plastic_velocity_gradient - plastic_velocity_gradient.T
        )
        plastic_velocity_gradient = (
            symmetric + gate_a_parameters.plastic_spin_scale * spin
        )
        fp[cell] = expm(plastic_velocity_gradient * dt_s) @ state.plastic_deformation_gradient[cell]
    next_state = DrivenCDDState(
        fp, state.initial_orientation, plus, minus, locked_plus, locked_minus,
        state.temperature_K, state.axial_true_strain + strain_increment,
        state.time_s + dt_s, state.accepted_steps + 1,
        state.accumulated_slip + response.actual_shear_rates_s_inv * dt_s,
    )

    mobile_after = float(np.sum(plus + minus) * dx)
    locked_after = float(np.sum(locked_plus + locked_minus) * dx)
    generated_content = float(np.sum(generated_total) * dx)
    removed_content = float(np.sum(removed_actual) * dx)
    limited_content = float(np.sum(removed_requested - removed_actual) * dx)
    same_transfer = float(np.sum(same_plus + same_minus) * dx)
    cross_transfer = float(np.sum(cross_plus + cross_minus) * dx)
    unlock_transfer = float(np.sum(unlocked_plus + unlocked_minus) * dx)
    before_signed = np.sum(
        state.mobile_plus_m2 - state.mobile_minus_m2
        + state.locked_plus_m2 - state.locked_minus_m2, axis=1
    ) * dx
    after_signed = np.sum(plus - minus + locked_plus - locked_minus, axis=1) * dx
    signed_residual = after_signed - before_signed
    directions, _ = default_slip_geometry()
    vector_residual = gate_a_parameters.burgers_m * np.sum(signed_residual[:, None] * directions, axis=0)
    ledger = CDDLedger(
        mobile_before, 0.0, generated_content, removed_content, limited_content,
        same_transfer, cross_transfer,
        unlock_transfer, mobile_after, locked_before, locked_after,
        (mobile_after + locked_after)
        - (mobile_before + locked_before + generated_content - removed_content),
        float(np.max(np.abs(signed_residual))), float(np.linalg.norm(vector_residual)),
    )
    return next_state, DrivenCDDResponse(
        response.resolved_shear_Pa, response.effective_shear_Pa,
        response.taylor_friction_Pa, response.patterning_active,
        response.mrssp_angles_rad, response.inactive_weight,
        response.gate_a_shear_rates_s_inv,
        response.actual_shear_rates_s_inv, response.velocity_plus_m_s,
        response.velocity_minus_m_s, response.self_consistent_stress_Pa,
        response.backstress_Pa, response.diffusion_stress_Pa,
        response.orientations, response.slip_dyads, substeps,
    ), ledger


def _is_spatially_homogeneous(state: DrivenCDDState) -> bool:
    fields = (
        state.mobile_plus_m2, state.mobile_minus_m2,
        state.locked_plus_m2, state.locked_minus_m2,
    )
    populations_uniform = all(
        np.allclose(field, field[:, :1], rtol=2.0e-14, atol=1.0)
        for field in fields
    )
    fp_uniform = np.allclose(
        state.plastic_deformation_gradient,
        state.plastic_deformation_gradient[:1], rtol=2.0e-14, atol=2.0e-15,
    )
    temperature_uniform = np.allclose(
        state.temperature_K, state.temperature_K[0], rtol=0.0, atol=1.0e-12
    )
    return bool(populations_uniform and fp_uniform and temperature_uniform)


def driven_cdd_step(
    state: DrivenCDDState,
    axial_rate_s_inv: float,
    strain_increment: float,
    parameters: DrivenCDDParameters,
    gate_a_parameters: BertinBCCParameters,
) -> tuple[DrivenCDDState, DrivenCDDResponse, CDDLedger]:
    """Advance one requested step with coupled internal accuracy control.

    A homogeneous state takes exactly one Gate-A step, preserving the exact
    reduction. A spatially nonuniform state is refreshed at a declared maximum
    load increment in addition to its density-flux CFL subcycling.
    """

    subdivisions = 1
    if not _is_spatially_homogeneous(state):
        subdivisions = max(
            1, int(math.ceil(
                abs(strain_increment) / parameters.max_coupled_strain_increment
            )),
        )
    if subdivisions == 1:
        return _driven_cdd_single_step(
            state, axial_rate_s_inv, strain_increment,
            parameters, gate_a_parameters,
        )
    current = state
    responses: list[DrivenCDDResponse] = []
    ledgers: list[CDDLedger] = []
    increment = strain_increment / subdivisions
    for _ in range(subdivisions):
        current, response, ledger = _driven_cdd_single_step(
            current, axial_rate_s_inv, increment,
            parameters, gate_a_parameters,
        )
        responses.append(response)
        ledgers.append(ledger)
    dx = parameters.domain_m / state.grid_points
    signed_before = np.sum(
        state.mobile_plus_m2 - state.mobile_minus_m2
        + state.locked_plus_m2 - state.locked_minus_m2, axis=1
    ) * dx
    signed_after = np.sum(
        current.mobile_plus_m2 - current.mobile_minus_m2
        + current.locked_plus_m2 - current.locked_minus_m2, axis=1
    ) * dx
    signed_residual = signed_after - signed_before
    directions, _ = default_slip_geometry()
    vector_residual = gate_a_parameters.burgers_m * np.sum(
        signed_residual[:, None] * directions, axis=0
    )
    first, last = ledgers[0], ledgers[-1]
    generated = sum(item.pair_multiplication_m_inv for item in ledgers)
    removed = sum(item.pair_annihilation_m_inv for item in ledgers)
    combined = CDDLedger(
        first.mobile_before_m_inv, 0.0, generated, removed,
        sum(item.annihilation_limited_m_inv for item in ledgers),
        sum(item.same_family_lock_transfer_m_inv for item in ledgers),
        sum(item.cross_family_lock_transfer_m_inv for item in ledgers),
        sum(item.unlock_transfer_m_inv for item in ledgers),
        last.mobile_after_m_inv, first.locked_before_m_inv,
        last.locked_after_m_inv,
        (last.mobile_after_m_inv + last.locked_after_m_inv)
        - (first.mobile_before_m_inv + first.locked_before_m_inv
           + generated - removed),
        float(np.max(np.abs(signed_residual))),
        float(np.linalg.norm(vector_residual)),
    )
    final_response = responses[-1]
    final_response = DrivenCDDResponse(
        final_response.resolved_shear_Pa, final_response.effective_shear_Pa,
        final_response.taylor_friction_Pa, final_response.patterning_active,
        final_response.mrssp_angles_rad, final_response.inactive_weight,
        final_response.gate_a_shear_rates_s_inv,
        final_response.actual_shear_rates_s_inv,
        final_response.velocity_plus_m_s, final_response.velocity_minus_m_s,
        final_response.self_consistent_stress_Pa,
        final_response.backstress_Pa, final_response.diffusion_stress_Pa,
        final_response.orientations, final_response.slip_dyads,
        sum(item.transport_substeps for item in responses),
    )
    return current, final_response, combined


def advance_driven_cdd(
    state: DrivenCDDState,
    axial_rate_s_inv: float,
    target_axial_strain: float,
    strain_increment: float,
    parameters: DrivenCDDParameters,
    gate_a_parameters: BertinBCCParameters,
) -> tuple[DrivenCDDState, tuple[CDDLedger, ...]]:
    """Advance to an aligned target strain with accepted-step ledgers."""

    if strain_increment <= 0.0:
        raise ValueError("strain_increment must be positive")
    distance = target_axial_strain - state.axial_true_strain
    if distance == 0.0:
        return state, ()
    direction = math.copysign(1.0, distance)
    if direction * axial_rate_s_inv <= 0.0:
        raise ValueError("axial rate must point toward the target strain")
    ledgers: list[CDDLedger] = []
    current = state
    while direction * (target_axial_strain - current.axial_true_strain) > 1.0e-14:
        increment = direction * min(
            strain_increment, abs(target_axial_strain - current.axial_true_strain)
        )
        current, _, ledger = driven_cdd_step(
            current, axial_rate_s_inv, increment,
            parameters, gate_a_parameters,
        )
        ledgers.append(ledger)
    return current, tuple(ledgers)


def hold_driven_cdd(
    state: DrivenCDDState,
    duration_s: float,
    time_increment_s: float,
    parameters: DrivenCDDParameters,
    gate_a_parameters: BertinBCCParameters,
) -> tuple[DrivenCDDState, tuple[CDDLedger, ...]]:
    """Relax at fixed total axial strain while retaining full CDD feedback.

    The Gate-A step uses its lower-envelope reference rate only to express the
    requested physical time as a strain increment.  Resetting total strain
    after every accepted substep makes the deformation clamp explicit; `Fp`,
    density sources, signed transport, orientation, and time all evolve.
    """

    if duration_s < 0.0 or time_increment_s <= 0.0:
        raise ValueError("hold duration must be nonnegative and increment positive")
    if duration_s == 0.0:
        return state, ()
    reference_rate = gate_a_parameters.reference_axial_rate_min_s_inv
    maximum_dt = parameters.max_coupled_strain_increment / reference_rate
    dt_nominal = min(time_increment_s, maximum_dt)
    fixed_strain = state.axial_true_strain
    current = state
    ledgers: list[CDDLedger] = []
    elapsed = 0.0
    while elapsed < duration_s - 1.0e-24:
        dt = min(dt_nominal, duration_s - elapsed)
        advanced, _, ledger = _driven_cdd_single_step(
            current, reference_rate, reference_rate * dt,
            parameters, gate_a_parameters,
        )
        current = DrivenCDDState(
            advanced.plastic_deformation_gradient, advanced.initial_orientation,
            advanced.mobile_plus_m2, advanced.mobile_minus_m2,
            advanced.locked_plus_m2, advanced.locked_minus_m2,
            advanced.temperature_K, fixed_strain, advanced.time_s,
            advanced.accepted_steps, advanced.accumulated_slip,
        )
        ledgers.append(ledger)
        elapsed += dt
    return current, tuple(ledgers)


def save_driven_checkpoint(path: str | Path, state: DrivenCDDState) -> None:
    np.savez(
        Path(path), schema=np.asarray("asb-drx-driven-cdd/v2"),
        plastic_deformation_gradient=state.plastic_deformation_gradient,
        initial_orientation=state.initial_orientation,
        mobile_plus_m2=state.mobile_plus_m2, mobile_minus_m2=state.mobile_minus_m2,
        locked_plus_m2=state.locked_plus_m2, locked_minus_m2=state.locked_minus_m2,
        temperature_K=state.temperature_K, axial_true_strain=np.asarray(state.axial_true_strain),
        time_s=np.asarray(state.time_s), accepted_steps=np.asarray(state.accepted_steps, dtype=np.int64),
        accumulated_slip=state.accumulated_slip,
    )


def load_driven_checkpoint(path: str | Path) -> DrivenCDDState:
    with np.load(Path(path), allow_pickle=False) as archive:
        if str(archive["schema"]) != "asb-drx-driven-cdd/v2":
            raise ValueError("unsupported driven CDD checkpoint schema")
        return DrivenCDDState(
            archive["plastic_deformation_gradient"], archive["initial_orientation"],
            archive["mobile_plus_m2"], archive["mobile_minus_m2"],
            archive["locked_plus_m2"], archive["locked_minus_m2"],
            archive["temperature_K"], float(archive["axial_true_strain"]),
            float(archive["time_s"]), int(archive["accepted_steps"]),
            archive["accumulated_slip"],
        )


def derived_nye_tensor(state: DrivenCDDState, domain_m: float) -> np.ndarray:
    return nye_tensor_1d(state.plastic_deformation_gradient - np.eye(3)[None, :, :], domain_m)


def groma_dimensionless_growth_rates(
    dimensionless_wavenumber: np.ndarray,
    dimensionless_shear_rate: float,
    alpha_prime: float,
    parameters: DrivenCDDParameters,
) -> np.ndarray:
    """Positive branch of Groma et al. (2016), Eq. (100), for k_y=0.

    The one-dimensional reduction has ``T(k)=0``.  Wavenumber is normalized
    by ``sqrt(rho)`` and time by the paper's dislocation time.  This function
    supplies a dimensionless rate for stability and mode selection; it does
    not invent the unavailable drag-time calibration.
    """

    k = np.asarray(dimensionless_wavenumber, dtype=float)
    a = parameters.diffusion_coefficient
    d = parameters.backstress_coefficient
    beta = (dimensionless_shear_rate + 2.0 * alpha_prime) * (
        dimensionless_shear_rate - alpha_prime
    )
    trace = (a + d) * k * k
    discriminant = np.maximum(
        (d - a) ** 2 * k**4 - 4.0 * beta * k * k, 0.0
    )
    return 0.5 * (-trace + np.sqrt(discriminant))


def homogeneous_dispersion_snapshot(
    state: DrivenCDDState,
    parameters: DrivenCDDParameters,
    gate_a_parameters: BertinBCCParameters,
) -> dict[str, object]:
    """Evaluate the declared driven linear mode prediction cell by cell."""

    response = evaluate_driven_cdd(state, parameters, gate_a_parameters)
    total = (
        state.mobile_plus_m2 + state.mobile_minus_m2
        + state.locked_plus_m2 + state.locked_minus_m2
    )
    forest = np.sum(total, axis=0)
    n = state.grid_points
    physical_k = 2.0 * np.pi * np.fft.rfftfreq(
        n, d=parameters.domain_m / n
    )
    family_records: list[dict[str, object]] = []
    for family in range(4):
        growth_samples = []
        gamma_prime_samples = []
        for cell in range(n):
            c11, c12, mu = gate_a_parameters.elastic_constants_Pa(
                float(state.temperature_K[cell])
            )
            poisson = c12 / (c11 + c12)
            g_prefactor = mu / (2.0 * math.pi * (1.0 - poisson))
            overdrive = max(
                response.effective_shear_Pa[family, cell]
                - response.taylor_friction_Pa[family, cell], 0.0
            )
            gamma_prime = overdrive / max(
                g_prefactor * gate_a_parameters.burgers_m
                * math.sqrt(forest[cell]), np.finfo(float).tiny
            )
            alpha_prime = (
                math.pi * (1.0 - poisson) * gate_a_parameters.taylor_alpha
            )
            if overdrive > 0.0:
                growth_samples.append(
                    groma_dimensionless_growth_rates(
                        physical_k / math.sqrt(forest[cell]), gamma_prime,
                        alpha_prime, parameters,
                    )
                )
            else:
                growth_samples.append(np.zeros_like(physical_k))
            gamma_prime_samples.append(gamma_prime)
        growth = np.mean(growth_samples, axis=0)
        growth[0] = 0.0
        mode = int(np.argmax(growth))
        family_records.append({
            "family": family,
            "dimensionless_shear_rate_mean": float(np.mean(gamma_prime_samples)),
            "unstable": bool(np.max(growth) > 0.0),
            "fastest_discrete_mode": mode,
            "fastest_wavelength_m": (
                parameters.domain_m / mode if mode > 0 else None
            ),
            "maximum_dimensionless_growth": float(np.max(growth)),
        })
    return {"families": family_records, "no_prescribed_wavelength": True}


def full_linearized_amplification_spectrum(
    homogeneous_state: DrivenCDDState,
    axial_rate_s_inv: float,
    strain_increment: float,
    parameters: DrivenCDDParameters,
    gate_a_parameters: BertinBCCParameters,
    *,
    relative_perturbation: float = 2.0e-6,
    modes: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Finite-difference the full Nye-compatible signed-population map.

    Unlike :func:`homogeneous_dispersion_snapshot`, this operator includes the
    implemented Gate-A source, MRSSP/orientation feedback, finite-volume flux,
    and multiplicative plastic update. Every signed-density perturbation also
    carries the quadrature slip perturbation required by
    ``partial_x gamma_a=-b*kappa_a``; independent incompatible `Fp` waves are
    not admissible state variables. Similarity scaling keeps eigenvalues
    independent of SI density magnitudes.
    """

    if not _is_spatially_homogeneous(homogeneous_state):
        raise ValueError("linearized spectrum requires a homogeneous state")
    if not 0.0 < relative_perturbation < 1.0e-3:
        raise ValueError("relative perturbation is outside the linear probe range")
    n = homogeneous_state.grid_points
    mode_numbers = tuple(range(1, n // 2)) if modes is None else tuple(modes)
    if not mode_numbers or any(mode < 1 or mode >= n // 2 for mode in mode_numbers):
        raise ValueError("linear probe modes must lie strictly below Nyquist")
    base, _, _ = _driven_cdd_single_step(
        homogeneous_state, axial_rate_s_inv, strain_increment,
        parameters, gate_a_parameters,
    )
    all_input_density = np.concatenate((
        homogeneous_state.mobile_plus_m2[:, 0],
        homogeneous_state.mobile_minus_m2[:, 0],
    ))
    reference_response = evaluate_driven_cdd(
        homogeneous_state, parameters, gate_a_parameters
    )
    active_families = [
        family for family in range(4)
        if bool(np.any(reference_response.patterning_active[family]))
    ]
    active_indices = active_families + [family + 4 for family in active_families]
    scales = all_input_density[active_indices]
    reference_dyads = reference_response.slip_dyads[:, 0]
    coordinate = np.arange(n)
    records: list[dict[str, object]] = []
    dt_s = strain_increment / axial_rate_s_inv
    for mode in mode_numbers:
        cosine = np.cos(2.0 * np.pi * mode * coordinate / n)
        matrix = np.zeros((len(active_indices), len(active_indices)), dtype=complex)
        physical_k = 2.0 * math.pi * mode / parameters.domain_m
        sine = np.sin(2.0 * np.pi * mode * coordinate / n)
        for column, population_index in enumerate(active_indices):
            plus = homogeneous_state.mobile_plus_m2.copy()
            minus = homogeneous_state.mobile_minus_m2.copy()
            fp = homogeneous_state.plastic_deformation_gradient.copy()
            amplitude = relative_perturbation * scales[column]
            population = plus if population_index < 4 else minus
            family = population_index % 4
            population[family] += amplitude * cosine
            kappa_sign = 1.0 if population_index < 4 else -1.0
            slip_amplitude = (
                -gate_a_parameters.burgers_m * kappa_sign
                * amplitude / physical_k
            )
            for cell in range(n):
                fp[cell] = expm(
                    slip_amplitude * sine[cell] * reference_dyads[family]
                ) @ fp[cell]
            perturbed = DrivenCDDState(
                fp, homogeneous_state.initial_orientation, plus, minus,
                homogeneous_state.locked_plus_m2,
                homogeneous_state.locked_minus_m2,
                homogeneous_state.temperature_K,
                homogeneous_state.axial_true_strain,
                homogeneous_state.time_s,
                homogeneous_state.accepted_steps,
            )
            advanced, _, _ = _driven_cdd_single_step(
                perturbed, axial_rate_s_inv, strain_increment,
                parameters, gate_a_parameters,
            )
            density_difference = np.concatenate((
                advanced.mobile_plus_m2 - base.mobile_plus_m2,
                advanced.mobile_minus_m2 - base.mobile_minus_m2,
            ))
            output = density_difference[active_indices]
            coefficient = np.fft.fft(output, axis=1)[:, mode] / n
            matrix[:, column] = coefficient / (
                0.5 * relative_perturbation * scales
            )
        eigenvalues = np.linalg.eigvals(matrix)
        dominant = eigenvalues[int(np.argmax(np.abs(eigenvalues)))]
        records.append({
            "mode": mode,
            "wavelength_m": parameters.domain_m / mode,
            "amplification_magnitude": float(abs(dominant)),
            "growth_rate_s_inv": float(math.log(max(abs(dominant), 1.0e-300)) / dt_s),
            "oscillation_rate_rad_s": float(np.angle(dominant) / dt_s),
        })
    fastest = max(records, key=lambda item: item["growth_rate_s_inv"])
    return {
        "state_dimension": len(active_indices),
        "active_families": active_families,
        "constraint": "partial_x gamma_a = -b kappa_a",
        "relative_perturbation": relative_perturbation,
        "records": records,
        "fastest_mode": fastest["mode"],
        "fastest_wavelength_m": fastest["wavelength_m"],
        "maximum_growth_rate_s_inv": fastest["growth_rate_s_inv"],
        "finite_mode_unstable": bool(
            fastest["mode"] > 1 and fastest["growth_rate_s_inv"] > 0.0
        ),
    }


def driven_structure_diagnostics(
    state: DrivenCDDState,
    parameters: DrivenCDDParameters,
    gate_a_parameters: BertinBCCParameters,
) -> dict[str, object]:
    total = np.sum(
        state.mobile_plus_m2 + state.mobile_minus_m2
        + state.locked_plus_m2 + state.locked_minus_m2, axis=0
    )
    kappa = state.mobile_plus_m2 - state.mobile_minus_m2
    alpha = derived_nye_tensor(state, parameters.domain_m)
    gnd = np.linalg.norm(alpha, axis=(1, 2)) / gate_a_parameters.burgers_m
    signed_power = np.sum(np.abs(np.fft.rfft(kappa, axis=1)) ** 2, axis=0)
    total_power = np.abs(np.fft.rfft(total - np.mean(total))) ** 2
    signed_power[0] = 0.0
    total_power[0] = 0.0
    signed_mode = int(np.argmax(signed_power)) if np.any(signed_power > 0.0) else 0
    total_mode = int(np.argmax(total_power)) if np.any(total_power > 0.0) else 0
    response = evaluate_driven_cdd(state, parameters, gate_a_parameters)
    rotations = response.orientations
    relative = np.einsum("nij,njk->nik", rotations, np.swapaxes(np.roll(rotations, 1, axis=0), 1, 2))
    jumps = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))
    gradient = jumps / (parameters.domain_m / state.grid_points)
    gnd_ratio = float(np.sqrt(np.mean(gnd * gnd)) / max(np.mean(total), 1.0))
    total_contrast = float(np.std(total) / max(np.mean(total), 1.0))
    signed_fraction = float(
        signed_power[signed_mode] / np.sum(signed_power)
    ) if signed_mode else 0.0
    low_gradient_fraction = float(np.mean(gradient < 0.25 * max(np.max(gradient), 1.0)))
    wall_mask = gnd >= 0.5 * max(float(np.max(gnd)), 1.0)
    wall_count = int(np.sum(wall_mask & ~np.roll(wall_mask, 1)))
    wall_width = (
        float(np.sum(wall_mask) * parameters.domain_m / state.grid_points / wall_count)
        if wall_count > 0 else None
    )
    if gnd_ratio < 1.0e-3:
        classification = (
            "total-density modulation" if total_contrast >= 0.02
            else "homogeneous/balanced SSD"
        )
    elif (
        signed_fraction < 0.08 or low_gradient_fraction < 0.35
        or total_contrast < 0.005 or wall_count == 0
    ):
        classification = "diffuse GND polarization"
    else:
        classification = "GND-rich wall precursor"
    return {
        "classification": classification,
        "compatible_LAGB_candidate": False,
        "total_density_mean_m2": float(np.mean(total)),
        "total_density_contrast": total_contrast,
        "gnd_rms_m2": float(np.sqrt(np.mean(gnd * gnd))),
        "gnd_to_total_ratio": gnd_ratio,
        "signed_structure_factor_peak_fraction": signed_fraction,
        "signed_dominant_mode": signed_mode,
        "signed_dominant_wavelength_m": (
            parameters.domain_m / signed_mode if signed_mode else None
        ),
        "total_dominant_mode": total_mode,
        "orientation_gradient_rms_m_inv": float(np.sqrt(np.mean(gradient * gradient))),
        "localized_orientation_jump_deg": float(np.degrees(np.max(jumps))),
        "low_orientation_gradient_interior_fraction": low_gradient_fraction,
        "wall_count": wall_count,
        "mean_wall_FWHM_m": wall_width,
        "physical_grain_count": state.physical_grain_count,
    }
