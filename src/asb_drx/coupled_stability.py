"""Frozen-time finite-wavenumber stability of the coupled antiplane model.

The nonzero Fourier-mode state is ``(gamma_p, T, rho_0, rho_1, phi)``, where
``eta_0=1-phi`` and ``eta_1=phi``.  Antiplane equilibrium eliminates the
displacement exactly and supplies ``delta sigma_x=-G P_xx delta gamma_p`` with
``P_xx=k_y^2/(k_x^2+k_y^2)``.  The zero mode belongs to the imposed-loading
problem and is intentionally excluded from this localizing-mode operator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .analytical import ExpFloorLaw, KB_J_PER_K
from .multi_order import interpolation_h, interpolation_h_prime
from .spatial_coupled import SpatialCoupledParameters


STATE_NAMES = ("plastic_shear", "temperature_K", "parent_density_m2",
               "child_density_m2", "child_order")


@dataclass(frozen=True)
class HomogeneousCoupledState:
    macroscopic_stress_Pa: float
    temperature_K: float
    parent_density_m2: float
    child_density_m2: float
    child_order: float

    def __post_init__(self) -> None:
        for name in ("macroscopic_stress_Pa", "temperature_K",
                     "parent_density_m2", "child_density_m2"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.child_order) or not 0.0 < self.child_order < 1.0:
            raise ValueError("child_order must be strictly between zero and one")


@dataclass(frozen=True)
class NetRateTangents:
    net_rate_s_inv: float
    macroscopic_stress_tangent_Pa_inv_s_inv: float
    temperature_tangent_K_inv_s_inv: float
    density_tangent_m2_s_inv: float
    forward_rate_s_inv: float
    reverse_rate_s_inv: float


@dataclass(frozen=True)
class CoupledStabilityMode:
    kx_m_inv: float
    ky_m_inv: float
    antiplane_projection: float
    storage_branch: str
    state_names: tuple[str, ...]
    jacobian: np.ndarray
    eigenvalues_s_inv: np.ndarray
    maximum_growth_rate_s_inv: float


def _temperature_log_tangent(
    law: ExpFloorLaw, local_stress_Pa: float, temperature_K: float
) -> float:
    barrier = law.barrier_J(local_stress_Pa, temperature_K)
    G0 = law.barrier_scale_J(temperature_K)
    dG0 = law.barrier_scale_temperature_derivative_J_K(temperature_K)
    if local_stress_Pa == 0.0:
        dbarrier = dG0
    else:
        tau_c = law.stress_scale_Pa(temperature_K)
        reduced = local_stress_Pa / tau_c
        exponential = math.exp(-law.shape_a * reduced**law.shape_n)
        fraction = law.floor_fraction + (1.0 - law.floor_fraction) * exponential
        dr_dT = (
            law.stress_temperature_coefficient * reduced
            / law.reference_temperature_K
        )
        dF_dr = (
            -(1.0 - law.floor_fraction) * exponential * law.shape_a
            * law.shape_n * reduced ** (law.shape_n - 1.0)
        )
        dbarrier = dG0 * fraction + G0 * dF_dr * dr_dT
    return (
        barrier / (KB_J_PER_K * temperature_K**2)
        - dbarrier / (KB_J_PER_K * temperature_K)
    )


def net_common_stress_rate_tangents(
    law: ExpFloorLaw,
    macroscopic_stress_Pa: float,
    density_m2: float,
    temperature_K: float,
) -> NetRateTangents:
    """Tangents of positive net flow at fixed macroscopic stress."""
    if not math.isfinite(macroscopic_stress_Pa) or macroscopic_stress_Pa <= 0.0:
        raise ValueError("macroscopic_stress_Pa must be finite and positive")
    ratio = law.taylor_ratio(density_m2)
    local_stress = macroscopic_stress_Pa / ratio
    forward = law.shear_rate_s_inv(local_stress, density_m2, temperature_K)
    reverse = law.shear_rate_s_inv(0.0, density_m2, temperature_K)
    net = forward - reverse
    if net <= 0.0:
        raise ValueError("positive stress must produce positive net flow")
    volume = law.activation_volume_m3(local_stress, temperature_K)
    stress_tangent = forward * volume / (
        KB_J_PER_K * temperature_K * ratio
    )
    forward_density_log_tangent = (
        law.density_exponent_p
        - volume * local_stress / (KB_J_PER_K * temperature_K)
    ) / (2.0 * density_m2)
    reverse_density_log_tangent = law.density_exponent_p / (2.0 * density_m2)
    density_tangent = (
        forward * forward_density_log_tangent
        - reverse * reverse_density_log_tangent
    )
    temperature_tangent = (
        forward * _temperature_log_tangent(law, local_stress, temperature_K)
        - reverse * _temperature_log_tangent(law, 0.0, temperature_K)
    )
    return NetRateTangents(
        net, stress_tangent, temperature_tangent, density_tangent,
        forward, reverse,
    )


def _h_second(value: float) -> float:
    return 60.0 * value * (1.0 - value) * (1.0 - 2.0 * value)


def _branch(
    state: HomogeneousCoupledState,
    parameters: SpatialCoupledParameters,
    requested: str,
) -> str:
    if requested not in ("auto", "uncapped", "capped"):
        raise ValueError("storage_branch must be 'auto', 'uncapped', or 'capped'")
    threshold = (
        parameters.stored_line_energy_J_m
        * parameters.forest_storage_per_plastic_strain_m2
    )
    scale = max(state.macroscopic_stress_Pa, threshold, 1.0)
    if abs(state.macroscopic_stress_Pa - threshold) <= 1.0e-12 * scale:
        raise ValueError("storage-cap switching surface has no unique Jacobian")
    actual = "uncapped" if state.macroscopic_stress_Pa > threshold else "capped"
    if requested != "auto" and requested != actual:
        raise ValueError(f"requested {requested} storage branch is inactive")
    return actual


def coupled_mode_rhs(
    perturbation: np.ndarray,
    law: ExpFloorLaw,
    state: HomogeneousCoupledState,
    kx_m_inv: float,
    ky_m_inv: float,
    parameters: SpatialCoupledParameters,
    *,
    storage_branch: str = "auto",
) -> np.ndarray:
    """Nonlinear frozen-mode RHS used to verify the analytical Jacobian."""
    x = np.asarray(perturbation, dtype=float)
    if x.shape != (5,) or not np.all(np.isfinite(x)):
        raise ValueError("perturbation must be a finite five-vector")
    k2 = kx_m_inv**2 + ky_m_inv**2
    if not math.isfinite(k2) or k2 <= 0.0:
        raise ValueError("a finite nonzero wave vector is required")
    projection = ky_m_inv**2 / k2
    branch = _branch(state, parameters, storage_branch)
    sigma = state.macroscopic_stress_Pa - parameters.shear_modulus_Pa * projection * x[0]
    temperature = state.temperature_K + x[1]
    rho0 = state.parent_density_m2 + x[2]
    rho1 = state.child_density_m2 + x[3]
    phi = state.child_order + x[4]
    if min(sigma, temperature, rho0, rho1, phi, 1.0 - phi) <= 0.0:
        raise ValueError("perturbation leaves the positive interior state")
    r0 = law.net_shear_rate_s_inv(sigma / law.taylor_ratio(rho0), rho0, temperature)
    r1 = law.net_shear_rate_s_inv(sigma / law.taylor_ratio(rho1), rho1, temperature)
    h0, h1 = interpolation_h(np.asarray((1.0 - phi, phi)))
    rate = h0 * r0 + h1 * r1
    E = parameters.stored_line_energy_J_m
    nominal_K = parameters.forest_storage_per_plastic_strain_m2
    K = nominal_K if branch == "uncapped" else sigma / E
    mechanical_heat = (sigma - E * K) * rate
    hp0, hp1 = interpolation_h_prime(np.asarray((1.0 - phi, phi)))
    local_difference = (
        2.0 * parameters.pair_penalty_J_m3 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
        + E * (rho1 * hp1 - rho0 * hp0)
    )
    difference = local_difference + 2.0 * parameters.gradient_coefficient_J_m * k2 * x[4]
    mobility = parameters.phase_mobility_m3_J_s
    phase_heat = 0.5 * mobility * difference**2
    C = parameters.volumetric_heat_capacity_J_m3_K
    alpha = parameters.thermal_conductivity_W_m_K / C
    return np.asarray((
        rate,
        (mechanical_heat + phase_heat) / C - alpha * k2 * x[1],
        K * r0,
        K * r1,
        -0.5 * mobility * difference,
    ))


def full_coupled_stability_mode(
    law: ExpFloorLaw,
    state: HomogeneousCoupledState,
    kx_m_inv: float,
    ky_m_inv: float,
    parameters: SpatialCoupledParameters,
    *,
    storage_branch: str = "auto",
) -> CoupledStabilityMode:
    """Return the analytical frozen-time 5x5 Fourier-mode operator."""
    k2 = kx_m_inv**2 + ky_m_inv**2
    if not math.isfinite(k2) or k2 <= 0.0:
        raise ValueError("a finite nonzero wave vector is required")
    projection = ky_m_inv**2 / k2
    branch = _branch(state, parameters, storage_branch)
    sigma = state.macroscopic_stress_Pa
    temperature = state.temperature_K
    phi = state.child_order
    rho = (state.parent_density_m2, state.child_density_m2)
    tangent = tuple(
        net_common_stress_rate_tangents(law, sigma, item, temperature)
        for item in rho
    )
    h0, h1 = interpolation_h(np.asarray((1.0 - phi, phi)))
    hp0, hp1 = interpolation_h_prime(np.asarray((1.0 - phi, phi)))
    weights = (float(h0), float(h1))
    rate = sum(weights[i] * tangent[i].net_rate_s_inv for i in range(2))
    rate_sigma = sum(
        weights[i] * tangent[i].macroscopic_stress_tangent_Pa_inv_s_inv
        for i in range(2)
    )
    rate_T = sum(weights[i] * tangent[i].temperature_tangent_K_inv_s_inv for i in range(2))
    rate_rho = (
        weights[0] * tangent[0].density_tangent_m2_s_inv,
        weights[1] * tangent[1].density_tangent_m2_s_inv,
    )
    rate_phi = float(hp1) * (tangent[1].net_rate_s_inv - tangent[0].net_rate_s_inv)
    stress_from_gamma = -parameters.shear_modulus_Pa * projection
    rate_gradient = np.asarray((
        rate_sigma * stress_from_gamma, rate_T, rate_rho[0], rate_rho[1], rate_phi,
    ))

    E = parameters.stored_line_energy_J_m
    nominal_K = parameters.forest_storage_per_plastic_strain_m2
    mobility = parameters.phase_mobility_m3_J_s
    pair = parameters.pair_penalty_J_m3
    D0 = (
        2.0 * pair * phi * (1.0 - phi) * (1.0 - 2.0 * phi)
        + E * (rho[1] * hp1 - rho[0] * hp0)
    )
    D_gradient = np.asarray((
        0.0,
        0.0,
        -E * float(hp0),
        E * float(hp1),
        2.0 * pair * (1.0 - 6.0 * phi + 6.0 * phi**2)
        + E * (rho[1] * _h_second(phi) + rho[0] * _h_second(1.0 - phi))
        + 2.0 * parameters.gradient_coefficient_J_m * k2,
    ))
    phase_heat_gradient = mobility * D0 * D_gradient

    J = np.zeros((5, 5), dtype=float)
    J[0] = rate_gradient
    if branch == "uncapped":
        K = nominal_K
        heat_factor = sigma - E * K
        mechanical_heat_gradient = heat_factor * rate_gradient
        mechanical_heat_gradient[0] += rate * stress_from_gamma
        for grain in range(2):
            J[2 + grain] = K * np.asarray((
                tangent[grain].macroscopic_stress_tangent_Pa_inv_s_inv * stress_from_gamma,
                tangent[grain].temperature_tangent_K_inv_s_inv,
                tangent[grain].density_tangent_m2_s_inv if grain == 0 else 0.0,
                tangent[grain].density_tangent_m2_s_inv if grain == 1 else 0.0,
                0.0,
            ))
    else:
        K = sigma / E
        mechanical_heat_gradient = np.zeros(5)
        for grain in range(2):
            J[2 + grain] = K * np.asarray((
                tangent[grain].macroscopic_stress_tangent_Pa_inv_s_inv * stress_from_gamma,
                tangent[grain].temperature_tangent_K_inv_s_inv,
                tangent[grain].density_tangent_m2_s_inv if grain == 0 else 0.0,
                tangent[grain].density_tangent_m2_s_inv if grain == 1 else 0.0,
                0.0,
            ))
            J[2 + grain, 0] += tangent[grain].net_rate_s_inv * stress_from_gamma / E
    C = parameters.volumetric_heat_capacity_J_m3_K
    J[1] = (mechanical_heat_gradient + phase_heat_gradient) / C
    J[1, 1] -= parameters.thermal_conductivity_W_m_K * k2 / C
    J[4] = -0.5 * mobility * D_gradient

    # A similarity scaling protects the eigensolve from the disparate physical
    # units without changing eigenvalues.
    scales = np.asarray((1.0, temperature, rho[0], rho[1], 1.0))
    scaled = J * scales[None, :] / scales[:, None]
    eigenvalues = np.linalg.eigvals(scaled)
    return CoupledStabilityMode(
        kx_m_inv, ky_m_inv, projection, branch, STATE_NAMES, J,
        eigenvalues, float(np.max(np.real(eigenvalues))),
    )
