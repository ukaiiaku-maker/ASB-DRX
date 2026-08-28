"""Finite-wavenumber thermal/storage stability of the EXP-floor common-stress limit."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .analytical import ExpFloorLaw, KB_J_PER_K


@dataclass(frozen=True)
class StabilityParameters:
    volumetric_heat_capacity_J_m3_K: float
    thermal_conductivity_W_m_K: float
    stored_line_energy_J_m: float
    forest_storage_per_plastic_strain_m2: float

    def __post_init__(self) -> None:
        for name in ("volumetric_heat_capacity_J_m3_K", "stored_line_energy_J_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("thermal_conductivity_W_m_K", "forest_storage_per_plastic_strain_m2"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class RateTangents:
    plastic_rate_s_inv: float
    temperature_tangent_s_inv_K: float
    density_tangent_m2_s_inv: float
    local_activation_stress_Pa: float


@dataclass(frozen=True)
class StabilityMode:
    wavenumber_m_inv: float
    jacobian_s_inv: np.ndarray
    eigenvalues_s_inv: np.ndarray
    maximum_growth_rate_s_inv: float


def common_stress_rate_tangents(
    law: ExpFloorLaw, macroscopic_stress_Pa: float,
    density_m2: float, temperature_K: float,
) -> RateTangents:
    if not math.isfinite(macroscopic_stress_Pa) or macroscopic_stress_Pa <= 0.0:
        raise ValueError("macroscopic_stress_Pa must be finite and positive")
    ratio = law.taylor_ratio(density_m2)
    local_stress = macroscopic_stress_Pa / ratio
    rate = law.shear_rate_s_inv(local_stress, density_m2, temperature_K)
    barrier = law.barrier_J(local_stress, temperature_K)
    volume = law.activation_volume_m3(local_stress, temperature_K)
    density_log_tangent = (
        law.density_exponent_p - volume * local_stress / (KB_J_PER_K * temperature_K)
    ) / (2.0 * density_m2)

    G0 = law.barrier_scale_J(temperature_K)
    tau_c = law.stress_scale_Pa(temperature_K)
    reduced_stress = local_stress / tau_c
    exponential = math.exp(-law.shape_a * reduced_stress**law.shape_n)
    barrier_fraction = law.floor_fraction + (1.0-law.floor_fraction)*exponential
    dG0_dT = -law.barrier_temperature_coefficient * G0 / law.reference_temperature_K
    dr_dT = law.stress_temperature_coefficient * reduced_stress / law.reference_temperature_K
    dF_dr = -(1.0-law.floor_fraction)*exponential*law.shape_a*law.shape_n*reduced_stress**(law.shape_n-1.0)
    dbarrier_dT = dG0_dT*barrier_fraction + G0*dF_dr*dr_dT
    temperature_log_tangent = barrier/(KB_J_PER_K*temperature_K**2) - dbarrier_dT/(KB_J_PER_K*temperature_K)
    return RateTangents(rate, rate*temperature_log_tangent, rate*density_log_tangent, local_stress)


def thermal_storage_mode(
    law: ExpFloorLaw, macroscopic_stress_Pa: float, density_m2: float,
    temperature_K: float, wavenumber_m_inv: float,
    parameters: StabilityParameters,
) -> StabilityMode:
    if not math.isfinite(wavenumber_m_inv) or wavenumber_m_inv <= 0.0:
        raise ValueError("wavenumber_m_inv must be finite and positive")
    tangent = common_stress_rate_tangents(law, macroscopic_stress_Pa, density_m2, temperature_K)
    heat_per_plastic_strain = macroscopic_stress_Pa - (
        parameters.stored_line_energy_J_m * parameters.forest_storage_per_plastic_strain_m2
    )
    if heat_per_plastic_strain < 0.0:
        raise ValueError("storage requests more energy than macroscopic plastic work")
    C = parameters.volumetric_heat_capacity_J_m3_K
    K = parameters.forest_storage_per_plastic_strain_m2
    alpha = parameters.thermal_conductivity_W_m_K / C
    jacobian = np.asarray([
        [heat_per_plastic_strain*tangent.temperature_tangent_s_inv_K/C-alpha*wavenumber_m_inv**2,
         heat_per_plastic_strain*tangent.density_tangent_m2_s_inv/C],
        [K*tangent.temperature_tangent_s_inv_K, K*tangent.density_tangent_m2_s_inv],
    ])
    eigenvalues = np.linalg.eigvals(jacobian)
    maximum = float(np.max(np.real(eigenvalues)))
    return StabilityMode(wavenumber_m_inv, jacobian, eigenvalues, maximum)


def local_thermal_storage_rhs(
    law: ExpFloorLaw, macroscopic_stress_Pa: float, density_m2: float,
    temperature_K: float, parameters: StabilityParameters,
) -> np.ndarray:
    tangent = common_stress_rate_tangents(law, macroscopic_stress_Pa, density_m2, temperature_K)
    heat_per_plastic_strain = macroscopic_stress_Pa-parameters.stored_line_energy_J_m*parameters.forest_storage_per_plastic_strain_m2
    if heat_per_plastic_strain < 0.0:
        raise ValueError("storage requests more energy than macroscopic plastic work")
    return np.asarray([
        heat_per_plastic_strain*tangent.plastic_rate_s_inv/parameters.volumetric_heat_capacity_J_m3_K,
        parameters.forest_storage_per_plastic_strain_m2*tangent.plastic_rate_s_inv,
    ])
