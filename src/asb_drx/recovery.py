"""Analytically constrained dynamic-recovery law and boundary design."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .analytical import ExpFloorLaw, KB_J_PER_K
from .coupled_stability import net_common_stress_rate_tangents


@dataclass(frozen=True)
class RecoveryLaw:
    """First-order density recovery with an Arrhenius relaxation time."""

    reference_temperature_K: float
    relaxation_time_ref_s: float
    activation_energy_J: float
    equilibrium_density_m2: float = 0.0

    def __post_init__(self) -> None:
        for name in ("reference_temperature_K", "relaxation_time_ref_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.activation_energy_J) or self.activation_energy_J < 0.0:
            raise ValueError("activation_energy_J must be finite and nonnegative")
        if not math.isfinite(self.equilibrium_density_m2) or self.equilibrium_density_m2 < 0.0:
            raise ValueError("equilibrium_density_m2 must be finite and nonnegative")

    def inverse_time_s_inv(self, temperature_K: float) -> float:
        if not math.isfinite(temperature_K) or temperature_K <= 0.0:
            raise ValueError("temperature_K must be finite and positive")
        exponent = -self.activation_energy_J / KB_J_PER_K * (
            1.0 / temperature_K - 1.0 / self.reference_temperature_K
        )
        return math.exp(exponent) / self.relaxation_time_ref_s

    def temperature_tangent_s_inv_K(self, temperature_K: float) -> float:
        inverse_time = self.inverse_time_s_inv(temperature_K)
        return inverse_time * self.activation_energy_J / (
            KB_J_PER_K * temperature_K**2
        )


@dataclass(frozen=True)
class RecoveryBoundaryPoint:
    temperature_K: float
    shear_rate_s_inv: float
    density_ratio_to_net_peak: float

    def __post_init__(self) -> None:
        for name in ("temperature_K", "shear_rate_s_inv", "density_ratio_to_net_peak"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class RecoveryBoundaryFit:
    law: RecoveryLaw
    anchors: tuple[RecoveryBoundaryPoint, RecoveryBoundaryPoint]
    anchor_storage_tangents_s_inv: tuple[float, float]
    maximum_log_closure_error: float


def post_peak_density_growth_rate_s_inv(
    flow_law: ExpFloorLaw,
    recovery_law: RecoveryLaw,
    forest_storage_per_plastic_strain_m2: float,
    point: RecoveryBoundaryPoint,
) -> float:
    peak = flow_law.net_peak(point.temperature_K, point.shear_rate_s_inv)
    density = point.density_ratio_to_net_peak * peak.density_m2
    stress = flow_law.net_macroscopic_strength_Pa(
        density, point.temperature_K, point.shear_rate_s_inv
    )
    rate_tangent = net_common_stress_rate_tangents(
        flow_law, stress, density, point.temperature_K
    ).density_tangent_m2_s_inv
    return (
        forest_storage_per_plastic_strain_m2 * rate_tangent
        - recovery_law.inverse_time_s_inv(point.temperature_K)
    )


def fit_recovery_law_to_boundary(
    flow_law: ExpFloorLaw,
    forest_storage_per_plastic_strain_m2: float,
    first: RecoveryBoundaryPoint,
    second: RecoveryBoundaryPoint,
    *,
    reference_temperature_K: float,
    equilibrium_density_m2: float = 0.0,
) -> RecoveryBoundaryFit:
    """Solve exactly for the two Arrhenius recovery parameters.

    At a compatible, isothermal perturbation the post-peak density eigenvalue
    is ``K dr/drho - 1/tau_rec(T)``. Two user-declared neutral boundary points
    therefore determine the activation energy and reference relaxation time
    without fitting the immutable EXP-floor parameters.
    """
    if not math.isfinite(forest_storage_per_plastic_strain_m2) or forest_storage_per_plastic_strain_m2 <= 0.0:
        raise ValueError("forest storage coefficient must be finite and positive")
    if first.temperature_K == second.temperature_K:
        raise ValueError("boundary anchors must have distinct temperatures")
    values = []
    for point in (first, second):
        peak = flow_law.net_peak(point.temperature_K, point.shear_rate_s_inv)
        density = point.density_ratio_to_net_peak * peak.density_m2
        stress = flow_law.net_macroscopic_strength_Pa(
            density, point.temperature_K, point.shear_rate_s_inv
        )
        tangent = net_common_stress_rate_tangents(
            flow_law, stress, density, point.temperature_K
        ).density_tangent_m2_s_inv
        storage_tangent = forest_storage_per_plastic_strain_m2 * tangent
        if storage_tangent <= 0.0:
            raise ValueError("recovery boundary anchors must be on the post-peak density branch")
        values.append(storage_tangent)
    inverse_temperature_difference = 1.0 / first.temperature_K - 1.0 / second.temperature_K
    activation = KB_J_PER_K * math.log(values[1] / values[0]) / inverse_temperature_difference
    if activation < 0.0:
        raise ValueError("anchors imply a negative recovery activation energy")
    inverse_time_ref = values[0] * math.exp(
        activation / KB_J_PER_K
        * (1.0 / first.temperature_K - 1.0 / reference_temperature_K)
    )
    law = RecoveryLaw(
        reference_temperature_K,
        1.0 / inverse_time_ref,
        activation,
        equilibrium_density_m2,
    )
    errors = tuple(
        abs(math.log(law.inverse_time_s_inv(point.temperature_K) / value))
        for point, value in zip((first, second), values)
    )
    return RecoveryBoundaryFit(law, (first, second), tuple(values), max(errors))
