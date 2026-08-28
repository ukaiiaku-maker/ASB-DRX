"""Prospective spatial cases anchored to the analytical peak boundary."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .fixtures import SingleGliderDDDParameterization
from .local_coupled import LocalCoupledState
from .multi_order import BinaryCircularLimit, diffuse_binary_circle
from .spatial_coupled import SpatialCoupledState


@dataclass(frozen=True)
class BoundarySpatialCase:
    temperature_K: float
    shear_rate_s_inv: float
    density_ratio: float
    domain_m: float = 1.6e-5
    density_perturbation_fraction: float = 0.01
    temperature_perturbation_K: float = 0.25
    nucleus_radius_over_critical: float = 1.35

    def __post_init__(self) -> None:
        for name in ("temperature_K", "shear_rate_s_inv", "density_ratio", "domain_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.density_perturbation_fraction < 1.0:
            raise ValueError("density_perturbation_fraction must be in [0, 1)")
        if not math.isfinite(self.temperature_perturbation_K) or self.temperature_perturbation_K < 0.0:
            raise ValueError("temperature_perturbation_K must be finite and nonnegative")
        if self.nucleus_radius_over_critical <= 1.0:
            raise ValueError("smoke nucleus must be supercritical in the sharp-interface limit")

    def build_state(
        self, points: int, fixture: SingleGliderDDDParameterization
    ) -> tuple[SpatialCoupledState, dict[str, float | str]]:
        if points < 8:
            raise ValueError("points must be at least 8")
        law = fixture.law()
        parameters = fixture.spatial_parameters()
        peak = law.net_peak(self.temperature_K, self.shear_rate_s_inv)
        nominal_density = self.density_ratio * peak.density_m2
        branch = (
            "pre_peak" if self.density_ratio < 1.0
            else "net_peak" if self.density_ratio == 1.0
            else "post_peak"
        )
        initial_stress = law.net_macroscopic_strength_Pa(
            nominal_density, self.temperature_K, self.shear_rate_s_inv
        )
        dx_m = self.domain_m / points
        interface_width_m = 2.0 * math.sqrt(
            parameters.gradient_coefficient_J_m / parameters.pair_penalty_J_m3
        )
        boundary_energy_J_m2 = math.sqrt(
            parameters.gradient_coefficient_J_m * parameters.pair_penalty_J_m3
        ) / 3.0
        target_critical_radius_m = 2.0 * interface_width_m
        density_relief_m2 = boundary_energy_J_m2 / (
            target_critical_radius_m * parameters.stored_line_energy_J_m
        )
        if density_relief_m2 >= nominal_density:
            raise ValueError("generic PF fixture requires a positive child density")
        limit = BinaryCircularLimit(
            boundary_energy_J_m2,
            parameters.stored_line_energy_J_m * density_relief_m2,
            1.0,
        )
        nucleus_radius_m = self.nucleus_radius_over_critical * limit.critical_radius_m
        if nucleus_radius_m + 2.0 * interface_width_m >= 0.5 * self.domain_m:
            raise ValueError("nucleus and diffuse interface do not fit inside the domain")
        fields = diffuse_binary_circle(points, dx_m, nucleus_radius_m, interface_width_m)
        coordinate = np.linspace(0.0, 2.0 * math.pi, points, endpoint=False)
        pattern = np.sin(coordinate)[:, None] * np.cos(coordinate)[None, :]
        parent_density = nominal_density * (
            1.0 + self.density_perturbation_fraction * pattern
        )
        child_density = parent_density - density_relief_m2
        density = np.stack((parent_density, child_density))
        temperature = self.temperature_K + self.temperature_perturbation_K * pattern
        state = SpatialCoupledState(
            stress_Pa=initial_stress,
            applied_shear=0.0,
            plastic_shear=np.zeros((points, points)),
            temperature_K=temperature,
            forest_density_m2=density,
            eta_fields=fields,
        )
        metadata: dict[str, float | str] = {
            "dx_m": dx_m,
            "peak_density_m2": peak.density_m2,
            "nominal_density_m2": nominal_density,
            "density_ratio": self.density_ratio,
            "branch": branch,
            "initial_stress_Pa": initial_stress,
            "interface_width_m": interface_width_m,
            "critical_radius_m": limit.critical_radius_m,
            "nucleus_radius_m": nucleus_radius_m,
            "density_relief_m2": density_relief_m2,
        }
        return state, metadata

    def build_local_state(
        self, points: int, fixture: SingleGliderDDDParameterization
    ) -> tuple[LocalCoupledState, dict[str, float | str]]:
        common_state, metadata = self.build_state(points, fixture)
        applied_shear = (
            common_state.stress_Pa / fixture.spatial_parameters().shear_modulus_Pa
            + float(np.mean(common_state.plastic_shear))
        )
        return (
            LocalCoupledState(
                applied_shear,
                common_state.plastic_shear,
                common_state.temperature_K,
                common_state.forest_density_m2,
                common_state.eta_fields,
                common_state.time_s,
                common_state.accepted_steps,
            ),
            metadata,
        )
