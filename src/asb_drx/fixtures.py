"""Declared generic fixtures with immutable provenance; not material calibrations."""

from __future__ import annotations

from dataclasses import dataclass

from .analytical import ExpFloorLaw, KB_J_PER_K
from .spatial_coupled import SpatialCoupledParameters
from .recovery import (
    RecoveryBoundaryPoint,
    RecoveryLaw,
    fit_recovery_law_to_boundary,
)

EV_J = 1.602176634e-19


@dataclass(frozen=True)
class SingleGliderDDDParameterization:
    source_path: str = "/Users/sdillon/Taylor_DDD/results/full_glider_expfloor_HT050_F020_overnight_peakmap_20260817/T1050/rho_3e16/seed86/clean_arrhenius_params.json"
    source_sha256: str = "14a7a3c7341da5f7d991c229af5efe7d2a4e1cb2ada4597b2cdad44efd8b2b2b"
    campaign_jobs_sha256: str = "d9d2d119fa9ffbb50c47024ad87ace0357284559e32ce3521d92c0c73dfce63d"
    enthalpy_eV: float = 0.50
    entropy_kB: float = -9.0
    stress_scale_GPa: float = 14.5
    floor_fraction: float = 0.20
    shape_a: float = 6.65607
    shape_n: float = 2.15276
    attempt_frequency_s_inv: float = 1.0e12
    burgers_m: float = 2.48e-10
    shear_modulus_Pa: float = 8.0e10
    taylor_geometry_factor: float = 2.0
    density_exponent_p: float = 4.0
    reference_temperature_K: float = 1000.0
    source_temperature_range_K: tuple[float, float] = (850.0, 1050.0)
    source_density_range_m2: tuple[float, float] = (1.0e15, 3.0e16)
    source_strain_rate_s_inv: float = 4.5

    def law(self) -> ExpFloorLaw:
        reference_barrier_J = (
            self.enthalpy_eV * EV_J
            - KB_J_PER_K * self.reference_temperature_K * self.entropy_kB
        )
        return ExpFloorLaw(
            barrier_ref_J=reference_barrier_J,
            stress_ref_Pa=self.stress_scale_GPa * 1.0e9,
            reference_temperature_K=self.reference_temperature_K,
            floor_fraction=self.floor_fraction,
            shape_a=self.shape_a,
            shape_n=self.shape_n,
            rate_prefactor_s_inv=self.attempt_frequency_s_inv,
            density_exponent_p=self.density_exponent_p,
            burgers_m=self.burgers_m,
            barrier_entropy_kB=self.entropy_kB,
            taylor_geometry_factor=self.taylor_geometry_factor,
        )

    def spatial_parameters(self) -> SpatialCoupledParameters:
        """DDD mechanics plus explicitly inherited generic PF/thermal fixtures."""
        line_energy = 0.5 * self.shear_modulus_Pa * self.burgers_m**2
        return SpatialCoupledParameters(
            shear_modulus_Pa=self.shear_modulus_Pa,
            volumetric_heat_capacity_J_m3_K=3.5e6,
            thermal_conductivity_W_m_K=5.0,
            stored_line_energy_J_m=line_energy,
            forest_storage_per_plastic_strain_m2=1.0e14,
            pair_penalty_J_m3=2.0e6,
            gradient_coefficient_J_m=1.0e-6,
            phase_mobility_m3_J_s=5.0e-7,
        )

    def recovery_law(self) -> RecoveryLaw:
        """Generic two-anchor recovery design; not a material calibration."""
        return fit_recovery_law_to_boundary(
            self.law(),
            self.spatial_parameters().forest_storage_per_plastic_strain_m2,
            RecoveryBoundaryPoint(850.0, 450.0, 2.0),
            RecoveryBoundaryPoint(1050.0, 45000.0, 2.0),
            reference_temperature_K=950.0,
        ).law
