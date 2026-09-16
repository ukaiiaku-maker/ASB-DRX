"""Dimensionally explicit physical/numerical ASB energy ledger.

Compatibility penalties can be useful constraint forces while remaining
outside the physical first law.  This module makes that distinction structural
and refuses residual-defined dissipation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math


ENERGY_TERM_CLASSIFICATION = {
    "bulk_stored_J_m3": "physical_helmholtz",
    "gradient_J_m3": "physical_helmholtz",
    "interface_J_m3": "physical_helmholtz",
    "elastic_J_m3": "physical_recoverable_elastic",
    "thermal_J_m3": "physical_thermal",
    "compatibility_alpha_penalty_J_m3": "numerical_constraint",
    "compatibility_gb_penalty_J_m3": "numerical_constraint",
}


@dataclass(frozen=True)
class DissipationRates:
    plastic_W_m3: float = 0.0
    recovery_W_m3: float = 0.0
    annihilation_W_m3: float = 0.0
    junction_W_m3: float = 0.0
    boundary_W_m3: float = 0.0
    thermal_W_m3: float = 0.0
    sink_W_m3: float = 0.0

    def __post_init__(self):
        for item in fields(self):
            value = float(getattr(self, item.name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"independent dissipation channel {item.name} is negative")

    @property
    def total_W_m3(self):
        return sum(float(getattr(self, item.name)) for item in fields(self))


@dataclass(frozen=True)
class ASBEnergyLedger:
    physical_free_energy_change_J_m3: float
    recoverable_elastic_change_J_m3: float
    thermal_energy_change_J_m3: float
    external_work_J_m3: float
    independent_dissipation_J_m3: float
    numerical_constraint_change_J_m3: float
    multiplier_work_J_m3: float

    @property
    def physical_first_law_residual_J_m3(self):
        return (self.external_work_J_m3
                -self.physical_free_energy_change_J_m3
                -self.recoverable_elastic_change_J_m3
                -self.thermal_energy_change_J_m3
                -self.independent_dissipation_J_m3)


def build_asb_energy_ledger(*, physical_free_energy_change_J_m3,
                            recoverable_elastic_change_J_m3,
                            thermal_energy_change_J_m3, external_work_J_m3,
                            independent_dissipation_J_m3,
                            compatibility_alpha_penalty_change_J_m3=0.0,
                            compatibility_gb_penalty_change_J_m3=0.0,
                            multiplier_work_J_m3=0.0):
    values = tuple(float(x) for x in (
        physical_free_energy_change_J_m3,
        recoverable_elastic_change_J_m3, thermal_energy_change_J_m3,
        external_work_J_m3, independent_dissipation_J_m3,
        compatibility_alpha_penalty_change_J_m3,
        compatibility_gb_penalty_change_J_m3, multiplier_work_J_m3))
    if not all(math.isfinite(x) for x in values):
        raise ValueError("ASB ledger values must be finite")
    if values[4] < 0.0:
        raise ValueError("dissipation must be independently nonnegative")
    return ASBEnergyLedger(
        values[0], values[1], values[2], values[3], values[4],
        values[5]+values[6], values[7])


def compatibility_dimensional_audit(coefficient_J_m, residual_m2,
                                    represented_volume_m3):
    """Audit ``0.5 A r²``: ``(J m)(m^-4)=J m^-3``."""
    coefficient = float(coefficient_J_m)
    residual = float(residual_m2)
    volume = float(represented_volume_m3)
    if coefficient < 0.0 or volume <= 0.0 or not all(map(math.isfinite,
                                                          (coefficient, residual, volume))):
        raise ValueError("compatibility dimensional inputs are invalid")
    density = 0.5*coefficient*residual*residual
    return {
        "coefficient_units": "J m",
        "residual_units": "m^-2",
        "energy_density_units": "J m^-3",
        "coefficient_J_m": coefficient,
        "residual_m2": residual,
        "energy_density_J_m3": density,
        "represented_energy_J": density*volume,
        "classification": "augmented_numerical_constraint",
        "eligible_as_physical_heat": False,
    }
