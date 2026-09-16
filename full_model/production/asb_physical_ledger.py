"""Dimensionally explicit physical/numerical ASB energy ledger.

Compatibility penalties can be useful constraint forces while remaining
outside the physical first law.  This module makes that distinction structural
and refuses residual-defined dissipation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Mapping

import numpy as np


ENERGY_TERM_CLASSIFICATION = {
    "bulk_stored_J_m3": "physical_helmholtz",
    "gradient_J_m3": "physical_helmholtz",
    "interface_J_m3": "physical_helmholtz",
    "elastic_J_m3": "physical_recoverable_elastic",
    "thermal_J_m3": "physical_thermal",
    "compatibility_alpha_penalty_J_m3": "numerical_constraint",
    "compatibility_gb_penalty_J_m3": "numerical_constraint",
}

CHANNEL_NAMES = (
    "plastic_drag", "mobile_forest_recovery", "neutral_pair_annihilation",
    "junction_relaxation", "boundary_recovery", "thermal_conduction",
    "declared_sinks",
)


@dataclass(frozen=True)
class AcceptedDissipationChannel:
    """One independently evaluated accepted process.

    ``affinity`` and ``extent_rate`` may be spatial fields.  The stored power
    is their volume mean, never a balance residual.  The affinity convention
    is the non-negative thermodynamic drop ``-A`` so that
    ``D=(-A)*dot(xi) >= 0``.
    """

    name: str
    affinity_J_per_extent: float
    extent_rate_per_m3_s: float
    dissipation_W_m3: float
    available: bool
    source: str

    def __post_init__(self):
        if self.name not in CHANNEL_NAMES:
            raise ValueError(f"unknown dissipation channel {self.name}")
        values = (self.affinity_J_per_extent, self.extent_rate_per_m3_s,
                  self.dissipation_W_m3)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"non-finite values in channel {self.name}")
        if any(float(value) < 0.0 for value in values):
            raise ValueError(f"negative accepted dissipation in channel {self.name}")
        if not self.source:
            raise ValueError(f"channel {self.name} lacks production provenance")


def accepted_channel(name, affinity_J_per_extent, extent_rate_per_m3_s, *,
                     source, available=True, component_axis=None):
    """Build a channel from independently supplied affinity/accepted extent.

    Array inputs preserve the correlation between local affinity and extent;
    multiplying their means would generally be incorrect.
    """
    affinity = np.asarray(affinity_J_per_extent, dtype=float)
    extent = np.asarray(extent_rate_per_m3_s, dtype=float)
    affinity, extent = np.broadcast_arrays(affinity, extent)
    if not np.all(np.isfinite(affinity)) or not np.all(np.isfinite(extent)):
        raise ValueError(f"non-finite affinity or extent in channel {name}")
    tolerance = 64.0*np.finfo(float).eps
    if np.min(affinity, initial=0.0) < -tolerance or np.min(extent, initial=0.0) < -tolerance:
        raise ValueError(f"negative affinity or accepted extent in channel {name}")
    affinity = np.maximum(affinity, 0.0)
    extent = np.maximum(extent, 0.0)
    product = affinity*extent
    if component_axis is not None:
        product = np.sum(product, axis=component_axis)
    return AcceptedDissipationChannel(
        name=name,
        affinity_J_per_extent=float(np.mean(affinity)),
        extent_rate_per_m3_s=float(np.mean(extent)),
        dissipation_W_m3=float(np.mean(product)),
        available=bool(available), source=str(source))


def unavailable_channel(name, source):
    return AcceptedDissipationChannel(name, 0.0, 0.0, 0.0, False, source)


@dataclass(frozen=True)
class PhysicalEnergyState:
    """Volume-averaged physical and numerical energies [J m^-3]."""

    local_line_correlation_J_m3: float = 0.0
    recoverable_elastic_J_m3: float = 0.0
    phase_interface_J_m3: float = 0.0
    physical_gb_disconnection_J_m3: float = 0.0
    thermal_J_m3: float = 0.0
    numerical_augmented_constraints_J_m3: float = 0.0
    multiplier_constraint_work_J_m3: float = 0.0

    def __post_init__(self):
        for item in fields(self):
            if not math.isfinite(float(getattr(self, item.name))):
                raise ValueError(f"non-finite energy {item.name}")

    @property
    def physical_stored_J_m3(self):
        return (self.local_line_correlation_J_m3
                + self.recoverable_elastic_J_m3
                + self.phase_interface_J_m3
                + self.physical_gb_disconnection_J_m3)


@dataclass(frozen=True)
class ProductionASBStepLedger:
    channels: tuple[AcceptedDissipationChannel, ...]
    energy_before: PhysicalEnergyState
    energy_after: PhysicalEnergyState
    external_work_J_m3: float
    deposited_heat_J_m3: float
    exported_heat_J_m3: float
    dt_s: float

    def __post_init__(self):
        names = tuple(channel.name for channel in self.channels)
        if names != CHANNEL_NAMES:
            raise ValueError("production ledger must contain all seven channels in canonical order")
        if self.dt_s <= 0.0 or not math.isfinite(self.dt_s):
            raise ValueError("accepted timestep must be finite and positive")
        for value in (self.external_work_J_m3, self.deposited_heat_J_m3,
                      self.exported_heat_J_m3):
            if not math.isfinite(float(value)):
                raise ValueError("non-finite production energy increment")
        if self.deposited_heat_J_m3 < 0.0 or self.exported_heat_J_m3 < 0.0:
            raise ValueError("heat deposition/export must be nonnegative")

    @property
    def all_channels_available(self):
        return all(channel.available for channel in self.channels)

    @property
    def dissipation_J_m3(self):
        return self.dt_s*sum(channel.dissipation_W_m3 for channel in self.channels)

    @property
    def physical_stored_change_J_m3(self):
        return self.energy_after.physical_stored_J_m3-self.energy_before.physical_stored_J_m3

    @property
    def thermal_change_J_m3(self):
        return self.energy_after.thermal_J_m3-self.energy_before.thermal_J_m3

    @property
    def numerical_constraint_change_J_m3(self):
        return (self.energy_after.numerical_augmented_constraints_J_m3
                -self.energy_before.numerical_augmented_constraints_J_m3)

    @property
    def first_law_residual_J_m3(self):
        # Dissipation is an internal conversion and is therefore not subtracted
        # again after it has appeared as deposited thermal energy.
        return (self.external_work_J_m3-self.physical_stored_change_J_m3
                -self.thermal_change_J_m3-self.exported_heat_J_m3)

    @property
    def dissipation_heat_residual_J_m3(self):
        # Fourier conduction redistributes heat and a declared bath sink
        # exports it; neither is a local mechanical heat source.  Compare
        # deposited heat only with the five constitutive conversion channels.
        mechanical = self.dt_s*sum(
            channel.dissipation_W_m3 for channel in self.channels[:5])
        return self.deposited_heat_J_m3-mechanical

    def to_dict(self):
        result = asdict(self)
        result["channels"] = [asdict(channel) for channel in self.channels]
        result.update(
            all_channels_available=self.all_channels_available,
            dissipation_J_m3=self.dissipation_J_m3,
            physical_stored_change_J_m3=self.physical_stored_change_J_m3,
            thermal_change_J_m3=self.thermal_change_J_m3,
            numerical_constraint_change_J_m3=self.numerical_constraint_change_J_m3,
            first_law_residual_J_m3=self.first_law_residual_J_m3,
            dissipation_heat_residual_J_m3=self.dissipation_heat_residual_J_m3)
        return result


def build_production_step_ledger(*, channels: Mapping[str, AcceptedDissipationChannel],
                                 energy_before, energy_after,
                                 external_work_J_m3, deposited_heat_J_m3,
                                 exported_heat_J_m3, dt_s):
    missing = set(CHANNEL_NAMES)-set(channels)
    extra = set(channels)-set(CHANNEL_NAMES)
    if missing or extra:
        raise ValueError(f"invalid channel set; missing={sorted(missing)}, extra={sorted(extra)}")
    ordered = tuple(channels[name] for name in CHANNEL_NAMES)
    if any(channel.name != name for channel, name in zip(ordered, CHANNEL_NAMES)):
        raise ValueError("channel key/name mismatch")
    return ProductionASBStepLedger(
        ordered, energy_before, energy_after, float(external_work_J_m3),
        float(deposited_heat_J_m3), float(exported_heat_J_m3), float(dt_s))


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
