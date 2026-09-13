"""Adapter from authoritative full-v34 fields to stateful embryo dynamics."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:
    from .stateful_embryos import (
        EmbryoEnvironment, EmbryoPopulation, circular_phase_support,
        evolve_embryo, phase_support_metrics,
    )
except ImportError:  # pragma: no cover
    from stateful_embryos import (
        EmbryoEnvironment, EmbryoPopulation, circular_phase_support,
        evolve_embryo, phase_support_metrics,
    )


@dataclass(frozen=True)
class FullFieldEnvironment:
    temperature_K: np.ndarray
    stored_relief_J_m3: np.ndarray
    orientation_penalty_J_m3: np.ndarray
    compatibility_penalty_J_m3: np.ndarray
    gb_contact: np.ndarray
    wall_contact: np.ndarray
    gnd_contact: np.ndarray

    def __post_init__(self):
        arrays = [np.asarray(getattr(self, name), dtype=float) for name in (
            "temperature_K", "stored_relief_J_m3", "orientation_penalty_J_m3",
            "compatibility_penalty_J_m3", "gb_contact", "wall_contact", "gnd_contact")]
        if arrays[0].ndim != 2 or any(array.shape != arrays[0].shape for array in arrays):
            raise ValueError("full embryo-environment fields must share one 2-D grid")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("full embryo-environment fields must be finite")
        if np.any(arrays[0] <= 0.0):
            raise ValueError("temperature must be positive")
        if any(np.any(array < 0.0) for array in arrays[1:4]):
            raise ValueError("energy-density fields must be nonnegative")
        if any(np.any((array < 0.0) | (array > 1.0)) for array in arrays[4:]):
            raise ValueError("contact fields must lie in [0,1]")


@dataclass(frozen=True)
class PopulationStepLedger:
    embryo_count: int
    active_count: int
    promotable_count: int
    retired_count: int
    free_energy_change_J: float
    dissipated_energy_J: float
    closure_error_J: float
    maximum_halvings: int


def _weighted_mean(field, weight):
    denominator = float(np.sum(weight))
    if denominator <= 0.0:
        raise ValueError("embryo support has zero weight")
    return float(np.sum(np.asarray(field, dtype=float) * weight) / denominator)


def local_environment(record, fields, domain_lengths_m, interface_width_m,
                      purity_threshold):
    shape = np.asarray(fields.temperature_K).shape
    support = circular_phase_support(
        record, shape, domain_lengths_m, interface_width_m)
    dx_m = domain_lengths_m[0] / shape[0]
    dy_m = domain_lengths_m[1] / shape[1]
    area, purity = phase_support_metrics(support, dx_m, dy_m, purity_threshold)
    return EmbryoEnvironment(
        temperature_K=_weighted_mean(fields.temperature_K, support),
        stored_relief_J_m3=_weighted_mean(fields.stored_relief_J_m3, support),
        orientation_penalty_J_m3=_weighted_mean(fields.orientation_penalty_J_m3, support),
        compatibility_penalty_J_m3=_weighted_mean(fields.compatibility_penalty_J_m3, support),
        phase_support_area_m2=area, phase_purity=purity,
        gb_contact_fraction=_weighted_mean(fields.gb_contact, support),
        wall_contact_fraction=_weighted_mean(fields.wall_contact, support),
        gnd_contact_fraction=_weighted_mean(fields.gnd_contact, support))


def advance_population(population, *, fields, step, time_s, proposed_dt_s,
                       domain_lengths_m, interface_width_m, purity_threshold,
                       parameters):
    """Advance all records against one frozen authoritative full-field state."""
    if not isinstance(population, EmbryoPopulation):
        raise TypeError("population must be an EmbryoPopulation")
    records = []
    free_change = 0.0
    dissipation = 0.0
    closure = 0.0
    max_halvings = 0
    for record in population.records:
        environment = local_environment(
            record, fields, domain_lengths_m, interface_width_m, purity_threshold)
        result = evolve_embryo(
            record, step, time_s, proposed_dt_s, environment, parameters)
        records.append(result.record)
        free_change += result.ledger.free_energy_change_J
        dissipation += result.ledger.dissipated_energy_J
        closure += result.ledger.closure_error_J
        max_halvings = max(max_halvings, result.halvings)
    new_population = EmbryoPopulation(population.next_id, tuple(records))
    ledger = PopulationStepLedger(
        embryo_count=len(records),
        active_count=sum(record.status == "active" for record in records),
        promotable_count=sum(record.status == "promotable" for record in records),
        retired_count=sum(record.status == "retired" for record in records),
        free_energy_change_J=free_change, dissipated_energy_J=dissipation,
        closure_error_J=closure, maximum_halvings=max_halvings)
    if not math.isclose(-free_change, dissipation + closure, rel_tol=0.0,
                        abs_tol=256*math.ulp(max(abs(free_change), 1e-300))):
        raise RuntimeError("embryo population energy ledger does not close")
    return new_population, ledger
