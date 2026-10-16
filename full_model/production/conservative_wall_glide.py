"""Local conservative glide of signed wall line down its chemical potential.

The operator moves the existing positive and negative wall populations on
nearest-neighbour edges.  It does not create line, relabel Burgers families,
change lattice orientation, or reset plastic history.  The chemical potential
is the derivative of the authoritative common-wall energy, including the
ordered-wall/Frank--Bilby mismatch.  A checkerboard edge update preserves
nonnegativity and every signed family inventory exactly; the complete energy
is nevertheless re-evaluated and remains the acceptance gate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .common_tensorial_wall import wall_free_energy_derivatives
from .complete_multigrain_energy import evaluate_complete_multigrain_energy
from .structural_boundary import (
    StructuralBoundaryParameters, exp_floor_event_rate_Hz,
)


@dataclass(frozen=True)
class ConservativeWallGlideParameters:
    activation: StructuralBoundaryParameters = StructuralBoundaryParameters(
        activation_enthalpy_eV=.75, barrier_floor_eV=.05,
        attempt_frequency_Hz=1.0e7, activation_segment_m=2.5e-9)
    minimum_wall_order: float = .20
    owner_support_threshold: float = .75
    maximum_edge_fraction: float = .25
    maximum_backtracks: int = 20


@dataclass(frozen=True)
class ConservativeWallGlideDecision:
    accepted: bool
    classification: str
    candidate_state: object
    committed_state: object
    moved_line_cell_sum_m2: float
    inventory_residual_m2: float
    maximum_rate_s: float
    backtracks: int
    delta_helmholtz_J: float
    generated_heat_J: float
    first_law_residual_J: float
    relative_first_law_residual: float
    heat_source_J_by_cell: np.ndarray


def _checkerboard_glide(population, chemical_J_m, temperature_K, domain,
                        interval_s, parameters):
    value = np.asarray(population, dtype=float).copy()
    chemical = np.asarray(chemical_J_m, dtype=float)
    temperature = np.asarray(temperature_K, dtype=float)
    shape = value.shape[:2]
    if any(size % 2 for size in shape):
        raise ValueError("checkerboard wall glide requires even grid extents")
    moved = 0.0
    maximum_rate = 0.0
    edge_dt = float(interval_s)/4.0
    for axis in (0, 1):
        coordinate = np.arange(shape[axis]).reshape(
            (-1, 1) if axis == 0 else (1, -1))
        for parity in (0, 1):
            neighbor_value = np.roll(value, -1, axis=axis)
            neighbor_mu = np.roll(chemical, -1, axis=axis)
            neighbor_temperature = np.roll(temperature, -1, axis=axis)
            eligible = (domain & np.roll(domain, -1, axis=axis)
                        &(coordinate % 2 == parity))
            difference = chemical-neighbor_mu
            affinity = np.abs(difference)
            edge_temperature = .5*(temperature+neighbor_temperature)
            rate = exp_floor_event_rate_Hz(
                affinity, edge_temperature[..., None], parameters.activation)
            maximum_rate = max(maximum_rate, float(np.max(rate[eligible]))
                               if np.any(eligible) else 0.0)
            fraction = np.minimum(
                -np.expm1(-rate*edge_dt), parameters.maximum_edge_fraction)
            donor = np.where(difference >= 0.0, value, neighbor_value)
            flux = np.where(
                eligible[..., None], np.sign(difference)*fraction*donor, 0.0)
            value = value-flux+np.roll(flux, 1, axis=axis)
            moved += float(np.sum(np.abs(flux), dtype=np.longdouble))
    if np.min(value) < -1e-12*max(float(np.max(population)), 1.0):
        raise RuntimeError("conservative wall glide violated nonnegativity")
    return np.maximum(value, 0.0), moved, maximum_rate


def evaluate_conservative_wall_glide(
        state, *, interval_s, mean_strain, spacing_m,
        represented_thickness_m, systems, wall_parameters,
        energy_kwargs=None, parameters=ConservativeWallGlideParameters(),
        maximum_relative_first_law_residual=.05):
    """Advance local wall-line glide and atomically audit its heat release."""
    dt = float(interval_s)
    spacing = float(spacing_m)
    thickness = float(represented_thickness_m)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("positive finite wall-glide interval required")
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    options.pop("mean_strain", None)
    common = dict(
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall_parameters, mean_strain=np.asarray(mean_strain),
        **options)
    before = evaluate_complete_multigrain_energy(state, **common)
    proposed_owners = []
    moved = 0.0
    maximum_rate = 0.0
    activity = np.zeros(state.supports.shape[1:])
    before_inventory = []
    for support, owner in zip(state.supports, state.owners):
        domain = ((np.asarray(support) > parameters.owner_support_threshold)
                  &(np.asarray(owner.wall_order)
                    >= parameters.minimum_wall_order))
        chemical = wall_free_energy_derivatives(
            owner, wall_parameters, options.get("topologies", ()), systems)
        fields = {}
        owner_activity = np.zeros_like(activity)
        for sign in ("plus", "minus"):
            name = f"wall_{sign}_m2"
            population = np.asarray(getattr(owner, name))
            before_inventory.append(np.sum(
                population, axis=(0, 1), dtype=np.longdouble))
            candidate, sign_moved, sign_rate = _checkerboard_glide(
                population, chemical[f"wall_{sign}_mu_J_m"],
                owner.temperature_K, domain, dt, parameters)
            fields[name] = candidate
            moved += sign_moved
            maximum_rate = max(maximum_rate, sign_rate)
            owner_activity += np.sum(np.abs(candidate-population), axis=2)
        proposed_owners.append(replace(owner, **fields))
        activity += np.asarray(support)*owner_activity
    proposed = replace(state, owners=tuple(proposed_owners))
    proposed.validate()

    resolution = 4096*np.finfo(float).eps*max(
        abs(before.helmholtz_J), 1e-300)
    scale = 1.0
    backtracks = 0
    cold = state
    cold_energy = before
    if moved > 0.0:
        for backtracks in range(parameters.maximum_backtracks+1):
            owners = []
            for old, trial in zip(state.owners, proposed.owners):
                owners.append(replace(
                    old,
                    wall_plus_m2=(old.wall_plus_m2+scale*(
                        trial.wall_plus_m2-old.wall_plus_m2)),
                    wall_minus_m2=(old.wall_minus_m2+scale*(
                        trial.wall_minus_m2-old.wall_minus_m2))))
            cold = replace(state, owners=tuple(owners))
            cold.validate()
            cold_energy = evaluate_complete_multigrain_energy(cold, **common)
            if cold_energy.helmholtz_J-before.helmholtz_J <= resolution:
                break
            scale *= .5
    delta = cold_energy.helmholtz_J-before.helmholtz_J
    downhill = delta <= resolution
    heat = max(-delta, 0.0) if moved > 0.0 and downhill else 0.0
    cell_volume = spacing**2*thickness
    heat_cells = np.zeros_like(activity)
    if heat > 0.0 and float(np.sum(activity)) > 0.0:
        heat_cells = heat*activity/float(np.sum(activity, dtype=np.longdouble))
        rise = heat_cells/(wall_parameters.volumetric_heat_capacity_J_m3_K
                           *cell_volume)
        committed = replace(cold, owners=tuple(
            replace(owner, temperature_K=np.asarray(owner.temperature_K)+rise)
            for owner in cold.owners))
        committed.validate()
        hot = evaluate_complete_multigrain_energy(committed, **common)
        residual = hot.internal_J-before.internal_J
        relative = abs(residual)/max(abs(delta), heat, resolution, 1e-300)
    else:
        committed = state
        residual = relative = 0.0
    after_inventory = []
    for owner in cold.owners:
        after_inventory.extend((
            np.sum(owner.wall_plus_m2, axis=(0, 1), dtype=np.longdouble),
            np.sum(owner.wall_minus_m2, axis=(0, 1), dtype=np.longdouble)))
    inventory_residual = float(np.max(np.abs(
        np.asarray(after_inventory)-np.asarray(before_inventory))))
    accepted = bool(moved > 0.0 and downhill
                    and relative <= maximum_relative_first_law_residual)
    if moved <= 0.0:
        classification = "INACTIVE_NO_ORDERED_WALL_GLIDE"
    elif not downhill:
        classification = "REJECTED_CONSERVATIVE_WALL_GLIDE_UPHILL"
    elif relative > maximum_relative_first_law_residual:
        classification = "REJECTED_CONSERVATIVE_WALL_GLIDE_FIRST_LAW"
    else:
        classification = "ACCEPTED_CONSERVATIVE_ARRHENIUS_WALL_GLIDE"
    return ConservativeWallGlideDecision(
        accepted, classification, cold, committed if accepted else state,
        scale*moved if accepted else 0.0, inventory_residual,
        maximum_rate, backtracks, delta,
        heat if accepted else 0.0, residual if accepted else 0.0,
        relative if accepted else 0.0,
        heat_cells if accepted else np.zeros_like(heat_cells))
