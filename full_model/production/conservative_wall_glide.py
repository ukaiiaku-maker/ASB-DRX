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
    maximum_edge_fraction: float = .02
    maximum_internal_substeps: int = 4096
    maximum_backtracks: int = 20
    ordered_wall_mobility_power: float = 2.0

    def __post_init__(self):
        if not 0.0 < float(self.maximum_edge_fraction) < 1.0:
            raise ValueError("maximum edge fraction must lie in (0,1)")
        if int(self.maximum_internal_substeps) < 1:
            raise ValueError("positive wall-glide substep capacity required")
        if float(self.ordered_wall_mobility_power) < 0.0:
            raise ValueError("ordered-wall mobility power must be nonnegative")


@dataclass(frozen=True)
class ConservativeWallGlideDecision:
    accepted: bool
    classification: str
    candidate_state: object
    committed_state: object
    moved_line_cell_sum_m2: float
    inventory_residual_m2: float
    maximum_rate_s: float
    internal_substeps: int
    backtracks: int
    delta_helmholtz_J: float
    generated_heat_J: float
    first_law_residual_J: float
    relative_first_law_residual: float
    heat_source_J_by_cell: np.ndarray


def _checkerboard_glide_pass(population, chemical_J_m, temperature_K, domain,
                             edge_interval_s, axis, parity, parameters,
                             mobility=None, support=None):
    value = np.asarray(population, dtype=float).copy()
    chemical = np.asarray(chemical_J_m, dtype=float)
    temperature = np.asarray(temperature_K, dtype=float)
    eta = (np.ones(value.shape[:2], dtype=float) if support is None
           else np.asarray(support, dtype=float))
    if eta.shape != value.shape[:2] or np.any(eta < 0.0):
        raise ValueError("wall-glide support must be nonnegative on the grid")
    extensive = eta[..., None]*value
    shape = value.shape[:2]
    if any(size % 2 for size in shape):
        raise ValueError("checkerboard wall glide requires even grid extents")
    coordinate = np.arange(shape[axis]).reshape(
        (-1, 1) if axis == 0 else (1, -1))
    neighbor_extensive = np.roll(extensive, -1, axis=axis)
    neighbor_mu = np.roll(chemical, -1, axis=axis)
    neighbor_temperature = np.roll(temperature, -1, axis=axis)
    eligible = (domain & np.roll(domain, -1, axis=axis)
                &(coordinate % 2 == parity))
    difference = chemical-neighbor_mu
    affinity = np.abs(difference)
    edge_temperature = .5*(temperature+neighbor_temperature)
    rate = exp_floor_event_rate_Hz(
        affinity, edge_temperature[..., None], parameters.activation)
    if mobility is not None:
        local_mobility = np.asarray(mobility, dtype=float)
        edge_mobility = .5*(
            local_mobility+np.roll(local_mobility, -1, axis=axis))
        rate = rate*edge_mobility[..., None]
    maximum_rate = (float(np.max(rate[eligible])) if np.any(eligible) else 0.0)
    fraction = np.minimum(
        -np.expm1(-rate*float(edge_interval_s)),
        parameters.maximum_edge_fraction)
    # df/drho is also d(eta*f)/d(eta*rho).  Transport the physical extensive
    # inventory eta*rho with that unweighted chemical potential.
    donor = np.where(difference >= 0.0, extensive, neighbor_extensive)
    flux = np.where(
        eligible[..., None], np.sign(difference)*fraction*donor, 0.0)
    next_extensive = extensive-flux+np.roll(flux, 1, axis=axis)
    delta = next_extensive-extensive
    value += np.divide(
        delta, eta[..., None], out=np.zeros_like(delta),
        where=eta[..., None] > 256.0*np.finfo(float).eps)
    moved = float(np.sum(np.abs(flux), dtype=np.longdouble))
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
    maximum_temperature = max(
        float(np.max(owner.temperature_K)) for owner in state.owners)
    maximum_mobility = max(float(np.max(
        np.maximum(1.0-np.asarray(owner.wall_order), 0.0)**
        parameters.ordered_wall_mobility_power)) for owner in state.owners)
    eV_J = 1.602176634e-19
    floor_affinity = (parameters.activation.activation_enthalpy_eV*eV_J
                      /parameters.activation.activation_segment_m)
    rate_ceiling = maximum_mobility*float(exp_floor_event_rate_Hz(
        floor_affinity, maximum_temperature, parameters.activation))
    edge_exposure_limit = -math.log1p(-parameters.maximum_edge_fraction)
    internal_substeps = max(1, int(math.ceil(
        rate_ceiling*dt/(4.0*edge_exposure_limit))))
    if internal_substeps > int(parameters.maximum_internal_substeps):
        raise RuntimeError(
            "conservative wall glide exceeded its physical-clock substeps")
    substep_s = dt/internal_substeps
    working = state
    moved = 0.0
    maximum_rate = 0.0
    activity = np.zeros(state.supports.shape[1:])
    before_inventory = []
    for support, owner in zip(state.supports, state.owners):
        before_inventory.extend((
            np.sum(np.asarray(support)[..., None]*owner.wall_plus_m2,
                   axis=(0, 1), dtype=np.longdouble),
            np.sum(np.asarray(support)[..., None]*owner.wall_minus_m2,
                   axis=(0, 1), dtype=np.longdouble)))
    for _ in range(internal_substeps):
        for axis in (0, 1):
            for parity in (0, 1):
                proposed_owners = []
                for support, owner in zip(working.supports, working.owners):
                    domain = (
                        (np.asarray(support) > parameters.owner_support_threshold)
                        &(np.asarray(owner.wall_order)
                          >= parameters.minimum_wall_order))
                    mobility = np.maximum(
                        1.0-np.asarray(owner.wall_order), 0.0)**(
                            parameters.ordered_wall_mobility_power)
                    chemical = wall_free_energy_derivatives(
                        owner, wall_parameters, options.get("topologies", ()),
                        systems)
                    fields = {}
                    owner_activity = np.zeros_like(activity)
                    for sign in ("plus", "minus"):
                        name = f"wall_{sign}_m2"
                        population = np.asarray(getattr(owner, name))
                        candidate, sign_moved, sign_rate = (
                            _checkerboard_glide_pass(
                                population,
                                chemical[f"wall_{sign}_mu_J_m"],
                                owner.temperature_K, domain,
                                substep_s/4.0, axis, parity, parameters,
                                mobility=mobility, support=support))
                        fields[name] = candidate
                        moved += sign_moved
                        maximum_rate = max(maximum_rate, sign_rate)
                        owner_activity += np.sum(
                            np.abs(candidate-population), axis=2)
                    proposed_owners.append(replace(owner, **fields))
                    activity += np.asarray(support)*owner_activity
                working = replace(working, owners=tuple(proposed_owners))
                working.validate()
    proposed = working
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
    for support, owner in zip(state.supports, cold.owners):
        after_inventory.extend((
            np.sum(np.asarray(support)[..., None]*owner.wall_plus_m2,
                   axis=(0, 1), dtype=np.longdouble),
            np.sum(np.asarray(support)[..., None]*owner.wall_minus_m2,
                   axis=(0, 1), dtype=np.longdouble)))
    inventory_residual = float(np.max(np.abs(
        np.asarray(after_inventory)-np.asarray(before_inventory))))
    inventory_scale = max(
        float(np.max(np.abs(np.asarray(before_inventory)))), 1.0)
    inventory_closes = bool(
        inventory_residual <= 4096*np.finfo(float).eps*inventory_scale)
    published_change = any(
        not np.array_equal(np.asarray(getattr(old, name)),
                           np.asarray(getattr(new, name)))
        for old, new in zip(state.owners, cold.owners)
        for name in ("wall_plus_m2", "wall_minus_m2"))
    accepted = bool(moved > 0.0 and published_change and downhill
                    and inventory_closes
                    and relative <= maximum_relative_first_law_residual)
    if moved <= 0.0:
        classification = "INACTIVE_NO_ORDERED_WALL_GLIDE"
    elif not published_change:
        classification = "INACTIVE_BELOW_NUMERICAL_WALL_GLIDE_RESOLUTION"
    elif not downhill:
        classification = "REJECTED_CONSERVATIVE_WALL_GLIDE_UPHILL"
    elif not inventory_closes:
        classification = "REJECTED_CONSERVATIVE_WALL_GLIDE_BALANCE"
    elif relative > maximum_relative_first_law_residual:
        classification = "REJECTED_CONSERVATIVE_WALL_GLIDE_FIRST_LAW"
    else:
        classification = "ACCEPTED_CONSERVATIVE_ARRHENIUS_WALL_GLIDE"
    return ConservativeWallGlideDecision(
        accepted, classification, cold, committed if accepted else state,
        scale*moved if accepted else 0.0, inventory_residual,
        maximum_rate, internal_substeps, backtracks, delta,
        heat if accepted else 0.0, residual if accepted else 0.0,
        relative if accepted else 0.0,
        heat_cells if accepted else np.zeros_like(heat_cells))
