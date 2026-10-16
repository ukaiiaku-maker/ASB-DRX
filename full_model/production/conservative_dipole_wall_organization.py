"""Conservative expansion of neutral mobile pairs into signed wall dipoles.

The event is a reduced representation of loop expansion: a co-located
positive/negative mobile pair is consumed at one cell, one sign is captured
there, and the opposite sign is captured on the neighbouring ordered wall.
Each Burgers-family plus and minus inventory and total line are conserved
exactly, while their first spatial moment may change.  This supplies the
missing route from a neutral Arrhenius pair source to GND-bearing wall
content without allocating a grain, prescribing an orientation, or resetting
plastic history.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy import ndimage

from .common_tensorial_wall import wall_free_energy_derivatives
from .complete_multigrain_energy import evaluate_complete_multigrain_energy
from .structural_boundary import (
    StructuralBoundaryParameters, exp_floor_event_rate_Hz,
)


@dataclass(frozen=True)
class ConservativeDipoleWallParameters:
    activation: StructuralBoundaryParameters = StructuralBoundaryParameters(
        activation_enthalpy_eV=.80, barrier_floor_eV=.05,
        attempt_frequency_Hz=1.0e7, activation_segment_m=2.5e-9)
    minimum_wall_order: float = .20
    owner_support_threshold: float = .75
    mobile_donor_support_threshold: float = 1.0e-8
    # Physical reach over which a boundary-emitted loop can be captured by
    # an existing ordered wall. Expressed in metres so grid refinement cannot
    # change the mechanism silently.
    wall_capture_distance_m: float = 5.0e-7
    maximum_edge_fraction: float = .02
    maximum_internal_substeps: int = 4096
    maximum_backtracks: int = 20

    def __post_init__(self):
        if not 0.0 <= self.mobile_donor_support_threshold < 1.0:
            raise ValueError("mobile donor support threshold must lie in [0,1)")
        if (not math.isfinite(self.wall_capture_distance_m)
                or self.wall_capture_distance_m < 0.0):
            raise ValueError("wall capture distance must be finite and nonnegative")


def _dipole_capture_domain(support, wall_order, spacing_m, parameters):
    """Return the owner-local physical capture zone around ordered wall."""
    owner_support = np.asarray(support)
    ordered_core = (
        (owner_support > parameters.owner_support_threshold)
        &(np.asarray(wall_order) >= parameters.minimum_wall_order))
    capture_cells = int(math.ceil(
        parameters.wall_capture_distance_m/float(spacing_m)))
    capture_zone = (
        ndimage.binary_dilation(ordered_core, iterations=capture_cells)
        if capture_cells > 0 else ordered_core)
    return (capture_zone
            &(owner_support > parameters.mobile_donor_support_threshold))


@dataclass(frozen=True)
class ConservativeDipoleWallDecision:
    accepted: bool
    classification: str
    candidate_state: object
    committed_state: object
    separated_pair_cell_sum_m2: float
    plus_inventory_residual_m2: float
    minus_inventory_residual_m2: float
    line_balance_residual_m2: float
    maximum_rate_s: float
    internal_substeps: int
    backtracks: int
    delta_helmholtz_J: float
    generated_heat_J: float
    first_law_residual_J: float
    relative_first_law_residual: float
    heat_source_J_by_cell: np.ndarray


def _checkerboard_pair_separation_pass(
        owner, chemical, temperature_K, domain, edge_interval_s, axis, parity,
        parameters, offset_cells=1):
    """Apply disjoint nearest-neighbour loop-expansion events for one pass."""
    mp = np.asarray(owner.mobile_plus_m2, dtype=float).copy()
    mm = np.asarray(owner.mobile_minus_m2, dtype=float).copy()
    wp = np.asarray(owner.wall_plus_m2, dtype=float).copy()
    wm = np.asarray(owner.wall_minus_m2, dtype=float).copy()
    shape = mp.shape[:2]
    if any(size % 2 for size in shape):
        raise ValueError("checkerboard dipole separation requires even extents")
    coordinate = np.arange(shape[axis]).reshape(
        (-1, 1) if axis == 0 else (1, -1))
    offset = int(offset_cells)
    if offset < 1 or offset >= shape[axis]:
        raise ValueError("dipole separation offset is outside the grid")
    eligible = (np.asarray(domain, dtype=bool)
                &np.roll(domain, -offset, axis=axis)
                &(coordinate % 2 == parity))
    mobile_mu = (np.asarray(chemical["mobile_plus_mu_J_m"])
                 +np.asarray(chemical["mobile_minus_mu_J_m"]))
    # A deposits + at the donor and - at its forward neighbour.  B reverses
    # those signs.  Both consume the same local neutral mobile pair.
    delta_a = (np.asarray(chemical["wall_plus_mu_J_m"])
               +np.roll(chemical["wall_minus_mu_J_m"], -offset, axis=axis)
               -mobile_mu)
    delta_b = (np.asarray(chemical["wall_minus_mu_J_m"])
               +np.roll(chemical["wall_plus_mu_J_m"], -offset, axis=axis)
               -mobile_mu)
    choose_a = delta_a <= delta_b
    delta = np.minimum(delta_a, delta_b)
    affinity = np.maximum(-delta, 0.0)
    edge_temperature = .5*(
        np.asarray(temperature_K)[..., None]
        +np.roll(temperature_K, -1, axis=axis)[..., None])
    rate = exp_floor_event_rate_Hz(
        affinity, edge_temperature, parameters.activation)
    maximum_rate = (float(np.max(rate[eligible]))
                    if np.any(eligible) else 0.0)
    fraction = np.minimum(
        -np.expm1(-rate*float(edge_interval_s)),
        parameters.maximum_edge_fraction)
    neutral = np.minimum(mp, mm)
    extent = np.where(
        eligible[..., None]&(delta < 0.0), fraction*neutral, 0.0)
    a = np.where(choose_a, extent, 0.0)
    b = extent-a
    mp -= extent
    mm -= extent
    wp += a+np.roll(b, offset, axis=axis)
    wm += b+np.roll(a, offset, axis=axis)
    tolerance = 1e-12*max(float(np.max(neutral)), 1.0)
    if min(float(np.min(mp)), float(np.min(mm)),
           float(np.min(wp)), float(np.min(wm))) < -tolerance:
        raise RuntimeError("dipole-wall separation violated nonnegativity")
    updated = replace(
        owner, mobile_plus_m2=np.maximum(mp, 0.0),
        mobile_minus_m2=np.maximum(mm, 0.0),
        wall_plus_m2=np.maximum(wp, 0.0),
        wall_minus_m2=np.maximum(wm, 0.0))
    return updated, float(np.sum(extent, dtype=np.longdouble)), maximum_rate


def _inventories(state):
    plus = minus = line = np.longdouble(0.0)
    for owner in state.owners:
        p = (np.asarray(owner.mobile_plus_m2)
             +np.asarray(owner.forest_plus_m2)
             +np.asarray(owner.wall_plus_m2))
        m = (np.asarray(owner.mobile_minus_m2)
             +np.asarray(owner.forest_minus_m2)
             +np.asarray(owner.wall_minus_m2))
        plus += np.sum(p, dtype=np.longdouble)
        minus += np.sum(m, dtype=np.longdouble)
        line += np.sum(p+m, dtype=np.longdouble)
    return plus, minus, line


def evaluate_conservative_dipole_wall_organization(
        state, *, interval_s, mean_strain, spacing_m,
        represented_thickness_m, systems, wall_parameters,
        energy_kwargs=None, parameters=ConservativeDipoleWallParameters(),
        maximum_relative_first_law_residual=.05):
    """Advance neutral-pair separation and gate the atomic energy ledger."""
    dt = float(interval_s)
    spacing = float(spacing_m)
    thickness = float(represented_thickness_m)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("positive finite dipole-wall interval required")
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    options.pop("mean_strain", None)
    common = dict(
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall_parameters, mean_strain=np.asarray(mean_strain),
        **options)
    before = evaluate_complete_multigrain_energy(state, **common)
    initial_inventory = _inventories(state)
    maximum_temperature = max(
        float(np.max(owner.temperature_K)) for owner in state.owners)
    eV_J = 1.602176634e-19
    floor_affinity = (parameters.activation.activation_enthalpy_eV*eV_J
                      /parameters.activation.activation_segment_m)
    rate_ceiling = float(exp_floor_event_rate_Hz(
        floor_affinity, maximum_temperature, parameters.activation))
    exposure = -math.log1p(-parameters.maximum_edge_fraction)
    internal_substeps = max(1, int(math.ceil(
        rate_ceiling*dt/(4.0*exposure))))
    if internal_substeps > int(parameters.maximum_internal_substeps):
        raise RuntimeError("dipole-wall event exceeded physical-clock substeps")
    substep_s = dt/internal_substeps
    working = state
    separated = maximum_rate = 0.0
    activity = np.zeros(state.supports.shape[1:])
    for _ in range(internal_substeps):
        for axis in (0, 1):
            for parity in (0, 1):
                owners = []
                for support, owner in zip(working.supports, working.owners):
                    domain = _dipole_capture_domain(
                        support, owner.wall_order, spacing, parameters)
                    chemical = wall_free_energy_derivatives(
                        owner, wall_parameters, options.get("topologies", ()),
                        systems)
                    candidate, moved, rate = _checkerboard_pair_separation_pass(
                        owner, chemical, owner.temperature_K, domain,
                        substep_s/4.0, axis, parity, parameters)
                    activity += np.asarray(support)*np.sum(
                        np.abs(candidate.wall_plus_m2-owner.wall_plus_m2)
                        +np.abs(candidate.wall_minus_m2-owner.wall_minus_m2),
                        axis=2)
                    separated += moved
                    maximum_rate = max(maximum_rate, rate)
                    owners.append(candidate)
                working = replace(working, owners=tuple(owners))
                working.validate()
    proposed = working
    resolution = 4096*np.finfo(float).eps*max(
        abs(before.helmholtz_J), 1e-300)
    scale = 1.0
    backtracks = 0
    cold = state
    cold_energy = before
    if separated > 0.0:
        for backtracks in range(parameters.maximum_backtracks+1):
            owners = []
            for old, trial in zip(state.owners, proposed.owners):
                owners.append(replace(
                    old,
                    mobile_plus_m2=old.mobile_plus_m2+scale*(
                        trial.mobile_plus_m2-old.mobile_plus_m2),
                    mobile_minus_m2=old.mobile_minus_m2+scale*(
                        trial.mobile_minus_m2-old.mobile_minus_m2),
                    wall_plus_m2=old.wall_plus_m2+scale*(
                        trial.wall_plus_m2-old.wall_plus_m2),
                    wall_minus_m2=old.wall_minus_m2+scale*(
                        trial.wall_minus_m2-old.wall_minus_m2)))
            cold = replace(state, owners=tuple(owners))
            cold.validate()
            cold_energy = evaluate_complete_multigrain_energy(cold, **common)
            if cold_energy.helmholtz_J-before.helmholtz_J <= resolution:
                break
            scale *= .5
    delta = cold_energy.helmholtz_J-before.helmholtz_J
    downhill = delta <= resolution
    heat = max(-delta, 0.0) if separated > 0.0 and downhill else 0.0
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
        hot_energy = evaluate_complete_multigrain_energy(committed, **common)
        residual = hot_energy.internal_J-before.internal_J
        relative = abs(residual)/max(abs(delta), heat, resolution, 1e-300)
    else:
        committed = state
        residual = relative = 0.0
    final_inventory = _inventories(cold)
    plus_residual = float(final_inventory[0]-initial_inventory[0])
    minus_residual = float(final_inventory[1]-initial_inventory[1])
    line_residual = float(final_inventory[2]-initial_inventory[2])
    inventory_scale = max(float(initial_inventory[2]), 1.0)
    balanced = max(abs(plus_residual), abs(minus_residual),
                   abs(line_residual)) <= 1e-12*inventory_scale
    accepted = bool(separated > 0.0 and downhill and balanced
                    and relative <= maximum_relative_first_law_residual)
    if accepted:
        classification = "ACCEPTED_CONSERVATIVE_DIPOLE_WALL_ORGANIZATION"
    elif separated <= 0.0:
        classification = "INACTIVE_NO_DOWNHILL_DIPOLE_WALL_EVENT"
    elif not balanced:
        classification = "REJECTED_DIPOLE_WALL_BALANCE"
    elif not downhill:
        classification = "INACTIVE_NO_FINITE_DOWNHILL_DIPOLE_WALL_STEP"
    else:
        classification = "REJECTED_DIPOLE_WALL_FIRST_LAW"
    return ConservativeDipoleWallDecision(
        accepted, classification, proposed, committed if accepted else state,
        separated*scale if accepted else 0.0, plus_residual, minus_residual,
        line_residual, maximum_rate, internal_substeps, backtracks, delta,
        heat if accepted else 0.0, residual if accepted else 0.0,
        relative if accepted else 0.0,
        heat_cells if accepted else np.zeros_like(heat_cells))
