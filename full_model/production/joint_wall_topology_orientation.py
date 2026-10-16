"""Finite wall-topology event coupled to its own Nye-implied orientation.

This optional V67 operator moves existing signed mobile excess into a closed
ordered wall and reconstructs the conjugate common-axis orientation from the
candidate wall Nye field.  It neither inserts a grain orientation nor creates
line.  Candidate owner, centre, and radius compete through the complete
Helmholtz functional; an uphill event is never published.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy import ndimage

from .collective_wall_nucleation import (
    CollectiveWallNucleationParameters, _collective_wall_candidate,
    _periodic_radius,
)
from .complete_multigrain_energy import evaluate_complete_multigrain_energy
from .orientation_wall_compatibility import (
    ordered_wall_nye_from_common, reconstruct_orientation_from_ordered_nye,
)
from .structural_boundary import (
    StructuralBoundaryParameters, exp_floor_event_rate_Hz,
)


@dataclass(frozen=True)
class JointWallTopologyOrientationParameters:
    activation: StructuralBoundaryParameters = StructuralBoundaryParameters(
        activation_enthalpy_eV=.80, barrier_floor_eV=.05,
        attempt_frequency_Hz=1e7, activation_segment_m=2.5e-9)
    candidate_radii_m: tuple[float, ...] = (
        4.5e-7, 7.5e-7, 1.05e-6, 1.35e-6, 1.65e-6)
    ring_width_m: float = 4.5e-7
    maximum_coordinate_per_interval: float = .05
    order_increment: float = .01
    owner_support_threshold: float = .75
    boundary_clearance_m: float = 2e-7
    maximum_backtracks: int = 16

    def __post_init__(self):
        if not 0.0 < self.maximum_coordinate_per_interval <= 1.0:
            raise ValueError("joint topology coordinate must lie in (0,1]")
        if not math.isfinite(self.ring_width_m) or self.ring_width_m <= 0.0:
            raise ValueError("joint topology ring width must be positive")
        if any(not math.isfinite(value) or value <= 0.0
               for value in self.candidate_radii_m):
            raise ValueError("joint topology radii must be positive finite")
        if self.maximum_backtracks < 0:
            raise ValueError("joint topology backtracks cannot be negative")


@dataclass(frozen=True)
class JointWallTopologyOrientationDecision:
    accepted: bool
    classification: str
    candidate_state: object
    committed_state: object
    selected_owner_id: int | None
    selected_center: tuple[int, int] | None
    selected_radius_m: float
    candidate_count: int
    downhill_candidate_count: int
    affinity_J_m: float
    event_rate_s: float
    realized_coordinate: float
    transferred_line_cell_sum_m2: float
    maximum_orientation_change_rad: float
    plus_inventory_residual_m2: float
    minus_inventory_residual_m2: float
    line_balance_residual_m2: float
    backtracks: int
    delta_helmholtz_J: float
    generated_heat_J: float
    first_law_residual_J: float
    relative_first_law_residual: float
    heat_source_J_by_cell: np.ndarray


def _inventories(state):
    plus = minus = np.longdouble(0.0)
    for support, owner in zip(state.supports, state.owners):
        weight = np.asarray(support)[..., None]
        plus += np.sum(weight*(owner.mobile_plus_m2+owner.wall_plus_m2),
                       dtype=np.longdouble)
        minus += np.sum(weight*(owner.mobile_minus_m2+owner.wall_minus_m2),
                        dtype=np.longdouble)
    return plus, minus


def joint_wall_orientation_candidate(
        state, owner_index, ring, coordinate, systems, spacing_m,
        boundary_clearance_m, order_increment):
    """Build one conservative wall event and its inventory-implied rotation."""
    wall, transferred = _collective_wall_candidate(
        state, owner_index, ring, order_increment, coordinate)
    old = state.owners[owner_index]
    candidate = wall.owners[owner_index]
    support = np.asarray(state.supports[owner_index])
    domain = support > .75
    cells = int(math.ceil(float(boundary_clearance_m)/float(spacing_m)))
    if cells:
        domain = ndimage.binary_erosion(domain, iterations=cells)
    if np.count_nonzero(domain) < 2:
        return wall, transferred, 0.0
    # Price only the orientation response conjugate to the *new* wall
    # transfer.  The retained state need not already be the least-squares
    # orientation reconstruction of its ordered Nye field.  Relaxing that
    # pre-existing mismatch as part of this reaction would give a finite
    # orientation jump at zero transfer and would spuriously make a wall
    # event downhill.  Subtract the baseline reconstruction so the coupled
    # path is continuous at coordinate zero.  Both wall transfer and this
    # incremental reconstruction are already proportional to ``coordinate``;
    # applying the coordinate again would incorrectly suppress the response
    # quadratically.
    old_alpha = ordered_wall_nye_from_common(old, systems)
    old_reconstructed, _ = reconstruct_orientation_from_ordered_nye(
        old_alpha, old.orientation_rad, domain, spacing_m)
    alpha = ordered_wall_nye_from_common(candidate, systems)
    reconstructed, _ = reconstruct_orientation_from_ordered_nye(
        alpha, old.orientation_rad, domain, spacing_m)
    increment = np.arctan2(
        np.sin(reconstructed-old_reconstructed),
        np.cos(reconstructed-old_reconstructed))
    orientation = np.asarray(old.orientation_rad).copy()
    orientation[domain] += increment[domain]
    orientation += float(np.mean(old.orientation_rad)-np.mean(orientation))
    owners = list(wall.owners)
    owners[owner_index] = replace(candidate, orientation_rad=orientation)
    joint = replace(wall, owners=tuple(owners))
    joint.validate()
    return joint, transferred, float(np.max(np.abs(
        orientation-np.asarray(old.orientation_rad))))


def evaluate_joint_wall_topology_orientation(
        state, *, interval_s, mean_strain, spacing_m,
        represented_thickness_m, systems, wall_parameters,
        energy_kwargs=None,
        parameters=JointWallTopologyOrientationParameters(),
        maximum_relative_first_law_residual=.05):
    """Select and publish at most one downhill joint topology transaction."""
    dt = float(interval_s)
    spacing = float(spacing_m)
    thickness = float(represented_thickness_m)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("positive finite joint topology interval required")
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    options.pop("mean_strain", None)
    common = dict(
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall_parameters, mean_strain=np.asarray(mean_strain),
        **options)
    before = evaluate_complete_multigrain_energy(state, **common)
    initial_plus, initial_minus = _inventories(state)
    cell_volume = spacing**2*thickness
    probe = float(parameters.maximum_coordinate_per_interval)
    candidates = []
    candidate_count = 0
    for owner_index, (grain_id, support, owner) in enumerate(zip(
            state.grain_ids, state.supports, state.owners)):
        domain = np.asarray(support) > parameters.owner_support_threshold
        mobile_excess = np.sum(np.abs(
            np.asarray(owner.mobile_plus_m2)
            -np.asarray(owner.mobile_minus_m2)), axis=2)
        score = np.where(domain, mobile_excess, -np.inf)
        if not np.any(np.isfinite(score)) or float(np.max(score)) <= 0.0:
            continue
        center = tuple(map(int, np.unravel_index(np.argmax(score), score.shape)))
        radial = _periodic_radius(domain.shape, center, spacing)
        for radius in parameters.candidate_radii_m:
            ring = (np.exp(-.5*((radial-radius)/parameters.ring_width_m)**2)
                    *domain)
            if float(np.sum(ring)) < 8.0:
                continue
            trial, transferred, rotation = joint_wall_orientation_candidate(
                state, owner_index, ring, probe, systems, spacing,
                parameters.boundary_clearance_m, parameters.order_increment)
            if transferred <= 0.0:
                continue
            candidate_count += 1
            trial_energy = evaluate_complete_multigrain_energy(trial, **common)
            delta = trial_energy.helmholtz_J-before.helmholtz_J
            moved_line_m = 0.0
            changed = trial.owners[owner_index]
            moved_line_m = float(np.sum(
                np.asarray(support)[..., None]
                *(np.asarray(owner.mobile_plus_m2)-changed.mobile_plus_m2
                  +np.asarray(owner.mobile_minus_m2)-changed.mobile_minus_m2),
                dtype=np.longdouble)*cell_volume)
            affinity = -delta/moved_line_m if moved_line_m > 0.0 else 0.0
            if delta < 0.0 and affinity > 0.0:
                temperature = float(np.sum(owner.temperature_K*ring)/np.sum(ring))
                rate = float(exp_floor_event_rate_Hz(
                    affinity, temperature, parameters.activation))
                candidates.append((
                    affinity*rate, affinity, rate, int(grain_id), owner_index,
                    center, float(radius), ring, transferred, rotation))
    zero = np.zeros(state.supports.shape[1:])
    if not candidates:
        return JointWallTopologyOrientationDecision(
            False, "INACTIVE_NO_DOWNHILL_JOINT_WALL_TOPOLOGY", state, state,
            None, None, 0.0, candidate_count, 0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, zero)
    selected = max(candidates, key=lambda item: item[0])
    (_, affinity, rate, grain_id, owner_index, center, radius, ring,
     _, _) = selected
    kinetic_fraction = -math.expm1(-rate*dt)
    coordinate = probe*kinetic_fraction
    scale = 1.0
    backtracks = 0
    cold = state
    cold_energy = before
    transferred = rotation = 0.0
    resolution = 4096*np.finfo(float).eps*max(
        abs(before.helmholtz_J), 1e-300)
    for backtracks in range(parameters.maximum_backtracks+1):
        cold, transferred, rotation = joint_wall_orientation_candidate(
            state, owner_index, ring, scale*coordinate, systems, spacing,
            parameters.boundary_clearance_m, parameters.order_increment)
        cold_energy = evaluate_complete_multigrain_energy(cold, **common)
        if cold_energy.helmholtz_J-before.helmholtz_J <= resolution:
            break
        scale *= .5
    coordinate *= scale
    delta = cold_energy.helmholtz_J-before.helmholtz_J
    downhill = delta <= resolution
    heat = max(-delta, 0.0) if coordinate > 0.0 and downhill else 0.0
    activity = np.asarray(ring)*float(transferred)
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
    final_plus, final_minus = _inventories(cold)
    plus_residual = float(final_plus-initial_plus)
    minus_residual = float(final_minus-initial_minus)
    line_residual = plus_residual+minus_residual
    inventory_scale = max(float(initial_plus+initial_minus), 1.0)
    balanced = max(abs(plus_residual), abs(minus_residual),
                   abs(line_residual)) <= 1e-12*inventory_scale
    accepted = bool(coordinate > 0.0 and downhill and balanced
                    and relative <= maximum_relative_first_law_residual)
    if accepted:
        classification = "ACCEPTED_JOINT_WALL_TOPOLOGY_ORIENTATION"
    elif not downhill:
        classification = "INACTIVE_NO_FINITE_DOWNHILL_JOINT_WALL_STEP"
    elif not balanced:
        classification = "REJECTED_JOINT_WALL_TOPOLOGY_BALANCE"
    else:
        classification = "REJECTED_JOINT_WALL_TOPOLOGY_FIRST_LAW"
    return JointWallTopologyOrientationDecision(
        accepted, classification, cold, committed if accepted else state,
        grain_id, center, radius, candidate_count, len(candidates), affinity,
        rate, coordinate if accepted else 0.0,
        transferred if accepted else 0.0,
        rotation if accepted else 0.0, plus_residual, minus_residual,
        line_residual, backtracks, delta,
        heat if accepted else 0.0, residual if accepted else 0.0,
        relative if accepted else 0.0,
        heat_cells if accepted else np.zeros_like(heat_cells))
