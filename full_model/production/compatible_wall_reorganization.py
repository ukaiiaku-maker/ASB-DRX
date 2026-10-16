"""Conservative signed-wall glide toward Frank--Bilby compatibility."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .complete_multigrain_energy import evaluate_complete_multigrain_energy
from .extensive_wall import orientation_gradient_frank_bilby_target_m1
from .structural_boundary import (
    StructuralBoundaryParameters, exp_floor_event_rate_Hz,
)
from .tensorial_nye import rotated_system_fields


@dataclass(frozen=True)
class CompatibleWallReorganizationParameters:
    activation: StructuralBoundaryParameters = StructuralBoundaryParameters(
        activation_enthalpy_eV=.75, barrier_floor_eV=.05,
        attempt_frequency_Hz=1.0e7, activation_segment_m=2.5e-9)
    minimum_wall_order: float = .20
    owner_support_threshold: float = .75
    maximum_fraction_per_interval: float = 1.0
    maximum_backtracks: int = 16


@dataclass(frozen=True)
class CompatibleWallReorganizationDecision:
    accepted: bool
    classification: str
    candidate_state: object
    committed_state: object
    owner_fraction_by_id: dict[int, float]
    owner_affinity_J_m_by_id: dict[int, float]
    signed_content_residual_m2: float
    line_balance_residual_m2: float
    backtracks: int
    delta_helmholtz_J: float
    generated_heat_J: float
    first_law_residual_J: float
    relative_first_law_residual: float
    heat_source_J_by_cell: np.ndarray


def _bounded_sum_projection(raw, capacity, required_sum):
    """Euclidean box projection with an exact prescribed scalar sum."""
    value = np.asarray(raw, dtype=float)
    bound = np.asarray(capacity, dtype=float)
    target = float(required_sum)
    if abs(target) > float(np.sum(bound, dtype=np.longdouble))*(1+1e-13):
        raise ValueError("signed wall inventory exceeds local line capacity")
    low = float(np.min(-bound-value))-1.0
    high = float(np.max(bound-value))+1.0
    for _ in range(100):
        middle = .5*(low+high)
        projected = np.clip(value+middle, -bound, bound)
        if float(np.sum(projected, dtype=np.longdouble)) < target:
            low = middle
        else:
            high = middle
    projected = np.clip(value+.5*(low+high), -bound, bound)
    # Distribute the remaining floating residual over unsaturated capacity.
    for _ in range(4):
        residual = target-float(np.sum(projected, dtype=np.longdouble))
        if abs(residual) <= 32*np.finfo(float).eps*max(abs(target), 1.0):
            break
        room = np.maximum(bound-projected if residual > 0.0
                          else bound+projected, 0.0)
        total_room = float(np.sum(room, dtype=np.longdouble))
        if total_room <= 0.0:
            break
        projected += np.sign(residual)*room*min(abs(residual)/total_room, 1.0)
    return projected


def _compatible_signed_target(owner, support, systems, spacing_m, parameters):
    current = (np.asarray(owner.wall_plus_m2)
               -np.asarray(owner.wall_minus_m2))
    total = (np.asarray(owner.wall_plus_m2)
             +np.asarray(owner.wall_minus_m2))
    active = ((np.asarray(support) > parameters.owner_support_threshold)
              & (np.asarray(owner.wall_order) >= parameters.minimum_wall_order))
    if np.count_nonzero(active) < 2:
        return current.copy(), active
    target_alpha = orientation_gradient_frank_bilby_target_m1(
        owner.orientation_rad, spacing_m)[..., :, 2]
    burgers = rotated_system_fields(systems, owner.orientation_rad)[0]
    inverse = np.linalg.pinv(np.swapaxes(burgers, -2, -1))
    raw = np.einsum("...ai,...i->...a", inverse, target_alpha)
    raw = np.divide(
        raw, np.maximum(np.asarray(owner.wall_order)[..., None],
                        parameters.minimum_wall_order),
        out=np.zeros_like(raw), where=active[..., None])
    target = current.copy()
    for family in range(current.shape[2]):
        fixed_sum = float(np.sum(
            current[..., family][~active], dtype=np.longdouble))
        required = float(np.sum(
            current[..., family], dtype=np.longdouble))-fixed_sum
        target[..., family][active] = _bounded_sum_projection(
            raw[..., family][active], total[..., family][active], required)
    return target, active


def evaluate_compatible_wall_reorganization(
        state, *, interval_s, mean_strain, spacing_m,
        represented_thickness_m, systems, wall_parameters,
        energy_kwargs=None,
        parameters=CompatibleWallReorganizationParameters(),
        maximum_relative_first_law_residual=.05):
    """Move existing wall signs without creating line or rotating lattice."""
    dt = float(interval_s)
    spacing = float(spacing_m)
    thickness = float(represented_thickness_m)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("positive finite wall-reorganization interval required")
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    options.pop("mean_strain", None)
    common = dict(
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall_parameters, mean_strain=np.asarray(mean_strain),
        **options)
    before = evaluate_complete_multigrain_energy(state, **common)
    cell_volume = spacing**2*thickness
    targets = []
    active_masks = []
    fractions = {}
    affinities = {}
    for grain_id, support, owner in zip(
            state.grain_ids, state.supports, state.owners):
        target, active = _compatible_signed_target(
            owner, support, systems, spacing, parameters)
        targets.append(target)
        active_masks.append(active)
        total = owner.wall_plus_m2+owner.wall_minus_m2
        candidate_owner = replace(
            owner, wall_plus_m2=.5*(total+target),
            wall_minus_m2=.5*(total-target))
        owners = tuple(candidate_owner if gid == grain_id else item
                       for gid, item in zip(state.grain_ids, state.owners))
        trial = replace(state, owners=owners)
        trial.validate()
        trial_energy = evaluate_complete_multigrain_energy(trial, **common)
        participating_line_m = float(np.sum(
            np.asarray(support)[..., None]*active[..., None]*total,
            dtype=np.longdouble)*cell_volume)
        affinity = ((before.helmholtz_J-trial_energy.helmholtz_J)
                    /participating_line_m if participating_line_m > 0.0 else 0.0)
        temperature = (float(np.mean(owner.temperature_K[active]))
                       if np.any(active) else float(np.mean(owner.temperature_K)))
        rate = float(exp_floor_event_rate_Hz(
            max(affinity, 0.0), temperature, parameters.activation))
        fraction = min(-math.expm1(-rate*dt),
                       parameters.maximum_fraction_per_interval)
        if affinity <= 0.0 or np.allclose(
                target, owner.wall_plus_m2-owner.wall_minus_m2,
                rtol=0.0, atol=1e-6):
            fraction = 0.0
        fractions[int(grain_id)] = fraction
        affinities[int(grain_id)] = affinity

    def cold_at(scale):
        owners = []
        for grain_id, owner, target in zip(
                state.grain_ids, state.owners, targets):
            fraction = scale*fractions[int(grain_id)]
            old = owner.wall_plus_m2-owner.wall_minus_m2
            signed = old+fraction*(target-old)
            total = owner.wall_plus_m2+owner.wall_minus_m2
            owners.append(replace(
                owner, wall_plus_m2=.5*(total+signed),
                wall_minus_m2=.5*(total-signed)))
        candidate = replace(state, owners=tuple(owners))
        candidate.validate()
        return candidate

    active = any(value > 0.0 for value in fractions.values())
    resolution = 4096*np.finfo(float).eps*max(abs(before.helmholtz_J), 1e-300)
    scale = 1.0
    backtracks = 0
    cold = state
    cold_energy = before
    if active:
        for backtracks in range(parameters.maximum_backtracks+1):
            cold = cold_at(scale)
            cold_energy = evaluate_complete_multigrain_energy(cold, **common)
            if cold_energy.helmholtz_J-before.helmholtz_J <= resolution:
                break
            scale *= .5
    delta = cold_energy.helmholtz_J-before.helmholtz_J
    downhill = delta <= resolution
    heat = max(-delta, 0.0) if active and downhill else 0.0
    activity = np.zeros(state.supports.shape[1:])
    for support, old, new in zip(state.supports, state.owners, cold.owners):
        activity += np.asarray(support)*np.sum(np.abs(
            (new.wall_plus_m2-new.wall_minus_m2)
            -(old.wall_plus_m2-old.wall_minus_m2)), axis=2)
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
    before_signed = np.asarray([
        np.sum(owner.wall_plus_m2-owner.wall_minus_m2, axis=(0, 1))
        for owner in state.owners])
    after_signed = np.asarray([
        np.sum(owner.wall_plus_m2-owner.wall_minus_m2, axis=(0, 1))
        for owner in cold.owners])
    before_line = sum(float(np.sum(
        owner.wall_plus_m2+owner.wall_minus_m2, dtype=np.longdouble))
        for owner in state.owners)
    after_line = sum(float(np.sum(
        owner.wall_plus_m2+owner.wall_minus_m2, dtype=np.longdouble))
        for owner in cold.owners)
    signed_residual = float(np.max(np.abs(after_signed-before_signed)))
    line_residual = float(after_line-before_line)
    accepted = bool(active and downhill
                    and relative <= maximum_relative_first_law_residual)
    if not active:
        classification = "INACTIVE_NO_DOWNHILL_COMPATIBLE_WALL_GLIDE"
    elif not downhill:
        classification = "REJECTED_COMPATIBLE_WALL_GLIDE_UPHILL"
    elif relative > maximum_relative_first_law_residual:
        classification = "REJECTED_COMPATIBLE_WALL_GLIDE_FIRST_LAW"
    else:
        classification = "ACCEPTED_CONSERVATIVE_COMPATIBLE_WALL_GLIDE"
    fractions = {key: (scale*value if accepted else 0.0)
                 for key, value in fractions.items()}
    return CompatibleWallReorganizationDecision(
        accepted, classification, cold, committed if accepted else state,
        fractions, affinities, signed_residual, line_residual, backtracks,
        delta, heat if accepted else 0.0, residual if accepted else 0.0,
        relative if accepted else 0.0,
        heat_cells if accepted else np.zeros_like(heat_cells))
