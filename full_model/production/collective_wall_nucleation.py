"""Finite-amplitude source-centered closed-wall nucleation transaction."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .arrhenius_dislocation_sources import random_discrete_bulk_site_field
from .complete_multigrain_energy import evaluate_complete_multigrain_energy
from .structural_boundary import (
    StructuralBoundaryParameters, exp_floor_event_rate_Hz,
)


@dataclass(frozen=True)
class CollectiveWallNucleationParameters:
    activation: StructuralBoundaryParameters = StructuralBoundaryParameters(
        activation_enthalpy_eV=.80, barrier_floor_eV=.05,
        attempt_frequency_Hz=1.0e7, activation_segment_m=2.5e-9)
    source_site_density_m2: float = 1.0e12
    source_seed: int = 67
    candidate_radii_m: tuple[float, ...] = (
        4.5e-7, 7.5e-7, 1.05e-6, 1.35e-6, 1.65e-6)
    ring_half_thickness_m: float = 2.25e-7
    order_increment: float = .01
    owner_support_threshold: float = .75


@dataclass(frozen=True)
class CollectiveWallNucleationDecision:
    accepted: bool
    classification: str
    candidate_state: object
    committed_state: object
    selected_owner_id: int | None
    selected_center: tuple[int, int] | None
    selected_radius_m: float
    candidate_count: int
    downhill_candidate_count: int
    event_probability: float
    random_uniform: float
    transferred_line_cell_sum_m2: float
    signed_content_residual_m2: float
    line_balance_residual_m2: float
    delta_helmholtz_J: float
    generated_heat_J: float
    first_law_residual_J: float
    relative_first_law_residual: float
    heat_source_J_by_cell: np.ndarray


def _periodic_radius(shape, center, spacing_m):
    x, y = np.indices(shape)
    dx = np.minimum((x-center[0]) % shape[0], (center[0]-x) % shape[0])
    dy = np.minimum((y-center[1]) % shape[1], (center[1]-y) % shape[1])
    return spacing_m*np.sqrt(dx*dx+dy*dy)


def evaluate_collective_wall_nucleation(
        state, *, interval_s, mean_strain, spacing_m,
        represented_thickness_m, systems, wall_parameters,
        energy_kwargs=None, random_uniform=0.0,
        parameters=CollectiveWallNucleationParameters(),
        maximum_relative_first_law_residual=.05):
    """Price all declared closed rings and publish at most one atomic event."""
    dt = float(interval_s)
    spacing = float(spacing_m)
    thickness = float(represented_thickness_m)
    draw = float(random_uniform)
    if not math.isfinite(dt) or dt <= 0.0 or not 0.0 <= draw < 1.0:
        raise ValueError("wall-nucleation interval/draw is invalid")
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    options.pop("mean_strain", None)
    common = dict(
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall_parameters, mean_strain=np.asarray(mean_strain),
        **options)
    before = evaluate_complete_multigrain_energy(state, **common)
    sites = random_discrete_bulk_site_field(
        state.supports.shape[1:], spacing,
        parameters.source_site_density_m2, parameters.source_seed)
    centers = np.argwhere(sites > 0.0)
    best = None
    candidate_count = downhill_count = 0
    for owner_index, (grain_id, support, owner) in enumerate(zip(
            state.grain_ids, state.supports, state.owners)):
        domain = np.asarray(support) > parameters.owner_support_threshold
        excess_plus = np.maximum(
            np.asarray(owner.mobile_plus_m2)
            -np.asarray(owner.mobile_minus_m2), 0.0)
        excess_minus = np.maximum(
            np.asarray(owner.mobile_minus_m2)
            -np.asarray(owner.mobile_plus_m2), 0.0)
        for center_array in centers:
            center = tuple(map(int, center_array))
            radial = _periodic_radius(domain.shape, center, spacing)
            for radius in parameters.candidate_radii_m:
                ring = (domain
                        &(np.abs(radial-radius)
                          <=parameters.ring_half_thickness_m))
                if np.count_nonzero(ring) < 8:
                    continue
                transfer_plus = np.where(ring[..., None], excess_plus, 0.0)
                transfer_minus = np.where(ring[..., None], excess_minus, 0.0)
                transferred = float(np.sum(
                    transfer_plus+transfer_minus, dtype=np.longdouble))
                if transferred <= 0.0:
                    continue
                order = np.asarray(owner.wall_order).copy()
                order[ring] += parameters.order_increment*(1.0-order[ring])
                candidate_owner = replace(
                    owner,
                    mobile_plus_m2=np.asarray(owner.mobile_plus_m2)-transfer_plus,
                    mobile_minus_m2=np.asarray(owner.mobile_minus_m2)-transfer_minus,
                    wall_plus_m2=np.asarray(owner.wall_plus_m2)+transfer_plus,
                    wall_minus_m2=np.asarray(owner.wall_minus_m2)+transfer_minus,
                    wall_order=order)
                owners = list(state.owners)
                owners[owner_index] = candidate_owner
                candidate = replace(state, owners=tuple(owners))
                candidate.validate()
                energy = evaluate_complete_multigrain_energy(candidate, **common)
                delta = energy.helmholtz_J-before.helmholtz_J
                candidate_count += 1
                if delta <= 0.0:
                    downhill_count += 1
                    if best is None or delta < best[0]:
                        best = (delta, int(grain_id), center, float(radius),
                                ring, transferred, candidate)
    if best is None:
        zero = np.zeros(state.supports.shape[1:])
        return CollectiveWallNucleationDecision(
            False, "INACTIVE_NO_DOWNHILL_CLOSED_WALL_NUCLEUS",
            state, state, None, None, 0.0, candidate_count, 0, 0.0, draw,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, zero)
    delta, grain_id, center, radius, ring, transferred, candidate = best
    owner = candidate.owners[candidate.grain_ids.index(grain_id)]
    temperature = float(np.mean(np.asarray(owner.temperature_K)[ring]))
    rate = float(exp_floor_event_rate_Hz(
        0.0, temperature, parameters.activation))
    probability = -math.expm1(-rate*dt)
    if draw >= probability:
        zero = np.zeros(state.supports.shape[1:])
        return CollectiveWallNucleationDecision(
            False, "INACTIVE_ACTIVATED_CLOSED_WALL_NOT_SAMPLED",
            candidate, state, grain_id, center, radius, candidate_count,
            downhill_count, probability, draw, transferred, 0.0, 0.0,
            delta, 0.0, 0.0, 0.0, zero)
    heat = max(-delta, 0.0)
    heat_cells = np.zeros(state.supports.shape[1:])
    heat_cells[ring] = heat/np.count_nonzero(ring)
    cell_volume = spacing*spacing*thickness
    rise = heat_cells/(wall_parameters.volumetric_heat_capacity_J_m3_K
                       *cell_volume)
    hot = replace(candidate, owners=tuple(
        replace(item, temperature_K=np.asarray(item.temperature_K)+rise)
        for item in candidate.owners))
    hot.validate()
    after = evaluate_complete_multigrain_energy(hot, **common)
    residual = after.internal_J-before.internal_J
    scale = max(abs(delta), heat, 1e-300)
    relative = abs(residual)/scale
    original = state.owners[state.grain_ids.index(grain_id)]
    changed = candidate.owners[candidate.grain_ids.index(grain_id)]
    line_before = np.sum(
        original.mobile_plus_m2+original.mobile_minus_m2
        +original.wall_plus_m2+original.wall_minus_m2,
        dtype=np.longdouble)
    line_after = np.sum(
        changed.mobile_plus_m2+changed.mobile_minus_m2
        +changed.wall_plus_m2+changed.wall_minus_m2,
        dtype=np.longdouble)
    signed_before = np.sum(
        original.mobile_plus_m2-original.mobile_minus_m2
        +original.wall_plus_m2-original.wall_minus_m2,
        dtype=np.longdouble)
    signed_after = np.sum(
        changed.mobile_plus_m2-changed.mobile_minus_m2
        +changed.wall_plus_m2-changed.wall_minus_m2,
        dtype=np.longdouble)
    line_residual = float(line_after-line_before)
    signed_residual = float(signed_after-signed_before)
    accepted = bool(relative <= maximum_relative_first_law_residual)
    return CollectiveWallNucleationDecision(
        accepted,
        ("ACCEPTED_ACTIVATED_CLOSED_WALL_NUCLEUS" if accepted
         else "REJECTED_CLOSED_WALL_FIRST_LAW"),
        candidate, hot if accepted else state, grain_id, center, radius,
        candidate_count, downhill_count, probability, draw, transferred,
        signed_residual, line_residual, delta, heat if accepted else 0.0,
        residual if accepted else 0.0, relative if accepted else 0.0,
        heat_cells if accepted else np.zeros_like(heat_cells))
