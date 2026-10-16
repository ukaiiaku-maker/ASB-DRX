"""Finite-amplitude source-centered closed-wall nucleation transaction."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .arrhenius_dislocation_sources import random_discrete_bulk_site_field
from .complete_multigrain_energy import evaluate_complete_multigrain_energy
from .structural_boundary import StructuralBoundaryParameters


@dataclass(frozen=True)
class CollectiveWallNucleationParameters:
    activation: StructuralBoundaryParameters = StructuralBoundaryParameters(
        activation_enthalpy_eV=.80, barrier_floor_eV=.05,
        attempt_frequency_Hz=1.0e7, activation_segment_m=2.5e-9)
    source_site_density_m2: float = 1.0e12
    source_seed: int = 67
    candidate_radii_m: tuple[float, ...] = (
        4.5e-7, 7.5e-7, 1.05e-6, 1.35e-6, 1.65e-6)
    ring_width_m: float = 4.5e-7
    order_increment: float = .01
    owner_support_threshold: float = .75
    energy_path_samples: int = 21

    def __post_init__(self):
        if int(self.energy_path_samples) != self.energy_path_samples:
            raise ValueError("energy_path_samples must be an integer")
        if self.energy_path_samples < 2:
            raise ValueError("at least two energy-path samples are required")


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
    resolved_forward_barrier_J: float = 0.0
    intrinsic_forward_barrier_eV: float = 0.0
    forward_barrier_eV: float = 0.0
    reverse_barrier_eV: float = 0.0
    forward_event_rate_Hz: float = 0.0
    reverse_event_rate_Hz: float = 0.0
    detailed_balance_log_residual: float = 0.0
    energy_path_coordinate: tuple[float, ...] = ()
    energy_path_delta_helmholtz_J: tuple[float, ...] = ()


@dataclass(frozen=True)
class CollectiveWallDetailedBalance:
    """Transition-state audit for one resolved collective reaction path.

    The immutable EXP-floor form is retained.  The declared intrinsic barrier
    sits above the highest resolved point on the forward path.  A single
    transition-state energy then determines both directions, so the reverse
    rate is not an independently adjustable wall-dissolution law.
    """

    resolved_forward_barrier_J: float
    delta_helmholtz_J: float
    transition_state_eV: float
    forward_barrier_eV: float
    reverse_barrier_eV: float
    forward_rate_Hz: float
    reverse_rate_Hz: float
    detailed_balance_log_residual: float


def _periodic_radius(shape, center, spacing_m):
    x, y = np.indices(shape)
    dx = np.minimum((x-center[0]) % shape[0], (center[0]-x) % shape[0])
    dy = np.minimum((y-center[1]) % shape[1], (center[1]-y) % shape[1])
    return spacing_m*np.sqrt(dx*dx+dy*dy)


def collective_wall_detailed_balance(
        energy_path_delta_helmholtz_J, temperature_K, activation):
    """Price a sampled path with one saddle and exact detailed balance."""
    path = np.asarray(energy_path_delta_helmholtz_J, dtype=float)
    temperature = float(temperature_K)
    if (path.ndim != 1 or path.size < 2 or np.any(~np.isfinite(path))
            or not math.isfinite(temperature) or temperature <= 0.0):
        raise ValueError("a finite energy path and positive temperature are required")
    scale = max(float(np.max(np.abs(path))), 1e-300)
    if abs(float(path[0])) > 4096.0*np.finfo(float).eps*scale:
        raise ValueError("the energy path must start at zero")
    eV_J = 1.602176634e-19
    kB_J_K = 1.380649e-23
    resolved = max(float(np.max(path)), 0.0)
    endpoint_eV = float(path[-1])/eV_J
    intrinsic = float(activation.activation_enthalpy_eV)
    floor = float(activation.barrier_floor_eV)
    # The same transition state must remain at least one floor above either
    # endpoint.  This preserves EXP-floor kinetics without clipping the two
    # directions independently and thereby breaking microscopic reversibility.
    saddle_eV = max(intrinsic+resolved/eV_J, floor, endpoint_eV+floor)
    forward_barrier_eV = saddle_eV
    reverse_barrier_eV = saddle_eV-endpoint_eV
    prefactor = float(activation.attempt_frequency_Hz)
    thermal_eV = kB_J_K*temperature/eV_J
    forward_rate = prefactor*math.exp(-forward_barrier_eV/thermal_eV)
    reverse_rate = prefactor*math.exp(-reverse_barrier_eV/thermal_eV)
    expected = -endpoint_eV/thermal_eV
    residual = math.log(forward_rate/reverse_rate)-expected
    return CollectiveWallDetailedBalance(
        resolved, float(path[-1]), saddle_eV, forward_barrier_eV,
        reverse_barrier_eV, forward_rate, reverse_rate, residual)


def _collective_wall_candidate(state, owner_index, ring, order_increment,
                               coordinate=1.0):
    """Return one point on the conservative mobile-to-wall event path."""
    lam = float(coordinate)
    if not math.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("reaction coordinate must lie in [0, 1]")
    owner = state.owners[owner_index]
    excess_plus = np.maximum(
        np.asarray(owner.mobile_plus_m2)-np.asarray(owner.mobile_minus_m2), 0.0)
    excess_minus = np.maximum(
        np.asarray(owner.mobile_minus_m2)-np.asarray(owner.mobile_plus_m2), 0.0)
    transfer_plus = lam*np.asarray(ring)[..., None]*excess_plus
    transfer_minus = lam*np.asarray(ring)[..., None]*excess_minus
    order = np.asarray(owner.wall_order)+(
        lam*float(order_increment)*np.asarray(ring)
        *(1.0-np.asarray(owner.wall_order)))
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
    transferred = float(np.sum(
        transfer_plus+transfer_minus, dtype=np.longdouble))
    return candidate, transferred


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
        for center_array in centers:
            center = tuple(map(int, center_array))
            radial = _periodic_radius(domain.shape, center, spacing)
            for radius in parameters.candidate_radii_m:
                ring = (np.exp(-.5*((radial-radius)
                                    /parameters.ring_width_m)**2)
                        *domain)
                if float(np.sum(ring)) < 8.0:
                    continue
                candidate, transferred = _collective_wall_candidate(
                    state, owner_index, ring, parameters.order_increment)
                if transferred <= 0.0:
                    continue
                energy = evaluate_complete_multigrain_energy(candidate, **common)
                delta = energy.helmholtz_J-before.helmholtz_J
                candidate_count += 1
                if delta <= 0.0:
                    downhill_count += 1
                    if best is None or delta < best[0]:
                        best = (delta, int(grain_id), owner_index, center,
                                float(radius), ring, transferred, candidate)
    if best is None:
        zero = np.zeros(state.supports.shape[1:])
        return CollectiveWallNucleationDecision(
            False, "INACTIVE_NO_DOWNHILL_CLOSED_WALL_NUCLEUS",
            state, state, None, None, 0.0, candidate_count, 0, 0.0, draw,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, zero)
    (delta, grain_id, owner_index, center, radius, ring, transferred,
     candidate) = best
    owner = candidate.owners[candidate.grain_ids.index(grain_id)]
    temperature = float(np.sum(np.asarray(owner.temperature_K)*ring)
                        /np.sum(ring))
    coordinates = np.linspace(0.0, 1.0, parameters.energy_path_samples)
    path = []
    for coordinate in coordinates:
        point, _ = _collective_wall_candidate(
            state, owner_index, ring, parameters.order_increment, coordinate)
        point_energy = evaluate_complete_multigrain_energy(point, **common)
        path.append(point_energy.helmholtz_J-before.helmholtz_J)
    detailed_balance = collective_wall_detailed_balance(
        path, temperature, parameters.activation)
    probability = -math.expm1(-detailed_balance.forward_rate_Hz*dt)
    audit_fields = dict(
        resolved_forward_barrier_J=(
            detailed_balance.resolved_forward_barrier_J),
        intrinsic_forward_barrier_eV=(
            parameters.activation.activation_enthalpy_eV),
        forward_barrier_eV=detailed_balance.forward_barrier_eV,
        reverse_barrier_eV=detailed_balance.reverse_barrier_eV,
        forward_event_rate_Hz=detailed_balance.forward_rate_Hz,
        reverse_event_rate_Hz=detailed_balance.reverse_rate_Hz,
        detailed_balance_log_residual=(
            detailed_balance.detailed_balance_log_residual),
        energy_path_coordinate=tuple(map(float, coordinates)),
        energy_path_delta_helmholtz_J=tuple(map(float, path)))
    if draw >= probability:
        zero = np.zeros(state.supports.shape[1:])
        return CollectiveWallNucleationDecision(
            False, "INACTIVE_ACTIVATED_CLOSED_WALL_NOT_SAMPLED",
            candidate, state, grain_id, center, radius, candidate_count,
            downhill_count, probability, draw, transferred, 0.0, 0.0,
            delta, 0.0, 0.0, 0.0, zero, **audit_fields)
    heat = max(-delta, 0.0)
    heat_cells = np.zeros(state.supports.shape[1:])
    heat_cells = heat*ring/float(np.sum(ring, dtype=np.longdouble))
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
        heat_cells if accepted else np.zeros_like(heat_cells), **audit_fields)
