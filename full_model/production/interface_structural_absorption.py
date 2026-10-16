"""Atomic common-axis absorption of explicit interface Burgers content.

The moving-front model stores two distinct interface inventories: nonnegative
line length and a signed laboratory-frame Burgers-vector density.  This module
connects the part of that vector density compatible with a common-axis tilt
boundary to the conserved structural edge content.  The projection is local;
the tangential/incompatible vector remains in the interface reservoir.

This is deliberately not a grain-growth rule.  It changes neither supports nor
labels.  Owner rotations are derived from the resulting structural network and
the complete fixed-strain energy decides whether the zero-work event commits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .complete_multigrain_energy import evaluate_complete_multigrain_energy
from .structural_boundary import (
    StructuralBoundaryParameters, exp_floor_event_rate_Hz,
)
from .structural_interface_network import (
    JunctionDisclinationParameters, StructuralInterface,
    audit_network_compatibility, interface_lengths_m,
)
from .tensorial_nye import spectral_derivatives


@dataclass(frozen=True)
class InterfaceStructuralAbsorptionParameters:
    """Bounded kinetics for the common-axis reservoir reaction."""

    boundary: StructuralBoundaryParameters = StructuralBoundaryParameters()
    maximum_fraction_per_interval: float = 0.10
    normal_resolution: float = 1.0e-12
    complete_physical_clock: bool = False

    def __post_init__(self):
        if (not math.isfinite(self.maximum_fraction_per_interval)
                or not 0.0 < self.maximum_fraction_per_interval <= 1.0):
            raise ValueError("maximum absorption fraction must lie in (0,1]")
        if not math.isfinite(self.normal_resolution) or self.normal_resolution <= 0.0:
            raise ValueError("normal resolution must be positive finite")


@dataclass(frozen=True)
class InterfaceStructuralProjection:
    candidate_state: object
    candidate_interfaces: tuple[StructuralInterface, ...]
    structural_content_change: np.ndarray
    removed_line_length_m: float
    removed_signed_content: float
    signed_content_balance: float
    maximum_vector_balance_m1: float
    vector_balance_scale_m1: float
    relative_vector_balance: float
    incompatible_burgers_norm_before_m1: float
    incompatible_burgers_norm_after_m1: float


@dataclass(frozen=True)
class InterfaceStructuralAbsorptionDecision:
    accepted: bool
    classification: str
    candidate_state: object
    committed_state: object
    candidate_interfaces: tuple[StructuralInterface, ...]
    committed_interfaces: tuple[StructuralInterface, ...]
    kinetic_fraction: float
    event_rate_s: float
    affinity_J_m: float
    projection: InterfaceStructuralProjection
    energy_before: object
    energy_candidate: object
    energy_committed: object
    delta_helmholtz_J: float
    generated_heat_J: float
    first_law_residual_J: float
    relative_first_law_residual: float


def _uniform_owner_orientations(state, values):
    values = np.asarray(values, dtype=float)
    if values.shape != (len(state.owners),):
        raise ValueError("one common-axis orientation per owner is required")
    owners = tuple(
        replace(owner, orientation_rad=(
            np.asarray(owner.orientation_rad, dtype=float)
            +value-float(np.mean(owner.orientation_rad))))
        for owner, value in zip(state.owners, values))
    candidate = replace(state, owners=owners)
    candidate.validate()
    return candidate


def _structural_options(energy_kwargs, interfaces, burgers_m, parameters):
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    for key in ("structural_interfaces", "structural_burgers_m",
                "structural_parameters", "junction_disclination_parameters"):
        options.pop(key, None)
    options.update(
        structural_interfaces=interfaces,
        structural_burgers_m=burgers_m,
        structural_parameters=parameters.boundary,
        junction_disclination_parameters=JunctionDisclinationParameters())
    return options


def project_interface_burgers_to_structural_content(
        state, structural_interfaces, *, spacing_m, fraction,
        burgers_m, normal_resolution=1.0e-12):
    """Debit accessible line/vector reservoirs and credit structural edges.

    ``boundary_burgers_m1`` is projected on the local normal directed from
    ``owner_a`` to ``owner_b``.  The exact projected vector is removed.  The
    corresponding unsigned line length is removed proportionally from the
    explicit +/- family inventory, whose family basis is not reconstructible
    after multiple front passages.  No assertion that slip sign equals the
    common-axis structural sign is made; the vector reservoir owns that sign.
    """
    state.validate()
    spacing = float(spacing_m)
    reacted_fraction = float(fraction)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be positive finite")
    if not math.isfinite(reacted_fraction) or not 0.0 <= reacted_fraction <= 1.0:
        raise ValueError("fraction must lie in [0,1]")
    edge_by_id = {item.interface_id: (index, item)
                  for index, item in enumerate(structural_interfaces)}
    owner_index = {owner: index for index, owner in enumerate(state.grain_ids)}
    content_change = np.zeros(len(structural_interfaces), dtype=float)
    next_components = []
    removed_line_length = 0.0
    removed_signed_content = 0.0
    maximum_vector_balance = 0.0
    vector_balance_scale = 1.0
    incompatible_before = 0.0
    incompatible_after = 0.0
    cell_area = spacing**2
    b = float(burgers_m)

    for component in state.interfaces:
        if component.component_id not in edge_by_id:
            next_components.append(component)
            continue
        edge_index, edge = edge_by_id[component.component_id]
        if edge.owner_a not in owner_index or edge.owner_b not in owner_index:
            raise ValueError("structural edge references an unknown owner")
        eta = (np.asarray(state.supports[owner_index[edge.owner_b]])
               -np.asarray(state.supports[owner_index[edge.owner_a]]))
        gx, gy = spectral_derivatives(eta, spacing)
        magnitude = np.sqrt(gx*gx+gy*gy)
        active = magnitude > float(normal_resolution)*max(
            float(np.max(magnitude, initial=0.0)), 1.0)
        nx = np.divide(gx, magnitude, out=np.zeros_like(gx), where=active)
        ny = np.divide(gy, magnitude, out=np.zeros_like(gy), where=active)
        normal = np.stack((nx, ny, np.zeros_like(nx)), axis=-1)

        burgers = np.asarray(component.boundary_burgers_m1, dtype=float)
        if burgers.shape[-2:] != (3, 3):
            raise ValueError("interface Burgers reservoir must be grid+(3,3)")
        vector_balance_scale = max(
            vector_balance_scale,
            float(np.max(np.abs(burgers), initial=0.0)))
        projected = np.sum(burgers*normal[..., None, :], axis=-1)
        requested = reacted_fraction*np.abs(projected)/b
        plus = np.asarray(component.boundary_plus_m2, dtype=float)
        minus = np.asarray(component.boundary_minus_m2, dtype=float)
        available = np.sum(plus+minus, axis=-1)
        scale = np.minimum(
            1.0, np.divide(available, requested, out=np.ones_like(requested),
                           where=requested > 0.0))
        reacted_projection = reacted_fraction*projected*scale
        debit = np.abs(reacted_projection)/b
        total = plus+minus
        allocation = np.divide(
            debit[..., None], available[..., None],
            out=np.zeros_like(total), where=available[..., None] > 0.0)
        next_plus = np.maximum(plus-allocation*plus, 0.0)
        next_minus = np.maximum(minus-allocation*minus, 0.0)
        removed_vector = reacted_projection[..., None]*normal[..., None, :]
        next_burgers = burgers-removed_vector
        vector_residual = (burgers-next_burgers)-removed_vector
        maximum_vector_balance = max(
            maximum_vector_balance,
            float(np.max(np.abs(vector_residual), initial=0.0)))
        before_tangent = burgers-projected[..., None]*normal[..., None, :]
        after_projection = np.sum(
            next_burgers*normal[..., None, :], axis=-1)
        after_tangent = next_burgers-after_projection[..., None]*normal[..., None, :]
        incompatible_before += float(np.linalg.norm(before_tangent))
        incompatible_after += float(np.linalg.norm(after_tangent))
        delta_content = float(np.sum(
            reacted_projection, dtype=np.longdouble)*cell_area/b)
        content_change[edge_index] += delta_content
        removed_signed_content += delta_content
        removed_line_length += float(np.sum(
            debit, dtype=np.longdouble)*cell_area)
        next_components.append(replace(
            component, boundary_plus_m2=next_plus,
            boundary_minus_m2=next_minus,
            boundary_burgers_m1=next_burgers))

    next_interfaces = tuple(
        replace(item, signed_integrated_content=(
            item.signed_integrated_content+content_change[index]))
        for index, item in enumerate(structural_interfaces))
    candidate = replace(state, interfaces=tuple(next_components))
    candidate.validate()
    return InterfaceStructuralProjection(
        candidate, next_interfaces, content_change, removed_line_length,
        removed_signed_content, float(np.sum(content_change)-removed_signed_content),
        maximum_vector_balance, vector_balance_scale,
        maximum_vector_balance/vector_balance_scale,
        incompatible_before, incompatible_after)


def evaluate_interface_structural_absorption(
        state, structural_interfaces, *, mean_strain, interval_s, spacing_m,
        represented_thickness_m, wall_parameters, energy_kwargs=None,
        parameters=InterfaceStructuralAbsorptionParameters(),
        maximum_relative_first_law_residual=0.05):
    """Price and atomically commit one stationary common-axis reaction."""
    dt = float(interval_s)
    spacing = float(spacing_m)
    thickness = float(represented_thickness_m)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("reaction interval must be positive finite")
    lengths = interface_lengths_m(
        state.supports, structural_interfaces, state.grain_ids, spacing)
    orientations = np.asarray(
        [np.mean(owner.orientation_rad) for owner in state.owners], dtype=float)
    before_audit = audit_network_compatibility(
        state.grain_ids, structural_interfaces, lengths,
        parameters.boundary.burgers_m,
        reference_owner=state.grain_ids[0],
        reference_orientation_rad=orientations[0])
    if np.max(np.abs(before_audit.fitted_orientation_rad-orientations)) > 1e-10:
        raise ValueError("owner orientations are not synchronized with structural edges")

    common = dict(
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall_parameters, mean_strain=np.asarray(mean_strain))
    before_options = _structural_options(
        energy_kwargs, structural_interfaces, parameters.boundary.burgers_m,
        parameters)
    before_energy = evaluate_complete_multigrain_energy(
        state, **common, **before_options)

    full = project_interface_burgers_to_structural_content(
        state, structural_interfaces, spacing_m=spacing, fraction=1.0,
        burgers_m=parameters.boundary.burgers_m,
        normal_resolution=parameters.normal_resolution)
    full_lengths = interface_lengths_m(
        full.candidate_state.supports, full.candidate_interfaces,
        state.grain_ids, spacing)
    full_audit = audit_network_compatibility(
        state.grain_ids, full.candidate_interfaces, full_lengths,
        parameters.boundary.burgers_m,
        reference_owner=state.grain_ids[0],
        reference_orientation_rad=orientations[0])
    full_state = _uniform_owner_orientations(
        full.candidate_state, full_audit.fitted_orientation_rad)
    full_energy = evaluate_complete_multigrain_energy(
        full_state, **common, **_structural_options(
            energy_kwargs, full.candidate_interfaces,
            parameters.boundary.burgers_m, parameters))
    full_drop = before_energy.helmholtz_J-full_energy.helmholtz_J
    full_line_m = full.removed_line_length_m*thickness
    affinity = full_drop/max(full_line_m, np.finfo(float).tiny)
    temperature = float(np.mean([
        np.mean(owner.temperature_K) for owner in state.owners]))
    rate = float(exp_floor_event_rate_Hz(
        max(affinity, 0.0), temperature, parameters.boundary))
    uncapped_fraction = -math.expm1(-rate*dt)
    fraction = (uncapped_fraction if parameters.complete_physical_clock else
                min(uncapped_fraction,
                    parameters.maximum_fraction_per_interval))

    projection = project_interface_burgers_to_structural_content(
        state, structural_interfaces, spacing_m=spacing, fraction=fraction,
        burgers_m=parameters.boundary.burgers_m,
        normal_resolution=parameters.normal_resolution)
    candidate_audit = audit_network_compatibility(
        state.grain_ids, projection.candidate_interfaces, lengths,
        parameters.boundary.burgers_m,
        reference_owner=state.grain_ids[0],
        reference_orientation_rad=orientations[0])
    cold_candidate = _uniform_owner_orientations(
        projection.candidate_state, candidate_audit.fitted_orientation_rad)
    candidate_options = _structural_options(
        energy_kwargs, projection.candidate_interfaces,
        parameters.boundary.burgers_m, parameters)
    candidate_energy = evaluate_complete_multigrain_energy(
        cold_candidate, **common, **candidate_options)
    delta = candidate_energy.helmholtz_J-before_energy.helmholtz_J
    resolution = 4096.0*np.finfo(float).eps*max(
        abs(before_energy.helmholtz_J), abs(candidate_energy.helmholtz_J), 1e-300)
    active = projection.removed_line_length_m > 0.0 and fraction > 0.0
    downhill = delta <= resolution
    heat = max(-delta, 0.0) if active and downhill else 0.0
    if active and downhill:
        cells = int(np.prod(state.supports.shape[1:]))
        volume = cells*spacing**2*thickness
        rise = heat/(wall_parameters.volumetric_heat_capacity_J_m3_K*volume)
        committed_cold = replace(cold_candidate, owners=tuple(
            replace(owner, temperature_K=np.asarray(owner.temperature_K)+rise)
            for owner in cold_candidate.owners))
        committed_cold.validate()
        committed_energy = evaluate_complete_multigrain_energy(
            committed_cold, **common, **candidate_options)
        residual = committed_energy.internal_J-before_energy.internal_J
        scale = max(abs(delta), heat, resolution, 1e-300)
        relative = abs(residual)/scale
    else:
        committed_cold = state
        committed_energy = before_energy
        residual = 0.0
        relative = 0.0
    vector_roundoff = (4096.0*np.finfo(float).eps
                       *projection.vector_balance_scale_m1)
    balances_close = (
        abs(projection.signed_content_balance) <= 1e-10
        and projection.maximum_vector_balance_m1 <= vector_roundoff)
    accepted = bool(active and downhill and balances_close
                    and relative <= maximum_relative_first_law_residual)
    if not active:
        classification = "INACTIVE_NO_COMPATIBLE_INTERFACE_BURGERS"
    elif not downhill:
        classification = "REJECTED_COMPLETE_ENERGY_UPHILL"
    elif not balances_close:
        classification = "REJECTED_BURGERS_OR_LINE_BALANCE"
    elif relative > maximum_relative_first_law_residual:
        classification = "REJECTED_COMPLETE_FIRST_LAW"
    else:
        classification = "ACCEPTED_COMMON_AXIS_INTERFACE_ABSORPTION"
    return InterfaceStructuralAbsorptionDecision(
        accepted, classification, cold_candidate,
        committed_cold if accepted else state,
        projection.candidate_interfaces,
        projection.candidate_interfaces if accepted else tuple(structural_interfaces),
        fraction, rate, affinity, projection, before_energy, candidate_energy,
        committed_energy if accepted else before_energy, delta,
        heat if accepted else 0.0, residual if accepted else 0.0,
        relative if accepted else 0.0)
