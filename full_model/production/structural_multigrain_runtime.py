"""Energy-complete common-state publication of structural junction reactions."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .complete_multigrain_energy import evaluate_complete_multigrain_energy
from .multigrain_production import advance_multigrain_front_interval
from .structural_boundary import StructuralBoundaryParameters
from .structural_interface_network import (
    JunctionDisclinationParameters, JunctionReactionParameters,
    StructuralInterface, audit_network_compatibility, interface_lengths_m,
    relax_junction_cycle_interval,
)


@dataclass(frozen=True)
class StructuralRuntimeLedger:
    intervals: int = 0
    accepted_reactions: int = 0
    rejected_reactions: int = 0
    inactive_intervals: int = 0
    reacted_absolute_content: float = 0.0
    generated_heat_J: float = 0.0
    maximum_relative_first_law_residual: float = 0.0


@dataclass(frozen=True)
class StructuralMultigrainRuntime:
    interfaces: tuple[StructuralInterface, ...]
    ledger: StructuralRuntimeLedger = StructuralRuntimeLedger()


@dataclass(frozen=True)
class StructuralRuntimeDecision:
    accepted: bool
    classification: str
    kinetic_proposal: object
    complete_energy_before: object
    complete_energy_candidate: object
    complete_energy_committed: object
    owner_orientation_before_rad: np.ndarray
    owner_orientation_candidate_rad: np.ndarray
    generated_heat_J: float
    first_law_residual_J: float
    relative_first_law_residual: float


@dataclass(frozen=True)
class StructuralFrontRuntime:
    front_runtime: object
    structural_runtime: StructuralMultigrainRuntime


@dataclass(frozen=True)
class StructuralFrontDecision:
    accepted: bool
    classification: str
    front_decisions: tuple
    orientation_sync_energy_change_J: float
    junction_decision: StructuralRuntimeDecision | None


def _uniform_owner_orientations(state, orientation_rad):
    values = np.asarray(orientation_rad, dtype=float)
    if values.shape != (len(state.owners),):
        raise ValueError("one derived orientation per owner is required")
    owners = tuple(
        replace(owner, orientation_rad=np.full_like(
            owner.orientation_rad, values[index], dtype=float))
        for index, owner in enumerate(state.owners))
    candidate = replace(state, owners=owners)
    candidate.validate()
    return candidate


def synchronize_structural_owner_orientations(
        state, runtime, *, spacing_m,
        burgers_m=StructuralBoundaryParameters().burgers_m):
    """Explicitly derive uniform common-axis owner angles from structural state."""
    lengths = interface_lengths_m(
        state.supports, runtime.interfaces, state.grain_ids, spacing_m)
    audit = audit_network_compatibility(
        state.grain_ids, runtime.interfaces, lengths, burgers_m)
    return _uniform_owner_orientations(state, audit.fitted_orientation_rad)


def _deposit_uniform_heat(state, heat_J, spacing_m, thickness_m,
                          volumetric_heat_capacity_J_m3_K):
    cells = int(np.prod(state.supports.shape[1:]))
    volume = cells*float(spacing_m)**2*float(thickness_m)
    delta_temperature = float(heat_J)/(
        float(volumetric_heat_capacity_J_m3_K)*volume)
    owners = tuple(
        replace(owner, temperature_K=np.asarray(owner.temperature_K)
                +delta_temperature)
        for owner in state.owners)
    heated = replace(state, owners=owners)
    heated.validate()
    return heated


def advance_structural_junction_interval(
        state, runtime, *, dt_s, spacing_m, represented_thickness_m,
        wall_parameters, energy_kwargs=None,
        boundary_parameters=StructuralBoundaryParameters(),
        disclination_parameters=JunctionDisclinationParameters(),
        reaction_parameters=JunctionReactionParameters(),
        closure_fraction=0.05):
    """Publish one junction reaction through the full common-state energy.

    The low-level EXP-floor operator supplies only a bounded content proposal.
    Owner orientations are then *derived* by graph compatibility, and the
    complete multigrain functional makes the publication decision.  Accepted
    zero-work energy release is deposited into the common temperature state;
    any uphill or first-law-failing proposal rolls back atomically.
    """
    if not isinstance(runtime, StructuralMultigrainRuntime):
        raise TypeError("a structural multigrain runtime is required")
    lengths = interface_lengths_m(
        state.supports, runtime.interfaces, state.grain_ids, spacing_m)
    temperature = float(np.mean([
        np.mean(owner.temperature_K) for owner in state.owners]))
    proposal = relax_junction_cycle_interval(
        state.grain_ids, runtime.interfaces, lengths,
        boundary_parameters.burgers_m, temperature_K=temperature, dt_s=dt_s,
        represented_thickness_m=represented_thickness_m,
        boundary_parameters=boundary_parameters,
        disclination_parameters=disclination_parameters,
        reaction_parameters=reaction_parameters)
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    for forbidden in ("structural_interfaces", "structural_burgers_m",
                      "structural_parameters",
                      "junction_disclination_parameters"):
        if forbidden in options:
            raise ValueError(f"{forbidden} is owned by the structural runtime")

    def complete(candidate_state, interfaces):
        return evaluate_complete_multigrain_energy(
            candidate_state, spacing_m=spacing_m,
            represented_thickness_m=represented_thickness_m,
            wall_parameters=wall_parameters,
            structural_interfaces=interfaces,
            structural_burgers_m=boundary_parameters.burgers_m,
            structural_parameters=boundary_parameters,
            junction_disclination_parameters=disclination_parameters,
            **options)

    before_audit = audit_network_compatibility(
        state.grain_ids, runtime.interfaces, lengths,
        boundary_parameters.burgers_m)
    for owner, derived in zip(
            state.owners, before_audit.fitted_orientation_rad):
        if np.max(np.abs(np.asarray(owner.orientation_rad)-derived)) > 1e-12:
            raise ValueError(
                "owner orientation is not synchronized with structural content")
    before_state = state
    before_energy = complete(before_state, runtime.interfaces)
    candidate_interfaces = proposal.candidate_interfaces
    candidate_audit = audit_network_compatibility(
        state.grain_ids, candidate_interfaces, lengths,
        boundary_parameters.burgers_m)
    candidate_state = _uniform_owner_orientations(
        state, candidate_audit.fitted_orientation_rad)
    candidate_energy = complete(candidate_state, candidate_interfaces)
    delta = candidate_energy.internal_J-before_energy.internal_J
    resolution = 4096.0*np.finfo(float).eps*max(
        abs(before_energy.internal_J), abs(candidate_energy.internal_J), 1e-30)
    active = proposal.classification != "NO_CYCLE_REACTION"
    downhill = bool(delta <= resolution)
    tentative = bool(active and downhill)
    heat = max(-delta, 0.0) if tentative else 0.0
    committed_state = (_deposit_uniform_heat(
        candidate_state, heat, spacing_m, represented_thickness_m,
        wall_parameters.volumetric_heat_capacity_J_m3_K)
        if tentative else before_state)
    committed_interfaces = candidate_interfaces if tentative else runtime.interfaces
    committed_energy = complete(committed_state, committed_interfaces)
    residual = (committed_energy.internal_J-before_energy.internal_J)
    scale = max(abs(delta)+heat, resolution, 1e-300)
    relative = abs(residual)/scale
    accepted = bool(tentative and relative <= float(closure_fraction))
    if not active:
        classification = "NO_CYCLE_REACTION"
    elif not downhill:
        classification = "REJECTED_COMPLETE_ENERGY_UPHILL"
    elif not accepted:
        classification = "REJECTED_COMPLETE_FIRST_LAW"
    else:
        classification = "ACCEPTED_COMPLETE_JUNCTION_RELAXATION"
    if not accepted:
        committed_state = before_state
        committed_interfaces = runtime.interfaces
        committed_energy = before_energy
        heat = 0.0
        residual = 0.0
        relative = 0.0
    ledger = replace(
        runtime.ledger,
        intervals=runtime.ledger.intervals+1,
        accepted_reactions=(runtime.ledger.accepted_reactions+int(accepted)),
        rejected_reactions=(runtime.ledger.rejected_reactions
                            +int(active and not accepted)),
        inactive_intervals=(runtime.ledger.inactive_intervals+int(not active)),
        reacted_absolute_content=(runtime.ledger.reacted_absolute_content
                                  +(proposal.absolute_reacted_content
                                    if accepted else 0.0)),
        generated_heat_J=runtime.ledger.generated_heat_J+heat,
        maximum_relative_first_law_residual=max(
            runtime.ledger.maximum_relative_first_law_residual, relative))
    decision = StructuralRuntimeDecision(
        accepted, classification, proposal, before_energy, candidate_energy,
        committed_energy, before_audit.fitted_orientation_rad,
        candidate_audit.fitted_orientation_rad, heat, residual, relative)
    return (committed_state,
            StructuralMultigrainRuntime(tuple(committed_interfaces), ledger),
            decision)


def advance_structural_front_interval(
        state, runtime, *, kinetics, dt_s, maximum_substep_s, spacing_m,
        represented_thickness_m, wall_parameters, energy_kwargs=None,
        applied_shear_rate_s=0.0, systems=None,
        boundary_parameters=StructuralBoundaryParameters(),
        disclination_parameters=JunctionDisclinationParameters(),
        reaction_parameters=JunctionReactionParameters(),
        heat_deposition_mode="local_realized_event"):
    """Compose the complete front and structural reaction on one owner state.

    Conservative interface content remains unchanged during the front map.
    The derived common-axis owner rotation must be energy-neutral to numerical
    resolution in this reduced production path; otherwise the whole front
    interval rolls back rather than hiding unpriced mechanical work.  A
    separately priced junction reaction then runs on the synchronized state.
    """
    if not isinstance(runtime, StructuralFrontRuntime):
        raise TypeError("a structural-front runtime is required")
    structural = runtime.structural_runtime
    # Enforce the entry invariant before allowing the complete front to run.
    synchronized = synchronize_structural_owner_orientations(
        state, structural, spacing_m=spacing_m,
        burgers_m=boundary_parameters.burgers_m)
    for old, derived in zip(state.owners, synchronized.owners):
        if np.max(np.abs(np.asarray(old.orientation_rad)
                         -np.asarray(derived.orientation_rad))) > 1e-12:
            raise ValueError(
                "front entry orientation is not synchronized with structural content")
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    for forbidden in ("structural_interfaces", "structural_burgers_m",
                      "structural_parameters",
                      "junction_disclination_parameters"):
        if forbidden in options:
            raise ValueError(f"{forbidden} is owned by the structural runtime")
    structural_energy = dict(
        options,
        structural_interfaces=structural.interfaces,
        structural_burgers_m=boundary_parameters.burgers_m,
        structural_parameters=boundary_parameters,
        junction_disclination_parameters=disclination_parameters)
    front_state, front_runtime, front_decisions = (
        advance_multigrain_front_interval(
            state, runtime.front_runtime, kinetics=kinetics, dt_s=dt_s,
            maximum_substep_s=maximum_substep_s, spacing_m=spacing_m,
            represented_thickness_m=represented_thickness_m,
            wall_parameters=wall_parameters, energy_kwargs=structural_energy,
            applied_shear_rate_s=applied_shear_rate_s, systems=systems,
            heat_deposition_mode=heat_deposition_mode))
    synchronized_front = synchronize_structural_owner_orientations(
        front_state, structural, spacing_m=spacing_m,
        burgers_m=boundary_parameters.burgers_m)

    def complete(candidate):
        return evaluate_complete_multigrain_energy(
            candidate, spacing_m=spacing_m,
            represented_thickness_m=represented_thickness_m,
            wall_parameters=wall_parameters, **structural_energy)

    unsynchronized_energy = complete(front_state)
    synchronized_energy = complete(synchronized_front)
    sync_delta = synchronized_energy.internal_J-unsynchronized_energy.internal_J
    resolution = 4096.0*np.finfo(float).eps*max(
        abs(unsynchronized_energy.internal_J),
        abs(synchronized_energy.internal_J), 1e-30)
    if abs(sync_delta) > resolution:
        return state, runtime, StructuralFrontDecision(
            False, "REJECTED_UNPRICED_ORIENTATION_SYNC", tuple(front_decisions),
            sync_delta, None)
    junction_state, junction_runtime, junction_decision = (
        advance_structural_junction_interval(
            synchronized_front, structural, dt_s=dt_s, spacing_m=spacing_m,
            represented_thickness_m=represented_thickness_m,
            wall_parameters=wall_parameters, energy_kwargs=options,
            boundary_parameters=boundary_parameters,
            disclination_parameters=disclination_parameters,
            reaction_parameters=reaction_parameters))
    combined = StructuralFrontRuntime(front_runtime, junction_runtime)
    accepted = all(decision.accepted for decision in front_decisions)
    classification = ("ACCEPTED_STRUCTURAL_FRONT_INTERVAL" if accepted
                      else "FRONT_SUBEVENT_REJECTED")
    return junction_state, combined, StructuralFrontDecision(
        accepted, classification, tuple(front_decisions), sync_delta,
        junction_decision)
