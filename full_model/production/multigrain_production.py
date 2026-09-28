"""Recurrent zero-pressure production driver for a shared multi-grain state."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .arrhenius_kinetics import ActivatedProcess
from .complete_multigrain_energy import (
    MultiGrainDissipation, evaluate_joint_multigrain_transaction,
)
from .coupled_front_event import propose_bidirectional_front_event
from .moving_front import DefectState
from .common_tensorial_wall import accepted_euler_step
from .multigrain_common_state import (
    PhysicalTransferLaw, derive_physical_transfer_proposal,
    joint_material_transaction,
)


@dataclass(frozen=True)
class MultiGrainInterface:
    component_id: str
    grain_a_id: int
    grain_b_id: int


@dataclass(frozen=True)
class MultiGrainFrontKinetics:
    process: ActivatedProcess
    transfer_law: PhysicalTransferLaw
    h0_J: float
    critical_pressure_Pa: float
    exp_a: float
    exp_n: float
    exp_floor: float
    event_volume_m3: float
    event_length_m: float
    virtual_fraction: float = 1.0e-4
    maximum_fraction_per_step: float = 0.05
    closure_fraction: float = 0.01
    maximum_backtracks: int = 12

    def validate(self):
        self.transfer_law.validate()
        positive = (self.h0_J, self.critical_pressure_Pa,
                    self.event_volume_m3, self.event_length_m,
                    self.virtual_fraction, self.maximum_fraction_per_step,
                    self.closure_fraction)
        if any(not math.isfinite(float(value)) or float(value) <= 0.0
               for value in positive):
            raise ValueError("front kinetic scales must be positive and finite")
        if self.exp_n < 1.0 or self.exp_a < 0.0 or not 0.0 <= self.exp_floor <= 1.0:
            raise ValueError("invalid EXP-floor kinetics")
        if self.maximum_fraction_per_step > 1.0 or self.virtual_fraction > 1.0:
            raise ValueError("front fractions must not exceed unity")


@dataclass(frozen=True)
class MultiGrainProductionLedger:
    intervals: int = 0
    accepted_events: int = 0
    rejected_events: int = 0
    stationary_intervals: int = 0
    physical_time_s: float = 0.0
    applied_shear_strain: float = 0.0
    transformed_volume_m3: float = 0.0
    generated_heat_J: float = 0.0
    maximum_relative_energy_closure: float = 0.0


@dataclass(frozen=True)
class MultiGrainProductionRuntime:
    interfaces: tuple[MultiGrainInterface, ...]
    ledger: MultiGrainProductionLedger = MultiGrainProductionLedger()


@dataclass(frozen=True)
class MultiGrainProductionDecision:
    accepted: bool
    classification: str
    direction_by_interface: dict[str, str]
    pressure_by_interface_Pa: dict[str, float]
    requested_fraction_by_interface: dict[str, float]
    accepted_fraction_by_interface: dict[str, float]
    energy_decision: object | None
    backtracks: int


@dataclass(frozen=True)
class MultiGrainMechanicalDecision:
    accepted: bool
    minimum_step_scale: float
    external_plastic_work_J: float
    irreversible_heat_J: float
    maximum_line_balance_residual_m2_s: float
    maximum_substeps: int
    consumed_interval_s: float


def _masked_owner_update(before, after, active):
    from dataclasses import fields
    from .common_tensorial_wall import CommonWallState
    arrays = {}
    for item in fields(CommonWallState):
        old = np.asarray(getattr(before, item.name))
        new = np.asarray(getattr(after, item.name))
        mask = active[(...,)+(None,)*(old.ndim-active.ndim)]
        arrays[item.name] = np.where(mask, new, old)
    return CommonWallState(**arrays)


def advance_multigrain_mechanics(
        state, *, driving, systems, topologies, wall_parameters, dt_s,
        represented_thickness_m):
    """Advance each persistent material owner over the same physical interval.

    Only supported material evolves; dormant owner history is bitwise retained
    for later re-entry. Work and heat are integrated from the accepted common
    residual channels, not inferred from an endpoint energy difference.
    """
    dt = float(dt_s)
    cell_volume = (float(wall_parameters.spacing_m)**2
                   *float(represented_thickness_m))
    owners = []
    work = heat = 0.0
    minimum_scale = 1.0
    maximum_line_residual = 0.0
    maximum_substeps = 0
    for support, owner in zip(state.supports, state.owners):
        updated = owner
        remaining = dt
        substeps = 0
        failed_retries = 0
        owner_work = owner_heat = 0.0
        while remaining > 64.0*np.finfo(float).eps*dt:
            attempted = remaining
            while True:
                try:
                    trial, residual, scale = accepted_euler_step(
                        updated, driving, systems, topologies,
                        wall_parameters, attempted)
                    break
                except (ValueError, FloatingPointError):
                    attempted *= .5
                    failed_retries += 1
                    if (failed_retries > 64
                            or attempted <= 64.0*np.finfo(float).eps*dt):
                        raise RuntimeError(
                            "mechanical interval could not find an admissible substep")
            consumed = attempted*float(scale)
            if consumed <= 0.0:
                raise RuntimeError("mechanical interval made no physical-time progress")
            weight = np.asarray(support, dtype=float)
            owner_work += float(np.sum(
                weight*residual.plastic_power_W_m3,
                dtype=np.longdouble)*cell_volume*consumed)
            owner_heat += float(np.sum(
                weight*residual.heat_rate_W_m3,
                dtype=np.longdouble)*cell_volume*consumed)
            line_residual = residual.channel_rates_m2_s.get(
                "line_balance_residual_m2_s", 0.0)
            maximum_line_residual = max(
                maximum_line_residual,
                float(np.max(np.abs(line_residual))))
            minimum_scale = min(minimum_scale, float(scale))
            updated = trial
            remaining -= consumed
            substeps += 1
            if substeps > 4096:
                raise RuntimeError("mechanical physical interval exceeded substep budget")
        active = np.asarray(support) > 256.0*np.finfo(float).eps
        owners.append(_masked_owner_update(owner, updated, active))
        work += owner_work
        heat += owner_heat
        maximum_substeps = max(maximum_substeps, substeps)
    candidate = replace(state, owners=tuple(owners))
    candidate.validate()
    decision = MultiGrainMechanicalDecision(
        True, minimum_scale, work, heat, maximum_line_residual,
        maximum_substeps, dt)
    return candidate, decision


def _defect(owner):
    return DefectState(
        np.asarray(owner.mobile_plus_m2), np.asarray(owner.mobile_minus_m2),
        np.asarray(owner.forest_plus_m2)+np.asarray(owner.forest_minus_m2),
        np.sum(np.asarray(owner.wall_plus_m2)
               +np.asarray(owner.wall_minus_m2), axis=2))


def _contact_weight(state, donor_index, receiver_index):
    donor = np.asarray(state.supports[donor_index])
    receiver = np.asarray(state.supports[receiver_index])
    neighbor = np.maximum.reduce((
        receiver, np.roll(receiver, 1, axis=0),
        np.roll(receiver, -1, axis=0), np.roll(receiver, 1, axis=1),
        np.roll(receiver, -1, axis=1)))
    return np.minimum(donor, np.clip(neighbor, 0.0, 1.0))


def _virtual_channel(state, interface, donor_id, receiver_id, *, kinetics,
                     spacing_m, represented_thickness_m, wall_parameters,
                     energy_kwargs, interval_s, systems):
    index = {grain_id: position for position, grain_id in enumerate(state.grain_ids)}
    weight = _contact_weight(state, index[donor_id], index[receiver_id])
    request = kinetics.virtual_fraction*weight
    swept_cells = float(np.sum(request, dtype=np.longdouble))
    cell_volume = float(spacing_m)**2*float(represented_thickness_m)
    swept_volume = swept_cells*cell_volume
    if swept_volume <= 0.0:
        return None, weight
    proposal = derive_physical_transfer_proposal(
        state, interface_id=interface.component_id, donor_id=donor_id,
        receiver_id=receiver_id, requested_fraction=request,
        law=kinetics.transfer_law, interval_s=interval_s, systems=systems)
    capacity = joint_material_transaction(state, (proposal,))
    priced = evaluate_joint_multigrain_transaction(
        state, capacity, spacing_m=spacing_m,
        represented_thickness_m=represented_thickness_m,
        wall_parameters=wall_parameters, interval_s=interval_s,
        dissipation=MultiGrainDissipation(), energy_kwargs=energy_kwargs,
        absolute_tolerance_J=0.0)
    return (priced.decision.available_change_J, swept_volume), weight


def advance_multigrain_front(
        state, runtime, *, kinetics, dt_s, spacing_m,
        represented_thickness_m, wall_parameters, energy_kwargs=None,
        applied_shear_rate_s=0.0, systems=None):
    """Advance all incident boundaries from one immutable accepted state.

    Directional derivatives price each edge in the full shared functional.
    EXP-floor kinetics chooses the realized contour speed. Simultaneous donor
    competition is then resolved once, and mobility dissipation is evaluated
    as driving pressure times realized volume rather than as an energy
    residual. Failed finite events are backtracked atomically.
    """
    kinetics.validate()
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("production interval must be positive and finite")
    index = {grain_id: position for position, grain_id in enumerate(state.grain_ids)}
    temperature = float(np.mean([
        np.mean(owner.temperature_K) for owner in state.owners]))
    directions = {}
    pressures = {}
    base_requests = {}
    channel_records = {}
    for interface in runtime.interfaces:
        if interface.grain_a_id not in index or interface.grain_b_id not in index:
            raise ValueError("runtime interface references an unknown grain")
        ab, weight_ab = _virtual_channel(
            state, interface, interface.grain_a_id, interface.grain_b_id,
            kinetics=kinetics, spacing_m=spacing_m,
            represented_thickness_m=represented_thickness_m,
            wall_parameters=wall_parameters, energy_kwargs=energy_kwargs,
            interval_s=dt, systems=systems)
        ba, weight_ba = _virtual_channel(
            state, interface, interface.grain_b_id, interface.grain_a_id,
            kinetics=kinetics, spacing_m=spacing_m,
            represented_thickness_m=represented_thickness_m,
            wall_parameters=wall_parameters, energy_kwargs=energy_kwargs,
            interval_s=dt, systems=systems)
        if ab is None and ba is None:
            continue
        delta_ab = 0.0 if ab is None else ab[0]*kinetics.event_volume_m3/ab[1]
        delta_ba = 0.0 if ba is None else ba[0]*kinetics.event_volume_m3/ba[1]
        event = propose_bidirectional_front_event(
            _defect(state.owners[index[interface.grain_a_id]]),
            _defect(state.owners[index[interface.grain_b_id]]),
            event_volume_m3=kinetics.event_volume_m3,
            event_length_m=kinetics.event_length_m,
            line_energy_J_m=wall_parameters.line_energy_J_m,
            temperature_K=temperature, process=kinetics.process,
            h0_J=kinetics.h0_J,
            critical_pressure_Pa=kinetics.critical_pressure_Pa,
            exp_a=kinetics.exp_a, exp_n=kinetics.exp_n,
            exp_floor=kinetics.exp_floor,
            transmission_fraction=0.0, boundary_storage_fraction=0.0,
            neutral_sink_fraction=0.0,
            kinetic_free_energy_a_to_b_J=delta_ab,
            kinetic_free_energy_b_to_a_J=delta_ba,
            actual_reverse_edge=False,
            deterministic_rate_law="complete_dissipation")
        velocity = event.net_velocity_a_to_b_m_s
        if velocity == 0.0:
            continue
        if velocity > 0.0:
            donor, receiver, weight, virtual = (
                interface.grain_a_id, interface.grain_b_id, weight_ab, ab)
            direction = "a_to_b"
        else:
            donor, receiver, weight, virtual = (
                interface.grain_b_id, interface.grain_a_id, weight_ba, ba)
            direction = "b_to_a"
        if virtual is None:
            continue
        fraction = min(abs(velocity)*dt/float(spacing_m),
                       kinetics.maximum_fraction_per_step)
        request = fraction*weight
        pressure = max(-virtual[0]/virtual[1], 0.0)
        if pressure <= 0.0 or not np.any(request > 0.0):
            continue
        directions[interface.component_id] = direction
        pressures[interface.component_id] = pressure
        base_requests[interface.component_id] = request
        channel_records[interface.component_id] = (interface, donor, receiver)

    next_ledger = replace(
        runtime.ledger, intervals=runtime.ledger.intervals+1,
        physical_time_s=runtime.ledger.physical_time_s+dt,
        applied_shear_strain=(runtime.ledger.applied_shear_strain
                              +float(applied_shear_rate_s)*dt))
    if not base_requests:
        next_ledger = replace(
            next_ledger,
            stationary_intervals=next_ledger.stationary_intervals+1)
        return state, replace(runtime, ledger=next_ledger), MultiGrainProductionDecision(
            True, "STATIONARY_COMPLETE_AFFINITY", {}, {}, {}, {}, None, 0)

    options = {} if energy_kwargs is None else dict(energy_kwargs)
    result = None
    for backtrack in range(kinetics.maximum_backtracks+1):
        factor = 0.5**backtrack
        proposals = []
        for key, request in base_requests.items():
            interface, donor, receiver = channel_records[key]
            proposals.append(derive_physical_transfer_proposal(
                state, interface_id=interface.component_id, donor_id=donor,
                receiver_id=receiver, requested_fraction=factor*request,
                law=kinetics.transfer_law, interval_s=dt, systems=systems))
        capacity = joint_material_transaction(state, tuple(proposals))
        cell_volume = float(spacing_m)**2*float(represented_thickness_m)
        dissipation_J = sum(
            pressures[key]*float(np.sum(extent, dtype=np.longdouble))*cell_volume
            for key, extent in capacity.accepted_fraction_by_interface.items())
        tolerance_J = max(
            4096.0*np.finfo(float).eps*max(dissipation_J, 1e-300),
            kinetics.closure_fraction*dissipation_J)
        result = evaluate_joint_multigrain_transaction(
            state, capacity, spacing_m=spacing_m,
            represented_thickness_m=represented_thickness_m,
            wall_parameters=wall_parameters, interval_s=dt,
            dissipation=MultiGrainDissipation(
                boundary_mobility_J=dissipation_J),
            energy_kwargs=options, absolute_tolerance_J=tolerance_J)
        if result.decision.accepted:
            break
    requested = {key: float(np.sum(value, dtype=np.longdouble))
                 for key, value in base_requests.items()}
    accepted_fractions = ({key: float(np.sum(value, dtype=np.longdouble))
                           for key, value in capacity.accepted_fraction_by_interface.items()}
                          if result is not None else {})
    if result is None or not result.decision.accepted:
        next_ledger = replace(
            next_ledger, rejected_events=next_ledger.rejected_events+1)
        decision = None if result is None else result.decision
        classification = ("REJECTED_NO_TRANSACTION" if decision is None
                          else decision.classification)
        return state, replace(runtime, ledger=next_ledger), MultiGrainProductionDecision(
            False, classification, directions, pressures, requested,
            accepted_fractions, decision, kinetics.maximum_backtracks)
    swept_volume = sum(accepted_fractions.values())*(
        float(spacing_m)**2*float(represented_thickness_m))
    closure_scale = max(result.decision.generated_heat_J, 1e-300)
    relative_closure = abs(result.decision.first_law_residual_J)/closure_scale
    next_ledger = replace(
        next_ledger, accepted_events=next_ledger.accepted_events+1,
        transformed_volume_m3=next_ledger.transformed_volume_m3+swept_volume,
        generated_heat_J=(next_ledger.generated_heat_J
                          +result.decision.generated_heat_J),
        maximum_relative_energy_closure=max(
            next_ledger.maximum_relative_energy_closure, relative_closure))
    return (result.published_state, replace(runtime, ledger=next_ledger),
            MultiGrainProductionDecision(
                True, result.decision.classification, directions, pressures,
                requested, accepted_fractions, result.decision, backtrack))
