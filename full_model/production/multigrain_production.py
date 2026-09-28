"""Recurrent zero-pressure production driver for a shared multi-grain state."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .arrhenius_kinetics import ActivatedProcess
from .complete_multigrain_energy import (
    MultiGrainDissipation, evaluate_joint_multigrain_transaction,
    evaluate_multigrain_mechanical_interval,
)
from .coupled_front_event import propose_bidirectional_front_event
from .moving_front import DefectState
from .common_tensorial_wall import CommonWallDriving, accepted_euler_step
from .nonlocal_elasticity import solve_periodic_eigenstrain
from .tensorial_nye import rotated_system_fields
from .multigrain_common_state import (
    PhysicalTransferLaw, derive_physical_transfer_proposal,
    joint_material_transaction, reconstruct_multigrain_common,
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


@dataclass(frozen=True)
class EnergyQualifiedMechanicalResult:
    state: object
    operator_decisions: tuple[MultiGrainMechanicalDecision, ...]
    energy_balances: tuple[object, ...]
    subdivisions: int
    external_work_J: float
    internal_energy_change_J: float
    first_law_residual_J: float
    relative_first_law_residual: float


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
    # Temperature is a common Eulerian field, not a dormant grain history.
    # Owner kinetics deposits its local heat without conducting separately
    # across artificial owner-support discontinuities. Conduction is applied
    # once to the reconstructed common temperature below.
    local_parameters = replace(
        wall_parameters, thermal_diffusivity_m2_s=0.0, bath_rate_s=0.0)
    if driving.resolved_stress_Pa is None:
        mixture, _ = reconstruct_multigrain_common(
            state, wall_parameters.spacing_m)
        beta2 = np.asarray(mixture.beta_p)[..., :2, :2]
        eigenstrain = .5*(beta2+np.swapaxes(beta2, -1, -2))
        common_stress, _ = solve_periodic_eigenstrain(
            eigenstrain, np.asarray(driving.mean_strain, dtype=float),
            wall_parameters.spacing_m, wall_parameters.c11_Pa,
            wall_parameters.c12_Pa, wall_parameters.c44_Pa,
            iterations=wall_parameters.elastic_iterations)
    else:
        common_stress = None
    for support, owner in zip(state.supports, state.owners):
        active = np.asarray(support) > 256.0*np.finfo(float).eps
        if common_stress is None:
            owner_driving = driving
        else:
            _, directions, normals = rotated_system_fields(
                systems, owner.orientation_rad)
            schmid = .5*(
                np.einsum("...si,...sj->...sij", directions[..., :2],
                          normals[..., :2])
                +np.einsum("...si,...sj->...sij", normals[..., :2],
                           directions[..., :2]))
            owner_driving = CommonWallDriving(resolved_stress_Pa=np.einsum(
                "...ij,...sij->...s", common_stress, schmid))
        updated = owner
        remaining = dt
        substeps = 0
        suggested = remaining
        owner_work = owner_heat = 0.0
        while remaining > 64.0*np.finfo(float).eps*dt:
            attempted = min(remaining, suggested)
            local_retries = 0
            while True:
                try:
                    trial, residual, scale = accepted_euler_step(
                        updated, owner_driving, systems, topologies,
                        local_parameters, attempted, active_mask=active)
                    break
                except (ValueError, FloatingPointError):
                    attempted *= .5
                    local_retries += 1
                    if (local_retries > 64
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
            suggested = min(dt, max(attempted, consumed)*1.25)
            substeps += 1
            if substeps > 4096:
                raise RuntimeError("mechanical physical interval exceeded substep budget")
        owners.append(_masked_owner_update(owner, updated, active))
        work += owner_work
        heat += owner_heat
        maximum_substeps = max(maximum_substeps, substeps)
    common_temperature = sum(
        np.asarray(state.supports[index])*owner.temperature_K
        for index, owner in enumerate(owners))
    diffusivity = float(wall_parameters.thermal_diffusivity_m2_s)
    if diffusivity > 0.0:
        nx, ny = common_temperature.shape
        kx = 2*np.pi*np.fft.fftfreq(nx, d=wall_parameters.spacing_m)
        ky = 2*np.pi*np.fft.fftfreq(ny, d=wall_parameters.spacing_m)
        kx, ky = np.meshgrid(kx, ky, indexing="ij")
        spectrum = np.fft.fftn(common_temperature)
        common_temperature = np.real(np.fft.ifftn(
            np.exp(-diffusivity*(kx*kx+ky*ky)*dt)*spectrum))
    bath_rate = float(wall_parameters.bath_rate_s)
    if bath_rate > 0.0:
        decay = math.exp(-bath_rate*dt)
        common_temperature = (wall_parameters.bath_temperature_K
                              +decay*(common_temperature
                                      -wall_parameters.bath_temperature_K))
    owners = [replace(owner, temperature_K=common_temperature.copy())
              for owner in owners]
    candidate = replace(state, owners=tuple(owners))
    candidate.validate()
    decision = MultiGrainMechanicalDecision(
        True, minimum_scale, work, heat, maximum_line_residual,
        maximum_substeps, dt)
    return candidate, decision


def advance_energy_qualified_mechanics(
        state, *, mean_strain_before, mean_strain_candidate, systems,
        topologies, wall_parameters, dt_s, represented_thickness_m,
        energy_kwargs=None, maximum_relative_first_law_residual=.05,
        maximum_subdivisions=10):
    """Rollback and bisect a mechanical interval until every leaf qualifies."""
    strain0 = np.asarray(mean_strain_before, dtype=float)
    strain1 = np.asarray(mean_strain_candidate, dtype=float)

    def recurse(accepted, left, right, interval, depth):
        midpoint = .5*(left+right)
        candidate, operator = advance_multigrain_mechanics(
            accepted, driving=CommonWallDriving(mean_strain=midpoint),
            systems=systems, topologies=topologies,
            wall_parameters=wall_parameters, dt_s=interval,
            represented_thickness_m=represented_thickness_m)
        balance = evaluate_multigrain_mechanical_interval(
            accepted, candidate, mean_strain_before=left,
            mean_strain_candidate=right,
            spacing_m=wall_parameters.spacing_m,
            represented_thickness_m=represented_thickness_m,
            wall_parameters=wall_parameters, energy_kwargs=energy_kwargs)
        if balance.relative_first_law_residual <= float(
                maximum_relative_first_law_residual):
            return candidate, (operator,), (balance,), depth
        if depth >= int(maximum_subdivisions):
            raise RuntimeError(
                "mechanical complete-energy audit failed subdivision budget: "
                f"depth={depth}, interval_s={interval:.17g}, "
                f"relative_residual={balance.relative_first_law_residual:.17g}, "
                f"external_work_J={balance.external_work_J:.17g}, "
                f"internal_energy_change_J={balance.internal_energy_change_J:.17g}, "
                f"left_strain={np.asarray(left).tolist()}, "
                f"right_strain={np.asarray(right).tolist()}")
        middle = .5*(left+right)
        first_state, first_ops, first_balances, depth_a = recurse(
            accepted, left, middle, .5*interval, depth+1)
        final_state, second_ops, second_balances, depth_b = recurse(
            first_state, middle, right, .5*interval, depth+1)
        return (final_state, first_ops+second_ops,
                first_balances+second_balances, max(depth_a, depth_b))

    candidate, operators, balances, depth = recurse(
        state, strain0, strain1, float(dt_s), 0)
    work = sum(value.external_work_J for value in balances)
    delta_internal = (balances[-1].candidate.internal_J
                      -balances[0].before.internal_J)
    residual = delta_internal-work
    scale = max(abs(delta_internal), abs(work), 1e-300)
    relative = abs(residual)/scale
    if relative > float(maximum_relative_first_law_residual):
        raise RuntimeError(
            "subdivided mechanical interval fails cumulative first-law audit")
    return EnergyQualifiedMechanicalResult(
        candidate, operators, balances, depth, work, delta_internal,
        residual, relative)


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
