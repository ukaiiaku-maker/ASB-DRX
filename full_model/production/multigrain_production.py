"""Recurrent zero-pressure production driver for a shared multi-grain state."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import math

import numpy as np

from .arrhenius_kinetics import ActivatedProcess
from .complete_multigrain_energy import (
    MultiGrainDissipation, evaluate_joint_multigrain_transaction,
    evaluate_multigrain_mechanical_interval,
)
from .coupled_front_event import propose_bidirectional_front_event
from .moving_front import DefectState
from .common_tensorial_wall import (
    CommonWallDriving, accepted_euler_step, wall_residual,
)
from .nonlocal_elasticity import solve_periodic_eigenstrain
from .tensorial_nye import (
    nye_from_plastic_distortion, plastic_distortion_from_slip,
    rotated_system_fields, spectral_derivatives,
)
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
    temperature_override_K: float | None = None

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
        if (self.temperature_override_K is not None
                and (not math.isfinite(float(self.temperature_override_K))
                     or float(self.temperature_override_K) <= 0.0)):
            raise ValueError("front temperature override must be positive and finite")


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
    cumulative_mechanical_external_work_J: float = 0.0
    cumulative_mechanical_internal_energy_change_J: float = 0.0
    cumulative_mechanical_first_law_residual_J: float = 0.0
    maximum_mechanical_relative_first_law_residual: float = 0.0
    cumulative_front_first_law_residual_J: float = 0.0


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
    independent_pressure_by_interface_Pa: dict[str, float] = field(
        default_factory=dict)
    selected_velocity_by_interface_m_s: dict[str, float] = field(
        default_factory=dict)
    joint_pressure_factor: float = 1.0
    selected_rate_conjugate_to_recorded_force: bool = True


@dataclass(frozen=True)
class MultiGrainMechanicalDecision:
    accepted: bool
    minimum_step_scale: float
    external_plastic_work_J: float
    irreversible_heat_J: float
    maximum_line_balance_residual_m2_s: float
    maximum_substeps: int
    consumed_interval_s: float
    plastic_work_J_m3_cells: np.ndarray
    irreversible_heat_J_m3_cells: np.ndarray


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


def _masked_owner_update(before, after, active, systems, spacing_m):
    """Publish material history with a Curl-compatible derived Nye update.

    Masking an owner increment creates a product-rule Nye contribution at the
    edge of its support.  Simply masking the already-computed family Nye rate
    omits that contribution.  The correction below is fixed by the actual
    masked beta increment and is partitioned using the corresponding physical
    family slip increments; it is not inferred from a target wall pattern.
    """
    from dataclasses import fields
    from .common_tensorial_wall import CommonWallState
    arrays = {}
    for item in fields(CommonWallState):
        old = np.asarray(getattr(before, item.name))
        new = np.asarray(getattr(after, item.name))
        mask = active[(...,)+(None,)*(old.ndim-active.ndim)]
        arrays[item.name] = np.where(mask, new, old)
    mask = np.asarray(active, dtype=float)
    beta_increment = (np.asarray(after.beta_p)-np.asarray(before.beta_p))
    target_increment = nye_from_plastic_distortion(
        mask[..., None, None]*beta_increment, spacing_m)
    family_increment = (np.asarray(after.family_nye_m1)
                        -np.asarray(before.family_nye_m1))
    masked_family_increment = mask[..., None, None, None]*family_increment
    correction = target_increment-np.sum(masked_family_increment, axis=2)
    slip_increment = np.asarray(after.slip)-np.asarray(before.slip)
    family_weights = []
    for family in range(len(systems)):
        family_slip = np.zeros_like(slip_increment)
        family_slip[..., family] = slip_increment[..., family]
        family_beta = plastic_distortion_from_slip(
            family_slip, systems, before.orientation_rad)
        family_weights.append(np.linalg.norm(family_beta, axis=(-2, -1)))
    family_weights = np.stack(family_weights, axis=2)
    denominator = np.sum(family_weights, axis=2, keepdims=True)
    family_weights = np.divide(
        family_weights, denominator,
        out=np.full_like(family_weights, 1.0/len(systems)),
        where=denominator > 0.0)
    arrays["family_nye_m1"] = (
        np.asarray(before.family_nye_m1)+masked_family_increment
        +family_weights[..., None, None]*correction[..., None, :, :])
    return CommonWallState(**arrays)


def _owner_drivings_from_common_stress(
        state, driving, systems, wall_parameters):
    if driving.resolved_stress_Pa is not None:
        return tuple(replace(driving, material_support=support)
                     for support in state.supports)
    if driving.fixed_stress_tensor_Pa is not None:
        return tuple(replace(driving, material_support=support)
                     for support in state.supports)
    mixture, _ = reconstruct_multigrain_common(
        state, wall_parameters.spacing_m)
    beta2 = np.asarray(mixture.beta_p)[..., :2, :2]
    eigenstrain = .5*(beta2+np.swapaxes(beta2, -1, -2))
    common_stress, _ = solve_periodic_eigenstrain(
        eigenstrain, np.asarray(driving.mean_strain, dtype=float),
        wall_parameters.spacing_m, wall_parameters.c11_Pa,
        wall_parameters.c12_Pa, wall_parameters.c44_Pa,
        iterations=wall_parameters.elastic_iterations)
    owner_drivings = []
    for support in state.supports:
        owner_drivings.append(CommonWallDriving(
            fixed_stress_tensor_Pa=common_stress,
            material_support=np.asarray(support, dtype=float)))
    return tuple(owner_drivings)


def multigrain_instantaneous_dissipation_fields(
        state, *, driving, systems, topologies, wall_parameters):
    """Evaluate common-stress owner power/heat fields without advancing."""
    shape = state.supports.shape[1:]
    plastic = np.zeros(shape, dtype=float)
    heat = np.zeros(shape, dtype=float)
    free_energy = np.zeros(shape, dtype=float)
    owner_drivings = _owner_drivings_from_common_stress(
        state, driving, systems, wall_parameters)
    for support, owner, owner_driving in zip(
            state.supports, state.owners, owner_drivings):
        residual = wall_residual(
            owner, owner_driving, systems, topologies, wall_parameters)
        weight = np.asarray(support, dtype=float)
        plastic += weight*residual.plastic_power_W_m3
        heat += weight*residual.heat_rate_W_m3
        free_energy += weight*residual.free_energy_rate_W_m3
    return {"plastic_power_W_m3": plastic,
            "irreversible_heat_rate_W_m3": heat,
            "defect_free_energy_rate_W_m3": free_energy}


def advance_multigrain_mechanics(
        state, *, driving, systems, topologies, wall_parameters, dt_s,
        represented_thickness_m, maximum_internal_substeps=4096):
    """Advance each persistent material owner over the same physical interval.

    Only supported material evolves; dormant owner history is bitwise retained
    for later re-entry. Work and heat are integrated from the accepted common
    residual channels, not inferred from an endpoint energy difference.
    """
    dt = float(dt_s)
    cell_volume = (float(wall_parameters.spacing_m)**2
                   *float(represented_thickness_m))
    work = heat = 0.0
    plastic_work_density = np.zeros(state.supports.shape[1:], dtype=float)
    irreversible_heat_density = np.zeros_like(plastic_work_density)
    minimum_scale = 1.0
    maximum_line_residual = 0.0
    substeps = 0
    # All owners advance on one accepted clock.  Re-equilibrating the common
    # stress after each microstep supplies the elastic unloading that bounds a
    # finite-rate burst and keeps evolving crystal frames work-conjugate.
    local_parameters = replace(
        wall_parameters, thermal_diffusivity_m2_s=0.0, bath_rate_s=0.0)
    updated_state = state
    remaining = dt
    suggested = remaining
    diffusivity = float(wall_parameters.thermal_diffusivity_m2_s)
    bath_rate = float(wall_parameters.bath_rate_s)
    nx, ny = state.supports.shape[1:]
    kx = 2*np.pi*np.fft.fftfreq(nx, d=wall_parameters.spacing_m)
    ky = 2*np.pi*np.fft.fftfreq(ny, d=wall_parameters.spacing_m)
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    while remaining > 64.0*np.finfo(float).eps*dt:
        attempted = min(remaining, suggested)
        local_retries = 0
        while True:
            try:
                owner_drivings = _owner_drivings_from_common_stress(
                    updated_state, driving, systems, wall_parameters)
                trials = [accepted_euler_step(
                    owner, owner_driving, systems, topologies,
                    local_parameters, attempted,
                    active_mask=(np.asarray(support)
                                 >256.0*np.finfo(float).eps))
                    for support, owner, owner_driving in zip(
                        updated_state.supports, updated_state.owners,
                        owner_drivings)]
                break
            except (ValueError, FloatingPointError):
                attempted *= .5
                local_retries += 1
                if (local_retries > 64
                        or attempted <= 64.0*np.finfo(float).eps*dt):
                    raise RuntimeError(
                        "mechanical interval could not find an admissible substep")
        scale = min(float(item[2]) for item in trials)
        consumed = attempted*scale
        if consumed <= 0.0:
            raise RuntimeError("mechanical interval made no physical-time progress")
        owners = []
        heat_increment = np.zeros(state.supports.shape[1:], dtype=float)
        for support, owner, (_, residual, _) in zip(
                updated_state.supports, updated_state.owners, trials):
            weight = np.asarray(support, dtype=float)
            work_increment = weight*residual.plastic_power_W_m3*consumed
            owner_heat_increment = weight*residual.heat_rate_W_m3*consumed
            plastic_work_density += work_increment
            irreversible_heat_density += owner_heat_increment
            heat_increment += owner_heat_increment
            work += float(np.sum(work_increment, dtype=np.longdouble)
                          *cell_volume)
            heat += float(np.sum(owner_heat_increment, dtype=np.longdouble)
                          *cell_volume)
            line_residual = residual.channel_rates_m2_s.get(
                "line_balance_residual_m2_s", 0.0)
            maximum_line_residual = max(
                maximum_line_residual,
                float(np.max(np.abs(line_residual))))
            active = weight > 256.0*np.finfo(float).eps
            arrays = {}
            for item in fields(owner):
                value = np.asarray(getattr(owner, item.name))
                rate = np.asarray(getattr(residual.state_rate, item.name))
                mask = active[(...,)+(None,)*(value.ndim-active.ndim)]
                arrays[item.name] = np.where(
                    mask, value+consumed*rate, value)
            raw_trial = type(owner)(**arrays)
            raw_trial = replace(
                raw_trial, temperature_K=np.asarray(owner.temperature_K).copy())
            owners.append(_masked_owner_update(
                owner, raw_trial, active, systems,
                wall_parameters.spacing_m))
        common_temperature = sum(
            np.asarray(updated_state.supports[index])
            *owner.temperature_K
            for index, owner in enumerate(updated_state.owners))
        common_temperature += (
            heat_increment/wall_parameters.volumetric_heat_capacity_J_m3_K)
        if diffusivity > 0.0:
            common_temperature = np.real(np.fft.ifftn(
                np.exp(-diffusivity*(kx*kx+ky*ky)*consumed)
                *np.fft.fftn(common_temperature)))
        if bath_rate > 0.0:
            decay = math.exp(-bath_rate*consumed)
            common_temperature = (
                wall_parameters.bath_temperature_K
                +decay*(common_temperature-wall_parameters.bath_temperature_K))
        if np.any(~np.isfinite(common_temperature)) or np.any(
                common_temperature <= 0.0):
            raise ValueError("common heat update produced nonpositive temperature")
        owners = [replace(owner, temperature_K=common_temperature.copy())
                  for owner in owners]
        updated_state = replace(updated_state, owners=tuple(owners))
        updated_state.validate()
        remaining -= consumed
        suggested = min(dt, max(consumed, attempted*scale)*1.25)
        minimum_scale = min(minimum_scale, scale)
        substeps += 1
        if substeps > int(maximum_internal_substeps):
            raise RuntimeError("mechanical physical interval exceeded substep budget")
    candidate = updated_state
    candidate.validate()
    decision = MultiGrainMechanicalDecision(
        True, minimum_scale, work, heat, maximum_line_residual,
        substeps, dt, plastic_work_density,
        irreversible_heat_density)
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
        def bisect_interval():
            middle = .5*(left+right)
            first_state, first_ops, first_balances, depth_a = recurse(
                accepted, left, middle, .5*interval, depth+1)
            final_state, second_ops, second_balances, depth_b = recurse(
                first_state, middle, right, .5*interval, depth+1)
            return (final_state, first_ops+second_ops,
                    first_balances+second_balances, max(depth_a, depth_b))

        midpoint = .5*(left+right)
        try:
            candidate, operator = advance_multigrain_mechanics(
                accepted, driving=CommonWallDriving(mean_strain=midpoint),
                systems=systems, topologies=topologies,
                wall_parameters=wall_parameters, dt_s=interval,
                represented_thickness_m=represented_thickness_m,
                # This is a trial inside an outer rollback/bisection
                # controller.  Fail early rather than spending thousands of
                # constitutive substeps proving that a coarse trial is stiff.
                maximum_internal_substeps=128)
        except RuntimeError as error:
            adaptive_failure = any(text in str(error) for text in (
                "mechanical physical interval exceeded substep budget",
                "mechanical interval could not find an admissible substep",
            ))
            if not adaptive_failure or depth >= int(maximum_subdivisions):
                raise
            return bisect_interval()
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
        return bisect_interval()

    candidate, operators, balances, depth = recurse(
        state, strain0, strain1, float(dt_s), 0)
    work = sum(value.external_work_J for value in balances)
    # Sum the cancellation-stable leaf evaluations.  Re-subtracting the two
    # O(1e-11 J) endpoint totals here would discard the accuracy recovered by
    # the exact quadratic elastic identity used in each accepted leaf.
    delta_internal = sum(value.internal_energy_change_J
                         for value in balances)
    residual = delta_internal-work
    roundoff_floor = (4096.0*np.finfo(float).eps
                      *max(abs(balances[0].before.internal_J),
                           abs(balances[-1].candidate.internal_J), 1e-300))
    scale = max(abs(delta_internal), abs(work), roundoff_floor, 1e-300)
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


def _geometric_sweep_weight(state, donor_index, receiver_index, spacing_m):
    """Return the dimensionless level-set measure for one directed pair.

    ``fraction * weight`` integrates to the area swept by a contour moving
    ``fraction`` grid spacings. Pair contact partitions a donor contour at
    junctions without replacing its geometric ``|grad eta|`` measure.
    """
    donor = np.asarray(state.supports[donor_index], dtype=float)
    gx, gy = spectral_derivatives(donor, float(spacing_m))
    contour = float(spacing_m)*np.sqrt(gx*gx+gy*gy)
    contacts = np.stack([
        (_contact_weight(state, donor_index, other)
         if other != donor_index else np.zeros_like(donor))
        for other in range(len(state.owners))])
    total_contact = np.sum(contacts, axis=0)
    share = np.divide(
        contacts[receiver_index], total_contact,
        out=np.zeros_like(donor), where=total_contact > 1e-15)
    return contour*share


def _virtual_channel(state, interface, donor_id, receiver_id, *, kinetics,
                     spacing_m, represented_thickness_m, wall_parameters,
                     energy_kwargs, interval_s, systems):
    index = {grain_id: position for position, grain_id in enumerate(state.grain_ids)}
    weight = _geometric_sweep_weight(
        state, index[donor_id], index[receiver_id], spacing_m)
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
    temperature = (float(kinetics.temperature_override_K)
                   if kinetics.temperature_override_K is not None else
                   float(np.mean([
                       np.mean(owner.temperature_K) for owner in state.owners])))
    directions = {}
    pressures = {}
    selected_velocities = {}
    base_requests = {}
    channel_records = {}
    directional_channels = {}
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
        directional_channels[interface.component_id] = (
            interface,
            0.0 if ab is None else max(-ab[0]/ab[1], 0.0),
            0.0 if ba is None else max(-ba[0]/ba[1], 0.0),
            weight_ab, weight_ba)
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
        donor_index = index[donor]
        receiver_index = index[receiver]
        request = fraction*_geometric_sweep_weight(
            state, donor_index, receiver_index, spacing_m)
        # Preserve the same geometric direction used by the virtual price.
        # Donor support and competing outgoing channels are resolved jointly
        # by ``joint_material_transaction`` at every trial amplitude.  Clipping
        # here would permanently reshape the direction before backtracking,
        # so the finite event would no longer be conjugate to its price after
        # a boundary had swept through low-support cells.
        pressure = max(-virtual[0]/virtual[1], 0.0)
        if pressure <= 0.0 or not np.any(request > 0.0):
            continue
        directions[interface.component_id] = direction
        pressures[interface.component_id] = pressure
        selected_velocities[interface.component_id] = float(velocity)
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

    # Individual edge derivatives do not in general add at a diffuse triple
    # junction: all proposals act on the same partition-of-unity supports and
    # physical boundary reservoirs.  Price the selected *joint direction* at
    # an independently prescribed virtual amplitude, then apply one scalar
    # correction to the edge pressures.  This remains a directional
    # derivative evaluated before the realized event; it is not inferred from
    # the finite-event closure residual.
    # Price the *actual kinetic direction*: interface velocities generally
    # differ, so a bundle assigning the same virtual contour fraction to every
    # edge is not collinear with the realized request.  A single pressure
    # correction obtained from that different direction is non-conjugate once
    # several interfaces are active.  Scale the complete realized request to
    # a small virtual amplitude instead; donor competition is still handled
    # jointly and the direction is unchanged.
    cell_volume = float(spacing_m)**2*float(represented_thickness_m)

    def requests_at_force_scale(scale):
        trial_directions = {}; trial_pressures = {}; trial_requests = {}
        trial_records = {}; trial_velocities = {}
        for key, (interface, pressure_ab, pressure_ba,
                  weight_ab, weight_ba) in directional_channels.items():
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
                kinetic_free_energy_a_to_b_J=(
                    -float(scale)*pressure_ab*kinetics.event_volume_m3),
                kinetic_free_energy_b_to_a_J=(
                    -float(scale)*pressure_ba*kinetics.event_volume_m3),
                actual_reverse_edge=False,
                deterministic_rate_law="complete_dissipation")
            velocity = event.net_velocity_a_to_b_m_s
            if velocity == 0.0:
                continue
            if velocity > 0.0:
                donor = interface.grain_a_id; receiver = interface.grain_b_id
                weight = weight_ab; direction = "a_to_b"; pressure = pressure_ab
            else:
                donor = interface.grain_b_id; receiver = interface.grain_a_id
                weight = weight_ba; direction = "b_to_a"; pressure = pressure_ba
            fraction = min(abs(velocity)*dt/float(spacing_m),
                           kinetics.maximum_fraction_per_step)
            request = fraction*weight
            if pressure <= 0.0 or not np.any(request > 0.0):
                continue
            trial_directions[key] = direction
            trial_pressures[key] = pressure
            trial_requests[key] = request
            trial_records[key] = (interface, donor, receiver)
            trial_velocities[key] = float(velocity)
        return (trial_directions, trial_pressures, trial_requests,
                trial_records, trial_velocities)

    def complete_direction_factor(requests, records, unscaled_pressures):
        maximum_request = max(
            float(np.max(request)) for request in requests.values())
        virtual_scale = min(
            1.0, kinetics.virtual_fraction/max(maximum_request, 1e-300))
        proposals = []
        for key in requests:
            interface, donor, receiver = records[key]
            proposals.append(derive_physical_transfer_proposal(
                state, interface_id=interface.component_id, donor_id=donor,
                receiver_id=receiver,
                requested_fraction=virtual_scale*requests[key],
                law=kinetics.transfer_law, interval_s=dt, systems=systems))
        capacity = joint_material_transaction(state, tuple(proposals))
        price = evaluate_joint_multigrain_transaction(
            state, capacity, spacing_m=spacing_m,
            represented_thickness_m=represented_thickness_m,
            wall_parameters=wall_parameters, interval_s=dt,
            dissipation=MultiGrainDissipation(), energy_kwargs=energy_kwargs,
            absolute_tolerance_J=0.0).decision.available_change_J
        independent_work = sum(
            unscaled_pressures[key]
            *float(np.sum(extent, dtype=np.longdouble))*cell_volume
            for key, extent in capacity.accepted_fraction_by_interface.items())
        return (-price/independent_work
                if price < 0.0 and independent_work > 0.0 else 1.0)

    # The complete joint directional derivative changes the force supplied to
    # the nonlinear EXP-floor law.  Reusing velocities selected before that
    # correction is formally nonconjugate.  Solve the scalar projected-force
    # fixed point so the published rate is generated by the same corrected
    # force used for mobility dissipation.
    joint_pressure_factor = 1.0
    force_rate_converged = False
    for _ in range(24):
        (directions, independent_pressures, base_requests,
         channel_records, selected_velocities) = requests_at_force_scale(
             joint_pressure_factor)
        if not base_requests:
            break
        updated_factor = complete_direction_factor(
            base_requests, channel_records, independent_pressures)
        relative = abs(updated_factor-joint_pressure_factor)/max(
            abs(updated_factor), 1e-300)
        joint_pressure_factor = float(updated_factor)
        if relative <= 1e-10:
            force_rate_converged = True
            break
    (directions, independent_pressures, base_requests,
     channel_records, selected_velocities) = requests_at_force_scale(
         joint_pressure_factor)
    if base_requests:
        verification_factor = complete_direction_factor(
            base_requests, channel_records, independent_pressures)
        force_rate_converged = bool(force_rate_converged and np.isclose(
            verification_factor, joint_pressure_factor,
            rtol=1e-10, atol=1e-12))
    if not force_rate_converged:
        raise RuntimeError("joint front force-rate fixed point did not converge")
    pressures = {key: value*joint_pressure_factor
                 for key, value in independent_pressures.items()}

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
            next_ledger.maximum_relative_energy_closure, relative_closure),
        cumulative_front_first_law_residual_J=(
            next_ledger.cumulative_front_first_law_residual_J
            +result.decision.first_law_residual_J))
    return (result.published_state, replace(runtime, ledger=next_ledger),
            MultiGrainProductionDecision(
                True, result.decision.classification, directions, pressures,
                requested, accepted_fractions, result.decision, backtrack,
                independent_pressures, selected_velocities,
                float(joint_pressure_factor), force_rate_converged))


def advance_multigrain_front_interval(
        state, runtime, *, kinetics, dt_s, maximum_substep_s, spacing_m,
        represented_thickness_m, wall_parameters, energy_kwargs=None,
        applied_shear_rate_s=0.0, systems=None):
    """Advance a physical interval with recomputed finite front increments.

    ``maximum_fraction_per_step`` is a local contour-CFL bound, not a physical
    mobility.  Applying it once per caller step therefore makes the migration
    distance depend on that caller step.  This controller divides the physical
    interval into equal subintervals no longer than ``maximum_substep_s`` and
    recomputes affinity, mobility, competition, and dissipation after every
    accepted atomic transaction.  The runtime ledger is advanced by the
    subintervals themselves, so its accumulated time and applied strain remain
    the exact requested interval.

    A rejected atomic transaction is retained in the returned audit trail and
    does not erase earlier accepted subintervals.  This is intentionally not a
    retry with inferred heat or relaxed energy closure.
    """
    dt = float(dt_s)
    maximum = float(maximum_substep_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("production interval must be positive and finite")
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("front maximum substep must be positive and finite")
    ratio = dt/maximum
    # Suppress a spurious extra substep when a decimal input lies only a few
    # ulps above an integer ratio.  Equal subdivision then closes time exactly.
    count = max(1, int(math.ceil(ratio-16.0*np.finfo(float).eps*max(ratio, 1.0))))
    sub_dt = dt/count
    decisions = []
    evolved = state
    evolved_runtime = runtime
    for _ in range(count):
        evolved, evolved_runtime, decision = advance_multigrain_front(
            evolved, evolved_runtime, kinetics=kinetics, dt_s=sub_dt,
            spacing_m=spacing_m,
            represented_thickness_m=represented_thickness_m,
            wall_parameters=wall_parameters, energy_kwargs=energy_kwargs,
            applied_shear_rate_s=applied_shear_rate_s, systems=systems)
        decisions.append(decision)
    return evolved, evolved_runtime, tuple(decisions)
