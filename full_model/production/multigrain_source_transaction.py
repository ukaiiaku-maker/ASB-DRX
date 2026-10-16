"""Complete-energy Hall--Petch pair-source transaction for common owners."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .arrhenius_dislocation_sources import apply_conjugate_pair_source_event
from .arrhenius_kinetics import (
    ActivatedProcess, EV_J, activated_rate_array_s, exp_floor_enthalpy_j,
)
from .common_tensorial_wall import (
    CommonWallDriving, resolved_driving_components,
    wall_free_energy_derivatives, wall_total_free_energy_density_J_m3,
)
from .complete_multigrain_energy import evaluate_complete_multigrain_energy
from .multigrain_common_state import reconstruct_multigrain_common
from .multigrain_production import (
    _masked_owner_update, _owner_drivings_from_common_stress,
)


@dataclass(frozen=True)
class MultiGrainSourceDecision:
    accepted: bool
    classification: str
    candidate_state: object
    event_count: float
    created_line_cell_sum_m2: float
    signed_content_change_m2: float
    line_balance_residual_m2: float
    resolved_work_J: float
    defect_storage_change_J: float
    generated_heat_J: float
    delta_helmholtz_before_heat_J: float
    internal_energy_change_J: float
    first_law_residual_J: float
    relative_first_law_residual: float
    energy_resolution_floor_J: float
    heat_source_J_by_cell: np.ndarray
    subdivisions: int = 0


@dataclass(frozen=True)
class NeutralBoundarySinkParameters:
    """Thermal neutral-pair annihilation, optionally assisted by stress."""

    activation_enthalpy_eV: float = 1.40
    attempt_frequency_s: float = 1.0e12
    critical_stress_Pa: float = 1.0e9
    exp_a: float = 1.0
    exp_n: float = 2.0
    barrier_floor_fraction: float = 0.05/1.40
    maximum_fraction_per_interval: float = 0.10
    stress_assisted: bool = False
    complete_physical_clock: bool = False

    def __post_init__(self):
        values = (self.activation_enthalpy_eV, self.attempt_frequency_s,
                  self.critical_stress_Pa, self.exp_a, self.exp_n,
                  self.maximum_fraction_per_interval)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("neutral sink parameters must be positive finite")
        if self.exp_n < 1.0 or not 0.0 <= self.barrier_floor_fraction <= 1.0:
            raise ValueError("invalid neutral sink EXP-floor shape")
        if self.maximum_fraction_per_interval > 1.0:
            raise ValueError("neutral sink fraction cannot exceed one")


@dataclass(frozen=True)
class MultiGrainSinkDecision:
    accepted: bool
    classification: str
    candidate_state: object
    annihilated_pair_cell_sum_m2: float
    signed_content_change_m2: float
    line_balance_residual_m2: float
    defect_storage_change_J: float
    generated_heat_J: float
    first_law_residual_J: float
    relative_first_law_residual: float
    maximum_fraction: float
    heat_source_J_by_cell: np.ndarray


def evaluate_multigrain_pair_source_transaction(
        state, *, mean_strain, site_field, interval_s, source_parameters,
        systems, wall_parameters, represented_thickness_m,
        energy_kwargs=None, source_mode="unspecified",
        maximum_relative_first_law_residual=.05):
    """Propose and qualify one coupled neutral-pair/source-slip event.

    The event is performed at fixed total strain, so external boundary work is
    zero. ``tau * delta_gamma`` is an independently evaluated release of
    recoverable elastic work. Line-storage increase is subtracted locally; the
    nonnegative remainder is deposited as heat. The complete common energy is
    then evaluated without using its residual to define that heat.
    """
    dt = float(interval_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("source interval must be positive and finite")
    site = np.asarray(site_field, dtype=float)
    if site.shape != state.supports.shape[1:]:
        raise ValueError("source site field has wrong grid")
    spacing = float(wall_parameters.spacing_m)
    thickness = float(represented_thickness_m)
    cell_volume = spacing*spacing*thickness
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    options.pop("mean_strain", None)
    common_energy = dict(
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall_parameters, mean_strain=np.asarray(mean_strain),
        **options)
    before_energy = evaluate_complete_multigrain_energy(state, **common_energy)
    drivings = _owner_drivings_from_common_stress(
        state, CommonWallDriving(mean_strain=np.asarray(mean_strain)),
        systems, wall_parameters)
    proposed_owners = []
    ledgers = []
    work_density = np.zeros(site.shape)
    defect_change_density = np.zeros(site.shape)
    for support, owner, driving in zip(state.supports, state.owners, drivings):
        components = resolved_driving_components(
            owner, driving, systems, (), wall_parameters)
        chemical = wall_free_energy_derivatives(
            owner, wall_parameters, (), systems)
        # A source opportunity becomes an irreversible expanding event only
        # where its conjugate slip work pays the first-variation cost of the
        # positive/negative line pair.  The Arrhenius Hall--Petch hazard still
        # sets exposure on those sites; this mask prevents a deterministic
        # expected-value update from inventing stored line energy at a
        # thermodynamically inaccessible quenched source.
        work_per_event_J_m3 = (
            np.abs(components["raw_stress_Pa"])
            *source_parameters.burgers_m/source_parameters.grain_size_m)
        storage_per_event_J_m3 = (
            chemical["mobile_plus_mu_J_m"]
            +chemical["mobile_minus_mu_J_m"]
        )/source_parameters.grain_size_m**2
        accessible_family = work_per_event_J_m3 >= storage_per_event_J_m3
        proposed, ledger = apply_conjugate_pair_source_event(
            owner, site, components["raw_stress_Pa"], dt, systems,
            source_parameters, spacing_m=spacing, source_mode=source_mode,
            orientation_spin_weight=wall_parameters.orientation_spin_weight,
            active_mask=np.asarray(support) > 256*np.finfo(float).eps,
            active_family_mask=accessible_family)
        proposed = _masked_owner_update(
            owner, proposed,
            np.asarray(support) > 256*np.finfo(float).eps,
            systems, spacing)
        proposed_owners.append(proposed)
        ledgers.append(ledger)
        weight = np.asarray(support)
        work_density += weight*ledger.resolved_work_density_J_m3
        before_defect = wall_total_free_energy_density_J_m3(
            owner, wall_parameters, (), systems)
        after_defect = wall_total_free_energy_density_J_m3(
            proposed, wall_parameters, (), systems)
        defect_change_density += weight*(after_defect-before_defect)

    unheated = replace(state, owners=tuple(proposed_owners))
    unheated.validate()
    unheated_energy = evaluate_complete_multigrain_energy(
        unheated, **common_energy)
    raw_heat_density = work_density-defect_change_density
    local_scale = np.maximum.reduce((
        np.abs(work_density), np.abs(defect_change_density),
        np.ones_like(work_density)))
    # Source work is localized at the slip event whereas its compatible Nye
    # increment and recoverable elastic response are nonlocal.  Requiring the
    # work-minus-storage remainder to be nonnegative in every neighboring cell
    # wrongly rejects sparse sources even when the independently evaluated
    # global transaction is strongly dissipative.  Conserve that global
    # remainder and deposit it over the nonnegative resolved-work measure.
    raw_heat_sum = float(np.sum(raw_heat_density, dtype=np.longdouble))
    heat_tolerance = 1e-12*float(np.sum(local_scale, dtype=np.longdouble))
    local_heat_valid = bool(raw_heat_sum >= -heat_tolerance)
    positive_work = np.maximum(work_density, 0.0)
    work_sum = float(np.sum(positive_work, dtype=np.longdouble))
    if local_heat_valid and raw_heat_sum > 0.0 and work_sum > 0.0:
        heat_density = raw_heat_sum*positive_work/work_sum
    else:
        heat_density = np.zeros_like(raw_heat_density)
    common_before, _ = reconstruct_multigrain_common(unheated, spacing)
    temperature = (np.asarray(common_before.temperature_K)
                   +heat_density/wall_parameters.volumetric_heat_capacity_J_m3_K)
    heated_owners = tuple(
        replace(owner, temperature_K=temperature.copy())
        for owner in unheated.owners)
    candidate = replace(unheated, owners=heated_owners)
    candidate.validate()
    candidate_energy = evaluate_complete_multigrain_energy(
        candidate, **common_energy)

    work_J = float(np.sum(work_density, dtype=np.longdouble)*cell_volume)
    defect_J = float(np.sum(
        defect_change_density, dtype=np.longdouble)*cell_volume)
    heat_J = float(np.sum(heat_density, dtype=np.longdouble)*cell_volume)
    delta_helmholtz = unheated_energy.helmholtz_J-before_energy.helmholtz_J
    delta_internal = candidate_energy.internal_J-before_energy.internal_J
    residual = delta_internal  # fixed total strain: external boundary work is zero
    scale = max(abs(work_J), abs(defect_J), abs(heat_J),
                abs(delta_helmholtz), 1e-300)
    resolution_floor = (4096.0*np.finfo(float).eps
                        *max(abs(before_energy.internal_J),
                             abs(candidate_energy.internal_J), 1e-300))
    resolved = scale >= resolution_floor
    relative = abs(residual)/scale
    event_count = float(sum(np.sum(item.events_by_family, dtype=np.longdouble)
                            for item in ledgers))
    line_change = float(sum(item.line_content_change_m2 for item in ledgers))
    signed_change = float(sum(item.signed_content_change_m2 for item in ledgers))
    line_residual = float(sum(item.line_balance_residual_m2 for item in ledgers))
    downhill = delta_helmholtz <= 1e-12*scale
    accepted = bool(
        resolved and local_heat_valid and downhill
        and relative <= float(maximum_relative_first_law_residual))
    if not resolved:
        classification = "INACTIVE_BELOW_ENERGY_RESOLUTION"
    elif not local_heat_valid:
        classification = "REJECTED_INSUFFICIENT_RESOLVED_WORK"
    elif not downhill:
        classification = "REJECTED_UPHILL_COMPLETE_HELMHOLTZ"
    elif relative > float(maximum_relative_first_law_residual):
        classification = "REJECTED_FIRST_LAW_CLOSURE"
    else:
        classification = "ACCEPTED_CONJUGATE_PAIR_SOURCE"
    return MultiGrainSourceDecision(
        accepted, classification, candidate if accepted else state,
        event_count, line_change, signed_change, line_residual,
        work_J, defect_J, heat_J, delta_helmholtz, delta_internal,
        residual, relative, resolution_floor, heat_density*cell_volume)


def advance_energy_qualified_pair_source_transaction(
        state, *, maximum_subdivisions=6, **kwargs):
    """Apply the source transaction with rollback and physical-time bisection.

    Sparse source fields can concentrate an otherwise admissible interval
    measure into a large simultaneous burst.  Bisection represents the same
    activated hazard as sequential events at fixed imposed strain.  Every leaf
    is independently accepted by :func:`evaluate_multigrain_pair_source_transaction`;
    no rejected leaf is published and the summed leaf durations equal the
    requested physical interval exactly.
    """
    total_dt = float(kwargs["interval_s"])

    def recurse(accepted, interval, depth):
        trial = evaluate_multigrain_pair_source_transaction(
            accepted, **dict(kwargs, interval_s=interval))
        if trial.accepted or trial.classification.startswith("INACTIVE"):
            return trial
        recoverable = trial.classification in {
            "REJECTED_INSUFFICIENT_RESOLVED_WORK",
            "REJECTED_FIRST_LAW_CLOSURE",
            "REJECTED_UPHILL_COMPLETE_HELMHOLTZ",
        }
        if not recoverable or depth >= int(maximum_subdivisions):
            return trial
        left = recurse(accepted, .5*interval, depth+1)
        if not left.accepted:
            return left
        right = recurse(left.candidate_state, .5*interval, depth+1)
        if not right.accepted:
            return right
        scale = max(
            abs(left.resolved_work_J)+abs(right.resolved_work_J),
            abs(left.defect_storage_change_J)+abs(right.defect_storage_change_J),
            abs(left.generated_heat_J)+abs(right.generated_heat_J), 1e-300)
        residual = left.first_law_residual_J+right.first_law_residual_J
        return MultiGrainSourceDecision(
            True, "ACCEPTED_CONJUGATE_PAIR_SOURCE_SUBDIVIDED",
            right.candidate_state,
            left.event_count+right.event_count,
            left.created_line_cell_sum_m2+right.created_line_cell_sum_m2,
            left.signed_content_change_m2+right.signed_content_change_m2,
            left.line_balance_residual_m2+right.line_balance_residual_m2,
            left.resolved_work_J+right.resolved_work_J,
            left.defect_storage_change_J+right.defect_storage_change_J,
            left.generated_heat_J+right.generated_heat_J,
            left.delta_helmholtz_before_heat_J+right.delta_helmholtz_before_heat_J,
            left.internal_energy_change_J+right.internal_energy_change_J,
            residual, abs(residual)/scale,
            left.energy_resolution_floor_J+right.energy_resolution_floor_J,
            np.asarray(left.heat_source_J_by_cell)
            +np.asarray(right.heat_source_J_by_cell),
            max(left.subdivisions, right.subdivisions, depth+1))

    result = recurse(state, total_dt, 0)
    return result


def evaluate_multigrain_neutral_sink_transaction(
        state, *, mean_strain, site_field, interval_s, sink_parameters,
        systems, wall_parameters, represented_thickness_m,
        energy_kwargs=None, sink_mode="gb_tj_neutral_pair_sink",
        maximum_relative_first_law_residual=.05):
    """Energy-complete opposite-sign annihilation at boundary or control sites.

    Equal positive/negative mobile content is removed family by family, so the
    process has no net Burgers or atom-count source.  It represents a GB/TJ
    annihilation sink, not absorption into an untracked structural boundary.
    The independently evaluated defect-energy release is deposited locally as
    heat; no external work or unlimited vacancy reservoir is invoked.
    """
    dt = float(interval_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("sink interval must be positive and finite")
    site = np.asarray(site_field, dtype=float)
    if (site.shape != state.supports.shape[1:] or np.any(~np.isfinite(site))
            or np.any(site < 0.0) or np.mean(site) <= 0.0):
        raise ValueError("sink site field must be positive finite on the grid")
    site = site/np.mean(site)
    spacing = float(wall_parameters.spacing_m)
    thickness = float(represented_thickness_m)
    cell_volume = spacing*spacing*thickness
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    options.pop("mean_strain", None)
    common_energy = dict(
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall_parameters, mean_strain=np.asarray(mean_strain),
        **options)
    before_energy = evaluate_complete_multigrain_energy(state, **common_energy)
    drivings = _owner_drivings_from_common_stress(
        state, CommonWallDriving(mean_strain=np.asarray(mean_strain)),
        systems, wall_parameters)
    process = ActivatedProcess(
        "neutral-boundary-pair-sink", sink_parameters.attempt_frequency_s,
        0.0, sink_parameters.attempt_frequency_s)
    proposed_owners = []
    defect_drop_density = np.zeros(site.shape)
    annihilated_total = 0.0
    signed_change = 0.0
    line_change = 0.0
    line_inventory_scale = 0.0
    maximum_fraction = 0.0
    for support, owner, driving in zip(state.supports, state.owners, drivings):
        components = resolved_driving_components(
            owner, driving, systems, (), wall_parameters)
        stress = (np.abs(np.asarray(components["raw_stress_Pa"]))
                  if sink_parameters.stress_assisted
                  else np.zeros_like(components["raw_stress_Pa"]))
        enthalpy = exp_floor_enthalpy_j(
            stress, sink_parameters.activation_enthalpy_eV*EV_J,
            sink_parameters.critical_stress_Pa, sink_parameters.exp_a,
            sink_parameters.exp_n, sink_parameters.barrier_floor_fraction)
        rate = activated_rate_array_s(
            process, enthalpy, np.asarray(owner.temperature_K)[..., None])
        fraction = np.minimum(
            -np.expm1(-rate*dt*site[..., None]),
            sink_parameters.maximum_fraction_per_interval)
        active = np.asarray(support) > 256*np.finfo(float).eps
        fraction = np.where(active[..., None], fraction, 0.0)
        neutral = np.minimum(
            np.asarray(owner.mobile_plus_m2),
            np.asarray(owner.mobile_minus_m2))*fraction
        proposed = replace(
            owner,
            mobile_plus_m2=np.asarray(owner.mobile_plus_m2)-neutral,
            mobile_minus_m2=np.asarray(owner.mobile_minus_m2)-neutral)
        proposed.validate(systems, ())
        proposed_owners.append(proposed)
        weight = np.asarray(support)
        before_defect = wall_total_free_energy_density_J_m3(
            owner, wall_parameters, (), systems)
        after_defect = wall_total_free_energy_density_J_m3(
            proposed, wall_parameters, (), systems)
        defect_drop_density += weight*np.maximum(before_defect-after_defect, 0.0)
        annihilated_total += float(np.sum(
            weight[..., None]*neutral, dtype=np.longdouble))
        signed_change += float(np.sum(weight[..., None]*(
            (proposed.mobile_plus_m2-proposed.mobile_minus_m2)
            -(owner.mobile_plus_m2-owner.mobile_minus_m2)),
            dtype=np.longdouble))
        line_change += float(np.sum(weight[..., None]*(
            proposed.mobile_plus_m2+proposed.mobile_minus_m2
            -owner.mobile_plus_m2-owner.mobile_minus_m2),
            dtype=np.longdouble))
        maximum_fraction = max(maximum_fraction, float(np.max(fraction)))
        line_inventory_scale += float(np.sum(
            weight[..., None]*(np.asarray(owner.mobile_plus_m2)
                              +np.asarray(owner.mobile_minus_m2)),
            dtype=np.longdouble))
    unheated = replace(state, owners=tuple(proposed_owners))
    unheated.validate()
    cold_energy = evaluate_complete_multigrain_energy(unheated, **common_energy)
    heat_density = defect_drop_density
    heat_by_cell = heat_density*cell_volume
    common, _ = reconstruct_multigrain_common(unheated, spacing)
    temperature = (np.asarray(common.temperature_K)
                   +heat_density/wall_parameters.volumetric_heat_capacity_J_m3_K)
    candidate = replace(unheated, owners=tuple(
        replace(owner, temperature_K=temperature.copy())
        for owner in unheated.owners))
    candidate.validate()
    after_energy = evaluate_complete_multigrain_energy(candidate, **common_energy)
    defect_change = cold_energy.helmholtz_J-before_energy.helmholtz_J
    heat_J = float(np.sum(heat_by_cell, dtype=np.longdouble))
    residual = after_energy.internal_J-before_energy.internal_J
    scale = max(abs(defect_change), abs(heat_J), 1e-300)
    relative = abs(residual)/scale
    balance = line_change+2.0*annihilated_total
    balance_tolerance = max(
        1e-12*max(2.0*annihilated_total, 1.0),
        4096.0*np.finfo(float).eps*max(line_inventory_scale, 1.0))
    downhill = defect_change <= 1e-12*scale
    accepted = bool(
        maximum_fraction > 0.0 and downhill
        and relative <= maximum_relative_first_law_residual
        and abs(signed_change) <= balance_tolerance
        and abs(balance) <= balance_tolerance)
    if maximum_fraction == 0.0:
        classification = "INACTIVE_NEUTRAL_SINK"
    elif not downhill:
        classification = "REJECTED_UPHILL_NEUTRAL_SINK"
    elif relative > maximum_relative_first_law_residual:
        classification = "REJECTED_NEUTRAL_SINK_FIRST_LAW"
    elif not accepted:
        classification = "REJECTED_NEUTRAL_SINK_BALANCE"
    else:
        classification = "ACCEPTED_NEUTRAL_BOUNDARY_SINK"
    return MultiGrainSinkDecision(
        accepted, f"{classification}:{sink_mode}",
        candidate if accepted else state, annihilated_total, signed_change,
        balance, defect_change, heat_J, residual, relative, maximum_fraction,
        heat_by_cell)


def evaluate_interface_neutral_sink_transaction(
        state, *, mean_strain, interval_s, sink_parameters,
        wall_parameters, represented_thickness_m, energy_kwargs=None,
        sink_mode="interface_neutral_pair_sink",
        maximum_relative_first_law_residual=.05):
    """Annihilate only neutral line pairs already stored by moving fronts.

    This is a stationary GB reservoir reaction, not lattice absorption and not
    structural rotation.  Removing equal positive/negative content preserves
    every interface Burgers residual exactly.  The independently known line
    energy release is deposited locally as heat and checked against the same
    complete common-state functional used by front publication.
    """
    dt = float(interval_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("interface sink interval must be positive and finite")
    if sink_parameters.stress_assisted:
        raise ValueError(
            "interface neutral sink has no declared conjugate resolved stress")
    spacing = float(wall_parameters.spacing_m)
    thickness = float(represented_thickness_m)
    cell_volume = spacing*spacing*thickness
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    options.pop("mean_strain", None)
    common_energy = dict(
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall_parameters, mean_strain=np.asarray(mean_strain),
        **options)
    before_energy = evaluate_complete_multigrain_energy(state, **common_energy)
    common, _ = reconstruct_multigrain_common(state, spacing)
    process = ActivatedProcess(
        "interface-neutral-pair-sink", sink_parameters.attempt_frequency_s,
        0.0, sink_parameters.attempt_frequency_s)
    enthalpy = sink_parameters.activation_enthalpy_eV*EV_J
    rate = activated_rate_array_s(
        process, np.full_like(common.temperature_K, enthalpy),
        common.temperature_K)
    uncapped_fraction = -np.expm1(-rate*dt)
    fraction = (uncapped_fraction if sink_parameters.complete_physical_clock
                else np.minimum(
                    uncapped_fraction,
                    sink_parameters.maximum_fraction_per_interval))
    maximum_fraction = float(np.max(fraction, initial=0.0))
    interfaces = []
    heat_density = np.zeros_like(common.temperature_K, dtype=float)
    annihilated_total = 0.0
    signed_change = 0.0
    line_change = 0.0
    inventory_scale = 0.0
    line_energy = float(wall_parameters.line_energy_J_m)
    for interface in state.interfaces:
        plus = np.asarray(interface.boundary_plus_m2, dtype=float)
        minus = np.asarray(interface.boundary_minus_m2, dtype=float)
        neutral = np.minimum(plus, minus)*fraction[..., None, None]
        next_plus = plus-neutral
        next_minus = minus-neutral
        interfaces.append(replace(
            interface, boundary_plus_m2=next_plus,
            boundary_minus_m2=next_minus))
        pair_sum = np.sum(neutral, axis=(2, 3))
        heat_density += 2.0*line_energy*pair_sum
        annihilated_total += float(np.sum(neutral, dtype=np.longdouble))
        signed_change += float(np.sum(
            (next_plus-next_minus)-(plus-minus), dtype=np.longdouble))
        line_change += float(np.sum(
            next_plus+next_minus-plus-minus, dtype=np.longdouble))
        inventory_scale += float(np.sum(plus+minus, dtype=np.longdouble))
    unheated = replace(state, interfaces=tuple(interfaces))
    unheated.validate()
    cold_energy = evaluate_complete_multigrain_energy(unheated, **common_energy)
    heat_by_cell = heat_density*cell_volume
    temperature = (np.asarray(common.temperature_K)
                   +heat_density/wall_parameters.volumetric_heat_capacity_J_m3_K)
    candidate = replace(unheated, owners=tuple(
        replace(owner, temperature_K=temperature.copy())
        for owner in unheated.owners))
    candidate.validate()
    after_energy = evaluate_complete_multigrain_energy(candidate, **common_energy)
    defect_change = cold_energy.helmholtz_J-before_energy.helmholtz_J
    heat_J = float(np.sum(heat_by_cell, dtype=np.longdouble))
    residual = after_energy.internal_J-before_energy.internal_J
    scale = max(abs(defect_change), abs(heat_J), 1e-300)
    relative = abs(residual)/scale
    balance = line_change+2.0*annihilated_total
    tolerance = max(
        1e-12*max(2.0*annihilated_total, 1.0),
        4096.0*np.finfo(float).eps*max(inventory_scale, 1.0))
    active = annihilated_total > tolerance
    downhill = defect_change <= 1e-12*scale
    accepted = bool(
        active and downhill
        and relative <= maximum_relative_first_law_residual
        and abs(signed_change) <= tolerance and abs(balance) <= tolerance)
    if not active:
        classification = "INACTIVE_INTERFACE_NEUTRAL_SINK"
    elif not downhill:
        classification = "REJECTED_UPHILL_INTERFACE_NEUTRAL_SINK"
    elif relative > maximum_relative_first_law_residual:
        classification = "REJECTED_INTERFACE_NEUTRAL_SINK_FIRST_LAW"
    elif not accepted:
        classification = "REJECTED_INTERFACE_NEUTRAL_SINK_BALANCE"
    else:
        classification = "ACCEPTED_INTERFACE_NEUTRAL_SINK"
    return MultiGrainSinkDecision(
        accepted, f"{classification}:{sink_mode}",
        candidate if accepted else state, annihilated_total, signed_change,
        balance, defect_change, heat_J, residual, relative, maximum_fraction,
        heat_by_cell)


def evaluate_interface_vector_neutral_junction_transaction(
        state, *, mean_strain, interval_s, sink_parameters,
        wall_parameters, represented_thickness_m, energy_kwargs=None,
        sink_mode="interface_vector_neutral_junction_sink",
        maximum_relative_first_law_residual=.05):
    """Remove only interface line that is neutral in exact Burgers space.

    Multiple absorbed lattice families can react by GB glide/climb even when
    no same-family positive/negative pair exists.  The exact tensor
    ``boundary_burgers_m1`` remains bitwise unchanged.  For each stored row,
    ``|B|/b`` is a strict lower bound on line capable of carrying that Burgers
    vector; only scalar line above that bound is eligible.  Thus this channel
    cannot erase net Burgers content or manufacture a structural rotation.
    """
    dt = float(interval_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("interface junction interval must be positive finite")
    if sink_parameters.stress_assisted:
        raise ValueError(
            "vector-neutral interface reaction has no declared resolved stress")
    spacing = float(wall_parameters.spacing_m)
    thickness = float(represented_thickness_m)
    cell_volume = spacing*spacing*thickness
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    options.pop("mean_strain", None)
    common_energy = dict(
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall_parameters, mean_strain=np.asarray(mean_strain),
        **options)
    before_energy = evaluate_complete_multigrain_energy(state, **common_energy)
    common, _ = reconstruct_multigrain_common(state, spacing)
    process = ActivatedProcess(
        "interface-vector-neutral-junction", sink_parameters.attempt_frequency_s,
        0.0, sink_parameters.attempt_frequency_s)
    enthalpy = sink_parameters.activation_enthalpy_eV*EV_J
    rate = activated_rate_array_s(
        process, np.full_like(common.temperature_K, enthalpy),
        common.temperature_K)
    uncapped_fraction = -np.expm1(-rate*dt)
    fraction = (uncapped_fraction if sink_parameters.complete_physical_clock
                else np.minimum(
                    uncapped_fraction,
                    sink_parameters.maximum_fraction_per_interval))
    maximum_fraction = float(np.max(fraction, initial=0.0))
    interfaces = []
    heat_density = np.zeros_like(common.temperature_K, dtype=float)
    removed_total = 0.0
    line_change = 0.0
    inventory_scale = 0.0
    maximum_capacity_violation = 0.0
    line_energy = float(wall_parameters.line_energy_J_m)
    burgers_m = float(wall_parameters.burgers_m)
    for interface in state.interfaces:
        plus = np.asarray(interface.boundary_plus_m2, dtype=float)
        minus = np.asarray(interface.boundary_minus_m2, dtype=float)
        burgers = np.asarray(interface.boundary_burgers_m1, dtype=float)
        line_by_row = np.sum(plus+minus, axis=-1)
        required_by_row = np.linalg.norm(burgers, axis=-1)/burgers_m
        removable_by_row = np.maximum(line_by_row-required_by_row, 0.0)
        removed_by_row = fraction[..., None]*removable_by_row
        retained_fraction = np.divide(
            line_by_row-removed_by_row, line_by_row,
            out=np.ones_like(line_by_row), where=line_by_row > 0.0)
        next_plus = plus*retained_fraction[..., None]
        next_minus = minus*retained_fraction[..., None]
        next_line_by_row = np.sum(next_plus+next_minus, axis=-1)
        maximum_capacity_violation = max(
            maximum_capacity_violation,
            float(np.max(
                np.maximum(required_by_row-next_line_by_row, 0.0),
                initial=0.0)))
        interfaces.append(replace(
            interface, boundary_plus_m2=next_plus,
            boundary_minus_m2=next_minus,
            boundary_burgers_m1=burgers.copy()))
        removed_cell = np.sum(removed_by_row, axis=-1)
        heat_density += line_energy*removed_cell
        removed_total += float(np.sum(
            removed_by_row, dtype=np.longdouble))
        line_change += float(np.sum(
            next_plus+next_minus-plus-minus, dtype=np.longdouble))
        inventory_scale += float(np.sum(
            plus+minus, dtype=np.longdouble))
    unheated = replace(state, interfaces=tuple(interfaces))
    unheated.validate()
    cold_energy = evaluate_complete_multigrain_energy(unheated, **common_energy)
    heat_by_cell = heat_density*cell_volume
    temperature = (np.asarray(common.temperature_K)
                   +heat_density/wall_parameters.volumetric_heat_capacity_J_m3_K)
    candidate = replace(unheated, owners=tuple(
        replace(owner, temperature_K=temperature.copy())
        for owner in unheated.owners))
    candidate.validate()
    after_energy = evaluate_complete_multigrain_energy(candidate, **common_energy)
    defect_change = cold_energy.helmholtz_J-before_energy.helmholtz_J
    heat_J = float(np.sum(heat_by_cell, dtype=np.longdouble))
    residual = after_energy.internal_J-before_energy.internal_J
    scale = max(abs(defect_change), abs(heat_J), 1e-300)
    relative = abs(residual)/scale
    balance = line_change+removed_total
    tolerance = max(
        1e-12*max(removed_total, 1.0),
        4096.0*np.finfo(float).eps*max(inventory_scale, 1.0))
    active = removed_total > tolerance
    downhill = defect_change <= 1e-12*scale
    capacity_close = maximum_capacity_violation <= tolerance
    accepted = bool(
        active and downhill and capacity_close
        and relative <= maximum_relative_first_law_residual
        and abs(balance) <= tolerance)
    if not active:
        classification = "INACTIVE_INTERFACE_VECTOR_NEUTRAL_JUNCTION"
    elif not downhill:
        classification = "REJECTED_UPHILL_INTERFACE_VECTOR_NEUTRAL_JUNCTION"
    elif relative > maximum_relative_first_law_residual:
        classification = "REJECTED_INTERFACE_VECTOR_NEUTRAL_JUNCTION_FIRST_LAW"
    elif not capacity_close or abs(balance) > tolerance:
        classification = "REJECTED_INTERFACE_VECTOR_NEUTRAL_JUNCTION_BALANCE"
    else:
        classification = "ACCEPTED_INTERFACE_VECTOR_NEUTRAL_JUNCTION"
    # The legacy decision name is retained for ledger compatibility: one
    # "pair" equals two units of removed scalar line.  The authoritative
    # signed balance for this mode is the unchanged Burgers tensor.
    return MultiGrainSinkDecision(
        accepted, f"{classification}:{sink_mode}",
        candidate if accepted else state, 0.5*removed_total, 0.0,
        balance, defect_change, heat_J, residual, relative, maximum_fraction,
        heat_by_cell)
