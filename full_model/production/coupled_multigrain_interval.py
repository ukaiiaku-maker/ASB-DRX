"""One complete mechanics/thermal/front interval for adaptive production.

This module contains the physical interval map used by the V64 controller.
It preserves the established operator order and constitutive laws while making
the complete update callable on an immutable parent endpoint.  Trial ledgers,
heat, histories, and material states therefore remain private until the outer
accuracy controller publishes an accepted fine path.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Mapping

import numpy as np

from .common_tensorial_wall import (
    CommonWallDriving, resolved_driving_components, wall_residual,
)
from .coupled_interval_control import CoupledEndpoint, CoupledInterval
from .multigrain_common_state import (
    audit_multigrain_nye, reconstruct_multigrain_common,
)
from .multigrain_production import (
    MultiGrainProductionRuntime, _owner_drivings_from_common_stress,
    advance_energy_qualified_mechanics, advance_multigrain_front_interval,
    multigrain_instantaneous_dissipation_fields,
)
from .arrhenius_dislocation_sources import (
    boundary_tj_site_field, randomized_matched_site_field,
)
from .multigrain_source_transaction import (
    evaluate_interface_neutral_sink_transaction,
    evaluate_multigrain_neutral_sink_transaction,
    evaluate_multigrain_pair_source_transaction,
)
from .interface_structural_absorption import (
    evaluate_interface_structural_absorption,
)
from .lattice_interface_absorption import (
    evaluate_lattice_interface_absorption,
)
from .structural_interface_network import (
    JunctionDisclinationParameters, initialize_structural_interfaces,
    interface_lengths_m,
)
from .structural_multigrain_runtime import (
    StructuralMultigrainRuntime, synchronize_structural_owner_orientations,
)


@dataclass(frozen=True)
class CoupledMultigrainContext:
    systems: tuple
    topologies: tuple
    wall_parameters: object
    front_kinetics: object
    spacing_m: float
    represented_thickness_m: float
    shear_rate_s: float
    maximum_front_substep_s: float
    maximum_mechanical_subdivisions: int
    energy_kwargs: Mapping[str, object]
    heat_deposition_mode: str = "local_realized_event"
    front_enabled: bool = True
    warm_start_front_factor: bool = False
    source_parameters: object | None = None
    source_mode: str = "disabled"
    source_random_seed: int = 66
    source_tj_weight: float = 2.0
    sink_parameters: object | None = None
    sink_mode: str = "disabled"
    sink_random_seed: int = 66
    sink_tj_weight: float = 2.0
    interface_sink_parameters: object | None = None
    interface_sink_mode: str = "disabled"
    structural_absorption_parameters: object | None = None
    structural_absorption_mode: str = "disabled"
    lattice_absorption_parameters: object | None = None
    lattice_absorption_mode: str = "disabled"


def endpoint_observables(state, load: float,
                         context: CoupledMultigrainContext) -> dict:
    spacing = float(context.spacing_m)
    strain = np.array([[0.0, .5*load], [.5*load, 0.0]])
    driving = CommonWallDriving(mean_strain=strain)
    common, _ = reconstruct_multigrain_common(state, spacing)
    nye = audit_multigrain_nye(state, spacing)
    instantaneous = multigrain_instantaneous_dissipation_fields(
        state, driving=driving, systems=context.systems,
        topologies=context.topologies, wall_parameters=context.wall_parameters)
    owner_drivings = _owner_drivings_from_common_stress(
        state, driving, context.systems, context.wall_parameters)
    owner_speeds = []
    owner_slip_rates = []
    for support, owner, owner_driving in zip(
            state.supports, state.owners, owner_drivings):
        resolved = resolved_driving_components(
            owner, owner_driving, context.systems, context.topologies,
            context.wall_parameters)
        residual = wall_residual(
            owner, owner_driving, context.systems, context.topologies,
            context.wall_parameters)
        weight = np.asarray(support)[..., None]
        owner_speeds.append(weight*np.asarray(resolved["speed_m_s"]))
        owner_slip_rates.append(weight*np.asarray(residual.state_rate.slip))
    signed_total = sum(
        state.supports[index][..., None]*sum(
            np.asarray(getattr(owner, plus))-np.asarray(getattr(owner, minus))
            for plus, minus in (("mobile_plus_m2", "mobile_minus_m2"),
                                ("forest_plus_m2", "forest_minus_m2"),
                                ("wall_plus_m2", "wall_minus_m2")))
        for index, owner in enumerate(state.owners))
    return {
        "supports": np.asarray(state.supports),
        "common_temperature_K": np.asarray(common.temperature_K),
        "common_beta_p": np.asarray(common.beta_p),
        "signed_total_density_m2": np.asarray(signed_total),
        "exact_reconstructed_nye_m1": np.asarray(nye.exact_reconstructed_m1),
        "common_stress_tensor_Pa": np.asarray(
            owner_drivings[0].fixed_stress_tensor_Pa),
        "instantaneous_plastic_power_W_m3": np.asarray(
            instantaneous["plastic_power_W_m3"]),
        "instantaneous_signed_slip_rate_s": np.asarray(
            instantaneous["signed_slip_rate_s"]),
        "owner_weighted_speed_m_s": np.stack(owner_speeds, axis=2),
        "owner_weighted_signed_slip_rate_s": np.stack(
            owner_slip_rates, axis=2),
    }


def make_endpoint(state, runtime, load: float,
                  context: CoupledMultigrainContext, *,
                  numerical_hints=None) -> CoupledEndpoint:
    return CoupledEndpoint(
        state, runtime, float(load), endpoint_observables(state, load, context),
        {} if numerical_hints is None else dict(numerical_hints))


def advance_complete_multigrain_interval(
        parent: CoupledEndpoint, duration_s: float,
        context: CoupledMultigrainContext) -> CoupledInterval:
    """Apply exactly one complete coupled physical interval.

    Mechanical and moving-front operators both represent the same physical
    interval.  Only the front runtime owns the physical clock, so their
    durations are never added to one another.
    """
    dt = float(duration_s)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("complete coupled interval must be positive and finite")
    gamma0 = float(parent.load)
    gamma1 = gamma0+float(context.shear_rate_s)*dt
    strain0 = np.array([[0.0, .5*gamma0], [.5*gamma0, 0.0]])
    strain1 = np.array([[0.0, .5*gamma1], [.5*gamma1, 0.0]])
    runtime = parent.runtime
    structural_interfaces = tuple(runtime.structural_interfaces)
    if context.structural_absorption_mode != "disabled":
        if context.structural_absorption_mode != "common_axis":
            raise ValueError("unknown structural absorption mode")
        if context.structural_absorption_parameters is None:
            raise ValueError(
                "enabled structural absorption requires kinetic parameters")
        if not structural_interfaces:
            lengths = interface_lengths_m(
                parent.state.supports, runtime.interfaces,
                parent.state.grain_ids, context.spacing_m)
            orientations = np.asarray([
                np.mean(owner.orientation_rad) for owner in parent.state.owners])
            structural_interfaces = initialize_structural_interfaces(
                parent.state.grain_ids, orientations,
                tuple((item.grain_a_id, item.grain_b_id)
                      for item in runtime.interfaces), lengths,
                context.structural_absorption_parameters.boundary.burgers_m)
            runtime = replace(runtime, structural_interfaces=structural_interfaces)
    energy_base = dict(context.energy_kwargs)
    if structural_interfaces:
        energy_base.update(
            structural_interfaces=structural_interfaces,
            structural_burgers_m=(
                context.structural_absorption_parameters.boundary.burgers_m),
            structural_parameters=(
                context.structural_absorption_parameters.boundary),
            junction_disclination_parameters=JunctionDisclinationParameters())
    # React the reservoir present at the accepted interval start.  Doing this
    # before the mechanical map keeps the structural edge and owner angles
    # synchronized; front-generated content becomes available on the next
    # physical interval rather than being consumed in the event that created it.
    structural_absorption_started = time.perf_counter()
    structural_absorption = None
    mechanical_parent_state = parent.state
    if context.structural_absorption_mode != "disabled":
        structural_absorption = evaluate_interface_structural_absorption(
            mechanical_parent_state, structural_interfaces, mean_strain=strain0,
            interval_s=dt, spacing_m=context.spacing_m,
            represented_thickness_m=context.represented_thickness_m,
            wall_parameters=context.wall_parameters, energy_kwargs=energy_base,
            parameters=context.structural_absorption_parameters)
        if structural_absorption.classification in {
                "REJECTED_BURGERS_OR_LINE_BALANCE",
                "REJECTED_COMPLETE_FIRST_LAW"}:
            raise RuntimeError(
                "structural absorption failed a hard balance audit")
        if structural_absorption.accepted:
            mechanical_parent_state = structural_absorption.committed_state
            structural_interfaces = structural_absorption.committed_interfaces
        runtime = replace(runtime, structural_interfaces=structural_interfaces,
                          ledger=replace(
            runtime.ledger,
            structural_absorption_intervals=(
                runtime.ledger.structural_absorption_intervals
                +int(structural_absorption.accepted)),
            structural_absorption_inactive_intervals=(
                runtime.ledger.structural_absorption_inactive_intervals
                +int(structural_absorption.classification.startswith("INACTIVE"))),
            structural_absorption_rejected_intervals=(
                runtime.ledger.structural_absorption_rejected_intervals
                +int(structural_absorption.classification.startswith("REJECTED"))),
            structural_absorbed_line_length_m=(
                runtime.ledger.structural_absorbed_line_length_m
                +(structural_absorption.projection.removed_line_length_m
                  *context.represented_thickness_m
                  if structural_absorption.accepted else 0.0)),
            cumulative_structural_absorption_heat_J=(
                runtime.ledger.cumulative_structural_absorption_heat_J
                +(structural_absorption.generated_heat_J
                  if structural_absorption.accepted else 0.0)),
            maximum_structural_absorption_first_law_residual=max(
                runtime.ledger.maximum_structural_absorption_first_law_residual,
                structural_absorption.relative_first_law_residual
                if structural_absorption.accepted else 0.0)))
        energy_base.update(
            structural_interfaces=structural_interfaces,
            structural_burgers_m=(
                context.structural_absorption_parameters.boundary.burgers_m),
            structural_parameters=(
                context.structural_absorption_parameters.boundary),
            junction_disclination_parameters=JunctionDisclinationParameters())
    structural_absorption_wall_time = (
        time.perf_counter()-structural_absorption_started)
    energy_before = dict(energy_base, mean_strain=strain0)
    mechanical_started = time.perf_counter()
    mechanical = advance_energy_qualified_mechanics(
        mechanical_parent_state, mean_strain_before=strain0,
        mean_strain_candidate=strain1, systems=context.systems,
        topologies=context.topologies, wall_parameters=context.wall_parameters,
        dt_s=dt, represented_thickness_m=context.represented_thickness_m,
        energy_kwargs=energy_before, maximum_relative_first_law_residual=.05,
        maximum_subdivisions=context.maximum_mechanical_subdivisions)
    mechanical_wall_time = time.perf_counter()-mechanical_started
    mechanical_heat = sum(
        (np.asarray(item.irreversible_heat_J_m3_cells)
         for item in mechanical.operator_decisions),
        start=np.zeros(parent.state.supports.shape[1:], dtype=float))
    mechanical_work_density = sum(
        (np.asarray(item.plastic_work_J_m3_cells)
         for item in mechanical.operator_decisions),
        start=np.zeros(parent.state.supports.shape[1:], dtype=float))
    conduction = sum(
        (np.asarray(item.thermal_conduction_J_m3_cells)
         for item in mechanical.operator_decisions),
        start=np.zeros(parent.state.supports.shape[1:], dtype=float))
    bath = sum(
        (np.asarray(item.thermal_bath_exchange_J_m3_cells)
         for item in mechanical.operator_decisions),
        start=np.zeros(parent.state.supports.shape[1:], dtype=float))
    runtime = replace(runtime, ledger=replace(
        runtime.ledger,
        cumulative_mechanical_external_work_J=(
            runtime.ledger.cumulative_mechanical_external_work_J
            +mechanical.external_work_J),
        cumulative_mechanical_internal_energy_change_J=(
            runtime.ledger.cumulative_mechanical_internal_energy_change_J
            +mechanical.internal_energy_change_J),
        cumulative_mechanical_first_law_residual_J=(
            runtime.ledger.cumulative_mechanical_first_law_residual_J
            +mechanical.first_law_residual_J),
        maximum_mechanical_relative_first_law_residual=max(
            runtime.ledger.maximum_mechanical_relative_first_law_residual,
            mechanical.relative_first_law_residual)))
    # The Hall--Petch source is a parallel physical channel over this same
    # interval. It is applied after the mechanical split at the final imposed
    # strain and before the front. Its joint line-plus-slip proposal owns an
    # independent complete-energy/heat transaction.
    source_started = time.perf_counter()
    source = None
    source_state = mechanical.state
    if context.source_mode != "disabled":
        if context.source_parameters is None:
            raise ValueError("enabled source mode requires source parameters")
        boundary_sites = boundary_tj_site_field(
            source_state.supports, tj_site_weight=context.source_tj_weight)
        if context.source_mode == "gb_tj_only":
            source_sites = boundary_sites
        elif context.source_mode == "matched_random":
            source_sites = randomized_matched_site_field(
                boundary_sites, context.source_random_seed)
        else:
            raise ValueError("unknown dislocation source mode")
        source = evaluate_multigrain_pair_source_transaction(
            source_state, mean_strain=strain1, site_field=source_sites,
            interval_s=dt, source_parameters=context.source_parameters,
            systems=context.systems, wall_parameters=context.wall_parameters,
            represented_thickness_m=context.represented_thickness_m,
            energy_kwargs=energy_base, source_mode=context.source_mode)
        if source.classification in {
                "REJECTED_FIRST_LAW_CLOSURE"}:
            raise RuntimeError(
                "complete coupled interval source failed its hard energy audit")
        if source.accepted:
            source_state = source.candidate_state
        runtime = replace(runtime, ledger=replace(
            runtime.ledger,
            source_intervals=runtime.ledger.source_intervals+int(source.accepted),
            source_inactive_intervals=(runtime.ledger.source_inactive_intervals
                                       +int(not source.accepted)),
            source_rejected_intervals=runtime.ledger.source_rejected_intervals,
            source_event_count=(runtime.ledger.source_event_count
                                +(source.event_count if source.accepted else 0.0)),
            source_created_line_cell_sum_m2=(
                runtime.ledger.source_created_line_cell_sum_m2
                +(source.created_line_cell_sum_m2 if source.accepted else 0.0)),
            cumulative_source_resolved_work_J=(
                runtime.ledger.cumulative_source_resolved_work_J
                +(source.resolved_work_J if source.accepted else 0.0)),
            cumulative_source_generated_heat_J=(
                runtime.ledger.cumulative_source_generated_heat_J
                +(source.generated_heat_J if source.accepted else 0.0)),
            cumulative_source_first_law_residual_J=(
                runtime.ledger.cumulative_source_first_law_residual_J
                +(source.first_law_residual_J if source.accepted else 0.0)),
            maximum_source_relative_first_law_residual=max(
                runtime.ledger.maximum_source_relative_first_law_residual,
                source.relative_first_law_residual if source.accepted else 0.0),
            maximum_source_line_balance_residual_m2=max(
                runtime.ledger.maximum_source_line_balance_residual_m2,
                abs(source.line_balance_residual_m2) if source.accepted else 0.0)))
    source_wall_time = time.perf_counter()-source_started
    # The neutral sink is a distinct zero-work physical channel. It follows
    # line creation so a same-site source/sink hypothesis has an unambiguous
    # operator order and separate line/heat ledgers.
    sink_started = time.perf_counter()
    sink = None
    sink_state = source_state
    if context.sink_mode != "disabled":
        if context.sink_parameters is None:
            raise ValueError("enabled sink mode requires sink parameters")
        boundary_sites = boundary_tj_site_field(
            sink_state.supports, tj_site_weight=context.sink_tj_weight)
        if context.sink_mode == "gb_tj_only":
            sink_sites = boundary_sites
        elif context.sink_mode == "matched_random":
            sink_sites = randomized_matched_site_field(
                boundary_sites, context.sink_random_seed)
        else:
            raise ValueError("unknown dislocation sink mode")
        sink = evaluate_multigrain_neutral_sink_transaction(
            sink_state, mean_strain=strain1, site_field=sink_sites,
            interval_s=dt, sink_parameters=context.sink_parameters,
            systems=context.systems, wall_parameters=context.wall_parameters,
            represented_thickness_m=context.represented_thickness_m,
            energy_kwargs=energy_base, sink_mode=context.sink_mode)
        if sink.classification.startswith("REJECTED_NEUTRAL_SINK_FIRST_LAW"):
            raise RuntimeError(
                "complete coupled interval sink failed its hard energy audit")
        if sink.accepted:
            sink_state = sink.candidate_state
        runtime = replace(runtime, ledger=replace(
            runtime.ledger,
            sink_intervals=runtime.ledger.sink_intervals+int(sink.accepted),
            sink_inactive_intervals=(runtime.ledger.sink_inactive_intervals
                                     +int(not sink.accepted)),
            sink_rejected_intervals=(runtime.ledger.sink_rejected_intervals
                                     +int(sink.classification.startswith(
                                         "REJECTED"))),
            sink_annihilated_pair_cell_sum_m2=(
                runtime.ledger.sink_annihilated_pair_cell_sum_m2
                +(sink.annihilated_pair_cell_sum_m2 if sink.accepted else 0.0)),
            cumulative_sink_generated_heat_J=(
                runtime.ledger.cumulative_sink_generated_heat_J
                +(sink.generated_heat_J if sink.accepted else 0.0)),
            cumulative_sink_first_law_residual_J=(
                runtime.ledger.cumulative_sink_first_law_residual_J
                +(sink.first_law_residual_J if sink.accepted else 0.0)),
            maximum_sink_relative_first_law_residual=max(
                runtime.ledger.maximum_sink_relative_first_law_residual,
                sink.relative_first_law_residual if sink.accepted else 0.0),
            maximum_sink_line_balance_residual_m2=max(
                runtime.ledger.maximum_sink_line_balance_residual_m2,
                abs(sink.line_balance_residual_m2) if sink.accepted else 0.0)))
    sink_wall_time = time.perf_counter()-sink_started
    # Moving fronts own explicit interfacial line reservoirs.  Neutral pairs
    # already stored there can recover while the boundary is stationary; this
    # channel is separate from lattice absorption/rotation and therefore
    # preserves the signed interface Burgers residual exactly.
    interface_sink_started = time.perf_counter()
    interface_sink = None
    interface_sink_state = sink_state
    if context.interface_sink_mode != "disabled":
        if context.interface_sink_parameters is None:
            raise ValueError(
                "enabled interface sink requires interface sink parameters")
        if context.interface_sink_mode != "neutral_pair":
            raise ValueError("unknown interface sink mode")
        interface_sink = evaluate_interface_neutral_sink_transaction(
            interface_sink_state, mean_strain=strain1, interval_s=dt,
            sink_parameters=context.interface_sink_parameters,
            wall_parameters=context.wall_parameters,
            represented_thickness_m=context.represented_thickness_m,
            energy_kwargs=energy_base,
            sink_mode=context.interface_sink_mode)
        if interface_sink.classification.startswith(
                "REJECTED_INTERFACE_NEUTRAL_SINK_FIRST_LAW"):
            raise RuntimeError(
                "interface sink failed its hard energy audit")
        if interface_sink.accepted:
            interface_sink_state = interface_sink.candidate_state
        runtime = replace(runtime, ledger=replace(
            runtime.ledger,
            interface_sink_intervals=(runtime.ledger.interface_sink_intervals
                                      +int(interface_sink.accepted)),
            interface_sink_inactive_intervals=(
                runtime.ledger.interface_sink_inactive_intervals
                +int(not interface_sink.accepted)),
            interface_sink_rejected_intervals=(
                runtime.ledger.interface_sink_rejected_intervals
                +int(interface_sink.classification.startswith("REJECTED"))),
            interface_sink_annihilated_pair_cell_sum_m2=(
                runtime.ledger.interface_sink_annihilated_pair_cell_sum_m2
                +(interface_sink.annihilated_pair_cell_sum_m2
                  if interface_sink.accepted else 0.0)),
            cumulative_interface_sink_generated_heat_J=(
                runtime.ledger.cumulative_interface_sink_generated_heat_J
                +(interface_sink.generated_heat_J
                  if interface_sink.accepted else 0.0)),
            cumulative_interface_sink_first_law_residual_J=(
                runtime.ledger.cumulative_interface_sink_first_law_residual_J
                +(interface_sink.first_law_residual_J
                  if interface_sink.accepted else 0.0)),
            maximum_interface_sink_relative_first_law_residual=max(
                runtime.ledger.maximum_interface_sink_relative_first_law_residual,
                interface_sink.relative_first_law_residual
                if interface_sink.accepted else 0.0),
            maximum_interface_sink_line_balance_residual_m2=max(
                runtime.ledger.maximum_interface_sink_line_balance_residual_m2,
                abs(interface_sink.line_balance_residual_m2)
                if interface_sink.accepted else 0.0)))
    interface_sink_wall_time = time.perf_counter()-interface_sink_started
    lattice_absorption_started = time.perf_counter()
    lattice_absorption = None
    structural_state = interface_sink_state
    if context.lattice_absorption_mode != "disabled":
        if context.lattice_absorption_mode != "one_flux_mura":
            raise ValueError("unknown lattice absorption mode")
        if context.lattice_absorption_parameters is None:
            raise ValueError("enabled lattice absorption requires parameters")
        lattice_absorption = evaluate_lattice_interface_absorption(
            structural_state, mean_strain=strain1, interval_s=dt,
            spacing_m=context.spacing_m,
            represented_thickness_m=context.represented_thickness_m,
            systems=context.systems, wall_parameters=context.wall_parameters,
            energy_kwargs=energy_base,
            parameters=context.lattice_absorption_parameters)
        if lattice_absorption.classification in {
                "REJECTED_LINE_BURGERS_OR_MURA_BALANCE",
                "REJECTED_COMPLETE_FIRST_LAW"}:
            raise RuntimeError("lattice absorption failed a hard balance audit")
        if lattice_absorption.accepted:
            structural_state = lattice_absorption.committed_state
        absorbed = context.represented_thickness_m*(
            lattice_absorption.projection.removed_plus_line_per_thickness
            +lattice_absorption.projection.removed_minus_line_per_thickness)
        runtime = replace(runtime, ledger=replace(
            runtime.ledger,
            lattice_absorption_intervals=(
                runtime.ledger.lattice_absorption_intervals
                +int(lattice_absorption.accepted)),
            lattice_absorption_inactive_intervals=(
                runtime.ledger.lattice_absorption_inactive_intervals
                +int(lattice_absorption.classification.startswith("INACTIVE"))),
            lattice_absorption_rejected_intervals=(
                runtime.ledger.lattice_absorption_rejected_intervals
                +int(lattice_absorption.classification.startswith("REJECTED"))),
            lattice_absorbed_line_length_m=(
                runtime.ledger.lattice_absorbed_line_length_m
                +(absorbed if lattice_absorption.accepted else 0.0)),
            cumulative_lattice_absorption_heat_J=(
                runtime.ledger.cumulative_lattice_absorption_heat_J
                +(lattice_absorption.generated_heat_J
                  if lattice_absorption.accepted else 0.0)),
            maximum_lattice_absorption_first_law_residual=max(
                runtime.ledger.maximum_lattice_absorption_first_law_residual,
                lattice_absorption.relative_first_law_residual
                if lattice_absorption.accepted else 0.0)))
    lattice_absorption_wall_time = (
        time.perf_counter()-lattice_absorption_started)
    energy_after = dict(energy_base, mean_strain=strain1)
    front_started = time.perf_counter()
    if context.front_enabled:
        state, runtime, fronts = advance_multigrain_front_interval(
            structural_state, runtime, kinetics=context.front_kinetics,
            dt_s=dt, maximum_substep_s=context.maximum_front_substep_s,
            spacing_m=context.spacing_m,
            represented_thickness_m=context.represented_thickness_m,
            wall_parameters=context.wall_parameters,
            energy_kwargs=energy_after,
            applied_shear_rate_s=context.shear_rate_s,
            systems=context.systems,
            heat_deposition_mode=context.heat_deposition_mode,
            initial_joint_pressure_factor=(
                float(parent.numerical_hints.get(
                    "front_joint_pressure_factor", 1.0))
                if context.warm_start_front_factor else 1.0),
            warm_start_joint_pressure_factor=(
                context.warm_start_front_factor))
        if not all(item.accepted and not item.contour_cfl_clipped
                   for item in fronts):
            raise RuntimeError(
                "complete coupled interval contains a rejected or clipped front trial")
    else:
        state = structural_state
        fronts = ()
        runtime = MultiGrainProductionRuntime(runtime.interfaces, replace(
            runtime.ledger, intervals=runtime.ledger.intervals+1,
            physical_time_s=runtime.ledger.physical_time_s+dt,
            applied_shear_strain=(runtime.ledger.applied_shear_strain
                                  +context.shear_rate_s*dt),
            stationary_intervals=runtime.ledger.stationary_intervals+1),
            runtime.structural_interfaces)
    if structural_interfaces:
        state = synchronize_structural_owner_orientations(
            state, StructuralMultigrainRuntime(structural_interfaces),
            spacing_m=context.spacing_m,
            burgers_m=context.structural_absorption_parameters.boundary.burgers_m)
    front_wall_time = time.perf_counter()-front_started
    front_sources = [
        np.asarray(item.heat_source_J_by_cell, dtype=float)
        for item in fronts if item.heat_source_J_by_cell is not None]
    front_heat = sum(
        front_sources,
        start=np.zeros(parent.state.supports.shape[1:], dtype=float))
    if not np.isclose(
            runtime.ledger.physical_time_s-parent.runtime.ledger.physical_time_s,
            dt, rtol=0.0, atol=64*np.finfo(float).eps*max(
                abs(runtime.ledger.physical_time_s),
                abs(parent.runtime.ledger.physical_time_s), dt, 1e-300)):
        raise RuntimeError("complete coupled map failed its physical clock")
    if not np.isclose(runtime.ledger.applied_shear_strain
                      -parent.runtime.ledger.applied_shear_strain,
                      context.shear_rate_s*dt, rtol=0.0,
                      atol=64*np.finfo(float).eps*max(abs(gamma1), 1.0)):
        raise RuntimeError("complete coupled map failed its load clock")
    observable_started = time.perf_counter()
    numerical_hints = dict(parent.numerical_hints)
    if fronts:
        numerical_hints["front_joint_pressure_factor"] = float(
            fronts[-1].joint_pressure_factor)
    endpoint = make_endpoint(
        state, runtime, gamma1, context, numerical_hints=numerical_hints)
    observable_wall_time = time.perf_counter()-observable_started
    integrals = {
        "mechanical_external_work_J": mechanical.external_work_J,
        "mechanical_internal_energy_change_J": mechanical.internal_energy_change_J,
        "mechanical_irreversible_heat_J": sum(
            item.irreversible_heat_J for item in mechanical.operator_decisions),
        "mechanical_plastic_work_J": sum(
            item.external_plastic_work_J for item in mechanical.operator_decisions),
        "front_heat_J": float(np.sum(front_heat, dtype=np.longdouble)),
        "source_resolved_work_J": (0.0 if source is None or not source.accepted
                                    else source.resolved_work_J),
        "source_heat_J": (0.0 if source is None or not source.accepted
                           else source.generated_heat_J),
        "source_heat_J_by_cell": (
            np.zeros(parent.state.supports.shape[1:], dtype=float)
            if source is None or not source.accepted
            else np.asarray(source.heat_source_J_by_cell)),
        "sink_heat_J": (0.0 if sink is None or not sink.accepted
                         else sink.generated_heat_J),
        "sink_heat_J_by_cell": (
            np.zeros(parent.state.supports.shape[1:], dtype=float)
            if sink is None or not sink.accepted
            else np.asarray(sink.heat_source_J_by_cell)),
        "interface_sink_heat_J": (
            0.0 if interface_sink is None or not interface_sink.accepted
            else interface_sink.generated_heat_J),
        "interface_sink_heat_J_by_cell": (
            np.zeros(parent.state.supports.shape[1:], dtype=float)
            if interface_sink is None or not interface_sink.accepted
            else np.asarray(interface_sink.heat_source_J_by_cell)),
        "structural_absorption_heat_J": (
            0.0 if structural_absorption is None
            or not structural_absorption.accepted
            else structural_absorption.generated_heat_J),
        "lattice_absorption_heat_J": (
            0.0 if lattice_absorption is None or not lattice_absorption.accepted
            else lattice_absorption.generated_heat_J),
        "lattice_absorption_heat_J_by_cell": (
            np.zeros(parent.state.supports.shape[1:], dtype=float)
            if lattice_absorption is None or not lattice_absorption.accepted
            else np.asarray(lattice_absorption.heat_source_J_by_cell)),
        "mechanical_heat_J_m3": mechanical_heat,
        "mechanical_work_J_m3": mechanical_work_density,
        "front_heat_J_by_cell": front_heat,
        "thermal_conduction_J_m3": conduction,
        "thermal_bath_exchange_J_m3": bath,
    }
    audit = {
        "mechanical_subdivision_depth": mechanical.subdivisions,
        "mechanical_internal_substeps": sum(
            item.maximum_substeps for item in mechanical.operator_decisions),
        "mechanical_consumed_interval_s": sum(
            item.consumed_interval_s for item in mechanical.operator_decisions),
        "mechanical_relative_first_law_residual": (
            mechanical.relative_first_law_residual),
        "front_subintervals": len(fronts),
        "front_rejected_subintervals": sum(not item.accepted for item in fronts),
        "front_maximum_unclipped_fraction": max(
            (item.maximum_unclipped_fraction for item in fronts), default=0.0),
        "front_directions": ({} if not fronts else
                             fronts[-1].direction_by_interface),
        "front_selected_velocities_m_s": ({} if not fronts else
                                           fronts[-1].selected_velocity_by_interface_m_s),
        "front_force_rate_closure_classifications": [
            item.force_rate_closure_classification for item in fronts],
        "front_force_rate_closure_brackets": [
            item.force_rate_closure_bracket for item in fronts
            if item.force_rate_closure_bracket is not None],
        "front_joint_pressure_factors": [
            item.joint_pressure_factor for item in fronts],
        "front_force_rate_iterations": [
            item.force_rate_iterations for item in fronts],
        "front_complete_energy_evaluations": [
            item.complete_energy_evaluations for item in fronts],
        "front_independent_directional_prices": [
            item.independent_directional_prices for item in fronts],
        "front_joint_capacity_solves": [
            item.joint_capacity_solves for item in fronts],
        "source_mode": context.source_mode,
        "source_classification": ("DISABLED" if source is None
                                  else source.classification),
        "source_event_count": (0.0 if source is None or not source.accepted
                               else source.event_count),
        "source_created_line_cell_sum_m2": (
            0.0 if source is None or not source.accepted
            else source.created_line_cell_sum_m2),
        "source_relative_first_law_residual": (
            0.0 if source is None or not source.accepted
            else source.relative_first_law_residual),
        "source_line_balance_residual_m2": (
            0.0 if source is None or not source.accepted
            else source.line_balance_residual_m2),
        "sink_mode": context.sink_mode,
        "sink_classification": ("DISABLED" if sink is None
                                else sink.classification),
        "sink_annihilated_pair_cell_sum_m2": (
            0.0 if sink is None or not sink.accepted
            else sink.annihilated_pair_cell_sum_m2),
        "sink_relative_first_law_residual": (
            0.0 if sink is None or not sink.accepted
            else sink.relative_first_law_residual),
        "sink_line_balance_residual_m2": (
            0.0 if sink is None or not sink.accepted
            else sink.line_balance_residual_m2),
        "interface_sink_mode": context.interface_sink_mode,
        "interface_sink_classification": (
            "DISABLED" if interface_sink is None
            else interface_sink.classification),
        "interface_sink_annihilated_pair_cell_sum_m2": (
            0.0 if interface_sink is None or not interface_sink.accepted
            else interface_sink.annihilated_pair_cell_sum_m2),
        "interface_sink_relative_first_law_residual": (
            0.0 if interface_sink is None or not interface_sink.accepted
            else interface_sink.relative_first_law_residual),
        "interface_sink_line_balance_residual_m2": (
            0.0 if interface_sink is None or not interface_sink.accepted
            else interface_sink.line_balance_residual_m2),
        "structural_absorption_mode": context.structural_absorption_mode,
        "structural_absorption_classification": (
            "DISABLED" if structural_absorption is None
            else structural_absorption.classification),
        "structural_absorption_fraction": (
            0.0 if structural_absorption is None
            else structural_absorption.kinetic_fraction),
        "structural_absorption_content_change": (
            [] if structural_absorption is None
            else structural_absorption.projection.structural_content_change.tolist()),
        "structural_absorption_relative_first_law_residual": (
            0.0 if structural_absorption is None
            or not structural_absorption.accepted
            else structural_absorption.relative_first_law_residual),
        "lattice_absorption_mode": context.lattice_absorption_mode,
        "lattice_absorption_classification": (
            "DISABLED" if lattice_absorption is None
            else lattice_absorption.classification),
        "lattice_absorption_fraction": (
            0.0 if lattice_absorption is None
            else lattice_absorption.kinetic_fraction),
        "lattice_absorption_removed_line_m": (
            0.0 if lattice_absorption is None
            else context.represented_thickness_m*(
                lattice_absorption.projection.removed_plus_line_per_thickness
                +lattice_absorption.projection.removed_minus_line_per_thickness)),
        "lattice_absorption_mura_relative_residual": (
            0.0 if lattice_absorption is None
            else lattice_absorption.projection.maximum_mura_relative_residual),
        "lattice_absorption_relative_first_law_residual": (
            0.0 if lattice_absorption is None or not lattice_absorption.accepted
            else lattice_absorption.relative_first_law_residual),
        "cost_breakdown_s": {
            "mechanical_map": mechanical_wall_time,
            "source_map": source_wall_time,
            "sink_map": sink_wall_time,
            "interface_sink_map": interface_sink_wall_time,
            "structural_absorption_map": structural_absorption_wall_time,
            "lattice_absorption_map": lattice_absorption_wall_time,
            "front_map": front_wall_time,
            "endpoint_observables": observable_wall_time,
            "accounted_total": (mechanical_wall_time+source_wall_time+sink_wall_time
                                +interface_sink_wall_time
                                +structural_absorption_wall_time+front_wall_time
                                +lattice_absorption_wall_time
                                +observable_wall_time),
        },
    }
    return CoupledInterval(endpoint, dt, integrals, audit)
