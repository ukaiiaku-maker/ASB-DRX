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
    boundary_tj_site_field, random_bulk_site_field,
    random_discrete_bulk_site_field,
    randomized_matched_site_field,
)
from .multigrain_source_transaction import (
    advance_energy_qualified_pair_source_transaction,
    evaluate_interface_neutral_sink_transaction,
    evaluate_interface_vector_neutral_junction_transaction,
    evaluate_multigrain_neutral_sink_transaction,
    evaluate_multigrain_pair_source_transaction,
)
from .interface_structural_absorption import (
    evaluate_interface_structural_absorption,
)
from .lattice_interface_absorption import (
    evaluate_lattice_interface_absorption,
)
from .interface_mobile_pair_emission import (
    evaluate_interface_mobile_pair_emission,
)
from .wall_orientation_relaxation import (
    evaluate_wall_orientation_relaxation,
)
from .implicit_wall_order_relaxation import (
    ImplicitWallOrderRelaxationParameters,
    evaluate_implicit_wall_order_relaxation,
)
from .compatible_wall_reorganization import (
    CompatibleWallReorganizationParameters,
    evaluate_compatible_wall_reorganization,
)
from .wall_neutral_pair_recovery import (
    WallNeutralPairRecoveryParameters, evaluate_wall_neutral_pair_recovery,
)
from .joint_wall_topology_orientation import (
    JointWallTopologyOrientationParameters,
    evaluate_joint_wall_topology_orientation,
)
from .conservative_wall_glide import (
    ConservativeWallGlideParameters, evaluate_conservative_wall_glide,
)
from .conservative_dipole_wall_organization import (
    ConservativeDipoleWallParameters,
    evaluate_conservative_dipole_wall_organization,
)
from .collective_wall_nucleation import (
    CollectiveWallNucleationParameters, evaluate_collective_wall_nucleation,
)
from .structural_interface_network import (
    JunctionDisclinationParameters, JunctionReactionParameters,
    initialize_structural_interfaces,
    interface_lengths_m,
)
from .structural_multigrain_runtime import (
    StructuralMultigrainRuntime, advance_structural_junction_interval,
    synchronize_structural_owner_orientations,
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
    maximum_mechanical_internal_substeps: int = 128
    heat_deposition_mode: str = "local_realized_event"
    front_enabled: bool = True
    warm_start_front_factor: bool = False
    source_parameters: object | None = None
    source_mode: str = "disabled"
    source_random_seed: int = 66
    source_tj_weight: float = 2.0
    source_bulk_site_density_m2: float = 1.0e12
    maximum_source_subdivisions: int = 6
    interface_mobile_emission_mode: str = "disabled"
    sink_parameters: object | None = None
    sink_mode: str = "disabled"
    sink_random_seed: int = 66
    sink_tj_weight: float = 2.0
    interface_sink_parameters: object | None = None
    interface_sink_mode: str = "disabled"
    structural_absorption_parameters: object | None = None
    structural_absorption_mode: str = "disabled"
    structural_junction_parameters: object | None = None
    structural_junction_mode: str = "disabled"
    lattice_absorption_parameters: object | None = None
    lattice_absorption_mode: str = "disabled"
    wall_orientation_relaxation_parameters: object | None = None
    wall_orientation_relaxation_mode: str = "disabled"
    wall_order_relaxation_parameters: object | None = None
    wall_order_relaxation_mode: str = "disabled"
    compatible_wall_reorganization_parameters: object | None = None
    compatible_wall_reorganization_mode: str = "disabled"
    wall_neutral_pair_recovery_parameters: object | None = None
    wall_neutral_pair_recovery_mode: str = "disabled"
    joint_wall_topology_orientation_parameters: object | None = None
    joint_wall_topology_orientation_mode: str = "disabled"
    conservative_wall_glide_parameters: object | None = None
    conservative_wall_glide_mode: str = "disabled"
    conservative_dipole_wall_parameters: object | None = None
    conservative_dipole_wall_mode: str = "disabled"
    collective_wall_nucleation_parameters: object | None = None
    collective_wall_nucleation_mode: str = "disabled"
    collective_wall_nucleation_seed: int = 67


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
    # Absorbed edge content can be incompatible around a triple-junction
    # cycle.  Its conservative redistribution is a distinct Arrhenius
    # reaction: front motion may neither erase that content nor pay its energy
    # implicitly.  Execute the already energy-complete reaction before the
    # mechanical/front maps and publish only its separately ledgered heat.
    structural_junction_started = time.perf_counter()
    structural_junction = None
    if context.structural_junction_mode != "disabled":
        if context.structural_junction_mode != "arrhenius_cycle":
            raise ValueError("unknown structural junction mode")
        if context.structural_absorption_parameters is None:
            raise ValueError(
                "structural junction reaction requires structural boundary parameters")
        if not structural_interfaces:
            raise ValueError(
                "structural junction reaction requires initialized interfaces")
        parameters = (JunctionReactionParameters()
                      if context.structural_junction_parameters is None
                      else context.structural_junction_parameters)
        (mechanical_parent_state, junction_runtime,
         structural_junction) = advance_structural_junction_interval(
            mechanical_parent_state,
            StructuralMultigrainRuntime(structural_interfaces),
            dt_s=dt, spacing_m=context.spacing_m,
            represented_thickness_m=context.represented_thickness_m,
            wall_parameters=context.wall_parameters,
            energy_kwargs=dict(context.energy_kwargs, mean_strain=strain0),
            boundary_parameters=(
                context.structural_absorption_parameters.boundary),
            disclination_parameters=JunctionDisclinationParameters(),
            reaction_parameters=parameters)
        if structural_junction.classification == "REJECTED_COMPLETE_FIRST_LAW":
            raise RuntimeError(
                "structural junction reaction failed its hard energy audit: "
                f"relative={structural_junction.relative_first_law_residual:.17g}, "
                f"residual_J={structural_junction.first_law_residual_J:.17g}, "
                f"heat_J={structural_junction.generated_heat_J:.17g}, "
                f"candidate_delta_J="
                f"{structural_junction.complete_energy_candidate.internal_J-structural_junction.complete_energy_before.internal_J:.17g}")
        structural_interfaces = tuple(junction_runtime.interfaces)
        runtime = replace(
            runtime, structural_interfaces=structural_interfaces,
            ledger=replace(
                runtime.ledger,
                structural_junction_intervals=(
                    runtime.ledger.structural_junction_intervals
                    +int(structural_junction.accepted)),
                structural_junction_inactive_intervals=(
                    runtime.ledger.structural_junction_inactive_intervals
                    +int(structural_junction.classification
                         == "NO_CYCLE_REACTION")),
                structural_junction_rejected_intervals=(
                    runtime.ledger.structural_junction_rejected_intervals
                    +int(structural_junction.classification.startswith(
                        "REJECTED"))),
                structural_junction_reacted_absolute_content=(
                    runtime.ledger.structural_junction_reacted_absolute_content
                    +(structural_junction.kinetic_proposal.
                      absolute_reacted_content
                      if structural_junction.accepted else 0.0)),
                cumulative_structural_junction_heat_J=(
                    runtime.ledger.cumulative_structural_junction_heat_J
                    +(structural_junction.generated_heat_J
                      if structural_junction.accepted else 0.0)),
                maximum_structural_junction_first_law_residual=max(
                    runtime.ledger.
                    maximum_structural_junction_first_law_residual,
                    structural_junction.relative_first_law_residual
                    if structural_junction.accepted else 0.0)))
        energy_base.update(
            structural_interfaces=structural_interfaces,
            structural_burgers_m=(
                context.structural_absorption_parameters.boundary.burgers_m),
            structural_parameters=(
                context.structural_absorption_parameters.boundary),
            junction_disclination_parameters=JunctionDisclinationParameters())
    structural_junction_wall_time = (
        time.perf_counter()-structural_junction_started)
    interface_mobile_emission_started = time.perf_counter()
    interface_mobile_emission = None
    if context.interface_mobile_emission_mode != "disabled":
        if context.interface_mobile_emission_mode != "arrhenius_neutral_pair":
            raise ValueError("unknown interface mobile-emission mode")
        if context.source_parameters is None:
            raise ValueError(
                "interface mobile emission requires Hall--Petch source parameters")
        interface_mobile_emission = evaluate_interface_mobile_pair_emission(
            mechanical_parent_state, mean_strain=strain0, interval_s=dt,
            systems=context.systems, wall_parameters=context.wall_parameters,
            represented_thickness_m=context.represented_thickness_m,
            source_parameters=context.source_parameters,
            energy_kwargs=energy_base)
        if interface_mobile_emission.classification in {
                "REJECTED_LINE_BURGERS_OR_MURA_BALANCE",
                "REJECTED_COMPLETE_FIRST_LAW"}:
            raise RuntimeError(
                "interface mobile emission failed a hard balance audit")
        if interface_mobile_emission.accepted:
            mechanical_parent_state = (
                interface_mobile_emission.committed_state)
        projection = interface_mobile_emission.projection
        runtime = replace(runtime, ledger=replace(
            runtime.ledger,
            interface_mobile_emission_intervals=(
                runtime.ledger.interface_mobile_emission_intervals
                +int(interface_mobile_emission.accepted)),
            interface_mobile_emission_inactive_intervals=(
                runtime.ledger.interface_mobile_emission_inactive_intervals
                +int(interface_mobile_emission.classification.startswith(
                    "INACTIVE"))),
            interface_mobile_emission_rejected_intervals=(
                runtime.ledger.interface_mobile_emission_rejected_intervals
                +int(interface_mobile_emission.classification.startswith(
                    "REJECTED"))),
            interface_mobile_emitted_pair_cell_sum_m2=(
                runtime.ledger.interface_mobile_emitted_pair_cell_sum_m2
                +(projection.created_lattice_pair_cell_sum_m2
                  if interface_mobile_emission.accepted else 0.0)),
            cumulative_interface_mobile_emission_heat_J=(
                runtime.ledger.cumulative_interface_mobile_emission_heat_J
                +(interface_mobile_emission.generated_heat_J
                  if interface_mobile_emission.accepted else 0.0)),
            maximum_interface_mobile_emission_first_law_residual=max(
                runtime.ledger.
                maximum_interface_mobile_emission_first_law_residual,
                interface_mobile_emission.relative_first_law_residual
                if interface_mobile_emission.accepted else 0.0),
            maximum_interface_mobile_emission_line_balance_residual_m2=max(
                runtime.ledger.
                maximum_interface_mobile_emission_line_balance_residual_m2,
                abs(projection.line_balance_residual_cell_sum_m2)),
            maximum_interface_mobile_emission_mura_residual=max(
                runtime.ledger.maximum_interface_mobile_emission_mura_residual,
                projection.maximum_mura_relative_residual)))
    interface_mobile_emission_wall_time = (
        time.perf_counter()-interface_mobile_emission_started)
    energy_before = dict(energy_base, mean_strain=strain0)
    collective_wall_started = time.perf_counter()
    collective_wall = None
    if context.collective_wall_nucleation_mode != "disabled":
        if context.collective_wall_nucleation_mode not in {
                "activated_closed_wall", "continuous_closed_wall"}:
            raise ValueError("unknown collective wall-nucleation mode")
        parameters = (CollectiveWallNucleationParameters()
                      if context.collective_wall_nucleation_parameters is None
                      else context.collective_wall_nucleation_parameters)
        draw = float(np.random.default_rng(
            int(context.collective_wall_nucleation_seed)
            +int(runtime.ledger.intervals)).random())
        collective_wall = evaluate_collective_wall_nucleation(
            mechanical_parent_state, interval_s=dt, mean_strain=strain0,
            spacing_m=context.spacing_m,
            represented_thickness_m=context.represented_thickness_m,
            systems=context.systems, wall_parameters=context.wall_parameters,
            energy_kwargs=energy_base, random_uniform=draw,
            parameters=parameters)
        if collective_wall.classification == "REJECTED_CLOSED_WALL_FIRST_LAW":
            raise RuntimeError("collective wall nucleus failed its hard energy audit")
        if collective_wall.accepted:
            mechanical_parent_state = collective_wall.committed_state
        runtime = replace(runtime, ledger=replace(
            runtime.ledger,
            collective_wall_nucleation_intervals=(
                runtime.ledger.collective_wall_nucleation_intervals
                +int(collective_wall.accepted)),
            collective_wall_nucleation_inactive_intervals=(
                runtime.ledger.collective_wall_nucleation_inactive_intervals
                +int(collective_wall.classification.startswith("INACTIVE"))),
            collective_wall_nucleation_rejected_intervals=(
                runtime.ledger.collective_wall_nucleation_rejected_intervals
                +int(collective_wall.classification.startswith("REJECTED"))),
            cumulative_collective_wall_nucleation_heat_J=(
                runtime.ledger.cumulative_collective_wall_nucleation_heat_J
                +(collective_wall.generated_heat_J
                  if collective_wall.accepted else 0.0)),
            maximum_collective_wall_nucleation_first_law_residual=max(
                runtime.ledger.maximum_collective_wall_nucleation_first_law_residual,
                collective_wall.relative_first_law_residual
                if collective_wall.accepted else 0.0)))
    collective_wall_wall_time = time.perf_counter()-collective_wall_started
    mechanical_started = time.perf_counter()
    mechanical_wall_parameters = context.wall_parameters
    if context.wall_order_relaxation_mode != "disabled":
        # The independently audited implicit transaction owns q for this
        # physical interval; transport/reactions/slip retain their production
        # equations and cannot double-advance the stiff ordering channel.
        mechanical_wall_parameters = replace(
            context.wall_parameters, wall_order_enabled=False)
    mechanical = advance_energy_qualified_mechanics(
        mechanical_parent_state, mean_strain_before=strain0,
        mean_strain_candidate=strain1, systems=context.systems,
        topologies=context.topologies, wall_parameters=mechanical_wall_parameters,
        dt_s=dt, represented_thickness_m=context.represented_thickness_m,
        energy_kwargs=energy_before, maximum_relative_first_law_residual=.05,
        maximum_subdivisions=context.maximum_mechanical_subdivisions,
        maximum_internal_substeps=(
            context.maximum_mechanical_internal_substeps),
        preserve_owner_mean_orientation=bool(structural_interfaces))
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
    multiplication_raw = sum(
        (np.asarray(item.raw_multiplication_line_increment_m2_cells)
         for item in mechanical.operator_decisions),
        start=np.zeros(parent.state.supports.shape[1:], dtype=float))
    multiplication_accepted = sum(
        (np.asarray(item.accepted_multiplication_line_increment_m2_cells)
         for item in mechanical.operator_decisions),
        start=np.zeros(parent.state.supports.shape[1:], dtype=float))
    multiplication_budget_active_time = sum(
        (np.asarray(item.multiplication_budget_active_time_s_cells)
         for item in mechanical.operator_decisions),
        start=np.zeros(parent.state.supports.shape[1:], dtype=float))
    multiplication_raw_owner_family = sum(
        (np.asarray(
            item.raw_multiplication_line_increment_m2_owner_family_cells)
         for item in mechanical.operator_decisions),
        start=np.zeros(
            (len(parent.state.owners),)+parent.state.supports.shape[1:]
            +(len(context.systems),), dtype=float))
    multiplication_accepted_owner_family = sum(
        (np.asarray(
            item.accepted_multiplication_line_increment_m2_owner_family_cells)
         for item in mechanical.operator_decisions),
        start=np.zeros_like(multiplication_raw_owner_family))
    signed_slip_owner_family = sum(
        (np.asarray(item.signed_slip_increment_owner_family_cells)
         for item in mechanical.operator_decisions),
        start=np.zeros_like(multiplication_raw_owner_family))
    absolute_slip_owner_family = sum(
        (np.asarray(item.absolute_slip_increment_owner_family_cells)
         for item in mechanical.operator_decisions),
        start=np.zeros_like(multiplication_raw_owner_family))
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
        if context.source_mode == "gb_tj_only":
            source_sites = boundary_tj_site_field(
                source_state.supports,
                tj_site_weight=context.source_tj_weight)
        elif context.source_mode == "matched_random":
            boundary_sites = boundary_tj_site_field(
                source_state.supports,
                tj_site_weight=context.source_tj_weight)
            source_sites = randomized_matched_site_field(
                boundary_sites, context.source_random_seed)
        elif context.source_mode == "random_bulk":
            # A physical bulk-source hypothesis, not the shuffled GB/TJ
            # geography control.  This remains meaningful for a single
            # crystal, where the boundary measure is identically zero.
            source_sites = random_bulk_site_field(
                source_state.supports.shape[1:],
                context.source_random_seed)
        elif context.source_mode == "random_discrete_bulk":
            source_sites = random_discrete_bulk_site_field(
                source_state.supports.shape[1:], context.spacing_m,
                context.source_bulk_site_density_m2,
                context.source_random_seed)
        else:
            raise ValueError("unknown dislocation source mode")
        source = advance_energy_qualified_pair_source_transaction(
            source_state, mean_strain=strain1, site_field=source_sites,
            interval_s=dt, source_parameters=context.source_parameters,
            systems=context.systems, wall_parameters=context.wall_parameters,
            represented_thickness_m=context.represented_thickness_m,
            energy_kwargs=energy_base, source_mode=context.source_mode,
            maximum_subdivisions=context.maximum_source_subdivisions,
            preserve_owner_mean_orientation=bool(structural_interfaces))
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
    wall_order_started = time.perf_counter()
    wall_order = None
    ordered_state = source_state
    if context.wall_order_relaxation_mode != "disabled":
        if context.wall_order_relaxation_mode != "implicit_arrhenius":
            raise ValueError("unknown wall-order relaxation mode")
        parameters = (ImplicitWallOrderRelaxationParameters()
                      if context.wall_order_relaxation_parameters is None
                      else context.wall_order_relaxation_parameters)
        wall_order = evaluate_implicit_wall_order_relaxation(
            ordered_state, interval_s=dt, mean_strain=strain1,
            spacing_m=context.spacing_m,
            represented_thickness_m=context.represented_thickness_m,
            topologies=context.topologies, systems=context.systems,
            wall_parameters=context.wall_parameters,
            energy_kwargs=energy_base, parameters=parameters)
        if wall_order.classification == "REJECTED_WALL_ORDER_FIRST_LAW":
            raise RuntimeError("wall-order relaxation failed its hard energy audit")
        if wall_order.accepted:
            ordered_state = wall_order.committed_state
        runtime = replace(runtime, ledger=replace(
            runtime.ledger,
            wall_order_relaxation_intervals=(
                runtime.ledger.wall_order_relaxation_intervals
                +int(wall_order.accepted)),
            wall_order_relaxation_inactive_intervals=(
                runtime.ledger.wall_order_relaxation_inactive_intervals
                +int(wall_order.classification.startswith("INACTIVE"))),
            wall_order_relaxation_rejected_intervals=(
                runtime.ledger.wall_order_relaxation_rejected_intervals
                +int(wall_order.classification.startswith("REJECTED"))),
            cumulative_wall_order_relaxation_heat_J=(
                runtime.ledger.cumulative_wall_order_relaxation_heat_J
                +(wall_order.generated_heat_J if wall_order.accepted else 0.0)),
            maximum_wall_order_relaxation_first_law_residual=max(
                runtime.ledger.maximum_wall_order_relaxation_first_law_residual,
                wall_order.relative_first_law_residual
                if wall_order.accepted else 0.0)))
    wall_order_wall_time = time.perf_counter()-wall_order_started
    dipole_wall_started = time.perf_counter()
    dipole_wall = None
    dipole_state = ordered_state
    if context.conservative_dipole_wall_mode != "disabled":
        if context.conservative_dipole_wall_mode != "neutral_pair_expansion":
            raise ValueError("unknown conservative dipole-wall mode")
        parameters = (ConservativeDipoleWallParameters()
                      if context.conservative_dipole_wall_parameters is None
                      else context.conservative_dipole_wall_parameters)
        dipole_wall = evaluate_conservative_dipole_wall_organization(
            dipole_state, interval_s=dt, mean_strain=strain1,
            spacing_m=context.spacing_m,
            represented_thickness_m=context.represented_thickness_m,
            systems=context.systems, wall_parameters=context.wall_parameters,
            energy_kwargs=energy_base, parameters=parameters)
        if dipole_wall.classification.startswith("REJECTED"):
            raise RuntimeError(
                "conservative dipole-wall event failed a hard audit")
        if dipole_wall.accepted:
            dipole_state = dipole_wall.committed_state
        maximum_inventory_residual = max(
            abs(dipole_wall.plus_inventory_residual_m2),
            abs(dipole_wall.minus_inventory_residual_m2),
            abs(dipole_wall.line_balance_residual_m2))
        runtime = replace(runtime, ledger=replace(
            runtime.ledger,
            conservative_dipole_wall_intervals=(
                runtime.ledger.conservative_dipole_wall_intervals
                +int(dipole_wall.accepted)),
            conservative_dipole_wall_inactive_intervals=(
                runtime.ledger.conservative_dipole_wall_inactive_intervals
                +int(dipole_wall.classification.startswith("INACTIVE"))),
            conservative_dipole_wall_rejected_intervals=(
                runtime.ledger.conservative_dipole_wall_rejected_intervals
                +int(dipole_wall.classification.startswith("REJECTED"))),
            conservative_dipole_wall_pair_cell_sum_m2=(
                runtime.ledger.conservative_dipole_wall_pair_cell_sum_m2
                +(dipole_wall.separated_pair_cell_sum_m2
                  if dipole_wall.accepted else 0.0)),
            cumulative_conservative_dipole_wall_heat_J=(
                runtime.ledger.cumulative_conservative_dipole_wall_heat_J
                +(dipole_wall.generated_heat_J
                  if dipole_wall.accepted else 0.0)),
            maximum_conservative_dipole_wall_first_law_residual=max(
                runtime.ledger.
                maximum_conservative_dipole_wall_first_law_residual,
                dipole_wall.relative_first_law_residual
                if dipole_wall.accepted else 0.0),
            maximum_conservative_dipole_wall_inventory_residual_m2=max(
                runtime.ledger.
                maximum_conservative_dipole_wall_inventory_residual_m2,
                maximum_inventory_residual)))
    dipole_wall_wall_time = time.perf_counter()-dipole_wall_started
    wall_glide_started = time.perf_counter()
    wall_glide = None
    glide_state = dipole_state
    if context.conservative_wall_glide_mode != "disabled":
        if context.conservative_wall_glide_mode != "chemical_edge_glide":
            raise ValueError("unknown conservative wall-glide mode")
        parameters = (ConservativeWallGlideParameters()
                      if context.conservative_wall_glide_parameters is None
                      else context.conservative_wall_glide_parameters)
        wall_glide = evaluate_conservative_wall_glide(
            glide_state, interval_s=dt, mean_strain=strain1,
            spacing_m=context.spacing_m,
            represented_thickness_m=context.represented_thickness_m,
            systems=context.systems, wall_parameters=context.wall_parameters,
            energy_kwargs=energy_base, parameters=parameters)
        if wall_glide.classification in {
                "REJECTED_CONSERVATIVE_WALL_GLIDE_FIRST_LAW",
                "REJECTED_CONSERVATIVE_WALL_GLIDE_BALANCE"}:
            raise RuntimeError("conservative wall glide failed its hard energy audit")
        if wall_glide.accepted:
            glide_state = wall_glide.committed_state
        runtime = replace(runtime, ledger=replace(
            runtime.ledger,
            conservative_wall_glide_intervals=(
                runtime.ledger.conservative_wall_glide_intervals
                +int(wall_glide.accepted)),
            conservative_wall_glide_inactive_intervals=(
                runtime.ledger.conservative_wall_glide_inactive_intervals
                +int(wall_glide.classification.startswith("INACTIVE"))),
            conservative_wall_glide_rejected_intervals=(
                runtime.ledger.conservative_wall_glide_rejected_intervals
                +int(wall_glide.classification.startswith("REJECTED"))),
            conservative_wall_glide_line_cell_sum_m2=(
                runtime.ledger.conservative_wall_glide_line_cell_sum_m2
                +(wall_glide.moved_line_cell_sum_m2
                  if wall_glide.accepted else 0.0)),
            cumulative_conservative_wall_glide_heat_J=(
                runtime.ledger.cumulative_conservative_wall_glide_heat_J
                +(wall_glide.generated_heat_J if wall_glide.accepted else 0.0)),
            maximum_conservative_wall_glide_first_law_residual=max(
                runtime.ledger.maximum_conservative_wall_glide_first_law_residual,
                wall_glide.relative_first_law_residual
                if wall_glide.accepted else 0.0),
            maximum_conservative_wall_glide_inventory_residual_m2=max(
                runtime.ledger.maximum_conservative_wall_glide_inventory_residual_m2,
                abs(wall_glide.inventory_residual_m2))))
    wall_glide_wall_time = time.perf_counter()-wall_glide_started
    wall_recovery_started = time.perf_counter()
    wall_recovery = None
    recovered_state = glide_state
    if context.wall_neutral_pair_recovery_mode != "disabled":
        if context.wall_neutral_pair_recovery_mode != "arrhenius_climb":
            raise ValueError("unknown wall neutral-pair recovery mode")
        parameters = (WallNeutralPairRecoveryParameters()
                      if context.wall_neutral_pair_recovery_parameters is None
                      else context.wall_neutral_pair_recovery_parameters)
        wall_recovery = evaluate_wall_neutral_pair_recovery(
            recovered_state, interval_s=dt, mean_strain=strain1,
            spacing_m=context.spacing_m,
            represented_thickness_m=context.represented_thickness_m,
            systems=context.systems, wall_parameters=context.wall_parameters,
            energy_kwargs=energy_base, parameters=parameters)
        if wall_recovery.classification.startswith("REJECTED"):
            raise RuntimeError(
                "wall neutral-pair recovery failed a hard balance audit")
        if wall_recovery.accepted:
            recovered_state = wall_recovery.committed_state
        maximum_balance_residual = max(
            abs(wall_recovery.signed_content_residual_m2),
            abs(wall_recovery.line_balance_residual_m2))
        runtime = replace(runtime, ledger=replace(
            runtime.ledger,
            wall_neutral_pair_recovery_intervals=(
                runtime.ledger.wall_neutral_pair_recovery_intervals
                +int(wall_recovery.accepted)),
            wall_neutral_pair_recovery_inactive_intervals=(
                runtime.ledger.wall_neutral_pair_recovery_inactive_intervals
                +int(wall_recovery.classification.startswith("INACTIVE"))),
            wall_neutral_pair_recovery_rejected_intervals=(
                runtime.ledger.wall_neutral_pair_recovery_rejected_intervals
                +int(wall_recovery.classification.startswith("REJECTED"))),
            wall_neutral_pair_recovery_annihilated_pair_cell_sum_m2=(
                runtime.ledger.
                wall_neutral_pair_recovery_annihilated_pair_cell_sum_m2
                +(wall_recovery.annihilated_pair_cell_sum_m2
                  if wall_recovery.accepted else 0.0)),
            cumulative_wall_neutral_pair_recovery_heat_J=(
                runtime.ledger.cumulative_wall_neutral_pair_recovery_heat_J
                +(wall_recovery.generated_heat_J
                  if wall_recovery.accepted else 0.0)),
            maximum_wall_neutral_pair_recovery_first_law_residual=max(
                runtime.ledger.
                maximum_wall_neutral_pair_recovery_first_law_residual,
                wall_recovery.relative_first_law_residual
                if wall_recovery.accepted else 0.0),
            maximum_wall_neutral_pair_recovery_balance_residual_m2=max(
                runtime.ledger.
                maximum_wall_neutral_pair_recovery_balance_residual_m2,
                maximum_balance_residual)))
    wall_recovery_wall_time = time.perf_counter()-wall_recovery_started
    wall_reorganization_started = time.perf_counter()
    wall_reorganization = None
    # Redundant neutral content recovers first; the compatible reaction cone
    # then redistributes only the surviving signed structural excess.
    reorganized_state = recovered_state
    if context.compatible_wall_reorganization_mode != "disabled":
        if context.compatible_wall_reorganization_mode != "frank_bilby_projection":
            raise ValueError("unknown compatible wall-reorganization mode")
        parameters = (CompatibleWallReorganizationParameters()
                      if context.compatible_wall_reorganization_parameters is None
                      else context.compatible_wall_reorganization_parameters)
        wall_reorganization = evaluate_compatible_wall_reorganization(
            reorganized_state, interval_s=dt, mean_strain=strain1,
            spacing_m=context.spacing_m,
            represented_thickness_m=context.represented_thickness_m,
            systems=context.systems, wall_parameters=context.wall_parameters,
            energy_kwargs=energy_base, parameters=parameters)
        if wall_reorganization.classification == (
                "REJECTED_COMPATIBLE_WALL_GLIDE_FIRST_LAW"):
            raise RuntimeError(
                "compatible wall reorganization failed its hard energy audit")
        if wall_reorganization.accepted:
            reorganized_state = wall_reorganization.committed_state
        runtime = replace(runtime, ledger=replace(
            runtime.ledger,
            compatible_wall_reorganization_intervals=(
                runtime.ledger.compatible_wall_reorganization_intervals
                +int(wall_reorganization.accepted)),
            compatible_wall_reorganization_inactive_intervals=(
                runtime.ledger.compatible_wall_reorganization_inactive_intervals
                +int(wall_reorganization.classification.startswith("INACTIVE"))),
            compatible_wall_reorganization_rejected_intervals=(
                runtime.ledger.compatible_wall_reorganization_rejected_intervals
                +int(wall_reorganization.classification.startswith("REJECTED"))),
            cumulative_compatible_wall_reorganization_heat_J=(
                runtime.ledger.cumulative_compatible_wall_reorganization_heat_J
                +(wall_reorganization.generated_heat_J
                  if wall_reorganization.accepted else 0.0)),
            maximum_compatible_wall_reorganization_first_law_residual=max(
                runtime.ledger.maximum_compatible_wall_reorganization_first_law_residual,
                wall_reorganization.relative_first_law_residual
                if wall_reorganization.accepted else 0.0)))
    wall_reorganization_wall_time = (
        time.perf_counter()-wall_reorganization_started)
    joint_topology_started = time.perf_counter()
    joint_topology = None
    joint_state = reorganized_state
    if context.joint_wall_topology_orientation_mode != "disabled":
        if context.joint_wall_topology_orientation_mode != "nye_reconstructed_ring":
            raise ValueError("unknown joint wall-topology/orientation mode")
        parameters = (JointWallTopologyOrientationParameters()
                      if context.joint_wall_topology_orientation_parameters is None
                      else context.joint_wall_topology_orientation_parameters)
        joint_topology = evaluate_joint_wall_topology_orientation(
            joint_state, interval_s=dt, mean_strain=strain1,
            spacing_m=context.spacing_m,
            represented_thickness_m=context.represented_thickness_m,
            systems=context.systems, wall_parameters=context.wall_parameters,
            energy_kwargs=energy_base, parameters=parameters)
        if joint_topology.classification.startswith("REJECTED"):
            raise RuntimeError(
                "joint wall topology/orientation failed a hard audit")
        if joint_topology.accepted:
            joint_state = joint_topology.committed_state
        maximum_balance_residual = max(
            abs(joint_topology.plus_inventory_residual_m2),
            abs(joint_topology.minus_inventory_residual_m2),
            abs(joint_topology.line_balance_residual_m2))
        runtime = replace(runtime, ledger=replace(
            runtime.ledger,
            joint_wall_topology_orientation_intervals=(
                runtime.ledger.joint_wall_topology_orientation_intervals
                +int(joint_topology.accepted)),
            joint_wall_topology_orientation_inactive_intervals=(
                runtime.ledger.joint_wall_topology_orientation_inactive_intervals
                +int(joint_topology.classification.startswith("INACTIVE"))),
            joint_wall_topology_orientation_rejected_intervals=(
                runtime.ledger.joint_wall_topology_orientation_rejected_intervals
                +int(joint_topology.classification.startswith("REJECTED"))),
            joint_wall_topology_orientation_transferred_line_cell_sum_m2=(
                runtime.ledger.
                joint_wall_topology_orientation_transferred_line_cell_sum_m2
                +(joint_topology.transferred_line_cell_sum_m2
                  if joint_topology.accepted else 0.0)),
            cumulative_joint_wall_topology_orientation_heat_J=(
                runtime.ledger.cumulative_joint_wall_topology_orientation_heat_J
                +(joint_topology.generated_heat_J
                  if joint_topology.accepted else 0.0)),
            maximum_joint_wall_topology_orientation_first_law_residual=max(
                runtime.ledger.
                maximum_joint_wall_topology_orientation_first_law_residual,
                joint_topology.relative_first_law_residual
                if joint_topology.accepted else 0.0),
            maximum_joint_wall_topology_orientation_balance_residual_m2=max(
                runtime.ledger.
                maximum_joint_wall_topology_orientation_balance_residual_m2,
                maximum_balance_residual)))
    joint_topology_wall_time = time.perf_counter()-joint_topology_started
    # The neutral sink is a distinct zero-work physical channel. It follows
    # line creation so a same-site source/sink hypothesis has an unambiguous
    # operator order and separate line/heat ledgers.
    sink_started = time.perf_counter()
    sink = None
    sink_state = joint_state
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
        if context.interface_sink_mode not in {
                "neutral_pair", "vector_neutral_junction"}:
            raise ValueError("unknown interface sink mode")
        interface_sink_evaluator = (
            evaluate_interface_vector_neutral_junction_transaction
            if context.interface_sink_mode == "vector_neutral_junction"
            else evaluate_interface_neutral_sink_transaction)
        interface_sink = interface_sink_evaluator(
            interface_sink_state, mean_strain=strain1, interval_s=dt,
            sink_parameters=context.interface_sink_parameters,
            wall_parameters=context.wall_parameters,
            represented_thickness_m=context.represented_thickness_m,
            energy_kwargs=energy_base,
            sink_mode=context.interface_sink_mode)
        if ("FIRST_LAW" in interface_sink.classification
                and interface_sink.classification.startswith("REJECTED")):
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
    # The retained structural graph owns owner-mean common-axis rotations;
    # mechanical plastic spin and source kinematics own only intragranular
    # fluctuations until an explicit GB accommodation transaction is added.
    # Reapply that existing constraint before the same-interval structural
    # handoff (rather than only after front motion), otherwise an accepted
    # high-rate spin increment leaves the edge state stale and the atomic
    # absorption audit correctly refuses to proceed.
    pre_post_structural_orientation_sync_max_rad = 0.0
    if structural_interfaces:
        means_before_sync = np.asarray([
            np.mean(owner.orientation_rad)
            for owner in structural_state.owners])
        structural_state = synchronize_structural_owner_orientations(
            structural_state,
            StructuralMultigrainRuntime(structural_interfaces),
            spacing_m=context.spacing_m,
            burgers_m=(
                context.structural_absorption_parameters.boundary.burgers_m))
        means_after_sync = np.asarray([
            np.mean(owner.orientation_rad)
            for owner in structural_state.owners])
        pre_post_structural_orientation_sync_max_rad = float(np.max(
            np.abs(means_after_sync-means_before_sync), initial=0.0))
        if pre_post_structural_orientation_sync_max_rad > 1e-10:
            raise RuntimeError(
                "unpriced owner-mean orientation change before structural handoff")
    # Complete the newly captured lattice -> interface -> structural handoff
    # inside the same physical interval.  The start-of-interval transaction
    # clears retained reservoir content; this second transaction sees only
    # content delivered by the mechanical/lattice map above.  With the exact
    # Arrhenius clocks enabled, neither operation advances a numerical fixed
    # fraction per outer call.
    structural_absorption_post = None
    structural_junction_post = None
    if (context.structural_absorption_mode != "disabled"
            and lattice_absorption is not None
            and lattice_absorption.accepted):
        structural_absorption_post = evaluate_interface_structural_absorption(
            structural_state, structural_interfaces, mean_strain=strain1,
            interval_s=dt, spacing_m=context.spacing_m,
            represented_thickness_m=context.represented_thickness_m,
            wall_parameters=context.wall_parameters, energy_kwargs=energy_base,
            parameters=context.structural_absorption_parameters)
        if structural_absorption_post.classification in {
                "REJECTED_BURGERS_OR_LINE_BALANCE",
                "REJECTED_COMPLETE_FIRST_LAW"}:
            raise RuntimeError(
                "post-capture structural absorption failed a hard balance audit: "
                f"{structural_absorption_post.classification}; "
                "signed_content_balance="
                f"{structural_absorption_post.projection.signed_content_balance:.17g}; "
                "maximum_vector_balance_m1="
                f"{structural_absorption_post.projection.maximum_vector_balance_m1:.17g}; "
                "relative_vector_balance="
                f"{structural_absorption_post.projection.relative_vector_balance:.17g}")
        if structural_absorption_post.accepted:
            structural_state = structural_absorption_post.committed_state
            structural_interfaces = (
                structural_absorption_post.committed_interfaces)
        runtime = replace(runtime, structural_interfaces=structural_interfaces,
                          ledger=replace(
            runtime.ledger,
            structural_absorption_intervals=(
                runtime.ledger.structural_absorption_intervals
                +int(structural_absorption_post.accepted)),
            structural_absorption_inactive_intervals=(
                runtime.ledger.structural_absorption_inactive_intervals
                +int(structural_absorption_post.classification.startswith(
                    "INACTIVE"))),
            structural_absorption_rejected_intervals=(
                runtime.ledger.structural_absorption_rejected_intervals
                +int(structural_absorption_post.classification.startswith(
                    "REJECTED"))),
            structural_absorbed_line_length_m=(
                runtime.ledger.structural_absorbed_line_length_m
                +(structural_absorption_post.projection.removed_line_length_m
                  *context.represented_thickness_m
                  if structural_absorption_post.accepted else 0.0)),
            cumulative_structural_absorption_heat_J=(
                runtime.ledger.cumulative_structural_absorption_heat_J
                +(structural_absorption_post.generated_heat_J
                  if structural_absorption_post.accepted else 0.0)),
            maximum_structural_absorption_first_law_residual=max(
                runtime.ledger.maximum_structural_absorption_first_law_residual,
                structural_absorption_post.relative_first_law_residual
                if structural_absorption_post.accepted else 0.0)))
        energy_base.update(
            structural_interfaces=structural_interfaces,
            structural_burgers_m=(
                context.structural_absorption_parameters.boundary.burgers_m),
            structural_parameters=(
                context.structural_absorption_parameters.boundary),
            junction_disclination_parameters=JunctionDisclinationParameters())
    if (context.structural_junction_mode != "disabled"
            and structural_absorption_post is not None
            and structural_absorption_post.accepted):
        parameters = (JunctionReactionParameters()
                      if context.structural_junction_parameters is None
                      else context.structural_junction_parameters)
        (structural_state, junction_runtime,
         structural_junction_post) = advance_structural_junction_interval(
            structural_state,
            StructuralMultigrainRuntime(structural_interfaces),
            dt_s=dt, spacing_m=context.spacing_m,
            represented_thickness_m=context.represented_thickness_m,
            wall_parameters=context.wall_parameters,
            energy_kwargs=dict(context.energy_kwargs, mean_strain=strain1),
            boundary_parameters=(
                context.structural_absorption_parameters.boundary),
            disclination_parameters=JunctionDisclinationParameters(),
            reaction_parameters=parameters)
        if structural_junction_post.classification == "REJECTED_COMPLETE_FIRST_LAW":
            raise RuntimeError(
                "post-capture structural junction reaction failed its hard "
                "energy audit: "
                f"relative="
                f"{structural_junction_post.relative_first_law_residual:.17g}, "
                f"residual_J="
                f"{structural_junction_post.first_law_residual_J:.17g}, "
                f"heat_J={structural_junction_post.generated_heat_J:.17g}, "
                f"candidate_delta_J="
                f"{structural_junction_post.complete_energy_candidate.internal_J-structural_junction_post.complete_energy_before.internal_J:.17g}")
        structural_interfaces = tuple(junction_runtime.interfaces)
        runtime = replace(
            runtime, structural_interfaces=structural_interfaces,
            ledger=replace(
                runtime.ledger,
                structural_junction_intervals=(
                    runtime.ledger.structural_junction_intervals
                    +int(structural_junction_post.accepted)),
                structural_junction_inactive_intervals=(
                    runtime.ledger.structural_junction_inactive_intervals
                    +int(structural_junction_post.classification
                         == "NO_CYCLE_REACTION")),
                structural_junction_rejected_intervals=(
                    runtime.ledger.structural_junction_rejected_intervals
                    +int(structural_junction_post.classification.startswith(
                        "REJECTED"))),
                structural_junction_reacted_absolute_content=(
                    runtime.ledger.structural_junction_reacted_absolute_content
                    +(structural_junction_post.kinetic_proposal.
                      absolute_reacted_content
                      if structural_junction_post.accepted else 0.0)),
                cumulative_structural_junction_heat_J=(
                    runtime.ledger.cumulative_structural_junction_heat_J
                    +(structural_junction_post.generated_heat_J
                      if structural_junction_post.accepted else 0.0)),
                maximum_structural_junction_first_law_residual=max(
                    runtime.ledger.
                    maximum_structural_junction_first_law_residual,
                    structural_junction_post.relative_first_law_residual
                    if structural_junction_post.accepted else 0.0)))
        energy_base.update(
            structural_interfaces=structural_interfaces,
            structural_burgers_m=(
                context.structural_absorption_parameters.boundary.burgers_m),
            structural_parameters=(
                context.structural_absorption_parameters.boundary),
            junction_disclination_parameters=JunctionDisclinationParameters())
    wall_orientation_started = time.perf_counter()
    wall_orientation = None
    if structural_interfaces:
        wall_entry_synchronized = synchronize_structural_owner_orientations(
            structural_state,
            StructuralMultigrainRuntime(structural_interfaces),
            spacing_m=context.spacing_m,
            burgers_m=(
                context.structural_absorption_parameters.boundary.burgers_m))
        wall_entry_mismatch = float(np.max(np.abs(np.asarray([
            np.mean(owner.orientation_rad)
            for owner in wall_entry_synchronized.owners])-np.asarray([
                np.mean(owner.orientation_rad)
                for owner in structural_state.owners])), initial=0.0))
        if wall_entry_mismatch > 1e-10:
            raise RuntimeError(
                "unpriced owner-mean orientation change before ordered-wall "
                f"relaxation: {wall_entry_mismatch:.17g} rad")
    if context.wall_orientation_relaxation_mode != "disabled":
        if context.wall_orientation_relaxation_mode != "ordered_nye":
            raise ValueError("unknown wall orientation relaxation mode")
        if context.wall_orientation_relaxation_parameters is None:
            raise ValueError(
                "enabled wall orientation relaxation requires parameters")
        wall_orientation = evaluate_wall_orientation_relaxation(
            structural_state, interval_s=dt, spacing_m=context.spacing_m,
            represented_thickness_m=context.represented_thickness_m,
            systems=context.systems, wall_parameters=context.wall_parameters,
            energy_kwargs=dict(energy_base, mean_strain=strain1),
            parameters=context.wall_orientation_relaxation_parameters,
            preserve_owner_mean_orientation=bool(structural_interfaces))
        if wall_orientation.classification == "REJECTED_COMPLETE_FIRST_LAW":
            raise RuntimeError(
                "wall orientation relaxation failed its hard energy audit")
        if wall_orientation.accepted:
            structural_state = wall_orientation.committed_state
        if structural_interfaces:
            wall_exit_synchronized = synchronize_structural_owner_orientations(
                structural_state,
                StructuralMultigrainRuntime(structural_interfaces),
                spacing_m=context.spacing_m,
                burgers_m=(context.structural_absorption_parameters.boundary.
                           burgers_m))
            wall_exit_mismatch = float(np.max(np.abs(np.asarray([
                np.mean(owner.orientation_rad)
                for owner in wall_exit_synchronized.owners])-np.asarray([
                    np.mean(owner.orientation_rad)
                    for owner in structural_state.owners])), initial=0.0))
            if wall_exit_mismatch > 1e-10:
                raise RuntimeError(
                    "unpriced owner-mean orientation change in ordered-wall "
                    f"relaxation: {wall_exit_mismatch:.17g} rad")
        runtime = replace(runtime, ledger=replace(
            runtime.ledger,
            wall_orientation_relaxation_intervals=(
                runtime.ledger.wall_orientation_relaxation_intervals
                +int(wall_orientation.accepted)),
            wall_orientation_relaxation_inactive_intervals=(
                runtime.ledger.wall_orientation_relaxation_inactive_intervals
                +int(wall_orientation.classification.startswith("INACTIVE"))),
            wall_orientation_relaxation_rejected_intervals=(
                runtime.ledger.wall_orientation_relaxation_rejected_intervals
                +int(wall_orientation.classification.startswith("REJECTED"))),
            cumulative_wall_orientation_relaxation_heat_J=(
                runtime.ledger.cumulative_wall_orientation_relaxation_heat_J
                +(wall_orientation.generated_heat_J
                  if wall_orientation.accepted else 0.0)),
            maximum_wall_orientation_relaxation_first_law_residual=max(
                runtime.ledger.maximum_wall_orientation_relaxation_first_law_residual,
                wall_orientation.relative_first_law_residual
                if wall_orientation.accepted else 0.0)))
    wall_orientation_wall_time = time.perf_counter()-wall_orientation_started
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
            synchronize_structural_owner_orientation=bool(
                structural_interfaces),
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
    final_structural_orientation_sync_max_rad = 0.0
    if structural_interfaces:
        means_before_final_sync = np.asarray([
            np.mean(owner.orientation_rad) for owner in state.owners])
        state = synchronize_structural_owner_orientations(
            state, StructuralMultigrainRuntime(structural_interfaces),
            spacing_m=context.spacing_m,
            burgers_m=context.structural_absorption_parameters.boundary.burgers_m)
        means_after_final_sync = np.asarray([
            np.mean(owner.orientation_rad) for owner in state.owners])
        final_structural_orientation_sync_max_rad = float(np.max(
            np.abs(means_after_final_sync-means_before_final_sync), initial=0.0))
        if final_structural_orientation_sync_max_rad > 1e-10:
            raise RuntimeError(
                "unpriced owner-mean orientation change before interval commit: "
                f"{final_structural_orientation_sync_max_rad:.17g} rad")
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
        "collective_wall_nucleation_heat_J": (
            0.0 if collective_wall is None or not collective_wall.accepted
            else collective_wall.generated_heat_J),
        "collective_wall_nucleation_heat_J_by_cell": (
            np.zeros(parent.state.supports.shape[1:], dtype=float)
            if collective_wall is None or not collective_wall.accepted
            else np.asarray(collective_wall.heat_source_J_by_cell)),
        "front_heat_J": float(np.sum(front_heat, dtype=np.longdouble)),
        "source_resolved_work_J": (0.0 if source is None or not source.accepted
                                    else source.resolved_work_J),
        "source_heat_J": (0.0 if source is None or not source.accepted
                           else source.generated_heat_J),
        "source_heat_J_by_cell": (
            np.zeros(parent.state.supports.shape[1:], dtype=float)
            if source is None or not source.accepted
            else np.asarray(source.heat_source_J_by_cell)),
        "interface_mobile_emission_heat_J": (
            0.0 if interface_mobile_emission is None
            or not interface_mobile_emission.accepted
            else interface_mobile_emission.generated_heat_J),
        "interface_mobile_emission_heat_J_by_cell": (
            np.zeros(parent.state.supports.shape[1:], dtype=float)
            if interface_mobile_emission is None
            or not interface_mobile_emission.accepted
            else np.asarray(interface_mobile_emission.heat_source_J_by_cell)),
        "wall_order_relaxation_heat_J": (
            0.0 if wall_order is None or not wall_order.accepted
            else wall_order.generated_heat_J),
        "wall_order_relaxation_heat_J_by_cell": (
            np.zeros(parent.state.supports.shape[1:], dtype=float)
            if wall_order is None or not wall_order.accepted
            else np.asarray(wall_order.heat_source_J_by_cell)),
        "conservative_dipole_wall_heat_J": (
            0.0 if dipole_wall is None or not dipole_wall.accepted
            else dipole_wall.generated_heat_J),
        "conservative_dipole_wall_heat_J_by_cell": (
            np.zeros(parent.state.supports.shape[1:], dtype=float)
            if dipole_wall is None or not dipole_wall.accepted
            else np.asarray(dipole_wall.heat_source_J_by_cell)),
        "conservative_wall_glide_heat_J": (
            0.0 if wall_glide is None or not wall_glide.accepted
            else wall_glide.generated_heat_J),
        "conservative_wall_glide_heat_J_by_cell": (
            np.zeros(parent.state.supports.shape[1:], dtype=float)
            if wall_glide is None or not wall_glide.accepted
            else np.asarray(wall_glide.heat_source_J_by_cell)),
        "wall_neutral_pair_recovery_heat_J": (
            0.0 if wall_recovery is None or not wall_recovery.accepted
            else wall_recovery.generated_heat_J),
        "wall_neutral_pair_recovery_heat_J_by_cell": (
            np.zeros(parent.state.supports.shape[1:], dtype=float)
            if wall_recovery is None or not wall_recovery.accepted
            else np.asarray(wall_recovery.heat_source_J_by_cell)),
        "compatible_wall_reorganization_heat_J": (
            0.0 if wall_reorganization is None or not wall_reorganization.accepted
            else wall_reorganization.generated_heat_J),
        "compatible_wall_reorganization_heat_J_by_cell": (
            np.zeros(parent.state.supports.shape[1:], dtype=float)
            if wall_reorganization is None or not wall_reorganization.accepted
            else np.asarray(wall_reorganization.heat_source_J_by_cell)),
        "joint_wall_topology_orientation_heat_J": (
            0.0 if joint_topology is None or not joint_topology.accepted
            else joint_topology.generated_heat_J),
        "joint_wall_topology_orientation_heat_J_by_cell": (
            np.zeros(parent.state.supports.shape[1:], dtype=float)
            if joint_topology is None or not joint_topology.accepted
            else np.asarray(joint_topology.heat_source_J_by_cell)),
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
            (0.0 if structural_absorption is None
             or not structural_absorption.accepted
             else structural_absorption.generated_heat_J)
            +(0.0 if structural_absorption_post is None
              or not structural_absorption_post.accepted
              else structural_absorption_post.generated_heat_J)),
        "structural_junction_heat_J": (
            (0.0 if structural_junction is None
             or not structural_junction.accepted
             else structural_junction.generated_heat_J)
            +(0.0 if structural_junction_post is None
              or not structural_junction_post.accepted
              else structural_junction_post.generated_heat_J)),
        "lattice_absorption_heat_J": (
            0.0 if lattice_absorption is None or not lattice_absorption.accepted
            else lattice_absorption.generated_heat_J),
        "lattice_absorption_heat_J_by_cell": (
            np.zeros(parent.state.supports.shape[1:], dtype=float)
            if lattice_absorption is None or not lattice_absorption.accepted
            else np.asarray(lattice_absorption.heat_source_J_by_cell)),
        "wall_orientation_relaxation_heat_J": (
            0.0 if wall_orientation is None or not wall_orientation.accepted
            else wall_orientation.generated_heat_J),
        "wall_orientation_relaxation_heat_J_by_cell": (
            np.zeros(parent.state.supports.shape[1:], dtype=float)
            if wall_orientation is None or not wall_orientation.accepted
            else np.asarray(wall_orientation.heat_source_J_by_cell)),
        "mechanical_heat_J_m3": mechanical_heat,
        "mechanical_work_J_m3": mechanical_work_density,
        "front_heat_J_by_cell": front_heat,
        "thermal_conduction_J_m3": conduction,
        "thermal_bath_exchange_J_m3": bath,
        "multiplication_raw_line_increment_m2": multiplication_raw,
        "multiplication_accepted_line_increment_m2": multiplication_accepted,
        "multiplication_budget_active_time_s": (
            multiplication_budget_active_time),
        "multiplication_raw_line_increment_m2_owner_family": (
            multiplication_raw_owner_family),
        "multiplication_accepted_line_increment_m2_owner_family": (
            multiplication_accepted_owner_family),
        "signed_slip_increment_owner_family": signed_slip_owner_family,
        "absolute_slip_increment_owner_family": absolute_slip_owner_family,
    }
    audit = {
        "mechanical_subdivision_depth": mechanical.subdivisions,
        "mechanical_internal_substeps": sum(
            item.maximum_substeps for item in mechanical.operator_decisions),
        "mechanical_consumed_interval_s": sum(
            item.consumed_interval_s for item in mechanical.operator_decisions),
        "multiplication_raw_line_increment_cell_sum_m2": float(np.sum(
            multiplication_raw, dtype=np.longdouble)),
        "multiplication_accepted_line_increment_cell_sum_m2": float(np.sum(
            multiplication_accepted, dtype=np.longdouble)),
        "multiplication_acceptance_fraction": float(np.sum(
            multiplication_accepted, dtype=np.longdouble)/max(
                float(np.sum(multiplication_raw, dtype=np.longdouble)),
                1e-300)),
        "multiplication_budget_active_cell_time_fraction": float(np.sum(
            multiplication_budget_active_time, dtype=np.longdouble)/max(
                dt*float(multiplication_budget_active_time.size), 1e-300)),
        "mechanical_relative_first_law_residual": (
            mechanical.relative_first_law_residual),
        "collective_wall_nucleation_mode": (
            context.collective_wall_nucleation_mode),
        "collective_wall_nucleation_classification": (
            "DISABLED" if collective_wall is None
            else collective_wall.classification),
        "collective_wall_nucleation_downhill_candidates": (
            0 if collective_wall is None
            else collective_wall.downhill_candidate_count),
        "collective_wall_nucleation_event_probability": (
            0.0 if collective_wall is None else collective_wall.event_probability),
        "collective_wall_nucleation_resolved_forward_barrier_J": (
            0.0 if collective_wall is None
            else collective_wall.resolved_forward_barrier_J),
        "collective_wall_nucleation_forward_barrier_eV": (
            0.0 if collective_wall is None
            else collective_wall.forward_barrier_eV),
        "collective_wall_nucleation_reverse_barrier_eV": (
            0.0 if collective_wall is None
            else collective_wall.reverse_barrier_eV),
        "collective_wall_nucleation_forward_event_rate_Hz": (
            0.0 if collective_wall is None
            else collective_wall.forward_event_rate_Hz),
        "collective_wall_nucleation_total_forward_event_rate_Hz": (
            0.0 if collective_wall is None
            else collective_wall.total_forward_event_rate_Hz),
        "collective_wall_nucleation_selected_channel_probability": (
            0.0 if collective_wall is None
            else collective_wall.selected_channel_probability),
        "collective_wall_nucleation_reverse_event_rate_Hz": (
            0.0 if collective_wall is None
            else collective_wall.reverse_event_rate_Hz),
        "collective_wall_nucleation_detailed_balance_log_residual": (
            0.0 if collective_wall is None
            else collective_wall.detailed_balance_log_residual),
        "collective_wall_nucleation_realized_continuous_coordinate": (
            0.0 if collective_wall is None
            else collective_wall.realized_continuous_coordinate),
        "collective_wall_nucleation_backtracks": (
            0 if collective_wall is None else collective_wall.backtracks),
        "collective_wall_nucleation_selected_radius_m": (
            0.0 if collective_wall is None else collective_wall.selected_radius_m),
        "collective_wall_nucleation_delta_helmholtz_J": (
            0.0 if collective_wall is None else collective_wall.delta_helmholtz_J),
        "collective_wall_nucleation_relative_first_law_residual": (
            0.0 if collective_wall is None or not collective_wall.accepted
            else collective_wall.relative_first_law_residual),
        "front_subintervals": len(fronts),
        "front_rejected_subintervals": sum(not item.accepted for item in fronts),
        "front_maximum_unclipped_fraction": max(
            (item.maximum_unclipped_fraction for item in fronts), default=0.0),
        "front_directions": ({} if not fronts else
                             fronts[-1].direction_by_interface),
        "front_selected_velocities_m_s": ({} if not fronts else
                                           fronts[-1].selected_velocity_by_interface_m_s),
        "front_pressures_Pa": ({} if not fronts else
                                fronts[-1].pressure_by_interface_Pa),
        "front_independent_pressures_Pa": ({} if not fronts else
            fronts[-1].independent_pressure_by_interface_Pa),
        "front_directional_pressure_pairs_Pa": ({} if not fronts else
            fronts[-1].directional_pressure_pairs_Pa),
        "front_directional_energy_component_slopes_Pa": ({} if not fronts else
            fronts[-1].directional_energy_component_slopes_Pa),
        "front_requested_fractions": ({} if not fronts else
                                       fronts[-1].requested_fraction_by_interface),
        "front_accepted_fractions": ({} if not fronts else
                                      fronts[-1].accepted_fraction_by_interface),
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
        "source_subdivisions": (0 if source is None else source.subdivisions),
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
        "interface_mobile_emission_mode": (
            context.interface_mobile_emission_mode),
        "interface_mobile_emission_classification": (
            "DISABLED" if interface_mobile_emission is None
            else interface_mobile_emission.classification),
        "interface_mobile_emitted_pair_cell_sum_m2": (
            0.0 if interface_mobile_emission is None
            or not interface_mobile_emission.accepted
            else interface_mobile_emission.projection.
            created_lattice_pair_cell_sum_m2),
        "interface_mobile_emission_line_balance_residual_m2": (
            0.0 if interface_mobile_emission is None
            else interface_mobile_emission.projection.
            line_balance_residual_cell_sum_m2),
        "interface_mobile_emission_mura_relative_residual": (
            0.0 if interface_mobile_emission is None
            else interface_mobile_emission.projection.
            maximum_mura_relative_residual),
        "interface_mobile_emission_relative_first_law_residual": (
            0.0 if interface_mobile_emission is None
            or not interface_mobile_emission.accepted
            else interface_mobile_emission.relative_first_law_residual),
        "wall_order_relaxation_mode": context.wall_order_relaxation_mode,
        "wall_order_relaxation_classification": (
            "DISABLED" if wall_order is None else wall_order.classification),
        "wall_order_relaxation_fractions": (
            {} if wall_order is None else wall_order.owner_fraction_by_id),
        "wall_order_relaxation_rates_s": (
            {} if wall_order is None else wall_order.owner_rate_s_by_id),
        "wall_order_relaxation_disordering_rates_s": (
            {} if wall_order is None
            else wall_order.owner_disordering_rate_s_by_id),
        "wall_order_relaxation_disordering_fractions": (
            {} if wall_order is None
            else wall_order.owner_disordering_fraction_by_id),
        "wall_order_relaxation_target_energy_changes_J": (
            {} if wall_order is None
            else wall_order.owner_target_energy_change_J_by_id),
        "wall_order_relaxation_backtracks": (
            0 if wall_order is None else wall_order.backtracks),
        "wall_order_relaxation_relative_first_law_residual": (
            0.0 if wall_order is None or not wall_order.accepted
            else wall_order.relative_first_law_residual),
        "conservative_dipole_wall_mode": (
            context.conservative_dipole_wall_mode),
        "conservative_dipole_wall_classification": (
            "DISABLED" if dipole_wall is None else dipole_wall.classification),
        "conservative_dipole_wall_separated_pair_cell_sum_m2": (
            0.0 if dipole_wall is None or not dipole_wall.accepted
            else dipole_wall.separated_pair_cell_sum_m2),
        "conservative_dipole_wall_plus_inventory_residual_m2": (
            0.0 if dipole_wall is None
            else dipole_wall.plus_inventory_residual_m2),
        "conservative_dipole_wall_minus_inventory_residual_m2": (
            0.0 if dipole_wall is None
            else dipole_wall.minus_inventory_residual_m2),
        "conservative_dipole_wall_line_residual_m2": (
            0.0 if dipole_wall is None else dipole_wall.line_balance_residual_m2),
        "conservative_dipole_wall_maximum_rate_s": (
            0.0 if dipole_wall is None else dipole_wall.maximum_rate_s),
        "conservative_dipole_wall_internal_substeps": (
            0 if dipole_wall is None else dipole_wall.internal_substeps),
        "conservative_dipole_wall_backtracks": (
            0 if dipole_wall is None else dipole_wall.backtracks),
        "conservative_dipole_wall_delta_helmholtz_J": (
            0.0 if dipole_wall is None else dipole_wall.delta_helmholtz_J),
        "conservative_dipole_wall_relative_first_law_residual": (
            0.0 if dipole_wall is None or not dipole_wall.accepted
            else dipole_wall.relative_first_law_residual),
        "conservative_wall_glide_mode": context.conservative_wall_glide_mode,
        "conservative_wall_glide_classification": (
            "DISABLED" if wall_glide is None else wall_glide.classification),
        "conservative_wall_glide_moved_line_cell_sum_m2": (
            0.0 if wall_glide is None or not wall_glide.accepted
            else wall_glide.moved_line_cell_sum_m2),
        "conservative_wall_glide_inventory_residual_m2": (
            0.0 if wall_glide is None else wall_glide.inventory_residual_m2),
        "conservative_wall_glide_maximum_rate_s": (
            0.0 if wall_glide is None else wall_glide.maximum_rate_s),
        "conservative_wall_glide_internal_substeps": (
            0 if wall_glide is None else wall_glide.internal_substeps),
        "conservative_wall_glide_backtracks": (
            0 if wall_glide is None else wall_glide.backtracks),
        "conservative_wall_glide_delta_helmholtz_J": (
            0.0 if wall_glide is None else wall_glide.delta_helmholtz_J),
        "conservative_wall_glide_relative_first_law_residual": (
            0.0 if wall_glide is None or not wall_glide.accepted
            else wall_glide.relative_first_law_residual),
        "wall_neutral_pair_recovery_mode": (
            context.wall_neutral_pair_recovery_mode),
        "wall_neutral_pair_recovery_classification": (
            "DISABLED" if wall_recovery is None
            else wall_recovery.classification),
        "wall_neutral_pair_recovery_annihilated_pair_cell_sum_m2": (
            0.0 if wall_recovery is None or not wall_recovery.accepted
            else wall_recovery.annihilated_pair_cell_sum_m2),
        "wall_neutral_pair_recovery_signed_residual_m2": (
            0.0 if wall_recovery is None
            else wall_recovery.signed_content_residual_m2),
        "wall_neutral_pair_recovery_line_residual_m2": (
            0.0 if wall_recovery is None
            else wall_recovery.line_balance_residual_m2),
        "wall_neutral_pair_recovery_maximum_rate_s": (
            0.0 if wall_recovery is None else wall_recovery.maximum_rate_s),
        "wall_neutral_pair_recovery_backtracks": (
            0 if wall_recovery is None else wall_recovery.backtracks),
        "wall_neutral_pair_recovery_delta_helmholtz_J": (
            0.0 if wall_recovery is None else wall_recovery.delta_helmholtz_J),
        "wall_neutral_pair_recovery_relative_first_law_residual": (
            0.0 if wall_recovery is None or not wall_recovery.accepted
            else wall_recovery.relative_first_law_residual),
        "compatible_wall_reorganization_mode": (
            context.compatible_wall_reorganization_mode),
        "compatible_wall_reorganization_classification": (
            "DISABLED" if wall_reorganization is None
            else wall_reorganization.classification),
        "compatible_wall_reorganization_fractions": (
            {} if wall_reorganization is None
            else wall_reorganization.owner_fraction_by_id),
        "compatible_wall_reorganization_affinities_J_m": (
            {} if wall_reorganization is None
            else wall_reorganization.owner_affinity_J_m_by_id),
        "compatible_wall_reorganization_signed_residual_m2": (
            0.0 if wall_reorganization is None
            else wall_reorganization.signed_content_residual_m2),
        "compatible_wall_reorganization_line_residual_m2": (
            0.0 if wall_reorganization is None
            else wall_reorganization.line_balance_residual_m2),
        "compatible_wall_reorganization_backtracks": (
            0 if wall_reorganization is None else wall_reorganization.backtracks),
        "compatible_wall_reorganization_delta_helmholtz_J": (
            0.0 if wall_reorganization is None
            else wall_reorganization.delta_helmholtz_J),
        "compatible_wall_reorganization_relative_first_law_residual": (
            0.0 if wall_reorganization is None or not wall_reorganization.accepted
            else wall_reorganization.relative_first_law_residual),
        "joint_wall_topology_orientation_mode": (
            context.joint_wall_topology_orientation_mode),
        "joint_wall_topology_orientation_classification": (
            "DISABLED" if joint_topology is None
            else joint_topology.classification),
        "joint_wall_topology_orientation_owner_id": (
            None if joint_topology is None else joint_topology.selected_owner_id),
        "joint_wall_topology_orientation_center": (
            None if joint_topology is None else joint_topology.selected_center),
        "joint_wall_topology_orientation_radius_m": (
            0.0 if joint_topology is None else joint_topology.selected_radius_m),
        "joint_wall_topology_orientation_downhill_candidates": (
            0 if joint_topology is None else joint_topology.downhill_candidate_count),
        "joint_wall_topology_orientation_affinity_J_m": (
            0.0 if joint_topology is None else joint_topology.affinity_J_m),
        "joint_wall_topology_orientation_event_rate_s": (
            0.0 if joint_topology is None else joint_topology.event_rate_s),
        "joint_wall_topology_orientation_coordinate": (
            0.0 if joint_topology is None else joint_topology.realized_coordinate),
        "joint_wall_topology_orientation_transferred_line_cell_sum_m2": (
            0.0 if joint_topology is None or not joint_topology.accepted
            else joint_topology.transferred_line_cell_sum_m2),
        "joint_wall_topology_orientation_maximum_rotation_rad": (
            0.0 if joint_topology is None or not joint_topology.accepted
            else joint_topology.maximum_orientation_change_rad),
        "joint_wall_topology_orientation_line_residual_m2": (
            0.0 if joint_topology is None
            else joint_topology.line_balance_residual_m2),
        "joint_wall_topology_orientation_delta_helmholtz_J": (
            0.0 if joint_topology is None else joint_topology.delta_helmholtz_J),
        "joint_wall_topology_orientation_relative_first_law_residual": (
            0.0 if joint_topology is None or not joint_topology.accepted
            else joint_topology.relative_first_law_residual),
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
        "structural_absorption_post_classification": (
            "DISABLED" if structural_absorption_post is None
            else structural_absorption_post.classification),
        "structural_absorption_post_fraction": (
            0.0 if structural_absorption_post is None
            else structural_absorption_post.kinetic_fraction),
        "structural_absorption_post_content_change": (
            [] if structural_absorption_post is None
            else structural_absorption_post.projection.
            structural_content_change.tolist()),
        "structural_absorption_post_relative_first_law_residual": (
            0.0 if structural_absorption_post is None
            or not structural_absorption_post.accepted
            else structural_absorption_post.relative_first_law_residual),
        "structural_junction_mode": context.structural_junction_mode,
        "structural_junction_classification": (
            "DISABLED" if structural_junction is None
            else structural_junction.classification),
        "structural_junction_reacted_fraction": (
            0.0 if structural_junction is None
            else structural_junction.kinetic_proposal.reacted_fraction),
        "structural_junction_backstress_Pa": (
            0.0 if structural_junction is None
            else structural_junction.kinetic_proposal.frank_backstress_Pa),
        "structural_junction_absolute_reacted_content": (
            0.0 if structural_junction is None
            or not structural_junction.accepted
            else structural_junction.kinetic_proposal.absolute_reacted_content),
        "structural_junction_relative_first_law_residual": (
            0.0 if structural_junction is None
            or not structural_junction.accepted
            else structural_junction.relative_first_law_residual),
        "structural_junction_post_classification": (
            "DISABLED" if structural_junction_post is None
            else structural_junction_post.classification),
        "structural_junction_post_reacted_fraction": (
            0.0 if structural_junction_post is None
            else structural_junction_post.kinetic_proposal.reacted_fraction),
        "structural_junction_post_absolute_reacted_content": (
            0.0 if structural_junction_post is None
            or not structural_junction_post.accepted
            else structural_junction_post.kinetic_proposal.
            absolute_reacted_content),
        "structural_junction_post_relative_first_law_residual": (
            0.0 if structural_junction_post is None
            or not structural_junction_post.accepted
            else structural_junction_post.relative_first_law_residual),
        "lattice_absorption_mode": context.lattice_absorption_mode,
        "lattice_absorption_classification": (
            "DISABLED" if lattice_absorption is None
            else lattice_absorption.classification),
        "lattice_absorption_fraction": (
            0.0 if lattice_absorption is None
            else lattice_absorption.kinetic_fraction),
        "lattice_absorption_event_rate_s": (
            0.0 if lattice_absorption is None
            else lattice_absorption.event_rate_s),
        "lattice_absorption_maximum_uncapped_fraction": (
            0.0 if lattice_absorption is None
            else lattice_absorption.maximum_uncapped_fraction),
        "lattice_absorption_selected_hazard_scale": (
            0.0 if lattice_absorption is None
            else lattice_absorption.selected_hazard_scale),
        "lattice_absorption_energy_search_evaluations": (
            0 if lattice_absorption is None
            else lattice_absorption.energy_search_evaluations),
        "lattice_absorption_kinetics_mode": (
            "DISABLED" if lattice_absorption is None
            else lattice_absorption.local_kinetics_mode),
        "lattice_absorption_arrival_limited": (
            False if lattice_absorption is None
            else lattice_absorption.arrival_limited),
        "lattice_absorption_maximum_arrival_exposure": (
            0.0 if lattice_absorption is None
            else lattice_absorption.maximum_arrival_exposure),
        "lattice_absorption_edge_fractions": (
            {} if lattice_absorption is None
            else lattice_absorption.edge_fraction_by_interface),
        "lattice_absorption_edge_affinities_J_m": (
            {} if lattice_absorption is None
            else lattice_absorption.edge_affinity_J_m_by_interface),
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
        "pre_post_structural_orientation_sync_max_rad": (
            pre_post_structural_orientation_sync_max_rad),
        "final_structural_orientation_sync_max_rad": (
            final_structural_orientation_sync_max_rad),
        "wall_orientation_relaxation_mode": (
            context.wall_orientation_relaxation_mode),
        "wall_orientation_relaxation_classification": (
            "DISABLED" if wall_orientation is None
            else wall_orientation.classification),
        "wall_orientation_relaxation_fractions": (
            {} if wall_orientation is None
            else wall_orientation.owner_fraction_by_id),
        "wall_orientation_relaxation_affinities_J_m": (
            {} if wall_orientation is None
            else wall_orientation.owner_affinity_J_m_by_id),
        "wall_orientation_relaxation_relative_first_law_residual": (
            0.0 if wall_orientation is None or not wall_orientation.accepted
            else wall_orientation.relative_first_law_residual),
        "wall_orientation_relaxation_backtracks": (
            0 if wall_orientation is None else wall_orientation.backtracks),
        "cost_breakdown_s": {
            "mechanical_map": mechanical_wall_time,
            "collective_wall_nucleation_map": collective_wall_wall_time,
            "source_map": source_wall_time,
            "interface_mobile_emission_map": (
                interface_mobile_emission_wall_time),
            "wall_order_relaxation_map": wall_order_wall_time,
            "conservative_dipole_wall_map": dipole_wall_wall_time,
            "conservative_wall_glide_map": wall_glide_wall_time,
            "wall_neutral_pair_recovery_map": wall_recovery_wall_time,
            "compatible_wall_reorganization_map": wall_reorganization_wall_time,
            "joint_wall_topology_orientation_map": joint_topology_wall_time,
            "sink_map": sink_wall_time,
            "interface_sink_map": interface_sink_wall_time,
            "structural_absorption_map": structural_absorption_wall_time,
            "structural_junction_map": structural_junction_wall_time,
            "lattice_absorption_map": lattice_absorption_wall_time,
            "wall_orientation_relaxation_map": wall_orientation_wall_time,
            "front_map": front_wall_time,
            "endpoint_observables": observable_wall_time,
            "accounted_total": (mechanical_wall_time+source_wall_time
                                +interface_mobile_emission_wall_time
                                +collective_wall_wall_time
                                +wall_order_wall_time+dipole_wall_wall_time
                                +wall_glide_wall_time
                                +wall_recovery_wall_time
                                +sink_wall_time
                                +wall_reorganization_wall_time
                                +joint_topology_wall_time
                                +interface_sink_wall_time
                                +structural_absorption_wall_time
                                +structural_junction_wall_time+front_wall_time
                                +lattice_absorption_wall_time
                                +wall_orientation_wall_time
                                +observable_wall_time),
        },
    }
    return CoupledInterval(endpoint, dt, integrals, audit)
