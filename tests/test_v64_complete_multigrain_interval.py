from dataclasses import replace

import numpy as np

from full_model.analysis.run_v58_three_grain_production import (
    initialize_network_state, network_interfaces,
)
from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters,
)
from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.coupled_multigrain_interval import (
    CoupledMultigrainContext, advance_complete_multigrain_interval, make_endpoint,
)
from full_model.production.multigrain_common_state import PhysicalTransferLaw
from full_model.production.interface_structural_absorption import (
    InterfaceStructuralAbsorptionParameters,
)
from full_model.production.structural_interface_network import (
    JunctionReactionParameters, initialize_structural_interfaces,
    interface_lengths_m,
)
from full_model.production.structural_multigrain_runtime import (
    StructuralMultigrainRuntime, synchronize_structural_owner_orientations,
)
from full_model.production.multigrain_source_transaction import (
    NeutralBoundarySinkParameters,
)
from full_model.production.multigrain_production import (
    MultiGrainFrontKinetics, MultiGrainProductionRuntime,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


def context(front_enabled=True):
    spacing = 5e-6/8
    wall = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=2,
        mobile_correlation_diffusivity_m2_s=0.0,
        maximum_fraction_per_step=.1, volumetric_heat_capacity_J_m3_K=3.8e6,
        thermal_diffusivity_m2_s=.15/3.8e6, bath_rate_s=0.0)
    kinetics = MultiGrainFrontKinetics(
        ActivatedProcess("test", 1e8, 0.0, 1e8),
        PhysicalTransferLaw(.55, .08, .02), .35*EV_J, 1e9,
        2.0, 1.5, .10, wall.burgers_m**3, wall.burgers_m,
        virtual_fraction=2e-4, maximum_fraction_per_step=.015,
        closure_fraction=.05, maximum_backtracks=14)
    return CoupledMultigrainContext(
        bcc_four_family_systems(), (), wall, kinetics, spacing, 4.96e-10,
        4e4, 1e-8, 10,
        {"topologies": (), "systems": bcc_four_family_systems(),
         "phase_barrier_J_m3": 5e6, "phase_gradient_J_m": 5e-7,
         "boundary_line_energy_J_m": wall.line_energy_J_m,
         "boundary_junction_energy_J_m": wall.junction_energy_J_m,
         "reference_temperature_K": 900.0},
        front_enabled=front_enabled)


def parent(ctx):
    state = initialize_network_state(
        8, 900.0, 4, spacing_m=ctx.spacing_m,
        interface_width_m=3.125e-7)
    runtime = MultiGrainProductionRuntime(network_interfaces(state))
    return make_endpoint(state, runtime, .012, ctx)


def test_complete_interval_advances_physical_and_load_clocks_once():
    # The small n=8 fixture exercises the exact coupled clock with the front
    # channel disabled; resolved-front acceptance is covered by the production
    # checkpoint tests and the V64 n48/n64 smoke trajectories.
    ctx = context(front_enabled=False)
    before = parent(ctx)
    result = advance_complete_multigrain_interval(before, 1e-10, ctx)
    assert np.isclose(result.endpoint.runtime.ledger.physical_time_s, 1e-10)
    assert np.isclose(result.endpoint.load-.012, 4e-6)
    assert np.isclose(result.endpoint.runtime.ledger.applied_shear_strain, 4e-6)
    assert result.audit["mechanical_consumed_interval_s"] == 1e-10
    assert before.runtime.ledger.physical_time_s == 0.0


def test_disabled_front_channel_has_exact_zero_front_heat():
    ctx = context(front_enabled=False)
    result = advance_complete_multigrain_interval(parent(ctx), 1e-10, ctx)
    assert result.integrals["front_heat_J"] == 0.0
    assert np.all(result.integrals["front_heat_J_by_cell"] == 0.0)
    assert result.audit["front_subintervals"] == 0
    assert result.audit["front_force_rate_closure_classifications"] == []
    assert result.audit["front_force_rate_closure_brackets"] == []


def test_enabled_conjugate_pair_source_is_in_complete_interval_and_ledger():
    ctx = replace(
        context(front_enabled=False),
        source_parameters=ArrheniusSourceParameters(grain_size_m=2.5e-6),
        source_mode="gb_tj_only")
    before = parent(ctx)
    result = advance_complete_multigrain_interval(before, 1e-10, ctx)
    assert result.audit["source_classification"] == (
        "ACCEPTED_CONJUGATE_PAIR_SOURCE")
    assert result.audit["source_event_count"] > 0.0
    assert result.integrals["source_resolved_work_J"] > 0.0
    assert result.integrals["source_heat_J"] > 0.0
    assert result.audit["source_relative_first_law_residual"] < .05
    assert (abs(result.audit["source_line_balance_residual_m2"])
            < 1e-6*result.audit["source_created_line_cell_sum_m2"])
    assert result.endpoint.runtime.ledger.source_intervals == 1
    assert result.endpoint.runtime.ledger.source_event_count > 0.0
    assert result.endpoint.runtime.ledger.physical_time_s == 1e-10


def test_random_bulk_source_is_available_without_a_grain_boundary():
    ctx = replace(
        context(front_enabled=False),
        source_parameters=ArrheniusSourceParameters(grain_size_m=2.5e-6),
        source_mode="random_bulk", source_random_seed=67)
    multi = parent(ctx)
    owner = multi.state.owners[0]
    one = replace(
        multi.state, grain_ids=(1,),
        supports=np.ones((1,)+owner.temperature_K.shape),
        owners=(owner,), interfaces=())
    one.validate()
    endpoint = make_endpoint(
        one, MultiGrainProductionRuntime(()), multi.load, ctx)
    result = advance_complete_multigrain_interval(endpoint, 1e-10, ctx)
    assert result.audit["source_mode"] == "random_bulk"
    assert result.audit["source_classification"] == (
        "ACCEPTED_CONJUGATE_PAIR_SOURCE")
    assert result.audit["source_event_count"] > 0.0
    # The exact signed update is performed atop O(1e15 m^-2) reservoirs;
    # subtraction of the O(1e5 m^-2) increment is roundoff-limited.
    assert (abs(result.audit["source_line_balance_residual_m2"])
            <1e-5*result.audit["source_created_line_cell_sum_m2"])


def test_discrete_random_bulk_source_is_available_without_a_grain_boundary():
    ctx = replace(
        context(front_enabled=False),
        source_parameters=ArrheniusSourceParameters(grain_size_m=2.5e-6),
        source_mode="random_discrete_bulk", source_random_seed=67,
        source_bulk_site_density_m2=1e12)
    multi = parent(ctx)
    owner = multi.state.owners[0]
    one = replace(
        multi.state, grain_ids=(1,),
        supports=np.ones((1,)+owner.temperature_K.shape),
        owners=(owner,), interfaces=())
    endpoint = make_endpoint(
        one, MultiGrainProductionRuntime(()), multi.load, ctx)
    result = advance_complete_multigrain_interval(endpoint, 1e-10, ctx)
    assert result.audit["source_classification"] == (
        "ACCEPTED_CONJUGATE_PAIR_SOURCE")
    assert result.audit["source_event_count"] > 0.0


def test_enabled_neutral_boundary_sink_is_separate_complete_channel():
    ctx = replace(
        context(front_enabled=False),
        sink_parameters=NeutralBoundarySinkParameters(),
        sink_mode="gb_tj_only")
    result = advance_complete_multigrain_interval(parent(ctx), 1e-10, ctx)
    assert result.audit["sink_classification"].startswith(
        "ACCEPTED_NEUTRAL_BOUNDARY_SINK")
    assert result.audit["sink_annihilated_pair_cell_sum_m2"] > 0.0
    assert result.integrals["sink_heat_J"] > 0.0
    assert result.audit["sink_relative_first_law_residual"] < .05
    assert (abs(result.audit["sink_line_balance_residual_m2"])
            <1e-10*result.audit["sink_annihilated_pair_cell_sum_m2"])
    assert result.endpoint.runtime.ledger.sink_intervals == 1
    assert result.endpoint.runtime.ledger.source_intervals == 0
    assert result.endpoint.runtime.ledger.physical_time_s == 1e-10


def test_common_axis_structural_absorption_is_persistent_production_state():
    ctx = replace(
        context(front_enabled=False),
        structural_absorption_parameters=(
            InterfaceStructuralAbsorptionParameters()),
        structural_absorption_mode="common_axis")
    result = advance_complete_multigrain_interval(parent(ctx), 1e-10, ctx)
    assert result.audit["structural_absorption_classification"] == (
        "INACTIVE_NO_COMPATIBLE_INTERFACE_BURGERS")
    assert len(result.endpoint.runtime.structural_interfaces) == len(
        result.endpoint.runtime.interfaces)
    assert result.audit["pre_post_structural_orientation_sync_max_rad"] < 1e-12
    assert result.endpoint.runtime.ledger.structural_absorption_inactive_intervals == 1
    assert result.endpoint.runtime.ledger.physical_time_s == 1e-10


def test_arrhenius_junction_reaction_is_in_complete_interval_and_ledger():
    ctx = replace(
        context(front_enabled=False), shear_rate_s=0.0,
        structural_absorption_parameters=(
            InterfaceStructuralAbsorptionParameters()),
        structural_junction_parameters=JunctionReactionParameters(),
        structural_junction_mode="arrhenius_cycle")
    endpoint = parent(ctx)
    lengths = interface_lengths_m(
        endpoint.state.supports, endpoint.runtime.interfaces,
        endpoint.state.grain_ids, ctx.spacing_m)
    orientations = np.asarray([
        np.mean(owner.orientation_rad) for owner in endpoint.state.owners])
    structural = list(initialize_structural_interfaces(
        endpoint.state.grain_ids, orientations,
        tuple((edge.grain_a_id, edge.grain_b_id)
              for edge in endpoint.runtime.interfaces),
        lengths, ctx.structural_absorption_parameters.boundary.burgers_m))
    structural[0] = replace(
        structural[0], signed_integrated_content=(
            1.02*structural[0].signed_integrated_content))
    synchronized = synchronize_structural_owner_orientations(
        endpoint.state, StructuralMultigrainRuntime(tuple(structural)),
        spacing_m=ctx.spacing_m,
        burgers_m=ctx.structural_absorption_parameters.boundary.burgers_m)
    endpoint = make_endpoint(
        synchronized, replace(endpoint.runtime,
                              structural_interfaces=tuple(structural)),
        endpoint.load, ctx)
    result = advance_complete_multigrain_interval(endpoint, 1e-8, ctx)
    assert result.audit["structural_junction_classification"] == (
        "ACCEPTED_COMPLETE_JUNCTION_RELAXATION")
    assert result.audit["structural_junction_absolute_reacted_content"] > 0.0
    assert result.integrals["structural_junction_heat_J"] > 0.0
    assert result.audit["structural_junction_relative_first_law_residual"] < .05
    assert result.endpoint.runtime.ledger.structural_junction_intervals == 1
