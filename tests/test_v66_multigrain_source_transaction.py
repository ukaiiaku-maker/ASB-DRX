from dataclasses import replace

import numpy as np

from full_model.analysis.run_v66_multigrain_handoff import _initial
from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters, boundary_tj_site_field,
)
from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.multigrain_source_transaction import (
    advance_energy_qualified_pair_source_transaction,
    NeutralBoundarySinkParameters,
    evaluate_interface_neutral_sink_transaction,
    evaluate_interface_vector_neutral_junction_transaction,
    evaluate_multigrain_neutral_sink_transaction,
    evaluate_multigrain_pair_source_transaction,
)
from full_model.production.multigrain_common_state import InterfaceComponentState
from full_model.production.tensorial_nye import bcc_four_family_systems


def _fixture():
    n = 12
    length = 4.8e-6
    state, _, _ = _initial(n, length, 1000.0, 2.0e7)
    systems = bcc_four_family_systems()
    wall = CommonWallParameters(
        spacing_m=length/n, elastic_iterations=2,
        mobile_correlation_diffusivity_m2_s=0.0,
        transport_scheme="upwind")
    energy = {
        "topologies": (), "systems": systems,
        "phase_barrier_J_m3": 5e6, "phase_gradient_J_m": 5e-7,
        "boundary_line_energy_J_m": wall.line_energy_J_m,
        "boundary_junction_energy_J_m": wall.junction_energy_J_m,
        "reference_temperature_K": 1000.0,
    }
    source = ArrheniusSourceParameters(grain_size_m=2.4e-6)
    sites = boundary_tj_site_field(state.supports, tj_site_weight=2.0)
    return state, systems, wall, energy, source, sites


def test_stressed_pair_source_closes_complete_energy_and_signed_line():
    state, systems, wall, energy, source, sites = _fixture()
    strain = np.array([[0.0, .01], [.01, 0.0]])
    result = evaluate_multigrain_pair_source_transaction(
        state, mean_strain=strain, site_field=sites, interval_s=2e-9,
        source_parameters=source, systems=systems, wall_parameters=wall,
        represented_thickness_m=4.96e-10, energy_kwargs=energy,
        source_mode="gb_tj_only")
    assert result.accepted
    assert result.classification == "ACCEPTED_CONJUGATE_PAIR_SOURCE"
    assert result.resolved_work_J > result.defect_storage_change_J > 0.0
    assert result.generated_heat_J > 0.0
    assert result.delta_helmholtz_before_heat_J < 0.0
    assert result.relative_first_law_residual < .05
    assert abs(result.signed_content_change_m2) < 1e-8*max(
        result.created_line_cell_sum_m2, 1.0)
    assert abs(result.line_balance_residual_m2) < 1e-8*max(
        result.created_line_cell_sum_m2, 1.0)


def test_structural_graph_source_constraint_preserves_owner_means_not_span():
    state, systems, wall, energy, source, sites = _fixture()
    before_means = np.asarray([
        np.mean(owner.orientation_rad) for owner in state.owners])
    result = evaluate_multigrain_pair_source_transaction(
        state, mean_strain=np.array([[0.0, .01], [.01, 0.0]]),
        site_field=sites, interval_s=2e-9, source_parameters=source,
        systems=systems, wall_parameters=wall,
        represented_thickness_m=4.96e-10, energy_kwargs=energy,
        source_mode="gb_tj_only", preserve_owner_mean_orientation=True)
    assert result.accepted
    after_means = np.asarray([
        np.mean(owner.orientation_rad)
        for owner in result.candidate_state.owners])
    np.testing.assert_allclose(after_means, before_means, rtol=0.0, atol=1e-15)
    assert any(np.ptp(owner.orientation_rad) > 0.0
               for owner in result.candidate_state.owners)


def test_unstressed_thermal_attempt_cannot_invent_line_storage_energy():
    state, systems, wall, energy, source, sites = _fixture()
    result = evaluate_multigrain_pair_source_transaction(
        state, mean_strain=np.zeros((2, 2)), site_field=sites,
        interval_s=2e-9, source_parameters=source, systems=systems,
        wall_parameters=wall, represented_thickness_m=4.96e-10,
        energy_kwargs=energy, source_mode="gb_tj_only")
    assert not result.accepted
    assert result.classification in {
        "REJECTED_INSUFFICIENT_RESOLVED_WORK",
        "INACTIVE_BELOW_ENERGY_RESOLUTION",
    }
    assert result.candidate_state is state
    assert result.generated_heat_J == 0.0


def test_sparse_source_burst_is_energy_qualified_by_time_subdivision():
    state, systems, wall, energy, source, sites = _fixture()
    sparse = np.zeros_like(sites)
    sparse[3, 4] = sparse.size
    kwargs = dict(
        mean_strain=np.array([[0.0, .01], [.01, 0.0]]),
        site_field=sparse, interval_s=2e-4,
        source_parameters=source, systems=systems, wall_parameters=wall,
        represented_thickness_m=4.96e-10, energy_kwargs=energy,
        source_mode="random_discrete_bulk")
    coarse = evaluate_multigrain_pair_source_transaction(state, **kwargs)
    assert not coarse.accepted
    accepted = advance_energy_qualified_pair_source_transaction(
        state, maximum_subdivisions=8, **kwargs)
    assert accepted.accepted
    assert accepted.classification == (
        "ACCEPTED_CONJUGATE_PAIR_SOURCE_SUBDIVIDED")
    assert accepted.subdivisions > 0
    assert accepted.event_count > 0.0
    assert accepted.relative_first_law_residual < .05


def test_neutral_boundary_sink_closes_line_energy_and_heat_at_zero_work():
    state, systems, wall, energy, _, sites = _fixture()
    result = evaluate_multigrain_neutral_sink_transaction(
        state, mean_strain=np.zeros((2, 2)), site_field=sites,
        interval_s=2e-9, sink_parameters=NeutralBoundarySinkParameters(),
        systems=systems, wall_parameters=wall,
        represented_thickness_m=4.96e-10, energy_kwargs=energy)
    assert result.accepted
    assert result.classification.startswith("ACCEPTED_NEUTRAL_BOUNDARY_SINK")
    assert result.annihilated_pair_cell_sum_m2 > 0.0
    assert result.defect_storage_change_J < 0.0
    assert result.generated_heat_J > 0.0
    assert result.relative_first_law_residual < .05
    assert abs(result.signed_content_change_m2) < 1e-12*max(
        result.annihilated_pair_cell_sum_m2, 1.0)
    assert abs(result.line_balance_residual_m2) < 1e-12*max(
        2.0*result.annihilated_pair_cell_sum_m2, 1.0)


def test_stationary_interface_sink_preserves_signed_burgers_and_closes_heat():
    state, _, wall, energy, _, _ = _fixture()
    grid = state.supports.shape[1:]
    plus = np.full(grid+(3, 4), 1.0e14)
    minus = np.full(grid+(3, 4), .8e14)
    burgers = np.arange(np.prod(grid)*9, dtype=float).reshape(grid+(3, 3))
    component = InterfaceComponentState(
        "edge-10-20", 10, 20, np.zeros(grid), np.zeros(grid), plus,
        minus, np.zeros(grid+(3, 1)), boundary_burgers_m1=burgers)
    state = replace(state, interfaces=(component,))
    result = evaluate_interface_neutral_sink_transaction(
        state, mean_strain=np.zeros((2, 2)), interval_s=2e-8,
        sink_parameters=NeutralBoundarySinkParameters(),
        wall_parameters=wall, represented_thickness_m=4.96e-10,
        energy_kwargs=energy)
    assert result.accepted
    assert result.classification.startswith("ACCEPTED_INTERFACE_NEUTRAL_SINK")
    assert result.annihilated_pair_cell_sum_m2 > 0.0
    assert result.defect_storage_change_J < 0.0
    assert result.generated_heat_J > 0.0
    assert result.relative_first_law_residual < .05
    np.testing.assert_array_equal(
        result.candidate_state.interfaces[0].boundary_burgers_m1, burgers)
    np.testing.assert_allclose(
        result.candidate_state.interfaces[0].boundary_plus_m2
        -result.candidate_state.interfaces[0].boundary_minus_m2,
        plus-minus, rtol=0.0, atol=1e-2)
    assert abs(result.line_balance_residual_m2) < 1e-12*max(
        2.0*result.annihilated_pair_cell_sum_m2, 1.0)

    exact = evaluate_interface_neutral_sink_transaction(
        state, mean_strain=np.zeros((2, 2)), interval_s=1e-3,
        sink_parameters=NeutralBoundarySinkParameters(
            maximum_fraction_per_interval=.1,
            complete_physical_clock=True),
        wall_parameters=wall, represented_thickness_m=4.96e-10,
        energy_kwargs=energy)
    assert exact.accepted
    assert exact.maximum_fraction > .1
    assert exact.annihilated_pair_cell_sum_m2 > (
        result.annihilated_pair_cell_sum_m2)
    np.testing.assert_array_equal(
        exact.candidate_state.interfaces[0].boundary_burgers_m1, burgers)


def test_vector_neutral_interface_junction_reaches_burgers_capacity_bound():
    state, _, wall, energy, _, _ = _fixture()
    grid = state.supports.shape[1:]
    plus = np.full(grid+(3, 4), 1.0e14)
    minus = np.full(grid+(3, 4), .8e14)
    burgers = (1e4*np.random.default_rng(66).normal(
        size=grid+(3, 3)))
    component = InterfaceComponentState(
        "edge-10-20", 10, 20, np.zeros(grid), np.zeros(grid), plus,
        minus, np.zeros(grid+(3, 1)), boundary_burgers_m1=burgers)
    state = replace(state, interfaces=(component,))
    result = evaluate_interface_vector_neutral_junction_transaction(
        state, mean_strain=np.zeros((2, 2)), interval_s=1e-3,
        sink_parameters=NeutralBoundarySinkParameters(
            complete_physical_clock=True),
        wall_parameters=wall, represented_thickness_m=4.96e-10,
        energy_kwargs=energy)
    assert result.accepted
    assert result.classification.startswith(
        "ACCEPTED_INTERFACE_VECTOR_NEUTRAL_JUNCTION")
    np.testing.assert_array_equal(
        result.candidate_state.interfaces[0].boundary_burgers_m1, burgers)
    after = result.candidate_state.interfaces[0]
    line_after = np.sum(
        after.boundary_plus_m2+after.boundary_minus_m2, axis=-1)
    required = np.linalg.norm(burgers, axis=-1)/wall.burgers_m
    assert np.min(line_after-required) >= -1.0
    assert np.sum(line_after) < np.sum(plus+minus)
    assert abs(result.line_balance_residual_m2) < 1e-12*max(
        2.0*result.annihilated_pair_cell_sum_m2, 1.0)
    assert result.relative_first_law_residual < .05


def test_neutral_sink_site_geography_changes_density_not_integral_balance():
    state, systems, wall, energy, _, sites = _fixture()
    random = np.random.default_rng(66).permutation(sites.ravel()).reshape(sites.shape)
    kwargs = dict(
        state=state, mean_strain=np.zeros((2, 2)), interval_s=2e-9,
        sink_parameters=NeutralBoundarySinkParameters(), systems=systems,
        wall_parameters=wall, represented_thickness_m=4.96e-10,
        energy_kwargs=energy)
    boundary = evaluate_multigrain_neutral_sink_transaction(
        site_field=sites, sink_mode="gb_tj", **kwargs)
    control = evaluate_multigrain_neutral_sink_transaction(
        site_field=random, sink_mode="matched_random", **kwargs)
    assert boundary.accepted and control.accepted
    assert abs(boundary.line_balance_residual_m2) < 1e-12*max(
        2*boundary.annihilated_pair_cell_sum_m2, 1.0)
    assert abs(control.line_balance_residual_m2) < 1e-12*max(
        2*control.annihilated_pair_cell_sum_m2, 1.0)
    # The exact-permutation control preserves the site histogram, while owner
    # masks and nonlinear local populations make the realized fields distinct.
    b = boundary.candidate_state.owners[0].mobile_plus_m2
    c = control.candidate_state.owners[0].mobile_plus_m2
    assert np.linalg.norm(b-c) > 0.0
