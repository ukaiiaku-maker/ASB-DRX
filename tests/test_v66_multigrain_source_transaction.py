import numpy as np

from full_model.analysis.run_v66_multigrain_handoff import _initial
from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters, boundary_tj_site_field,
)
from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.multigrain_source_transaction import (
    NeutralBoundarySinkParameters, evaluate_multigrain_neutral_sink_transaction,
    evaluate_multigrain_pair_source_transaction,
)
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
