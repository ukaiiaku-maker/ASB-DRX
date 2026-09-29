import numpy as np

from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallParameters, accepted_euler_step,
)
from full_model.production.multigrain_common_state import PhysicalTransferLaw
from full_model.production.complete_multigrain_energy import (
    evaluate_multigrain_mechanical_interval,
)
from full_model.production.multigrain_production import (
    MultiGrainFrontKinetics, MultiGrainInterface,
    MultiGrainProductionRuntime, advance_multigrain_front,
    advance_multigrain_front_interval,
    advance_energy_qualified_mechanics, advance_multigrain_mechanics,
    _geometric_sweep_weight, _masked_owner_update,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, nye_from_plastic_distortion,
)
from tests.test_full_model_v58_complete_multigrain_energy import _state
from full_model.analysis.spatial_localization import spatial_localization_metrics


def _front_parameters(spacing):
    wall = CommonWallParameters(spacing_m=spacing, elastic_iterations=1)
    kinetics = MultiGrainFrontKinetics(
        ActivatedProcess("multi-grain-front", 1e8, 0.0, 1e8),
        PhysicalTransferLaw(.5, .1, .05), .35*EV_J, 1e9, 2.0, 1.5,
        .1, wall.burgers_m**3, wall.burgers_m,
        virtual_fraction=1e-3, maximum_fraction_per_step=.02,
        closure_fraction=.05)
    return wall, kinetics


def test_zero_pressure_two_boundary_production_step_is_complete_and_joint():
    spacing = 2e-8
    state = _state(16)
    wall, kinetics = _front_parameters(spacing)
    runtime = MultiGrainProductionRuntime((
        MultiGrainInterface("10-20", 10, 20),
        MultiGrainInterface("10-30", 10, 30)))
    evolved, runtime, decision = advance_multigrain_front(
        state, runtime, kinetics=kinetics, dt_s=1e-7,
        spacing_m=spacing, represented_thickness_m=5e-10,
        wall_parameters=wall, energy_kwargs=dict(
            phase_barrier_J_m3=5e6, phase_gradient_J_m=5e-7,
            reference_temperature_K=900.0),
        systems=bcc_four_family_systems())
    assert decision.accepted
    assert decision.energy_decision.complete_functional
    assert decision.energy_decision.external_work_J == 0.0
    assert len(decision.accepted_fraction_by_interface) == 2
    assert runtime.ledger.accepted_events == 1
    assert runtime.ledger.generated_heat_J > 0.0
    assert runtime.ledger.maximum_relative_energy_closure < .05
    assert evolved.ledger.energy_accepted_transactions == 1


def test_front_physical_interval_subcycling_matches_manual_sequence():
    spacing = 2e-8
    state = _state(16)
    wall, kinetics = _front_parameters(spacing)
    initial_runtime = MultiGrainProductionRuntime((
        MultiGrainInterface("10-20", 10, 20),
        MultiGrainInterface("10-30", 10, 30)))
    options = dict(
        phase_barrier_J_m3=5e6, phase_gradient_J_m=5e-7,
        reference_temperature_K=900.0)
    automatic, automatic_runtime, decisions = (
        advance_multigrain_front_interval(
            state, initial_runtime, kinetics=kinetics, dt_s=1e-7,
            maximum_substep_s=2.5e-8, spacing_m=spacing,
            represented_thickness_m=5e-10, wall_parameters=wall,
            energy_kwargs=options, applied_shear_rate_s=2e4,
            systems=bcc_four_family_systems()))
    manual = state
    manual_runtime = initial_runtime
    for _ in range(4):
        manual, manual_runtime, _ = advance_multigrain_front(
            manual, manual_runtime, kinetics=kinetics, dt_s=2.5e-8,
            spacing_m=spacing, represented_thickness_m=5e-10,
            wall_parameters=wall, energy_kwargs=options,
            applied_shear_rate_s=2e4, systems=bcc_four_family_systems())
    assert len(decisions) == 4
    assert automatic_runtime == manual_runtime
    assert automatic_runtime.ledger.physical_time_s == 1e-7
    assert automatic_runtime.ledger.applied_shear_strain == 2e-3
    np.testing.assert_array_equal(automatic.supports, manual.supports)
    for automatic_owner, manual_owner in zip(
            automatic.owners, manual.owners):
        for name in automatic_owner.__dataclass_fields__:
            np.testing.assert_array_equal(
                getattr(automatic_owner, name), getattr(manual_owner, name))


def test_mechanical_step_evolves_supported_owner_and_preserves_dormant_history():
    spacing = 2e-8
    state = _state(12)
    systems = bcc_four_family_systems()
    wall = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=1,
        mobile_correlation_diffusivity_m2_s=0.0,
        thermal_diffusivity_m2_s=1e-7)
    stress = np.full((12, 12, 4), 8e8)
    dormant_before = state.owners[2].slip[:6].copy()
    evolved, decision = advance_multigrain_mechanics(
        state, driving=CommonWallDriving(resolved_stress_Pa=stress),
        systems=systems, topologies=(), wall_parameters=wall,
        dt_s=1e-12, represented_thickness_m=5e-10)
    assert decision.accepted
    assert decision.minimum_step_scale > 0.0
    assert decision.external_plastic_work_J >= 0.0
    # Grain 30 has no support in the upper half and cannot age while absent.
    np.testing.assert_array_equal(evolved.owners[2].slip[:6], dormant_before)
    assert np.max(np.abs(evolved.owners[0].slip-state.owners[0].slip)) > 0.0
    for owner_state in evolved.owners[1:]:
        np.testing.assert_array_equal(
            owner_state.temperature_K, evolved.owners[0].temperature_K)


def test_mechanical_identity_interval_has_exact_common_energy_closure():
    state = _state(8)
    spacing = 2e-8
    parameters = CommonWallParameters(spacing_m=spacing, elastic_iterations=1)
    strain = np.array([[0.0, .004], [.004, 0.0]])
    balance = evaluate_multigrain_mechanical_interval(
        state, state, mean_strain_before=strain,
        mean_strain_candidate=strain, spacing_m=spacing,
        represented_thickness_m=5e-10, wall_parameters=parameters,
        energy_kwargs=dict(reference_temperature_K=900.0))
    assert balance.external_work_J == 0.0
    assert balance.internal_energy_change_J == 0.0
    assert balance.first_law_residual_J == 0.0


def test_energy_qualified_mechanics_consumes_full_interval():
    spacing = 2e-8
    state = _state(8)
    systems = bcc_four_family_systems()
    parameters = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=1,
        mobile_correlation_diffusivity_m2_s=0.0,
        thermal_diffusivity_m2_s=1e-7)
    before = np.array([[0.0, .004], [.004, 0.0]])
    after = np.array([[0.0, .0041], [.0041, 0.0]])
    result = advance_energy_qualified_mechanics(
        state, mean_strain_before=before, mean_strain_candidate=after,
        systems=systems, topologies=(), wall_parameters=parameters,
        dt_s=1e-10, represented_thickness_m=5e-10,
        energy_kwargs=dict(reference_temperature_K=900.0),
        maximum_relative_first_law_residual=.05)
    assert sum(item.consumed_interval_s
               for item in result.operator_decisions) == 1e-10
    assert result.relative_first_law_residual <= .05


def test_energy_qualified_zero_load_hold_is_not_bisected_below_roundoff():
    spacing = 2e-8
    state = _state(8)
    systems = bcc_four_family_systems()
    parameters = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=1,
        mobile_correlation_diffusivity_m2_s=0.0,
        thermal_diffusivity_m2_s=1e-7)
    strain = np.array([[0.0, .004], [.004, 0.0]])
    result = advance_energy_qualified_mechanics(
        state, mean_strain_before=strain, mean_strain_candidate=strain,
        systems=systems, topologies=(), wall_parameters=parameters,
        dt_s=1e-10, represented_thickness_m=5e-10,
        energy_kwargs=dict(reference_temperature_K=900.0),
        maximum_relative_first_law_residual=.05)
    assert sum(item.consumed_interval_s
               for item in result.operator_decisions) == 1e-10
    assert result.relative_first_law_residual <= .05


def test_active_mask_cannot_age_dormant_owner_cells_or_limit_step():
    spacing = 2e-8
    state = _state(8).owners[0]
    systems = bcc_four_family_systems()
    parameters = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=1,
        mobile_correlation_diffusivity_m2_s=0.0)
    active = np.zeros((8, 8), dtype=bool); active[:4] = True
    stress = np.full((8, 8, 4), 7e8)
    evolved, _, scale = accepted_euler_step(
        state, CommonWallDriving(resolved_stress_Pa=stress), systems, (),
        parameters, 1e-12, active_mask=active)
    assert scale > 0.0
    for name in state.__dataclass_fields__:
        before = np.asarray(getattr(state, name))
        after = np.asarray(getattr(evolved, name))
        np.testing.assert_array_equal(after[~active], before[~active])


def test_masked_owner_update_retains_curl_nye_identity():
    spacing = 2e-8
    before = _state(8).owners[0]
    systems = bcc_four_family_systems()
    parameters = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=1,
        mobile_correlation_diffusivity_m2_s=0.0)
    stress = np.full((8, 8, 4), 7e8)
    after, _, _ = accepted_euler_step(
        before, CommonWallDriving(resolved_stress_Pa=stress), systems, (),
        parameters, 1e-12)
    active = np.zeros((8, 8), dtype=bool); active[:4] = True
    published = _masked_owner_update(
        before, after, active, systems, spacing)
    np.testing.assert_allclose(
        np.sum(published.family_nye_m1, axis=2),
        nye_from_plastic_distortion(published.beta_p, spacing),
        rtol=2e-13, atol=2e-8)


def test_periodic_localization_distinguishes_hotspot_from_band():
    hotspot = np.zeros((16, 16)); hotspot[7, 8] = 100.0
    point = spatial_localization_metrics(hotspot)
    assert not point["band_like"]
    assert point["largest_component_cells"] == 1

    band = np.ones((16, 16)); band[7, :] = 100.0
    line = spatial_localization_metrics(band)
    assert line["band_like"]
    assert line["periodic_winding"][1]
    assert line["signed_min_W_m3"] == 1.0


def test_owner_heat_is_applied_once_to_common_eulerian_temperature():
    spacing = 2e-8
    state = _state(8)
    systems = bcc_four_family_systems()
    parameters = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=1,
        mobile_correlation_diffusivity_m2_s=0.0,
        thermal_diffusivity_m2_s=0.0, bath_rate_s=0.0)
    stress = np.full((8, 8, 4), 7e8)
    before = sum(weight*owner_state.temperature_K for weight, owner_state
                 in zip(state.supports, state.owners))
    evolved, decision = advance_multigrain_mechanics(
        state, driving=CommonWallDriving(resolved_stress_Pa=stress),
        systems=systems, topologies=(), wall_parameters=parameters,
        dt_s=1e-12, represented_thickness_m=5e-10)
    expected = (before+decision.irreversible_heat_J_m3_cells
                /parameters.volumetric_heat_capacity_J_m3_K)
    for owner_state in evolved.owners:
        np.testing.assert_allclose(
            owner_state.temperature_K, expected, rtol=0.0, atol=2e-13)


def test_pair_sweep_uses_level_set_contour_measure():
    n = 32; spacing = 2e-8
    state = _state(n)
    x = np.arange(n)[:, None]
    donor = np.broadcast_to(.5+.45*np.cos(2*np.pi*x/n), (n, n)).copy()
    supports = np.zeros_like(state.supports)
    supports[0] = donor; supports[1] = 1.0-donor
    state = type(state)(state.grain_ids, supports, state.owners)
    weight = _geometric_sweep_weight(state, 0, 1, spacing)
    # The implementation uses the authoritative periodic spectral gradient;
    # compare its integral directly rather than a finite-difference surrogate.
    from full_model.production.tensorial_nye import spectral_derivatives
    sx, sy = spectral_derivatives(donor, spacing)
    np.testing.assert_allclose(
        np.sum(weight), spacing*np.sum(np.sqrt(sx*sx+sy*sy)), rtol=2e-14)
    assert np.sum(weight) > 1.5*n
