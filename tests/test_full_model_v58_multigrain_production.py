from dataclasses import replace

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
    multigrain_instantaneous_dissipation_fields,
    _geometric_sweep_weight, _masked_owner_update,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, nye_from_plastic_distortion,
)
from tests.test_full_model_v58_complete_multigrain_energy import _state
from full_model.analysis.spatial_localization import spatial_localization_metrics
from full_model.analysis.run_v58_three_grain_production import (
    initialize_network_state, network_interfaces,
)


def _front_parameters(spacing):
    wall = CommonWallParameters(spacing_m=spacing, elastic_iterations=1)
    kinetics = MultiGrainFrontKinetics(
        ActivatedProcess("multi-grain-front", 1e8, 0.0, 1e8),
        PhysicalTransferLaw(.5, .1, .05), .35*EV_J, 1e9, 2.0, 1.5,
        .1, wall.burgers_m**3, wall.burgers_m,
        virtual_fraction=1e-3, maximum_fraction_per_step=.02,
        closure_fraction=.05)
    return wall, kinetics


def test_four_grain_network_has_partition_cores_and_real_periodic_adjacency():
    state = initialize_network_state(
        32, 900.0, 4, spacing_m=5e-6/32,
        interface_width_m=3.125e-7)
    np.testing.assert_allclose(np.sum(state.supports, axis=0), 1.0)
    assert all(np.max(support) > .8 for support in state.supports)
    interfaces = network_interfaces(state)
    assert len(interfaces) >= 4
    assert all(item.grain_a_id in state.grain_ids
               and item.grain_b_id in state.grain_ids for item in interfaces)


def test_single_crystal_common_owner_has_declared_weak_thermal_band():
    state = initialize_network_state(
        32, 900.0, 1, spacing_m=5e-6/32,
        initial_temperature_band_K=1.0,
        initial_temperature_band_width_m=3.125e-7,
        single_crystal_band_normal="diagonal")
    assert state.grain_ids == (10,)
    assert not network_interfaces(state)
    np.testing.assert_allclose(state.supports, 1.0)
    temperature = state.owners[0].temperature_K
    assert np.isclose(temperature.mean(), 900.0)
    assert np.isclose(temperature.max()-900.0, 1.0)
    state.validate()


def test_single_crystal_density_band_preserves_inventory_and_signed_content():
    kwargs = dict(spacing_m=5e-6/32, interface_width_m=3.125e-7)
    reference = initialize_network_state(32, 900.0, 1, **kwargs)
    seeded = initialize_network_state(
        32, 900.0, 1, initial_density_band_fraction=.05,
        initial_density_band_width_m=3.125e-7,
        single_crystal_band_normal="diagonal", **kwargs)
    fields = (("mobile_plus_m2", "mobile_minus_m2"),
              ("forest_plus_m2", "forest_minus_m2"),
              ("wall_plus_m2", "wall_minus_m2"))
    reference_total = sum(np.sum(getattr(reference.owners[0], name))
                          for pair in fields for name in pair)
    seeded_total = sum(np.sum(getattr(seeded.owners[0], name))
                       for pair in fields for name in pair)
    assert np.isclose(seeded_total, reference_total, rtol=2e-15, atol=0.0)
    signed = sum(getattr(seeded.owners[0], plus)
                 -getattr(seeded.owners[0], minus) for plus, minus in fields)
    np.testing.assert_array_equal(signed, 0.0)
    np.testing.assert_array_equal(
        seeded.owners[0].beta_p, reference.owners[0].beta_p)
    np.testing.assert_array_equal(
        seeded.owners[0].family_nye_m1,
        reference.owners[0].family_nye_m1)
    assert np.ptp(seeded.owners[0].mobile_plus_m2[..., 0]) > 0.0


def test_four_grain_network_executes_one_joint_complete_event():
    spacing = 5e-6/16
    state = initialize_network_state(
        16, 900.0, 4, spacing_m=spacing, interface_width_m=6.25e-7)
    wall, kinetics = _front_parameters(spacing)
    evolved, runtime, decision = advance_multigrain_front(
        state, MultiGrainProductionRuntime(network_interfaces(state)),
        kinetics=kinetics, dt_s=1e-8, spacing_m=spacing,
        represented_thickness_m=5e-10, wall_parameters=wall,
        energy_kwargs=dict(phase_barrier_J_m3=5e6,
                           phase_gradient_J_m=5e-7,
                           reference_temperature_K=900.0),
        systems=bcc_four_family_systems())
    assert decision.accepted
    assert decision.selected_rate_conjugate_to_recorded_force
    assert decision.heat_source_J_by_cell is not None
    assert np.isclose(
        np.sum(decision.heat_source_J_by_cell, dtype=np.longdouble),
        decision.energy_decision.generated_heat_J, rtol=2e-15, atol=0.0)
    assert runtime.ledger.accepted_events == 1
    assert evolved.ledger.energy_accepted_transactions == 1


def test_front_temperature_override_is_independent_of_physical_heat_state():
    spacing = 5e-6/16
    state = initialize_network_state(
        16, 900.0, 4, spacing_m=spacing, interface_width_m=6.25e-7)
    wall, physical = _front_parameters(spacing)
    overridden = replace(physical, temperature_override_K=1200.0)
    kwargs = dict(
        dt_s=1e-8, spacing_m=spacing,
        represented_thickness_m=5e-10, wall_parameters=wall,
        energy_kwargs=dict(phase_barrier_J_m3=5e6,
                           phase_gradient_J_m=5e-7,
                           reference_temperature_K=900.0),
        systems=bcc_four_family_systems())
    runtime = MultiGrainProductionRuntime(network_interfaces(state))
    _, _, hot_decision = advance_multigrain_front(
        state, runtime, kinetics=physical, **kwargs)
    _, _, overridden_decision = advance_multigrain_front(
        state, runtime, kinetics=overridden, **kwargs)
    assert hot_decision.selected_velocity_by_interface_m_s
    assert overridden_decision.selected_velocity_by_interface_m_s.keys() == (
        hot_decision.selected_velocity_by_interface_m_s.keys())
    assert any(
        not np.isclose(overridden_decision.selected_velocity_by_interface_m_s[key],
                       hot_decision.selected_velocity_by_interface_m_s[key])
        for key in hot_decision.selected_velocity_by_interface_m_s)


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
    assert decision.independent_pressure_by_interface_Pa
    assert decision.selected_velocity_by_interface_m_s
    assert decision.joint_pressure_factor > 0.0
    assert decision.selected_rate_conjugate_to_recorded_force

    # After a finite first sweep, low-support cells and unequal edge speeds
    # make the next multi-edge direction nontrivial.  Its joint virtual price
    # must remain conjugate to that realized direction.
    evolved_again, runtime_again, second = advance_multigrain_front(
        evolved, runtime, kinetics=kinetics, dt_s=1e-7,
        spacing_m=spacing, represented_thickness_m=5e-10,
        wall_parameters=wall, energy_kwargs=dict(
            phase_barrier_J_m3=5e6, phase_gradient_J_m=5e-7,
            reference_temperature_K=900.0),
        systems=bcc_four_family_systems())
    assert second.accepted
    assert second.energy_decision.independent_dissipation
    assert runtime_again.ledger.accepted_events == 2
    assert evolved_again.ledger.energy_accepted_transactions == 2


def test_instantaneous_budget_exposes_signed_physical_channels_and_storage():
    spacing = 5e-6/16
    state = initialize_network_state(
        16, 900.0, 4, spacing_m=spacing, interface_width_m=6.25e-7)
    wall = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=1,
        thermal_diffusivity_m2_s=4e-8)
    fields = multigrain_instantaneous_dissipation_fields(
        state, driving=CommonWallDriving(
            mean_strain=np.array([[0.0, .01], [.01, 0.0]])),
        systems=bcc_four_family_systems(), topologies=(),
        wall_parameters=wall)
    channels = fields["dissipation_channels_W_m3"]
    np.testing.assert_allclose(
        sum(channels.values()), fields["irreversible_heat_rate_W_m3"],
        rtol=2e-13, atol=1e-4)
    np.testing.assert_allclose(
        fields["instantaneous_local_thermal_storage_W_m3"],
        fields["irreversible_heat_rate_W_m3"]
        +fields["thermal_conduction_W_m3"]
        +fields["thermal_bath_exchange_W_m3"])
    assert abs(np.sum(fields["thermal_conduction_W_m3"],
                      dtype=np.longdouble)) < 1e-6

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


def test_conduction_and_bath_are_independently_ledgered_as_thermal_increments():
    spacing = 2e-8
    state = _state(8)
    x = np.arange(8)[:, None]
    initial = 910.0+2.0*np.cos(2.0*np.pi*x/8.0)
    initial = np.broadcast_to(initial, (8, 8)).copy()
    state = replace(state, owners=tuple(
        replace(item, temperature_K=initial.copy()) for item in state.owners))
    parameters = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=1,
        mobile_correlation_diffusivity_m2_s=0.0,
        thermal_diffusivity_m2_s=1e-8, bath_rate_s=1e6,
        bath_temperature_K=900.0)
    zero_stress = np.zeros((8, 8, 4))
    evolved, decision = advance_multigrain_mechanics(
        state, driving=CommonWallDriving(resolved_stress_Pa=zero_stress),
        systems=bcc_four_family_systems(), topologies=(),
        wall_parameters=parameters, dt_s=1e-9,
        represented_thickness_m=5e-10)
    final = evolved.owners[0].temperature_K
    thermal_increment = (final-initial)*parameters.volumetric_heat_capacity_J_m3_K
    ledgered = (decision.irreversible_heat_J_m3_cells
                +decision.thermal_conduction_J_m3_cells
                +decision.thermal_bath_exchange_J_m3_cells)
    np.testing.assert_allclose(ledgered, thermal_increment, rtol=2e-13,
                               atol=2e-5)
    assert abs(np.sum(decision.thermal_conduction_J_m3_cells)) < 1e-4
    assert np.sum(decision.thermal_bath_exchange_J_m3_cells) < 0.0


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
