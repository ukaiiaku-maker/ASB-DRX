from dataclasses import replace

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, checkpoint_payload, compare_response_family,
    resolved_bicrystal, run_i3_cycle, state_from_payload,
)
from full_model.production.common_front_state import (
    reconstruct_common, state_arrays as common_front_arrays,
)
from full_model.production.common_tensorial_wall import CommonWallDriving


def _fixture():
    context = resolved_bicrystal(
        grid=16, length_m=3.2e-6, interface_width_m=4.0e-7,
        child_line_fraction=.35, temperature_K=1100.0)
    initial = context["state"]
    spacing = context["spacing_m"]
    x = np.arange(16)*spacing
    length = 16*spacing
    width = context["interface_width_m"]

    def phase_trial(normal_displacement_m):
        child = .5*(
            np.tanh((x-(.25*length-normal_displacement_m))/width)
            -np.tanh((x-(.75*length+normal_displacement_m))/width))
        child = np.broadcast_to(child[:, None], (16, 16))
        return np.stack((1.0-child, child), axis=2)

    driving = CommonWallDriving(
        mean_strain=np.array([[0.0, .01], [.01, 0.0]]),
        fixed_eigenstrain=np.zeros((16, 16, 2, 2)))
    controls = I3Controls(
        driving_pressure_a_to_b_Pa=1.0e8,
        applied_pressure_a_to_b_Pa=1.0e8,
        trial_dt_s=1.0e-10, front_dt_s=1.0e-3)
    return (context, initial, phase_trial(.05*spacing),
            phase_trial(-.05*spacing), driving, controls)


def test_response_family_is_same_state_and_reports_finite_reciprocal_metrics():
    context, initial, forward, reverse, driving, controls = _fixture()
    states, result = compare_response_family(
        context, initial, forward, reverse, driving, controls)
    cases = result["cases"]
    assert set(cases) == {
        "front_only", "mura_only", "combined",
        "prescribed_temperature", "reversed_contrast"}

    assert cases["front_only"]["maximum_abs_slip"] == 0.0
    assert cases["mura_only"]["sweep"]["absolute_m3"] == 0.0
    assert cases["combined"]["maximum_abs_slip"] > 0.0
    assert cases["combined"]["kinetic_normalization_is_grid_independent"]
    assert cases["combined"]["kinetic_event_volume_m3"] == (
        context["wall_parameters"].burgers_m**3)
    assert cases["combined"]["kinetic_event_length_m"] == (
        context["wall_parameters"].burgers_m)
    site_measure = cases["combined"]["physical_site_event_measure"]
    assert site_measure["physical_site_count"] > 0.0
    assert site_measure["site_count_per_interface_area_m2"] == (
        1.0/context["wall_parameters"].burgers_m**2)
    assert site_measure["expected_normal_velocity_m_s"] == (
        cases["combined"]["front_decision"]["net_velocity_a_to_b_m_s"])
    assert cases["combined"]["maximum_abs_beta_p"] > 0.0
    assert cases["combined"]["sweep"]["positive_m3"] > 0.0
    assert cases["combined"]["sweep"]["negative_m3"] == 0.0
    assert cases["reversed_contrast"]["sweep"]["negative_m3"] > 0.0
    assert cases["reversed_contrast"]["sweep"]["positive_m3"] == 0.0
    assert cases["combined"]["sweep"]["absolute_m3"] == abs(
        cases["combined"]["sweep"]["net_m3"])
    assert len(cases["combined"]["sweep"]["components"]) == 2
    assert all("normal_displacement_m" in row
               for row in cases["combined"]["sweep"]["components"])
    assert cases["combined"]["phase"]["maximum_abs_change"] > 0.0
    assert cases["combined"]["candidate_sweep_published"]
    assert cases["combined"]["complete_energy"]["front_decision"][
        "accepted"]
    assert abs(cases["combined"]["complete_energy"]["front_decision"][
        "first_law_residual_J"]) < 1e-24
    assert cases["combined"]["mura"]["family_selection_rule"] == (
        "complete_discrete_full_event_affinity_then_joint_backtrack")
    assert cases["combined"]["mura"]["minimum_heat_increment_J_m3"] >= 0.0
    assert cases["combined"]["actual_inventory_change"]["total_line_m"] != 0.0

    interaction = result["reciprocal_interactions"]
    assert np.isfinite(interaction["mura_to_front_net_sweep_change_m3"])
    assert interaction["front_to_next_mura_raw_stress_rms_change_Pa"] > 0.0
    assert interaction["front_to_next_mura_speed_rms_change_m_s"] > 0.0
    assert not interaction["next_mura_extent_not_executed"]
    np.testing.assert_array_equal(
        states["prescribed_temperature"].mechanical.common.temperature_K,
        initial.mechanical.common.temperature_K)


def test_checkpoint_reload_reproduces_same_complete_cycle_bitwise():
    context, initial, forward, _, driving, controls = _fixture()
    restored = state_from_payload(checkpoint_payload(initial, context), context)
    continuous, continuous_audit = run_i3_cycle(
        context, initial, forward, driving, controls)
    restarted, restarted_audit = run_i3_cycle(
        context, restored, forward, driving, controls)
    np.testing.assert_array_equal(continuous.eta, restarted.eta)
    for name, value in common_front_arrays(continuous.common_front).items():
        np.testing.assert_array_equal(
            value, common_front_arrays(restarted.common_front)[name])
    for group in ("common", "density", "reservoir_alignment"):
        a = getattr(continuous.mechanical, group)
        b = getattr(restarted.mechanical, group)
        for name in a.__dict__:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert continuous_audit == restarted_audit


def test_resolved_bicrystal_is_an_existing_two_owner_state_not_nucleation():
    context, initial, _, _, _, _ = _fixture()
    common, _ = reconstruct_common(initial.common_front, context["spacing_m"])
    assert initial.eta.shape == (16, 16, 2)
    np.testing.assert_allclose(np.sum(initial.eta, axis=2), 1.0)
    assert initial.common_front.front.parent_label == 0
    assert initial.common_front.front.child_label == 1
    assert initial.common_front.ledger.attempted_commits == 0
    assert np.ptp(common.mobile_plus_m2) > 0.0
    assert context["interface_width_m"] >= 2.0*context["spacing_m"]


def test_misoriented_bicrystal_has_declared_pure_owner_cores_without_fake_nye():
    context = resolved_bicrystal(
        grid=16, length_m=3.2e-6, interface_width_m=4.0e-7,
        misorientation_deg=20.0)
    state = context["state"]
    parent = state.common_front.parent
    child = state.common_front.child
    assert np.isclose(np.rad2deg(parent.orientation_rad[0, 0]), -10.0)
    assert np.isclose(np.rad2deg(child.orientation_rad[0, 0]), 10.0)
    assert np.all(parent.family_nye_m1 == 0.0)
    assert np.all(child.family_nye_m1 == 0.0)
    audit = context["boundary_initialization"]
    assert audit["kind"] == "misoriented_bicrystal"
    assert audit["declared_misorientation_deg"] == 20.0
    assert audit["plastic_nye_from_orientation_target"] is False


def test_v37_front_kinetic_family_controls_enter_production_event():
    context, initial, forward, _, driving, controls = _fixture()
    baseline_controls = replace(
        controls, mura_enabled=False, front_dt_s=5.0e-6,
        applied_pressure_a_to_b_Pa=0.0,
        driving_pressure_a_to_b_Pa=0.0)
    _, baseline = run_i3_cycle(
        context, initial, forward, driving, baseline_controls)
    _, half_available = run_i3_cycle(
        context, initial, forward, driving, replace(
            baseline_controls, front_symmetric_availability=.5))
    assert np.isclose(
        half_available["front_decision"]["net_velocity_a_to_b_m_s"],
        .5*baseline["front_decision"]["net_velocity_a_to_b_m_s"],
        rtol=2e-14, atol=0.0)

    zero_state, zero_available = run_i3_cycle(
        context, initial, forward, driving, replace(
            baseline_controls, front_symmetric_availability=0.0))
    assert zero_available["front_enabled"] is True
    assert zero_available["front_decision"] is None
    np.testing.assert_array_equal(zero_state.eta, initial.eta)

    _, altered_event = run_i3_cycle(
        context, initial, forward, driving, replace(
            baseline_controls, front_event_volume_b3=2.0,
            front_jump_length_b=.5))
    b = context["wall_parameters"].burgers_m
    assert altered_event["kinetic_event_volume_m3"] == 2.0*b**3
    assert altered_event["kinetic_event_length_m"] == .5*b
