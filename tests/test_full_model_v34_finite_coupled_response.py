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
    assert interaction["next_mura_extent_not_executed"]
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
