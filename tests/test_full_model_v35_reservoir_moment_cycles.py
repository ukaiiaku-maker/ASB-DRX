from dataclasses import replace

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import (
    checkpoint_payload, run_i3_cycle, state_from_payload,
)
from full_model.production.common_front_state import (
    apply_mechanical_increment, reconstruct_mechanical_state,
    state_arrays as common_front_arrays,
)
from full_model.production.coupled_front_production import (
    runtime_arrays, runtime_metadata_json,
)
from full_model.production.v24_mechanical_wall import (
    accepted_v24_mechanical_step,
)
from tests.test_full_model_v34_finite_coupled_response import _fixture


def _trial(context, displacement_cells):
    n = context["state"].eta.shape[0]
    spacing = context["spacing_m"]
    length = n*spacing
    width = context["interface_width_m"]
    x = np.arange(n)*spacing
    displacement = float(displacement_cells)*spacing
    child = .5*(
        np.tanh((x-(.25*length-displacement))/width)
        -np.tanh((x-(.75*length+displacement))/width))
    child = np.broadcast_to(child[:, None], (n, n))
    return np.stack((1.0-child, child), axis=2)


def _assert_state_exact(left, right):
    np.testing.assert_array_equal(left.eta, right.eta)
    for name, value in common_front_arrays(left.common_front).items():
        np.testing.assert_array_equal(
            value, common_front_arrays(right.common_front)[name])
    for name, value in runtime_arrays(left.front_runtime).items():
        np.testing.assert_array_equal(
            value, runtime_arrays(right.front_runtime)[name])
    assert runtime_metadata_json(left.front_runtime) == runtime_metadata_json(
        right.front_runtime)
    for group in ("common", "density", "reservoir_alignment"):
        a = getattr(left.mechanical, group)
        b = getattr(right.mechanical, group)
        for name in a.__dict__:
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_two_accepted_mura_front_cycles_and_midpoint_restart_are_exact():
    context, initial, _, _, driving, controls = _fixture()
    first_trial = _trial(context, .05)
    second_trial = _trial(context, .10)
    first, first_audit = run_i3_cycle(
        context, initial, first_trial, driving, controls)
    continuous, second_audit = run_i3_cycle(
        context, first, second_trial, driving, controls)
    restored = state_from_payload(checkpoint_payload(first, context), context)
    segmented, restarted_audit = run_i3_cycle(
        context, restored, second_trial, driving, controls)

    assert first_audit["candidate_sweep_published"]
    assert second_audit["candidate_sweep_published"]
    assert second_audit["sweep"]["positive_m3"] > 0.0
    assert second_audit["mura"]["event_scale"] > 0.0
    assert second_audit["next_mura_state_from_owned_reservoir_moments"]
    assert restarted_audit == second_audit
    _assert_state_exact(continuous, segmented)
    rebuilt = reconstruct_mechanical_state(
        continuous.common_front, context["spacing_m"], context["systems"],
        context["topologies"])
    for group in ("density", "reservoir_alignment"):
        for name in getattr(rebuilt, group).__dict__:
            np.testing.assert_array_equal(
                getattr(getattr(rebuilt, group), name),
                getattr(getattr(continuous.mechanical, group), name))


def test_rejected_second_front_rolls_back_exactly_after_accepted_second_mura():
    context, initial, _, _, driving, controls = _fixture()
    first, _ = run_i3_cycle(
        context, initial, _trial(context, .05), driving, controls)
    wrong_direction = replace(
        controls,
        driving_pressure_a_to_b_Pa=-controls.driving_pressure_a_to_b_Pa,
        applied_pressure_a_to_b_Pa=-controls.applied_pressure_a_to_b_Pa)
    rejected, audit = run_i3_cycle(
        context, first, _trial(context, .10), driving, wrong_direction)
    mura_only, control_audit = run_i3_cycle(
        context, first, first.eta, driving,
        replace(wrong_direction, front_enabled=False))
    assert not audit["candidate_sweep_published"]
    assert audit["front_decision"]["classification"] == (
        "REJECTED_BY_BIDIRECTIONAL_RATE")
    assert control_audit["mura"]["event_scale"] == audit["mura"][
        "event_scale"]
    _assert_state_exact(rejected, mura_only)


def test_retreat_and_revisit_preserve_evolved_owner_moments():
    context, state, _, _, driving, controls = _fixture()
    advanced, _ = run_i3_cycle(
        context, state, _trial(context, .05), driving, controls)
    reverse = replace(
        controls,
        driving_pressure_a_to_b_Pa=-controls.driving_pressure_a_to_b_Pa,
        applied_pressure_a_to_b_Pa=-controls.applied_pressure_a_to_b_Pa)
    retreated, retreat = run_i3_cycle(
        context, advanced, _trial(context, 0.0), driving, reverse)
    revisited, revisit = run_i3_cycle(
        context, retreated, _trial(context, .08), driving, controls)
    assert retreat["candidate_sweep_published"]
    assert retreat["sweep"]["negative_m3"] > 0.0
    assert revisit["candidate_sweep_published"]
    assert revisit["sweep"]["positive_m3"] > 0.0
    assert revisited.front_runtime.ledger.revisit_volume_m3 > 0.0
    assert np.linalg.norm(
        revisited.common_front.wake_alignment.mobile_plus_m2) > 0.0
    assert not np.array_equal(
        revisited.common_front.child_alignment.mobile_plus_m2,
        state.common_front.child_alignment.mobile_plus_m2)


def test_vanishing_support_retains_inactive_owner_history_exactly():
    context, initial, _, _, driving, controls = _fixture()
    front = initial.common_front.front
    chi = front.chi.copy()
    chi[:2, :] = 0.0
    modified_front = replace(
        front, chi=chi, processed_max=np.maximum(front.processed_max, chi),
        cleanup_max=np.maximum(front.cleanup_max, chi))
    owner_state = replace(initial.common_front, front=modified_front)
    mechanical = reconstruct_mechanical_state(
        owner_state, context["spacing_m"], context["systems"],
        context["topologies"])
    updated, _ = accepted_v24_mechanical_step(
        mechanical, driving, context["capture_support"],
        context["systems"], context["topologies"],
        context["wall_parameters"], context["extensive_parameters"],
        context["topology_kinetics"], controls.trial_dt_s,
        topology_route_enabled=False, mura_work_budget_mode="energy_limited")
    before = owner_state.child_alignment.mobile_plus_m2.copy()
    projected = apply_mechanical_increment(
        owner_state, updated, context["spacing_m"], context["systems"],
        context["topologies"])
    np.testing.assert_array_equal(
        projected.child_alignment.mobile_plus_m2[:2], before[:2])
    rebuilt = reconstruct_mechanical_state(
        projected, context["spacing_m"], context["systems"],
        context["topologies"])
    rebuilt.validate(context["systems"], context["topologies"])


def test_zero_applied_pressure_uses_nonreverse_channels_without_probe_work():
    context, initial, _, _, driving, controls = _fixture()
    zero_external = replace(
        controls, driving_pressure_a_to_b_Pa=0.0,
        applied_pressure_a_to_b_Pa=0.0,
        geometric_probe_pressure_Pa=1.0e8)
    result, audit = run_i3_cycle(
        context, initial, _trial(context, .05), driving, zero_external)
    assert audit["geometric_probe_is_nonphysical_and_unledgered"]
    assert audit["complete_directional_kinetics"]["a_to_b_event_J"] < 0.0
    assert audit["complete_directional_kinetics"]["b_to_a_event_J"] < 0.0
    assert audit["complete_directional_kinetics"]["actual_reverse_edge"] is False
    assert audit["complete_directional_kinetics"]["reverse_edge_status"] == (
        "DISTINCT_OUTGOING_ENDPOINTS_FROM_ONE_ACCEPTED_STATE")
    assert audit["front_decision"]["channel_a_to_b"][
        "acceptance_probability"] == 1.0
    assert audit["front_decision"]["channel_b_to_a"][
        "acceptance_probability"] == 1.0
    assert audit["front_decision"]["channel_a_to_b"][
        "transition_state_rate_s"] != audit["front_decision"][
            "channel_b_to_a"]["transition_state_rate_s"]
    assert audit["front_decision"]["net_velocity_a_to_b_m_s"] > 0.0
    assert audit["front_decision"]["accepted_signed_volume_m3"] > 0.0
    assert audit["complete_energy"]["front_decision"]["external_work_J"] == 0.0
    assert audit["candidate_sweep_published"]
    assert audit["actual_inventory_change"]["total_line_m"] != 0.0

    # The nonphysical proposal probe cannot choose a direction or do work:
    # changing its legacy diagnostic pressure leaves the physical result exact.
    alternate, alternate_audit = run_i3_cycle(
        context, initial, _trial(context, .05), driving,
        replace(zero_external, geometric_probe_pressure_Pa=-9.0e8))
    _assert_state_exact(result, alternate)
    assert audit["front_decision"] == alternate_audit["front_decision"]
    assert audit["complete_directional_kinetics"] == alternate_audit[
        "complete_directional_kinetics"]


def test_two_recurrent_cycles_share_one_accepted_physical_interval_each():
    context, initial, _, _, driving, controls = _fixture()
    common_clock = replace(
        controls, driving_pressure_a_to_b_Pa=2.0e8,
        applied_pressure_a_to_b_Pa=2.0e8,
        trial_dt_s=2.0e-9, front_dt_s=2.0e-9)
    first, audit1 = run_i3_cycle(
        context, initial, _trial(context, .05), driving, common_clock)
    second, audit2 = run_i3_cycle(
        context, first, _trial(context, .10), driving, common_clock)
    assert audit1["mura"]["accepted_dt_s"] == common_clock.front_dt_s
    assert audit2["mura"]["accepted_dt_s"] == common_clock.front_dt_s
    assert audit1["candidate_sweep_published"]
    assert audit2["candidate_sweep_published"]
    assert audit1["sweep"]["net_m3"] != 0.0
    assert audit2["sweep"]["net_m3"] != 0.0
    assert audit2["next_mura_state_from_owned_reservoir_moments"]
    assert second.common_front.ledger.accepted_commits == 2
