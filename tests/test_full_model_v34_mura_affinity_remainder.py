import numpy as np

from full_model.production.v24_mechanical_wall import accepted_v24_mechanical_step
from tests.test_full_model_v24_mechanical_wall import mechanical_fixture


def test_complete_affinity_selection_and_joint_candidate_are_ledgered():
    _, ledger = accepted_v24_mechanical_step(
        *mechanical_fixture(), dt_s=1e-9,
        mura_work_budget_mode="energy_limited")
    budget = ledger["mura_work_budget"]
    assert budget["family_selection_rule"] == (
        "complete_discrete_full_event_affinity_then_joint_backtrack")
    assert len(budget["family_candidate_audits"]) == 4
    for candidate in budget["family_candidate_audits"]:
        if candidate["result"] == "EVALUATED":
            assert candidate["selected"] == candidate["audit"]["admissible"]
    assert budget["joint_selected_family_candidate"] is budget["trials"][0]
    assert budget["accepted"]["admissible"]


def test_unreacted_remainder_is_deterministic_no_debt_rate_constraint():
    args = mechanical_fixture()
    first_state, first = accepted_v24_mechanical_step(
        *args, dt_s=1e-9, mura_work_budget_mode="energy_limited")
    second_state, second = accepted_v24_mechanical_step(
        *args, dt_s=1e-9, mura_work_budget_mode="energy_limited")
    first_budget = first["mura_work_budget"]
    second_budget = second["mura_work_budget"]
    assert first_budget["unreacted_remainder_semantics"] == (
        "deterministic_constrained_rate_no_debt_recomputed_next_step")
    expected = 1.0-first["mura_event_scale"]*np.asarray(
        first["mura_family_event_scales"])
    np.testing.assert_array_equal(
        first_budget["family_unreacted_fractions"], expected)
    assert first_budget["family_unreacted_fractions"] == second_budget[
        "family_unreacted_fractions"]
    for group in ("common", "density", "reservoir_alignment"):
        left = getattr(first_state, group); right = getattr(second_state, group)
        for name in left.__dict__:
            np.testing.assert_array_equal(getattr(left, name),
                                          getattr(right, name))


def test_legacy_off_control_does_not_apply_affinity_family_selection():
    _, ledger = accepted_v24_mechanical_step(
        *mechanical_fixture(), dt_s=1e-9,
        mura_work_budget_mode="legacy_reject")
    budget = ledger["mura_work_budget"]
    assert budget["family_selection_rule"] == "legacy_no_family_selection"
    assert budget["family_candidate_audits"] == []
    np.testing.assert_array_equal(ledger["mura_family_event_scales"],
                                  np.ones(4))
