import pytest
import numpy as np

from full_model.production.v24_mechanical_wall import (
    accepted_v24_mechanical_step, mechanical_checkpoint_arrays,
    mechanical_from_checkpoint_arrays, select_feasible_family_extent,
)
from tests.test_full_model_v24_mechanical_wall import mechanical_fixture


def row(extent, affinity=None, *, kinematic=False, resolved=True):
    if kinematic:
        return {"extent": extent, "result": "INADMISSIBLE_KINEMATICS",
                "admissible": False}
    return {"extent": extent, "result": "EVALUATED",
            "admissible": affinity >= 0.0,
            "complete_affinity_J_m3_cells": affinity,
            "sign_resolved": resolved}


@pytest.mark.parametrize("curve, expected", [
    ([row(1.0, 2.0), row(.5, 1.2)],
     (1.0, "FULL_EVENT_ADMISSIBLE")),
    ([row(1.0, -1.0), row(.5, .3), row(.25, .2)],
     (.5, "INITIAL_DIRECTION_DOWNHILL_FULL_EVENT_OVERSHOOTS")),
    ([row(1.0, kinematic=True), row(.5, .4), row(.25, .2)],
     (.5, "CAPACITY_LIMITED_EVENT")),
    ([row(1.0, -2.0), row(.5, -1.0), row(.25, -.4)],
     (0.0, "GENUINELY_UPHILL_SCREENED_DIRECTION")),
    ([row(1.0, -.1, resolved=False), row(.5, -.02, resolved=False)],
     (0.0, "UNRESOLVED_SUBTRACTIVE_CANCELLATION")),
])
def test_family_extent_classification(curve, expected):
    assert select_feasible_family_extent(curve) == expected


def test_family_extent_curve_requires_full_event_first():
    with pytest.raises(ValueError, match="begin with the full event"):
        select_feasible_family_extent([row(.5, 1.0)])


def test_feasible_extent_mode_is_complete_and_restart_exact():
    args = mechanical_fixture()
    first, ledger = accepted_v24_mechanical_step(
        *args, dt_s=1e-9,
        mura_work_budget_mode="energy_limited_feasible_extents",
        feasible_family_extent_levels=5)
    budget = ledger["mura_work_budget"]
    assert budget["family_selection_rule"] == (
        "complete_affinity_feasible_family_extent_then_joint_backtrack")
    for candidate in budget["family_candidate_audits"]:
        assert len(candidate["extent_curve"]) == 5
        assert candidate["extent_curve"][0]["extent"] == 1.0
        assert candidate["selected_extent"] in (
            0.0, 1.0, .5, .25, .125, .0625)
    restored = mechanical_from_checkpoint_arrays(
        mechanical_checkpoint_arrays(first), args[3], args[4])
    continuous, _ = accepted_v24_mechanical_step(
        first, *args[1:], dt_s=1e-9,
        mura_work_budget_mode="energy_limited_feasible_extents",
        feasible_family_extent_levels=5)
    restarted, _ = accepted_v24_mechanical_step(
        restored, *args[1:], dt_s=1e-9,
        mura_work_budget_mode="energy_limited_feasible_extents",
        feasible_family_extent_levels=5)
    for group in ("common", "density", "reservoir_alignment"):
        left = getattr(continuous, group); right = getattr(restarted, group)
        for name in left.__dict__:
            np.testing.assert_array_equal(getattr(left, name),
                                          getattr(right, name))
