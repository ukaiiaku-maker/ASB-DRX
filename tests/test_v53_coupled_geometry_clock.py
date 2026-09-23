from unittest.mock import patch

from full_model.production.coupled_geometry_clock import (
    accepted_common_clock_subcell_macro,
)


class DummyState:
    def __init__(self, label):
        self.label = label

    def validate(self, systems, topologies):
        return True


def arguments(state, duration=8e-9):
    return (state, duration, {"lower_x": 1.0, "upper_x": -1.0},
            (), (), object(), object(), object(), object(), object(), object())


def test_strang_common_clock_advances_external_time_by_h_not_2h():
    state = DummyState("initial")
    bulk_calls = []
    geometry_calls = []

    def bulk(current, *args, **kwargs):
        duration = args[7]
        bulk_calls.append(duration)
        return DummyState(current.label+"B"), {"accepted_dt_s": duration}

    def geometry(current, *args, **kwargs):
        duration = args[7]
        geometry_calls.append(duration)
        return DummyState(current.label+"G"), {
            "accepted": True, "consumed_duration_s": duration,
            "macro_complete": True,
        }

    with patch(
            "full_model.production.coupled_geometry_clock."
            "accepted_v24_mechanical_step", side_effect=bulk), patch(
            "full_model.production.coupled_geometry_clock."
            "accepted_state_dependent_subcell_x_faces", side_effect=geometry):
        result, audit = accepted_common_clock_subcell_macro(*arguments(state))
    assert result.label == "initialBGB"
    assert bulk_calls == [4e-9, 4e-9]
    assert geometry_calls == [8e-9]
    assert audit["physical_elapsed_time_s"] == 8e-9
    assert audit["bulk_operator_exposure_s"] == 8e-9
    assert audit["geometry_operator_exposure_s"] == 8e-9
    assert audit["elapsed_clock_is_H_not_2H"]


def test_unfillable_geometry_prefix_rolls_back_complete_macro():
    state = DummyState("initial")
    geometry_calls = {"count": 0}

    def bulk(current, *args, **kwargs):
        duration = args[7]
        return DummyState(current.label+"B"), {"accepted_dt_s": duration}

    def geometry(current, *args, **kwargs):
        geometry_calls["count"] += 1
        duration = args[7]
        if geometry_calls["count"] == 1:
            return DummyState(current.label+"P"), {
                "accepted": True, "consumed_duration_s": duration/2,
                "macro_complete": False,
            }
        return current, {
            "accepted": False, "consumed_duration_s": 0.0,
            "macro_complete": False,
        }

    with patch(
            "full_model.production.coupled_geometry_clock."
            "accepted_v24_mechanical_step", side_effect=bulk), patch(
            "full_model.production.coupled_geometry_clock."
            "accepted_state_dependent_subcell_x_faces", side_effect=geometry):
        result, audit = accepted_common_clock_subcell_macro(*arguments(state))
    assert result is state
    assert not audit["accepted"]
    assert audit["complete_macro_rollback"]
    assert audit["physical_elapsed_time_s"] == 0.0
    assert audit["unpublished_attempted_prefix_s"] == 4e-9
