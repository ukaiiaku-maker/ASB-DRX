from dataclasses import dataclass

import numpy as np
import pytest

from full_model.analysis.run_v64_adaptive_production import target_pending

from full_model.production.coupled_interval_control import (
    AdaptiveControllerState, CoupledEndpoint, CoupledInterval, ErrorMetric,
    advance_error_controlled_interval,
)


@dataclass(frozen=True)
class Runtime:
    time: float = 0.0
    work: float = 0.0


def endpoint(x=1.0, time=0.0):
    return CoupledEndpoint(
        np.array([x]), Runtime(time, x-1.0), time,
        {"x": np.array([x]), "rate": np.array([x*x])})


def nonlinear_map(counter):
    def advance(parent, h):
        counter.append((float(parent.load), h))
        x = float(parent.state[0])
        # Forward Euler for dx/dt=x; step doubling therefore has a nonzero,
        # duration-dependent physical error.
        evolved = x+h*x
        result = CoupledEndpoint(
            np.array([evolved]),
            Runtime(parent.runtime.time+h, parent.runtime.work+h*x),
            parent.load+h,
            {"x": np.array([evolved]), "rate": np.array([evolved*evolved])})
        return CoupledInterval(result, h, {"work": h*x}, {"accepted": True})
    return advance


def test_rejected_trial_is_private_and_first_half_is_reused():
    calls = []
    parent = endpoint()
    control = AdaptiveControllerState(.4, .025, .4)
    result = advance_error_controlled_interval(
        parent, control, nonlinear_map(calls),
        (ErrorMetric("x", "endpoint_increment", .08, 1e-12),))
    assert result.interval.endpoint.runtime.time == pytest.approx(
        result.decision["accepted_duration_s"])
    assert parent.runtime == Runtime()
    assert any(row["classification"] == "REJECTED_ACCURACY"
               for row in result.decision["attempts"])
    assert result.decision["attempts"][-1][
        "coarse_reused_from_rejected_first_half"] is True
    closure = result.decision["attempts"][-1][
        "trial_front_force_rate_closure"]
    assert set(closure) == {"coarse", "fine_first", "fine_second"}
    assert closure["coarse"]["classifications"] == []
    assert closure["coarse"]["set_valued_brackets"] == []
    assert result.controller.rejected_accuracy_trials >= 1


def test_two_consecutive_steps_advance_load_once_per_physical_interval():
    calls = []
    control = AdaptiveControllerState(.05, .0125, .05)
    first = advance_error_controlled_interval(
        endpoint(), control, nonlinear_map(calls),
        (ErrorMetric("x", "endpoint_increment", .2, 1e-12),))
    second = advance_error_controlled_interval(
        first.interval.endpoint, first.controller, nonlinear_map(calls),
        (ErrorMetric("x", "endpoint_increment", .2, 1e-12),))
    expected = (first.decision["accepted_duration_s"]
                +second.decision["accepted_duration_s"])
    assert second.interval.endpoint.load == pytest.approx(expected)
    assert second.interval.endpoint.runtime.time == pytest.approx(expected)


def test_two_half_tolerance_accepts_trigger_only_bounded_tested_growth():
    metric = (ErrorMetric("x", "endpoint_increment", .1, 1e-12),)
    control = AdaptiveControllerState(.2, .01, .5)
    first = advance_error_controlled_interval(
        endpoint(), control, nonlinear_map([]), metric)
    first_ratio = first.decision["attempts"][-1]["maximum_tolerance_ratio"]
    assert .25 < first_ratio < .5
    assert first.controller.proposed_step_s == pytest.approx(.2)
    second = advance_error_controlled_interval(
        first.interval.endpoint, first.controller, nonlinear_map([]), metric)
    assert second.controller.proposed_step_s == pytest.approx(.25)
    # The larger proposal is not pre-accepted: the next call must still build
    # its own coarse and two-half comparison before it can publish.
    assert second.decision["published_path"] == "two_complete_half_intervals"


def test_integral_and_rate_metrics_are_independent():
    result = advance_error_controlled_interval(
        endpoint(), AdaptiveControllerState(.05, .0125, .05),
        nonlinear_map([]), (
            ErrorMetric("rate", "endpoint_signal", .2, 1e-12),
            ErrorMetric("rate_change", "endpoint_increment", .2, 1e-12,
                        "rate"),
            ErrorMetric("work", "interval_integral", .2, 1e-12),
        ))
    errors = result.decision["attempts"][-1]["metric_errors"]
    assert errors["rate"]["source"] == "endpoint_signal"
    assert errors["rate_change"]["source"] == "endpoint_increment"
    assert errors["rate_change"]["observable_name"] == "rate"
    assert errors["work"]["source"] == "interval_integral"


def test_controller_round_trip_and_minimum_step_failure():
    control = AdaptiveControllerState(.1, .1, .1)
    assert AdaptiveControllerState.from_dict(control.to_dict()) == control
    with pytest.raises(RuntimeError, match="minimum step"):
        advance_error_controlled_interval(
            endpoint(), control, nonlinear_map([]),
            (ErrorMetric("x", "endpoint_increment", 1e-12, 1e-12),))


def test_target_roundoff_does_not_create_a_spurious_physical_interval():
    target = 1.7203125e-6
    assert not target_pending(target-5.0e-21, target)
    assert target_pending(target-1.0e-12, target)
