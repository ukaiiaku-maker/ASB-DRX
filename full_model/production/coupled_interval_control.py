"""Rollback-safe local accuracy control for complete coupled intervals.

The controller is deliberately agnostic to the constitutive model.  A supplied
interval map must be a pure operation on an immutable parent endpoint: trial
states and ledgers are private return values.  One interval of length ``h`` is
compared with two consecutive intervals of length ``h/2`` under the same load
history.  Only the fine endpoint is eligible for publication.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import time
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class CoupledEndpoint:
    state: object
    runtime: object
    load: float
    observables: Mapping[str, np.ndarray | float]
    numerical_hints: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CoupledInterval:
    endpoint: CoupledEndpoint
    duration_s: float
    integrals: Mapping[str, np.ndarray | float]
    audit: Mapping[str, object]


@dataclass(frozen=True)
class ErrorMetric:
    name: str
    source: str = "endpoint_increment"
    relative_tolerance: float = 0.02
    absolute_scale: float = 0.0
    observable_name: str | None = None

    def validate(self) -> None:
        if self.source not in {
                "endpoint_increment", "endpoint_signal", "interval_integral"}:
            raise ValueError(f"unknown accuracy metric source {self.source!r}")
        if (not math.isfinite(self.relative_tolerance)
                or self.relative_tolerance <= 0.0):
            raise ValueError("relative tolerance must be positive and finite")
        if not math.isfinite(self.absolute_scale) or self.absolute_scale < 0.0:
            raise ValueError("absolute scale must be nonnegative and finite")


@dataclass(frozen=True)
class AdaptiveControllerState:
    proposed_step_s: float
    minimum_step_s: float
    maximum_step_s: float
    accepted_intervals: int = 0
    rejected_accuracy_trials: int = 0
    failed_admissibility_trials: int = 0
    consecutive_low_error_accepts: int = 0

    def validate(self) -> None:
        values = (self.proposed_step_s, self.minimum_step_s,
                  self.maximum_step_s)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("adaptive steps must be positive and finite")
        if not self.minimum_step_s <= self.proposed_step_s <= self.maximum_step_s:
            raise ValueError("proposed adaptive step lies outside its bounds")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]):
        result = cls(**payload)
        result.validate()
        return result


@dataclass(frozen=True)
class AcceptedAdaptiveInterval:
    interval: CoupledInterval
    controller: AdaptiveControllerState
    decision: Mapping[str, object]


def _norm(value) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=float).ravel()))


def _metric_error(metric: ErrorMetric, parent: CoupledEndpoint,
                  coarse: CoupledInterval, fine: CoupledInterval) -> dict:
    metric.validate()
    field = metric.name if metric.observable_name is None else metric.observable_name
    if metric.source == "interval_integral":
        left = np.asarray(coarse.integrals[field], dtype=float)
        right = np.asarray(fine.integrals[field], dtype=float)
        parent_value = None
        scale = max(_norm(left), _norm(right), metric.absolute_scale)
    else:
        left = np.asarray(coarse.endpoint.observables[field], dtype=float)
        right = np.asarray(fine.endpoint.observables[field], dtype=float)
        parent_value = np.asarray(parent.observables[field], dtype=float)
        if metric.source == "endpoint_increment":
            scale = max(_norm(left-parent_value), _norm(right-parent_value),
                        metric.absolute_scale)
        else:
            scale = max(_norm(left), _norm(right), metric.absolute_scale)
    difference = _norm(left-right)
    relative = difference/max(scale, np.finfo(float).tiny)
    return {
        "source": metric.source,
        "observable_name": field,
        "absolute_l2": difference,
        "normalization_l2": scale,
        "normalized_error": relative,
        "relative_tolerance": metric.relative_tolerance,
        "absolute_scale": metric.absolute_scale,
        "passed": bool(relative <= metric.relative_tolerance),
        "effectively_absolute": bool(scale <= metric.absolute_scale),
    }


def _combine_fine(first: CoupledInterval, second: CoupledInterval) -> CoupledInterval:
    if set(first.integrals) != set(second.integrals):
        raise ValueError("fine subinterval integral fields differ")
    integrals = {
        name: np.asarray(first.integrals[name])+np.asarray(second.integrals[name])
        for name in first.integrals
    }
    return CoupledInterval(
        second.endpoint, first.duration_s+second.duration_s, integrals,
        {"first_half": first.audit, "second_half": second.audit})


def advance_error_controlled_interval(
        parent: CoupledEndpoint,
        controller: AdaptiveControllerState,
        interval_map: Callable[[CoupledEndpoint, float], CoupledInterval],
        metrics: tuple[ErrorMetric, ...], *,
        requested_maximum_s: float | None = None) -> AcceptedAdaptiveInterval:
    """Advance exactly one accepted interval and publish the fine path only.

    A rejected step's already-computed first half becomes the coarse candidate
    at the reduced duration.  This reuse is exact because it started from the
    same immutable parent and load history; the rejected second half remains
    private and is discarded.
    """
    controller.validate()
    if not metrics:
        raise ValueError("at least one physical accuracy metric is required")
    cap = (controller.maximum_step_s if requested_maximum_s is None else
           min(controller.maximum_step_s, float(requested_maximum_s)))
    h = min(controller.proposed_step_s, cap)
    effective_minimum = min(controller.minimum_step_s, cap)
    if h <= 0.0:
        raise ValueError("requested maximum interval must be positive")
    coarse_hint = None
    attempts = []
    accuracy_rejections = 0
    admissibility_failures = 0
    while True:
        if h < effective_minimum*(1.0-32*np.finfo(float).eps):
            raise RuntimeError(
                "coupled accuracy controller exhausted its minimum step")
        started = time.perf_counter()
        try:
            if coarse_hint is None:
                coarse = interval_map(parent, h)
                coarse_reused = False
            else:
                coarse = coarse_hint
                coarse_reused = True
            half = 0.5*h
            fine_first = interval_map(parent, half)
            fine_second = interval_map(fine_first.endpoint, half)
            fine = _combine_fine(fine_first, fine_second)
        except (RuntimeError, ValueError, FloatingPointError) as error:
            admissibility_failures += 1
            attempts.append({
                "duration_s": h, "classification": "ADMISSIBILITY_FAILURE",
                "error": str(error), "wall_time_s": time.perf_counter()-started,
            })
            h *= 0.5
            coarse_hint = None
            continue
        if not np.isclose(coarse.duration_s, h, rtol=0.0,
                          atol=32*np.finfo(float).eps*h):
            raise RuntimeError("coarse coupled map did not consume its interval")
        if not np.isclose(fine.duration_s, h, rtol=0.0,
                          atol=32*np.finfo(float).eps*h):
            raise RuntimeError("fine coupled map did not consume its interval")
        load_error = abs(float(coarse.endpoint.load)-float(fine.endpoint.load))
        errors = {metric.name: _metric_error(
            metric, parent, coarse, fine) for metric in metrics}
        passed = bool(load_error <= 64*np.finfo(float).eps*max(
            abs(float(parent.load)), abs(float(fine.endpoint.load)), 1.0)
            and all(row["passed"] for row in errors.values()))
        worst_ratio = max(
            row["normalized_error"]/row["relative_tolerance"]
            for row in errors.values())
        attempts.append({
            "duration_s": h,
            "classification": ("ACCEPTED_FINE_PATH" if passed
                               else "REJECTED_ACCURACY"),
            "coarse_reused_from_rejected_first_half": coarse_reused,
            "endpoint_load_absolute_difference": load_error,
            "metric_errors": errors,
            "trial_cost_breakdown_s": {
                "coarse": coarse.audit.get("cost_breakdown_s"),
                "fine_first": fine_first.audit.get("cost_breakdown_s"),
                "fine_second": fine_second.audit.get("cost_breakdown_s"),
            },
            "trial_front_force_rate_closure": {
                name: {
                    "classifications": interval.audit.get(
                        "front_force_rate_closure_classifications", []),
                    "set_valued_brackets": interval.audit.get(
                        "front_force_rate_closure_brackets", []),
                }
                for name, interval in (
                    ("coarse", coarse), ("fine_first", fine_first),
                    ("fine_second", fine_second))
            },
            "maximum_tolerance_ratio": worst_ratio,
            "wall_time_s": time.perf_counter()-started,
        })
        if passed:
            # Half-tolerance leaves a full factor-of-two error margin while
            # allowing the controller to escape unnecessarily conservative
            # plateaus.  Growth remains only 1.25x after two consecutive
            # accepts, and the proposed interval is independently tested and
            # rolled back before publication if that margin does not persist.
            low = worst_ratio <= 0.5
            low_count = (controller.consecutive_low_error_accepts+1
                         if low else 0)
            proposed = h
            # No convergence order is assumed.  Growth requires two successive
            # intervals comfortably below tolerance and remains deliberately
            # bounded; any failed proposal rolls back normally.
            if low_count >= 2:
                proposed = min(1.25*h, controller.maximum_step_s)
                low_count = 0
            updated = replace(
                controller, proposed_step_s=max(
                    controller.minimum_step_s, min(proposed,
                                                   controller.maximum_step_s)),
                accepted_intervals=controller.accepted_intervals+1,
                rejected_accuracy_trials=(
                    controller.rejected_accuracy_trials+accuracy_rejections),
                failed_admissibility_trials=(
                    controller.failed_admissibility_trials
                    +admissibility_failures),
                consecutive_low_error_accepts=low_count)
            decision = {
                "schema": "asb-drx-v64-complete-interval-control-v1",
                "accepted_duration_s": h,
                "published_path": "two_complete_half_intervals",
                "coarse_path_published": False,
                "attempts": attempts,
                "controller_before": controller.to_dict(),
                "controller_after": updated.to_dict(),
            }
            return AcceptedAdaptiveInterval(fine, updated, decision)
        accuracy_rejections += 1
        h *= 0.5
        coarse_hint = fine_first
