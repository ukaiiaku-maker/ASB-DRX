"""Mesh-invariant area-integrated hazard measure for full-v34 DRX.

Local creation kinetics provide a rate per physical site [s^-1].  Multiplying
by a declared areal site density [m^-2] gives hazard density [m^-2 s^-1].
Integration over physical cell area and time produces dimensionless expected
event count.  A single persistent Poisson clock owns the stochastic threshold;
Eulerian support motion never resets accumulated exposure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class AreaHazardState:
    exposure_field: np.ndarray
    residual_exposure: float
    total_exposure: float
    expected_raw_trigger_count: float
    next_threshold: float
    completed_events: int = 0
    threshold_redraws_after_event: int = 0
    threshold_redraws_other: int = 0
    discarded_exposure: float = 0.0
    transferred_exposure: float = 0.0
    newly_initialized_exposure: float = 0.0

    def __post_init__(self):
        field = np.asarray(self.exposure_field, dtype=float)
        if field.ndim != 2 or not np.all(np.isfinite(field)) or np.any(field < 0.0):
            raise ValueError("exposure field must be finite, nonnegative, and two-dimensional")
        for name in ("residual_exposure", "total_exposure", "expected_raw_trigger_count",
                     "next_threshold", "discarded_exposure", "transferred_exposure",
                     "newly_initialized_exposure"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.next_threshold <= 0.0:
            raise ValueError("Poisson threshold must be positive")
        if min(self.completed_events, self.threshold_redraws_after_event,
               self.threshold_redraws_other) < 0:
            raise ValueError("event and redraw counts must be nonnegative")


@dataclass(frozen=True)
class AreaHazardStep:
    state: AreaHazardState
    event_indices: tuple[tuple[int, int], ...]
    exposure_increment: float
    deferred_event_present: bool


def initialize_area_hazard(shape, rng):
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("hazard grid shape must be two-dimensional and nonempty")
    threshold = float(rng.exponential())
    return AreaHazardState(np.zeros(shape, dtype=float), 0.0, 0.0, 0.0, threshold)


def advance_area_hazard(state, local_rate_per_site_s, *, site_density_m2,
                        cell_area_m2, dt_s, rng, maximum_events=64):
    """Advance the exact frozen-rate Poisson clock over one interval.

    Event locations are independent draws from the normalized physical exposure
    increment. If the declared event-work bound is reached, unconsumed exposure
    and the current threshold remain in state for the next call.
    """
    if state.residual_exposure >= state.next_threshold:
        raise RuntimeError(
            "deferred Poisson event must be resolved before advancing physical fields")
    rate = np.asarray(local_rate_per_site_s, dtype=float)
    if rate.shape != state.exposure_field.shape or not np.all(np.isfinite(rate)) \
            or np.any(rate < 0.0):
        raise ValueError("local site rate must match the hazard grid and be nonnegative")
    for value, name in ((site_density_m2, "site density"),
                        (cell_area_m2, "cell area"), (dt_s, "timestep")):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if maximum_events < 1:
        raise ValueError("maximum_events must be positive")
    increment_field = rate*float(site_density_m2)*float(cell_area_m2)*float(dt_s)
    increment = float(np.sum(increment_field))
    exposure_field = np.asarray(state.exposure_field) + increment_field
    residual = state.residual_exposure + increment
    threshold = state.next_threshold
    weights = increment_field.ravel()
    weight_sum = float(np.sum(weights))
    events = []
    redraws = 0
    while residual >= threshold and len(events) < maximum_events:
        if weight_sum <= 0.0:
            raise RuntimeError("positive threshold crossing has no spatial exposure measure")
        flat = int(rng.choice(weights.size, p=weights/weight_sum))
        events.append(tuple(map(int, np.unravel_index(flat, rate.shape))))
        residual -= threshold
        threshold = float(rng.exponential())
        redraws += 1
    new_state = AreaHazardState(
        exposure_field=exposure_field, residual_exposure=residual,
        total_exposure=state.total_exposure + increment,
        expected_raw_trigger_count=state.expected_raw_trigger_count + increment,
        next_threshold=threshold,
        completed_events=state.completed_events + len(events),
        threshold_redraws_after_event=state.threshold_redraws_after_event + redraws,
        threshold_redraws_other=state.threshold_redraws_other,
        discarded_exposure=state.discarded_exposure,
        transferred_exposure=state.transferred_exposure,
        newly_initialized_exposure=state.newly_initialized_exposure)
    return AreaHazardStep(new_state, tuple(events), increment,
                          residual >= threshold)


def exposure_statistics(state):
    values = np.asarray(state.exposure_field, dtype=float)
    quantiles = np.quantile(values, (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0))
    return {
        "hazard_exposure_site_max": float(np.max(values)),
        "hazard_exposure_site_mean": float(np.mean(values)),
        "hazard_exposure_site_quantiles": tuple(map(float, quantiles)),
        "hazard_exposure_total": float(state.total_exposure),
        "expected_raw_trigger_count": float(state.expected_raw_trigger_count),
        "probability_at_least_one_raw_trigger": float(-math.expm1(-state.total_exposure)),
        "exposure_discarded": float(state.discarded_exposure),
        "exposure_transferred": float(state.transferred_exposure),
        "exposure_newly_initialized": float(state.newly_initialized_exposure),
        "threshold_redraws_after_event": int(state.threshold_redraws_after_event),
        "threshold_redraws_other": int(state.threshold_redraws_other),
    }
