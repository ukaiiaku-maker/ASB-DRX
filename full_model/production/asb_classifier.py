"""Strict conjunctive ASB classifier for matched full-model field histories."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ASBCriteria:
    maximum_active_fraction: float
    minimum_temperature_excess_K: float
    minimum_softening_fraction: float
    minimum_width_to_interface: float
    minimum_persistence_s: float
    refinement_tolerance: float

    def __post_init__(self):
        if not 0.0 < self.maximum_active_fraction < 1.0:
            raise ValueError("active fraction threshold must be in (0,1)")
        for name in ("minimum_temperature_excess_K", "minimum_softening_fraction",
                     "minimum_width_to_interface", "minimum_persistence_s",
                     "refinement_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.minimum_softening_fraction >= 1.0 or self.refinement_tolerance >= 1.0:
            raise ValueError("fractional thresholds must be below one")


@dataclass(frozen=True)
class ASBSnapshot:
    time_s: float
    active_fraction: float
    effective_width_m: float
    temperature_excess_K: float
    softening_fraction: float


@dataclass(frozen=True)
class ASBDecision:
    classified: bool
    onset_time_s: float | None
    persistence_s: float
    failed_criteria: tuple[str, ...]
    refinement_passed: bool


def localization_geometry(plastic_rate_s, dx_m, dy_m):
    rate = np.abs(np.asarray(plastic_rate_s, dtype=float))
    if rate.ndim != 2 or min(rate.shape) < 2 or not np.all(np.isfinite(rate)):
        raise ValueError("plastic rate must be a finite two-dimensional field")
    if dx_m <= 0.0 or dy_m <= 0.0:
        raise ValueError("cell dimensions must be positive")
    total = float(np.sum(rate))
    if total == 0.0:
        return 1.0, min(rate.shape[0]*dx_m, rate.shape[1]*dy_m)
    active_fraction = total**2/(rate.size*float(np.sum(rate**2)))
    marginal_x = np.sum(rate, axis=1)
    marginal_y = np.sum(rate, axis=0)
    width_x = dx_m*float(np.sum(marginal_x))**2/float(np.sum(marginal_x**2))
    width_y = dy_m*float(np.sum(marginal_y))**2/float(np.sum(marginal_y**2))
    return active_fraction, min(width_x, width_y)


def matched_history(plastic_rate_s, temperature_K, control_temperature_K,
                    stress_Pa, time_s, dx_m, dy_m):
    rates = np.asarray(plastic_rate_s, dtype=float)
    temperature = np.asarray(temperature_K, dtype=float)
    control = np.asarray(control_temperature_K, dtype=float)
    stress = np.asarray(stress_Pa, dtype=float)
    times = np.asarray(time_s, dtype=float)
    if rates.ndim != 3 or temperature.shape != rates.shape or control.shape != rates.shape:
        raise ValueError("matched rate and temperature histories must share (time,Nx,Ny)")
    if stress.shape != (rates.shape[0],) or times.shape != stress.shape:
        raise ValueError("stress and time histories must match field-history length")
    if not np.all(np.isfinite(stress)) or not np.all(np.isfinite(times)) \
            or np.any(np.diff(times) <= 0.0):
        raise ValueError("stress must be finite and time strictly increasing")
    peak = np.maximum.accumulate(np.abs(stress))
    snapshots = []
    for index, instant in enumerate(times):
        active, width = localization_geometry(rates[index], dx_m, dy_m)
        excess = float(np.max(temperature[index] - control[index]))
        softening = 0.0 if peak[index] == 0.0 else (peak[index]-abs(stress[index]))/peak[index]
        snapshots.append(ASBSnapshot(float(instant), active, width, excess, float(softening)))
    return tuple(snapshots)


def classify(history, interface_width_m, criteria, refinement_passed):
    if not history:
        raise ValueError("ASB history must be nonempty")
    if interface_width_m <= 0.0 or not math.isfinite(interface_width_m):
        raise ValueError("interface width must be finite and positive")
    start = None
    last_failed = ()
    for snapshot in history:
        failed = []
        if snapshot.active_fraction > criteria.maximum_active_fraction:
            failed.append("localized_plastic_rate_or_work")
        if snapshot.temperature_excess_K < criteria.minimum_temperature_excess_K:
            failed.append("matched_temperature_excess")
        if snapshot.softening_fraction < criteria.minimum_softening_fraction:
            failed.append("post_peak_softening")
        if snapshot.effective_width_m < criteria.minimum_width_to_interface*interface_width_m:
            failed.append("resolved_finite_width")
        if failed:
            start = None
            last_failed = tuple(failed)
            continue
        if start is None:
            start = snapshot.time_s
        persistence = snapshot.time_s - start
        time_tolerance = 64.0*math.ulp(max(abs(snapshot.time_s),
                                           abs(criteria.minimum_persistence_s), 1e-300))
        if persistence + time_tolerance >= criteria.minimum_persistence_s and refinement_passed:
            return ASBDecision(True, start, persistence, (), True)
    persistence = 0.0 if start is None else history[-1].time_s-start
    if not refinement_passed:
        last_failed = tuple(dict.fromkeys(last_failed + ("grid_and_timestep_refinement",)))
    elif start is not None and persistence < criteria.minimum_persistence_s:
        last_failed = ("physical_time_persistence",)
    return ASBDecision(False, None, persistence, last_failed, bool(refinement_passed))


def refinement_passes(coarse_onset_s, fine_onset_s, coarse_width_m, fine_width_m,
                      tolerance):
    values = (coarse_onset_s, fine_onset_s, coarse_width_m, fine_width_m, tolerance)
    if not all(math.isfinite(value) and value >= 0.0 for value in values) or tolerance >= 1.0:
        raise ValueError("refinement inputs are invalid")
    onset = abs(coarse_onset_s-fine_onset_s)/max(abs(fine_onset_s), 1e-300)
    width = abs(coarse_width_m-fine_width_m)/max(abs(fine_width_m), 1e-300)
    return onset <= tolerance and width <= tolerance
