"""Auditable localization observables and conjunctive ASB classifier."""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class LocalizationCriteria:
    maximum_active_fraction: float
    minimum_temperature_excess_K: float
    minimum_softening_fraction: float
    minimum_width_to_interface: float
    persistence_steps: int
    refinement_tolerance: float
    def __post_init__(self) -> None:
        if not 0.0 < self.maximum_active_fraction < 1.0: raise ValueError("maximum_active_fraction must be in (0,1)")
        for name in ("minimum_temperature_excess_K", "minimum_softening_fraction", "minimum_width_to_interface", "refinement_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0: raise ValueError(f"{name} must be finite and nonnegative")
        if self.minimum_softening_fraction >= 1.0 or self.refinement_tolerance >= 1.0: raise ValueError("fractional criteria must be below one")
        if self.persistence_steps < 1: raise ValueError("persistence_steps must be positive")

@dataclass(frozen=True)
class LocalizationSnapshot:
    active_fraction: float
    effective_width_m: float
    temperature_excess_K: float
    softening_fraction: float

@dataclass(frozen=True)
class LocalizationDecision:
    localized: bool
    onset_index: int | None
    persistent_steps: int
    failed_criteria: tuple[str, ...]

def plastic_localization_geometry(rate_s_inv: np.ndarray, dx_m: float) -> tuple[float, float]:
    rate = np.abs(np.asarray(rate_s_inv, dtype=float))
    if rate.ndim != 2 or min(rate.shape) < 2 or not np.all(np.isfinite(rate)): raise ValueError("rate must be a finite 2-D field")
    if not math.isfinite(dx_m) or dx_m <= 0.0: raise ValueError("dx_m must be positive")
    total = float(np.sum(rate))
    if total == 0.0: return 1.0, min(rate.shape) * dx_m
    active_fraction = total**2 / (rate.size * float(np.sum(rate**2)))
    marginals = (np.sum(rate, axis=0), np.sum(rate, axis=1))
    widths = [dx_m * float(np.sum(item)) ** 2 / float(np.sum(item**2)) for item in marginals]
    return active_fraction, min(widths)

def localization_history(
    plastic_rate_s_inv: np.ndarray, temperature_K: np.ndarray,
    control_temperature_K: np.ndarray, stress_Pa: np.ndarray, dx_m: float,
) -> tuple[LocalizationSnapshot, ...]:
    rates = np.asarray(plastic_rate_s_inv, dtype=float); temperatures = np.asarray(temperature_K, dtype=float)
    controls = np.asarray(control_temperature_K, dtype=float); stress = np.asarray(stress_Pa, dtype=float)
    if rates.ndim != 3 or temperatures.shape != rates.shape or controls.shape != rates.shape: raise ValueError("rate and temperature histories must share (time,y,x)")
    if stress.shape != (rates.shape[0],) or not np.all(np.isfinite(stress)): raise ValueError("stress history shape mismatch")
    peak = np.maximum.accumulate(np.abs(stress))
    output = []
    for index in range(rates.shape[0]):
        active, width = plastic_localization_geometry(rates[index], dx_m)
        excess = float(np.max(temperatures[index] - controls[index]))
        softening = 0.0 if peak[index] == 0.0 else (peak[index] - abs(stress[index])) / peak[index]
        output.append(LocalizationSnapshot(active, width, excess, float(softening)))
    return tuple(output)

def classify_localization(history: tuple[LocalizationSnapshot, ...], interface_width_m: float, criteria: LocalizationCriteria) -> LocalizationDecision:
    if not math.isfinite(interface_width_m) or interface_width_m <= 0.0: raise ValueError("interface_width_m must be positive")
    consecutive = 0; onset = None; last_failed: tuple[str, ...] = ()
    for index, item in enumerate(history):
        failed = []
        if item.active_fraction > criteria.maximum_active_fraction: failed.append("plastic_concentration")
        if item.temperature_excess_K < criteria.minimum_temperature_excess_K: failed.append("temperature_control_excess")
        if item.softening_fraction < criteria.minimum_softening_fraction: failed.append("post_peak_softening")
        if item.effective_width_m < criteria.minimum_width_to_interface * interface_width_m: failed.append("resolved_width")
        if failed: consecutive = 0; onset = None; last_failed = tuple(failed)
        else:
            if consecutive == 0: onset = index
            consecutive += 1
            if consecutive >= criteria.persistence_steps: return LocalizationDecision(True, onset, consecutive, ())
    return LocalizationDecision(False, None, consecutive, last_failed)

def refinement_passes(coarse_onset: float, fine_onset: float, coarse_width_m: float, fine_width_m: float, tolerance: float) -> bool:
    values = (coarse_onset, fine_onset, coarse_width_m, fine_width_m, tolerance)
    if not all(math.isfinite(v) and v >= 0.0 for v in values) or tolerance >= 1.0: raise ValueError("refinement inputs invalid")
    onset_scale = max(abs(fine_onset), 1.0e-30); width_scale = max(abs(fine_width_m), 1.0e-30)
    return abs(coarse_onset-fine_onset)/onset_scale <= tolerance and abs(coarse_width_m-fine_width_m)/width_scale <= tolerance
