"""Identifiability-aware calibration helpers for the analytical peak law."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares

from .analytical import ExpFloorLaw


_FIT_NAMES = (
    "barrier_ref_J",
    "stress_ref_Pa",
    "rate_prefactor_s_inv",
    "barrier_temperature_coefficient",
    "stress_temperature_coefficient",
)


@dataclass(frozen=True)
class PeakObservation:
    """A peak-strength observation and optional independently measured density."""

    temperature_K: float
    shear_rate_s_inv: float
    strength_Pa: float
    density_m2: float | None = None

    def __post_init__(self) -> None:
        for name in ("temperature_K", "shear_rate_s_inv", "strength_Pa"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.density_m2 is not None and (
            not math.isfinite(self.density_m2) or self.density_m2 <= 0.0
        ):
            raise ValueError("density_m2 must be finite and positive when supplied")


@dataclass(frozen=True)
class PeakFitResult:
    """Optimization result with a local numerical-identifiability audit."""

    law: ExpFloorLaw
    rms_log_residual: float
    jacobian_rank: int
    parameter_count: int
    singular_values: tuple[float, ...]
    condition_number: float
    identifiable: bool
    evaluations: int


def synthetic_peak_observations(
    law: ExpFloorLaw,
    temperatures_K: Iterable[float],
    shear_rates_s_inv: Iterable[float],
    *,
    include_density: bool,
) -> tuple[PeakObservation, ...]:
    """Generate exact analytical fixtures; this function is not physical data."""

    observations = []
    for temperature in temperatures_K:
        for rate in shear_rates_s_inv:
            peak = law.peak(float(temperature), float(rate))
            observations.append(
                PeakObservation(
                    temperature_K=float(temperature),
                    shear_rate_s_inv=float(rate),
                    strength_Pa=peak.macroscopic_strength_Pa,
                    density_m2=peak.density_m2 if include_density else None,
                )
            )
    return tuple(observations)


def fit_peak_scale_parameters(
    template: ExpFloorLaw,
    observations: Sequence[PeakObservation],
    *,
    starts: int = 7,
    seed: int = 20260828,
    rank_relative_tolerance: float = 1.0e-8,
) -> PeakFitResult:
    """Fit five peak-scale parameters and report local Jacobian rank.

    Barrier shape, density exponent, Burgers vector, and reference temperature
    remain fixed. Inputs are log-strength residuals plus log-density residuals
    where density was measured independently.
    """

    if not observations:
        raise ValueError("at least one observation is required")
    if starts < 1:
        raise ValueError("starts must be positive")

    lower = np.array(
        [math.log(0.05 * 1.602176634e-19), math.log(1.0e6), math.log(1.0e4), -5.0, -5.0]
    )
    upper = np.array(
        [math.log(10.0 * 1.602176634e-19), math.log(1.0e11), math.log(1.0e20), 5.0, 5.0]
    )
    center = _encode(template)
    if np.any(center <= lower) or np.any(center >= upper):
        raise ValueError("template scale parameters lie outside optimizer bounds")

    def residual(vector: np.ndarray) -> np.ndarray:
        law = _decode(template, vector)
        values: list[float] = []
        for observation in observations:
            try:
                peak = law.peak(observation.temperature_K, observation.shear_rate_s_inv)
                values.append(math.log(peak.macroscopic_strength_Pa / observation.strength_Pa))
                if observation.density_m2 is not None:
                    values.append(math.log(peak.density_m2 / observation.density_m2))
            except (OverflowError, ValueError):
                count = 2 if observation.density_m2 is not None else 1
                values.extend([1.0e6] * count)
        return np.asarray(values, dtype=float)

    rng = np.random.default_rng(seed)
    candidates = [center]
    for _ in range(starts - 1):
        candidates.append(lower + (upper - lower) * rng.uniform(0.05, 0.95, size=lower.size))

    solutions = [
        least_squares(residual, start, bounds=(lower, upper), xtol=1.0e-13, ftol=1.0e-13, gtol=1.0e-13)
        for start in candidates
    ]
    best = min(solutions, key=lambda item: float(np.dot(item.fun, item.fun)))
    singular = np.linalg.svd(best.jac, compute_uv=False)
    threshold = rank_relative_tolerance * singular[0]
    rank = int(np.count_nonzero(singular > threshold))
    condition = math.inf if singular[-1] <= 0.0 else float(singular[0] / singular[-1])
    return PeakFitResult(
        law=_decode(template, best.x),
        rms_log_residual=float(np.sqrt(np.mean(best.fun**2))),
        jacobian_rank=rank,
        parameter_count=len(_FIT_NAMES),
        singular_values=tuple(float(value) for value in singular),
        condition_number=condition,
        identifiable=rank == len(_FIT_NAMES),
        evaluations=int(sum(solution.nfev for solution in solutions)),
    )


def _encode(law: ExpFloorLaw) -> np.ndarray:
    return np.array(
        [
            math.log(law.barrier_ref_J),
            math.log(law.stress_ref_Pa),
            math.log(law.rate_prefactor_s_inv),
            law.barrier_temperature_coefficient,
            law.stress_temperature_coefficient,
        ],
        dtype=float,
    )


def _decode(template: ExpFloorLaw, vector: np.ndarray) -> ExpFloorLaw:
    return replace(
        template,
        barrier_ref_J=float(math.exp(vector[0])),
        stress_ref_Pa=float(math.exp(vector[1])),
        rate_prefactor_s_inv=float(math.exp(vector[2])),
        barrier_temperature_coefficient=float(vector[3]),
        stress_temperature_coefficient=float(vector[4]),
    )
