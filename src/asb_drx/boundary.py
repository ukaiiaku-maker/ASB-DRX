"""Prospective analytical boundary for the generic single-glider fixture."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .analytical import ExpFloorLaw


@dataclass(frozen=True)
class AnalyticalBoundaryPoint:
    """One point on the independent-law strength-maximum surface."""

    temperature_K: float
    shear_rate_s_inv: float
    peak_density_m2: float
    peak_strength_Pa: float
    density_m2: float
    density_ratio: float
    branch: str


@dataclass(frozen=True)
class AnalyticalPeakBoundary:
    """Classify density relative to the closed-form EXP-floor strength peak.

    This is a deliberately arbitrary, prospective regime boundary authorized for
    the generic campaign.  It separates the rising and falling branches of the
    independent-node analytical law.  The post-peak label is only a candidate
    region for collective/contact-capacity physics; it is not an ASB, DRX, or
    transparent-node classifier.
    """

    law: ExpFloorLaw
    equality_relative_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.equality_relative_tolerance)
            or self.equality_relative_tolerance < 0.0
        ):
            raise ValueError("equality_relative_tolerance must be finite and nonnegative")

    def classify(
        self,
        density_m2: float,
        temperature_K: float,
        shear_rate_s_inv: float,
    ) -> AnalyticalBoundaryPoint:
        if not math.isfinite(density_m2) or density_m2 <= 0.0:
            raise ValueError("density_m2 must be finite and positive")
        peak = self.law.peak(temperature_K, shear_rate_s_inv)
        ratio = density_m2 / peak.density_m2
        if math.isclose(
            ratio,
            1.0,
            rel_tol=self.equality_relative_tolerance,
            abs_tol=0.0,
        ):
            branch = "analytical_peak"
        elif ratio < 1.0:
            branch = "independent_rising_branch"
        else:
            branch = "post_peak_collective_candidate"
        return AnalyticalBoundaryPoint(
            temperature_K=temperature_K,
            shear_rate_s_inv=shear_rate_s_inv,
            peak_density_m2=peak.density_m2,
            peak_strength_Pa=peak.macroscopic_strength_Pa,
            density_m2=density_m2,
            density_ratio=ratio,
            branch=branch,
        )

    def surface(
        self,
        temperatures_K: tuple[float, ...],
        shear_rates_s_inv: tuple[float, ...],
    ) -> tuple[AnalyticalBoundaryPoint, ...]:
        if not temperatures_K or not shear_rates_s_inv:
            raise ValueError("boundary surface requires nonempty temperature and rate axes")
        points = []
        for rate in shear_rates_s_inv:
            for temperature in temperatures_K:
                peak = self.law.peak(temperature, rate)
                points.append(self.classify(peak.density_m2, temperature, rate))
        return tuple(points)
