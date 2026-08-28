"""Auditable classical candidate-admission kernel.

This module does not select a physical attempt prefactor, generate random
numbers, allocate order parameters, or replace the EXP-floor slip barrier.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from .grains import crystallographic_misorientation_rad


BOLTZMANN_J_K = 1.380649e-23
DecisionReason = Literal[
    "accepted",
    "unresolved_support",
    "subcritical_radius",
    "insufficient_misorientation",
    "thermal_draw_rejected",
]


@dataclass(frozen=True)
class CylindricalNucleationParameters:
    boundary_energy_J_m2: float
    stored_energy_driving_J_m3: float
    represented_thickness_m: float
    areal_attempt_rate_m2_s: float

    def __post_init__(self) -> None:
        for name in (
            "boundary_energy_J_m2",
            "stored_energy_driving_J_m3",
            "represented_thickness_m",
            "areal_attempt_rate_m2_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def critical_radius_m(self) -> float:
        return self.boundary_energy_J_m2 / self.stored_energy_driving_J_m3

    @property
    def escape_radius_m(self) -> float:
        """Radius above which cylindrical excess energy is negative."""

        return 2.0 * self.critical_radius_m

    @property
    def barrier_J(self) -> float:
        return (
            math.pi
            * self.represented_thickness_m
            * self.boundary_energy_J_m2**2
            / self.stored_energy_driving_J_m3
        )

    def excess_energy_J(self, radius_m: float) -> float:
        _check_positive("radius_m", radius_m)
        return self.represented_thickness_m * (
            2.0 * math.pi * radius_m * self.boundary_energy_J_m2
            - math.pi * radius_m**2 * self.stored_energy_driving_J_m3
        )

    def excess_energy_derivative_J_m(self, radius_m: float) -> float:
        _check_positive("radius_m", radius_m)
        return 2.0 * math.pi * self.represented_thickness_m * (
            self.boundary_energy_J_m2
            - radius_m * self.stored_energy_driving_J_m3
        )

    def event_probability(
        self, temperature_K: float, eligible_area_m2: float, interval_s: float
    ) -> float:
        """Poisson probability using an explicitly supplied areal prefactor."""

        _check_positive("temperature_K", temperature_K)
        _check_positive("eligible_area_m2", eligible_area_m2)
        _check_positive("interval_s", interval_s)
        log_expected_events = (
            math.log(self.areal_attempt_rate_m2_s)
            + math.log(eligible_area_m2)
            + math.log(interval_s)
            - self.barrier_J / (BOLTZMANN_J_K * temperature_K)
        )
        if log_expected_events < -745.0:
            return 0.0
        if log_expected_events > 36.0:
            return 1.0
        expected_events = math.exp(log_expected_events)
        return -math.expm1(-expected_events)


@dataclass(frozen=True)
class CandidateDecision:
    accepted: bool
    reason: DecisionReason
    event_probability: float
    critical_radius_m: float
    barrier_J: float
    misorientation_rad: float


def evaluate_candidate(
    parameters: CylindricalNucleationParameters,
    *,
    candidate_radius_m: float,
    minimum_resolved_radius_m: float,
    candidate_orientation_rad: float,
    parent_orientation_rad: float,
    minimum_misorientation_rad: float,
    symmetry_order: int,
    temperature_K: float,
    eligible_area_m2: float,
    interval_s: float,
    uniform_draw: float,
) -> CandidateDecision:
    """Evaluate a supplied trial without allocating a field or drawing RNG state."""

    _check_positive("candidate_radius_m", candidate_radius_m)
    _check_positive("minimum_resolved_radius_m", minimum_resolved_radius_m)
    if not math.isfinite(minimum_misorientation_rad) or minimum_misorientation_rad < 0.0:
        raise ValueError("minimum_misorientation_rad must be finite and nonnegative")
    if not math.isfinite(uniform_draw) or not 0.0 <= uniform_draw < 1.0:
        raise ValueError("uniform_draw must be finite and in [0, 1)")
    probability = parameters.event_probability(
        temperature_K, eligible_area_m2, interval_s
    )
    misorientation = crystallographic_misorientation_rad(
        candidate_orientation_rad, parent_orientation_rad, symmetry_order
    )
    if candidate_radius_m < minimum_resolved_radius_m:
        reason: DecisionReason = "unresolved_support"
    elif candidate_radius_m <= parameters.critical_radius_m:
        reason = "subcritical_radius"
    elif misorientation < minimum_misorientation_rad:
        reason = "insufficient_misorientation"
    elif uniform_draw >= probability:
        reason = "thermal_draw_rejected"
    else:
        reason = "accepted"
    return CandidateDecision(
        reason == "accepted",
        reason,
        probability,
        parameters.critical_radius_m,
        parameters.barrier_J,
        misorientation,
    )


def _check_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
