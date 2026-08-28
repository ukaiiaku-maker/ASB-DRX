"""Continuous collective-event ablations with explicit independent limits.

These closures are structural comparators, not calibrated production physics.
No closure in this module creates a grain or switches on at a density threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SequentialHitClosure:
    """Erlang renewal representation of a required sequence of hits."""

    hit_order: int

    def __post_init__(self) -> None:
        if self.hit_order < 1:
            raise ValueError("hit_order must be positive")

    def stage_rate_s_inv(self, independent_rate_s_inv: float) -> float:
        _check_rate(independent_rate_s_inv)
        return self.hit_order * independent_rate_s_inv

    @property
    def completion_wait_cv(self) -> float:
        return 1.0 / math.sqrt(self.hit_order)

    def mean_completion_rate_s_inv(self, independent_rate_s_inv: float) -> float:
        _check_rate(independent_rate_s_inv)
        return independent_rate_s_inv


@dataclass(frozen=True)
class RearmingContactClosure:
    """Alternating activation/rearm contact with a finite transparent interval."""

    rearm_time_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.rearm_time_s) or self.rearm_time_s < 0.0:
            raise ValueError("rearm_time_s must be finite and nonnegative")

    def stationary_completion_rate_s_inv(self, independent_rate_s_inv: float) -> float:
        _check_rate(independent_rate_s_inv)
        return independent_rate_s_inv / (
            1.0 + independent_rate_s_inv * self.rearm_time_s
        )

    def completion_wait_cv(self, independent_rate_s_inv: float) -> float:
        _check_rate(independent_rate_s_inv)
        activation_time = 1.0 / independent_rate_s_inv
        if self.rearm_time_s == 0.0:
            return 1.0
        return math.sqrt(activation_time**2 + self.rearm_time_s**2) / (
            activation_time + self.rearm_time_s
        )


@dataclass(frozen=True)
class ExponentialShotNoiseClosure:
    """Mean-field exponential-memory self-excitation (Hawkes ablation).

    Each event raises the conditional intensity by
    ``relative_kick * independent_rate`` and that excess decays with
    ``memory_time_s``. The dimensionless branching ratio is therefore
    ``relative_kick * independent_rate * memory_time``.
    """

    relative_kick: float
    memory_time_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.relative_kick) or self.relative_kick < 0.0:
            raise ValueError("relative_kick must be finite and nonnegative")
        if not math.isfinite(self.memory_time_s) or self.memory_time_s <= 0.0:
            raise ValueError("memory_time_s must be finite and positive")

    def branching_ratio(self, independent_rate_s_inv: float) -> float:
        _check_rate(independent_rate_s_inv)
        return self.relative_kick * independent_rate_s_inv * self.memory_time_s

    def stationary_mean_rate_s_inv(self, independent_rate_s_inv: float) -> float:
        branching = self.branching_ratio(independent_rate_s_inv)
        if branching >= 1.0:
            raise ValueError("no stationary mean at or above the branching threshold")
        return independent_rate_s_inv / (1.0 - branching)

    def linear_memory_growth_rate_s_inv(self, independent_rate_s_inv: float) -> float:
        return (self.branching_ratio(independent_rate_s_inv) - 1.0) / self.memory_time_s


def _check_rate(value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("independent rate must be finite and positive")
