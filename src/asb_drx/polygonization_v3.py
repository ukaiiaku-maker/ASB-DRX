"""Isolated signed-content wall maturation and polygonization fixture."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .arrhenius_v3 import ArrheniusMechanism


@dataclass(frozen=True)
class PolygonizationState:
    mobile_plus_m2: np.ndarray
    mobile_minus_m2: np.ndarray
    wall_plus_m2: np.ndarray
    wall_minus_m2: np.ndarray
    wall_maturity: float = 0.0

    def __post_init__(self) -> None:
        arrays = tuple(np.asarray(x, dtype=float) for x in (
            self.mobile_plus_m2, self.mobile_minus_m2,
            self.wall_plus_m2, self.wall_minus_m2,
        ))
        if arrays[0].ndim != 1 or any(x.shape != arrays[0].shape for x in arrays):
            raise ValueError("polygonization populations must share family shape")
        if any(np.any(~np.isfinite(x)) or np.any(x < 0.0) for x in arrays):
            raise ValueError("polygonization populations must be finite and nonnegative")
        if not math.isfinite(self.wall_maturity) or not 0.0 <= self.wall_maturity <= 1.0:
            raise ValueError("wall_maturity must lie in [0,1]")
        for name, value in zip((
            "mobile_plus_m2", "mobile_minus_m2",
            "wall_plus_m2", "wall_minus_m2",
        ), arrays):
            object.__setattr__(self, name, value.copy())

    @property
    def physical_grain_count(self) -> int:
        return 1


@dataclass(frozen=True)
class PolygonizationParameters:
    capture: ArrheniusMechanism
    climb_annihilation: ArrheniusMechanism
    wall_ordering: ArrheniusMechanism
    burgers_m: float
    wall_width_m: float
    line_energy_J_m: float
    capture_driving_stress_Pa: float = 0.0

    def __post_init__(self) -> None:
        for name in ("burgers_m", "wall_width_m", "line_energy_J_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.capture_driving_stress_Pa) or self.capture_driving_stress_Pa < 0.0:
            raise ValueError("capture_driving_stress_Pa must be finite and nonnegative")


@dataclass(frozen=True)
class PolygonizationLedger:
    line_content_before_m2: float
    captured_m2: float
    annihilated_m2: float
    line_content_after_m2: float
    line_balance_residual_m2: float
    maximum_signed_burgers_residual_m2: float
    released_line_energy_J_m3: float


def polygonization_step(
    state: PolygonizationState,
    temperature_K: float,
    dt_s: float,
    parameters: PolygonizationParameters,
) -> tuple[PolygonizationState, PolygonizationLedger]:
    """Advance exact bounded transfer/annihilation/order fractions."""

    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    before_arrays = (
        state.mobile_plus_m2, state.mobile_minus_m2,
        state.wall_plus_m2, state.wall_minus_m2,
    )
    before_total = float(sum(np.sum(x) for x in before_arrays))
    before_signed = (
        state.mobile_plus_m2 - state.mobile_minus_m2
        + state.wall_plus_m2 - state.wall_minus_m2
    )
    capture_rate = parameters.capture.one_way_rate_s_inv(
        parameters.capture_driving_stress_Pa, temperature_K
    )
    capture_fraction = -math.expm1(-capture_rate * dt_s)
    capture_plus = capture_fraction * state.mobile_plus_m2
    capture_minus = capture_fraction * state.mobile_minus_m2
    mobile_plus = state.mobile_plus_m2 - capture_plus
    mobile_minus = state.mobile_minus_m2 - capture_minus
    wall_plus = state.wall_plus_m2 + capture_plus
    wall_minus = state.wall_minus_m2 + capture_minus

    climb_rate = parameters.climb_annihilation.one_way_rate_s_inv(
        0.0, temperature_K
    )
    climb_fraction = -math.expm1(-climb_rate * dt_s)
    pairs = np.minimum(mobile_plus, mobile_minus)
    removed_each_sign = climb_fraction * pairs
    mobile_plus -= removed_each_sign
    mobile_minus -= removed_each_sign
    annihilated = float(2.0 * np.sum(removed_each_sign))

    wall_excess = float(np.sum(np.abs(wall_plus - wall_minus)))
    order_rate = parameters.wall_ordering.one_way_rate_s_inv(0.0, temperature_K)
    maturity = state.wall_maturity
    if wall_excess > 0.0:
        maturity = 1.0 - (1.0 - maturity) * math.exp(-order_rate * dt_s)
    advanced = PolygonizationState(
        mobile_plus, mobile_minus, wall_plus, wall_minus, maturity
    )
    after_total = float(sum(np.sum(x) for x in (
        mobile_plus, mobile_minus, wall_plus, wall_minus
    )))
    after_signed = mobile_plus - mobile_minus + wall_plus - wall_minus
    captured = float(np.sum(capture_plus + capture_minus))
    return advanced, PolygonizationLedger(
        before_total, captured, annihilated, after_total,
        after_total - (before_total - annihilated),
        float(np.max(np.abs(after_signed - before_signed))),
        annihilated * parameters.line_energy_J_m,
    )


def frank_bilby_misorientation_rad(
    state: PolygonizationState,
    parameters: PolygonizationParameters,
) -> np.ndarray:
    """Simple-tilt family angles derived from wall excess content."""

    net_line_m_inv = (
        state.wall_plus_m2 - state.wall_minus_m2
    ) * parameters.wall_width_m
    argument = 0.5 * parameters.burgers_m * np.abs(net_line_m_inv)
    if np.any(argument > 1.0):
        raise ValueError("wall inventory is outside the simple Frank--Bilby range")
    return 2.0 * np.arcsin(argument)


def frank_bilby_residual(
    state: PolygonizationState,
    parameters: PolygonizationParameters,
) -> np.ndarray:
    theta = frank_bilby_misorientation_rad(state, parameters)
    net_line_m_inv = np.abs(
        state.wall_plus_m2 - state.wall_minus_m2
    ) * parameters.wall_width_m
    return 2.0 * np.sin(0.5 * theta) - parameters.burgers_m * net_line_m_inv
