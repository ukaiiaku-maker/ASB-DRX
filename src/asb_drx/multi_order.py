"""Constrained multi-order-parameter verification kernel.

The fields form a pointwise simplex.  This isolated kernel verifies variational
signs and grain competition; it is not a calibrated nucleation model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MultiOrderParameters:
    pair_penalty_J_m3: float
    gradient_coefficient_J_m: float
    mobility_m3_J_s: float
    bulk_energy_J_m3: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in (
            "pair_penalty_J_m3",
            "gradient_coefficient_J_m",
            "mobility_m3_J_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if len(self.bulk_energy_J_m3) < 2:
            raise ValueError("at least two bulk energies are required")
        if not all(math.isfinite(value) for value in self.bulk_energy_J_m3):
            raise ValueError("bulk energies must be finite")


@dataclass(frozen=True)
class MultiOrderState:
    eta_fields: np.ndarray
    time_s: float
    accepted_steps: int

    def __post_init__(self) -> None:
        _check_simplex_fields(self.eta_fields)
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("time_s must be finite and nonnegative")
        if self.accepted_steps < 0:
            raise ValueError("accepted_steps must be nonnegative")


@dataclass(frozen=True)
class AcceptedMultiOrderStep:
    eta_fields: np.ndarray
    free_energy_J_m: float
    accepted_dt_s: float
    halvings: int


@dataclass(frozen=True)
class BinaryCircularLimit:
    boundary_energy_J_m2: float
    bulk_driving_J_m3: float
    kinetic_coefficient_m4_J_s: float

    def __post_init__(self) -> None:
        for name in (
            "boundary_energy_J_m2",
            "bulk_driving_J_m3",
            "kinetic_coefficient_m4_J_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def critical_radius_m(self) -> float:
        return self.boundary_energy_J_m2 / self.bulk_driving_J_m3

    def radial_velocity_m_s(self, radius_m: float) -> float:
        if not math.isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError("radius_m must be finite and positive")
        return self.kinetic_coefficient_m4_J_s * (
            self.bulk_driving_J_m3 - self.boundary_energy_J_m2 / radius_m
        )


def binary_boundary_energy_J_m2(parameters: MultiOrderParameters) -> float:
    """Planar two-field boundary energy for equal bulk energies."""

    return math.sqrt(
        parameters.gradient_coefficient_J_m * parameters.pair_penalty_J_m3
    ) / 3.0


def interpolation_h(fields: np.ndarray) -> np.ndarray:
    return fields**3 * (10.0 - 15.0 * fields + 6.0 * fields**2)


def interpolation_h_prime(fields: np.ndarray) -> np.ndarray:
    return 30.0 * fields**2 * (1.0 - fields) ** 2


def multi_order_free_energy_J_m(
    eta_fields: np.ndarray, dx_m: float, parameters: MultiOrderParameters
) -> float:
    fields = _check_simplex_fields(eta_fields)
    _check_inputs(fields, dx_m, parameters)
    squared = fields**2
    pair_sum = 0.5 * ((np.sum(squared, axis=0)) ** 2 - np.sum(squared**2, axis=0))
    density = parameters.pair_penalty_J_m3 * pair_sum
    density += np.tensordot(
        np.asarray(parameters.bulk_energy_J_m3), interpolation_h(fields), axes=(0, 0)
    )
    gradient_x = (np.roll(fields, -1, axis=2) - fields) / dx_m
    gradient_y = (np.roll(fields, -1, axis=1) - fields) / dx_m
    density += 0.5 * parameters.gradient_coefficient_J_m * np.sum(
        gradient_x**2 + gradient_y**2, axis=0
    )
    return float(dx_m**2 * np.sum(density))


def multi_order_chemical_potential_J_m3(
    eta_fields: np.ndarray, dx_m: float, parameters: MultiOrderParameters
) -> np.ndarray:
    """Unconstrained derivative; subtract its field mean for simplex flow."""

    fields = _check_simplex_fields(eta_fields)
    _check_inputs(fields, dx_m, parameters)
    squared_sum = np.sum(fields**2, axis=0, keepdims=True)
    local = 2.0 * parameters.pair_penalty_J_m3 * fields * (squared_sum - fields**2)
    local += (
        np.asarray(parameters.bulk_energy_J_m3)[:, None, None]
        * interpolation_h_prime(fields)
    )
    laplacian = (
        np.roll(fields, -1, axis=1)
        + np.roll(fields, 1, axis=1)
        + np.roll(fields, -1, axis=2)
        + np.roll(fields, 1, axis=2)
        - 4.0 * fields
    ) / dx_m**2
    return local - parameters.gradient_coefficient_J_m * laplacian


def energy_checked_multi_order_step(
    eta_fields: np.ndarray,
    dx_m: float,
    proposed_dt_s: float,
    parameters: MultiOrderParameters,
    *,
    maximum_halvings: int = 40,
    energy_tolerance_J_m: float = 0.0,
) -> AcceptedMultiOrderStep:
    fields = _check_simplex_fields(eta_fields)
    _check_inputs(fields, dx_m, parameters)
    if not math.isfinite(proposed_dt_s) or proposed_dt_s <= 0.0:
        raise ValueError("proposed_dt_s must be finite and positive")
    if maximum_halvings < 0:
        raise ValueError("maximum_halvings must be nonnegative")
    if not math.isfinite(energy_tolerance_J_m) or energy_tolerance_J_m < 0.0:
        raise ValueError("energy_tolerance_J_m must be finite and nonnegative")
    old_energy = multi_order_free_energy_J_m(fields, dx_m, parameters)
    chemical = multi_order_chemical_potential_J_m3(fields, dx_m, parameters)
    projected = chemical - np.mean(chemical, axis=0, keepdims=True)
    dt_s = proposed_dt_s
    for halvings in range(maximum_halvings + 1):
        candidate = fields - dt_s * parameters.mobility_m3_J_s * projected
        if np.all(np.isfinite(candidate)) and np.all(
            (candidate >= -1.0e-14) & (candidate <= 1.0 + 1.0e-14)
        ):
            candidate = np.clip(candidate, 0.0, 1.0)
            candidate /= np.sum(candidate, axis=0, keepdims=True)
            candidate_energy = multi_order_free_energy_J_m(candidate, dx_m, parameters)
            if candidate_energy <= old_energy + energy_tolerance_J_m:
                return AcceptedMultiOrderStep(candidate, candidate_energy, dt_s, halvings)
        dt_s *= 0.5
    raise RuntimeError("no energy-nonincreasing admissible multi-order step found")


def advance_multi_order(
    state: MultiOrderState,
    dx_m: float,
    proposed_dt_s: float,
    steps: int,
    parameters: MultiOrderParameters,
) -> MultiOrderState:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    fields = np.array(state.eta_fields, copy=True)
    time_s = state.time_s
    accepted_steps = state.accepted_steps
    for _ in range(steps):
        accepted = energy_checked_multi_order_step(
            fields, dx_m, proposed_dt_s, parameters
        )
        fields = accepted.eta_fields
        time_s += accepted.accepted_dt_s
        accepted_steps += 1
    return MultiOrderState(fields, time_s, accepted_steps)


def diffuse_binary_circle(
    grid_points: int, dx_m: float, radius_m: float, interface_length_m: float
) -> np.ndarray:
    """Two simplex fields with the child (index one) inside a periodic circle."""

    if grid_points < 8:
        raise ValueError("grid_points must be at least eight")
    for name, value in (
        ("dx_m", dx_m),
        ("radius_m", radius_m),
        ("interface_length_m", interface_length_m),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    coordinate = (np.arange(grid_points, dtype=float) + 0.5 - grid_points / 2.0) * dx_m
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    child = 0.5 * (
        1.0 - np.tanh((np.sqrt(x**2 + y**2) - radius_m) / interface_length_m)
    )
    return np.stack((1.0 - child, child))


def equivalent_child_radius_m(eta_fields: np.ndarray, dx_m: float) -> float:
    fields = _check_simplex_fields(eta_fields)
    if fields.shape[0] != 2:
        raise ValueError("equivalent child radius requires exactly two fields")
    if not math.isfinite(dx_m) or dx_m <= 0.0:
        raise ValueError("dx_m must be finite and positive")
    return math.sqrt(float(dx_m**2 * np.sum(fields[1])) / math.pi)


def save_multi_order_checkpoint(path: Path, state: MultiOrderState) -> None:
    np.savez(
        Path(path),
        eta_fields=state.eta_fields,
        time_s=np.asarray(state.time_s),
        accepted_steps=np.asarray(state.accepted_steps, dtype=np.int64),
    )


def load_multi_order_checkpoint(path: Path) -> MultiOrderState:
    with np.load(Path(path), allow_pickle=False) as payload:
        return MultiOrderState(
            np.array(payload["eta_fields"], copy=True),
            float(payload["time_s"]),
            int(payload["accepted_steps"]),
        )


def _check_simplex_fields(eta_fields: np.ndarray) -> np.ndarray:
    fields = np.asarray(eta_fields, dtype=float)
    if fields.ndim != 3 or fields.shape[0] < 2 or min(fields.shape[1:]) < 1:
        raise ValueError("eta_fields must have shape (at least two labels, rows, columns)")
    if not np.all(np.isfinite(fields)) or np.any(fields < 0.0) or np.any(fields > 1.0):
        raise ValueError("eta_fields must be finite and in [0, 1]")
    if not np.allclose(np.sum(fields, axis=0), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("eta_fields must sum pointwise to one")
    return fields


def _check_inputs(
    fields: np.ndarray, dx_m: float, parameters: MultiOrderParameters
) -> None:
    if not math.isfinite(dx_m) or dx_m <= 0.0:
        raise ValueError("dx_m must be finite and positive")
    if fields.shape[0] != len(parameters.bulk_energy_J_m3):
        raise ValueError("one bulk energy is required per field")
