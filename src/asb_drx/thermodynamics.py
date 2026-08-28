"""Material-agnostic thermodynamic verification kernels.

These routines establish dimensions, signs, conservation, and limiting behavior.
They are not a calibrated production phase-field solver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class GrainEnergyParameters:
    """Two-state grain free-energy parameters in SI units.

    eta=0 denotes parent material and eta=1 denotes recrystallized material.
    The bulk driving energy is positive when the recrystallized state is lower.
    """

    well_height_J_m3: float
    gradient_coefficient_J_m: float
    bulk_driving_J_m3: float
    mobility_m3_J_s: float

    def __post_init__(self) -> None:
        for name in (
            "well_height_J_m3",
            "gradient_coefficient_J_m",
            "mobility_m3_J_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.bulk_driving_J_m3) or self.bulk_driving_J_m3 < 0.0:
            raise ValueError("bulk_driving_J_m3 must be finite and nonnegative")


@dataclass(frozen=True)
class AcceptedStep:
    eta: np.ndarray
    free_energy_J_m2: float
    accepted_dt_s: float
    halvings: int


def interpolation_h(eta: np.ndarray) -> np.ndarray:
    """Quintic interpolation with zero slope/curvature at both pure states."""

    return eta**3 * (10.0 - 15.0 * eta + 6.0 * eta**2)


def interpolation_h_prime(eta: np.ndarray) -> np.ndarray:
    return 30.0 * eta**2 * (1.0 - eta) ** 2


def local_free_energy_J_m3(eta: np.ndarray, parameters: GrainEnergyParameters) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    return (
        parameters.well_height_J_m3 * eta**2 * (1.0 - eta) ** 2
        - parameters.bulk_driving_J_m3 * interpolation_h(eta)
    )


def free_energy_1d_J_m2(
    eta: np.ndarray, dx_m: float, parameters: GrainEnergyParameters
) -> float:
    """Periodic 1-D free energy per unit transverse area."""

    eta = _check_eta_field(eta)
    _check_dx(dx_m)
    forward_gradient = (np.roll(eta, -1) - eta) / dx_m
    density = local_free_energy_J_m3(eta, parameters)
    density += 0.5 * parameters.gradient_coefficient_J_m * forward_gradient**2
    return float(dx_m * np.sum(density))


def chemical_potential_J_m3(
    eta: np.ndarray, dx_m: float, parameters: GrainEnergyParameters
) -> np.ndarray:
    """Discrete variational derivative of the periodic 1-D free energy."""

    eta = _check_eta_field(eta)
    _check_dx(dx_m)
    local_derivative = (
        2.0
        * parameters.well_height_J_m3
        * eta
        * (1.0 - eta)
        * (1.0 - 2.0 * eta)
        - parameters.bulk_driving_J_m3 * interpolation_h_prime(eta)
    )
    laplacian = (np.roll(eta, -1) - 2.0 * eta + np.roll(eta, 1)) / dx_m**2
    return local_derivative - parameters.gradient_coefficient_J_m * laplacian


def energy_checked_allen_cahn_step(
    eta: np.ndarray,
    dx_m: float,
    proposed_dt_s: float,
    parameters: GrainEnergyParameters,
    *,
    maximum_halvings: int = 40,
    energy_tolerance_J_m2: float = 0.0,
) -> AcceptedStep:
    """Explicit Allen--Cahn step accepted only when discrete energy does not rise."""

    eta = _check_eta_field(eta)
    _check_dx(dx_m)
    if not math.isfinite(proposed_dt_s) or proposed_dt_s <= 0.0:
        raise ValueError("proposed_dt_s must be finite and positive")
    if maximum_halvings < 0:
        raise ValueError("maximum_halvings must be nonnegative")
    if not math.isfinite(energy_tolerance_J_m2) or energy_tolerance_J_m2 < 0.0:
        raise ValueError("energy_tolerance_J_m2 must be finite and nonnegative")

    old_energy = free_energy_1d_J_m2(eta, dx_m, parameters)
    chemical_potential = chemical_potential_J_m3(eta, dx_m, parameters)
    dt_s = proposed_dt_s
    for halvings in range(maximum_halvings + 1):
        candidate = eta - dt_s * parameters.mobility_m3_J_s * chemical_potential
        if np.all(np.isfinite(candidate)) and np.all((candidate >= 0.0) & (candidate <= 1.0)):
            candidate_energy = free_energy_1d_J_m2(candidate, dx_m, parameters)
            if candidate_energy <= old_energy + energy_tolerance_J_m2:
                return AcceptedStep(candidate, candidate_energy, dt_s, halvings)
        dt_s *= 0.5
    raise RuntimeError("no energy-nonincreasing admissible Allen--Cahn step found")


@dataclass(frozen=True)
class DislocationReservoirs:
    """Nonnegative line-length densities [m^-2] kept as distinct reservoirs."""

    mobile_m2: float
    forest_m2: float
    wall_m2: float
    gb_m2: float

    def __post_init__(self) -> None:
        for name in ("mobile_m2", "forest_m2", "wall_m2", "gb_m2"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def total_m2(self) -> float:
        return self.mobile_m2 + self.forest_m2 + self.wall_m2 + self.gb_m2

    def conservative_transfer(
        self,
        source: Literal["mobile_m2", "forest_m2", "wall_m2", "gb_m2"],
        destination: Literal["mobile_m2", "forest_m2", "wall_m2", "gb_m2"],
        amount_m2: float,
    ) -> "DislocationReservoirs":
        if source == destination:
            raise ValueError("source and destination must differ")
        if not math.isfinite(amount_m2) or amount_m2 < 0.0:
            raise ValueError("amount_m2 must be finite and nonnegative")
        if amount_m2 > getattr(self, source):
            raise ValueError("transfer exceeds source reservoir")
        return replace(
            self,
            **{
                source: getattr(self, source) - amount_m2,
                destination: getattr(self, destination) + amount_m2,
            },
        )


@dataclass(frozen=True)
class WorkLedger:
    """Incremental energy partition per unit volume [J m^-3]."""

    mechanical_work_J_m3: float
    stored_dislocation_J_m3: float
    interface_J_m3: float
    residual_gb_J_m3: float
    accommodation_J_m3: float
    heat_J_m3: float
    closure_error_J_m3: float


def close_work_ledger(
    mechanical_work_J_m3: float,
    *,
    stored_dislocation_J_m3: float = 0.0,
    interface_J_m3: float = 0.0,
    residual_gb_J_m3: float = 0.0,
    accommodation_J_m3: float = 0.0,
    tolerance_J_m3: float = 1.0e-10,
) -> WorkLedger:
    """Assign heat as the residual and reject energetically impossible partitions."""

    entries = {
        "mechanical_work_J_m3": mechanical_work_J_m3,
        "stored_dislocation_J_m3": stored_dislocation_J_m3,
        "interface_J_m3": interface_J_m3,
        "residual_gb_J_m3": residual_gb_J_m3,
        "accommodation_J_m3": accommodation_J_m3,
        "tolerance_J_m3": tolerance_J_m3,
    }
    for name, value in entries.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    allocated = (
        stored_dislocation_J_m3
        + interface_J_m3
        + residual_gb_J_m3
        + accommodation_J_m3
    )
    heat = mechanical_work_J_m3 - allocated
    if heat < -tolerance_J_m3:
        raise ValueError("declared stored/dissipated channels exceed mechanical work")
    heat = max(heat, 0.0)
    closure = mechanical_work_J_m3 - allocated - heat
    return WorkLedger(
        mechanical_work_J_m3,
        stored_dislocation_J_m3,
        interface_J_m3,
        residual_gb_J_m3,
        accommodation_J_m3,
        heat,
        closure,
    )


@dataclass(frozen=True)
class CircularNucleusLimit:
    """Sharp-interface circular-nucleus limit for a 2-D nucleus per unit depth."""

    gb_energy_J_m2: float
    bulk_driving_J_m3: float
    gb_mobility_m4_J_s: float

    def __post_init__(self) -> None:
        for name in ("gb_energy_J_m2", "bulk_driving_J_m3", "gb_mobility_m4_J_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def critical_radius_m(self) -> float:
        return self.gb_energy_J_m2 / self.bulk_driving_J_m3

    def excess_energy_J_m(self, radius_m: float) -> float:
        self._check_radius(radius_m)
        return (
            2.0 * math.pi * radius_m * self.gb_energy_J_m2
            - math.pi * radius_m**2 * self.bulk_driving_J_m3
        )

    def radius_rate_m_s(self, radius_m: float) -> float:
        self._check_radius(radius_m)
        return self.gb_mobility_m4_J_s * (
            self.bulk_driving_J_m3 - self.gb_energy_J_m2 / radius_m
        )

    @staticmethod
    def _check_radius(radius_m: float) -> None:
        if not math.isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError("radius_m must be finite and positive")


def _check_eta_field(eta: np.ndarray) -> np.ndarray:
    array = np.asarray(eta, dtype=float)
    if array.ndim != 1 or array.size < 3:
        raise ValueError("eta must be a one-dimensional field with at least three points")
    if not np.all(np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("eta must be finite and satisfy 0 <= eta <= 1")
    return array


def _check_dx(dx_m: float) -> None:
    if not math.isfinite(dx_m) or dx_m <= 0.0:
        raise ValueError("dx_m must be finite and positive")
