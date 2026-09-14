"""Positive-state dislocation free energy used by every material phase.

This module isolates the local v34 stored-energy branches from the monolithic
driver so that their values and derivatives can be verified independently.
Density has units m^-2, line coefficients J m^-1, and every returned energy
density has units J m^-3.

The positive ``rho log rho`` coefficient is treated as the existing
coarse-grained configurational/correlation contribution.  It is *not* also
claimed as the elastic outer-cutoff logarithm, whose logarithmic coefficient
would have the opposite sign.  The constant part of the line self energy is
represented once, by ``line_coefficient_J_m * rho``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DislocationFreeEnergyParameters:
    line_coefficient_J_m: float
    log_coefficient_J_m: float
    reference_density_m2: float
    ordering_amplitude_J_m3: float
    ordering_density_scale_m2: float
    ordering_center_ratio: float
    ordering_width_ratio: float
    low_density_strength: float = 0.0
    low_density_width_ratio: float = 0.03

    def __post_init__(self) -> None:
        positive = (
            "line_coefficient_J_m", "log_coefficient_J_m",
            "reference_density_m2", "ordering_density_scale_m2",
            "ordering_width_ratio", "low_density_width_ratio",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("ordering_amplitude_J_m3", "ordering_center_ratio",
                     "low_density_strength"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


def _density(rho_m2):
    rho = np.asarray(rho_m2, dtype=float)
    if np.any(~np.isfinite(rho)) or np.any(rho < 0.0):
        raise ValueError("density must be finite and nonnegative")
    return rho


def logarithmic_energy_J_m3(rho_m2, coefficient_J_m, reference_density_m2):
    """Return ``C rho [log(rho/rho_ref)-1]`` with its exact zero limit."""

    rho = _density(rho_m2)
    if coefficient_J_m <= 0.0 or reference_density_m2 <= 0.0:
        raise ValueError("logarithmic coefficient and reference must be positive")
    out = np.zeros_like(rho)
    positive = rho > 0.0
    out[positive] = coefficient_J_m * rho[positive] * (
        np.log(rho[positive] / reference_density_m2) - 1.0
    )
    return out


def logarithmic_chemical_potential_J_m(
    rho_m2, coefficient_J_m, reference_density_m2, *, evaluation_floor_m2=None
):
    """Return ``d Phi_log/d rho``; a floor regularizes evaluation only."""

    rho = _density(rho_m2)
    if evaluation_floor_m2 is None:
        if np.any(rho == 0.0):
            raise ValueError("chemical potential is singular at zero density")
        evaluated = rho
    else:
        floor = float(evaluation_floor_m2)
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError("evaluation floor must be finite and positive")
        evaluated = np.maximum(rho, floor)
    return coefficient_J_m * np.log(evaluated / reference_density_m2)


def logarithmic_hessian_J_m3_per_m4(rho_m2, coefficient_J_m):
    """Return the positive curvature ``C/rho`` for strictly positive density."""

    rho = _density(rho_m2)
    if coefficient_J_m <= 0.0 or np.any(rho == 0.0):
        raise ValueError("positive coefficient and strictly positive density required")
    return coefficient_J_m / rho


def free_energy_components_J_m3(rho_m2, parameters):
    """Return the four local v34 branches without hiding reference shifts."""

    rho = _density(rho_m2)
    p = parameters
    line = p.line_coefficient_J_m * rho
    logarithmic = logarithmic_energy_J_m3(
        rho, p.log_coefficient_J_m, p.reference_density_m2
    )
    r = rho / p.ordering_density_scale_m2
    z = (r - p.ordering_center_ratio) / p.ordering_width_ratio
    ordering = -p.ordering_amplitude_J_m3 * np.exp(-0.5 * z*z)
    low_density = (
        p.line_coefficient_J_m * p.ordering_density_scale_m2
        * p.low_density_strength * p.low_density_width_ratio
        * np.exp(-r / p.low_density_width_ratio)
    )
    return {
        "line": line, "logarithmic": logarithmic,
        "ordering": ordering, "low_density": low_density,
    }


def free_energy_J_m3(rho_m2, parameters):
    parts = free_energy_components_J_m3(rho_m2, parameters)
    return sum(parts.values())


def phase_owned_free_energy_J_m3(rho_m2, wall_density_m2, parameters):
    """Weight ordering by the explicit organized-wall reservoir fraction.

    This is the common phase-owned scalar functional.  ``rho_wall/rho`` is the
    existing full-v34 equivalent of an explicit wall order variable: a dense
    but disordered mobile/forest tangle receives no Gaussian ordering credit.
    """

    rho = _density(rho_m2)
    wall = np.asarray(wall_density_m2, dtype=float)
    rho, wall = np.broadcast_arrays(rho, wall)
    if np.any(~np.isfinite(wall)) or np.any(wall < 0.0) or np.any(wall > rho):
        raise ValueError("wall density must be finite and lie within total density")
    parts = free_energy_components_J_m3(rho, parameters)
    q = np.divide(wall, rho, out=np.zeros_like(rho), where=rho > 0.0)
    h = q*q*(3.0-2.0*q)
    parts["ordering"] = h*parts["ordering"]
    return sum(parts.values())


def chemical_potential_J_m(rho_m2, parameters, *, evaluation_floor_m2=None):
    rho = _density(rho_m2)
    p = parameters
    log_mu = logarithmic_chemical_potential_J_m(
        rho, p.log_coefficient_J_m, p.reference_density_m2,
        evaluation_floor_m2=evaluation_floor_m2,
    )
    r = rho / p.ordering_density_scale_m2
    z = (r - p.ordering_center_ratio) / p.ordering_width_ratio
    gaussian = np.exp(-0.5*z*z)
    ordering_mu = (
        p.ordering_amplitude_J_m3 * gaussian * z
        / (p.ordering_width_ratio * p.ordering_density_scale_m2)
    )
    low_mu = -p.line_coefficient_J_m * p.low_density_strength * np.exp(
        -r / p.low_density_width_ratio
    )
    return p.line_coefficient_J_m + log_mu + ordering_mu + low_mu


def hessian_J_m3_per_m4(rho_m2, parameters):
    rho = _density(rho_m2)
    p = parameters
    log_hessian = logarithmic_hessian_J_m3_per_m4(rho, p.log_coefficient_J_m)
    r = rho / p.ordering_density_scale_m2
    z = (r - p.ordering_center_ratio) / p.ordering_width_ratio
    gaussian = np.exp(-0.5*z*z)
    ordering_hessian = (
        p.ordering_amplitude_J_m3 * gaussian * (1.0-z*z)
        / (p.ordering_width_ratio*p.ordering_density_scale_m2)**2
    )
    low_hessian = (
        p.line_coefficient_J_m * p.low_density_strength
        * np.exp(-r/p.low_density_width_ratio)
        / (p.ordering_density_scale_m2*p.low_density_width_ratio)
    )
    return log_hessian + ordering_hessian + low_hessian
