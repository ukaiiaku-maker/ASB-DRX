"""Explicit bounded wall-order state for intragranular DRX qualification.

``q=0`` is a disordered tangle and ``q=1`` is an ordered wall.  Multiplying
the inherited v34 ordering branch by a smooth order interpolation makes the
old scalar ordering dip the fully ordered limit instead of silently treating
every density fluctuation as a wall.  The local functional contains no spatial
wavelength; its optional gradient coefficient only prices wall interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class WallOrderingParameters:
    ordering_amplitude_J_m3: float
    density_scale_m2: float
    center_ratio: float
    width_ratio: float
    order_barrier_J_m3: float
    wall_partition_coefficient_J_m: float
    target_wall_density_m2: float
    gradient_coefficient_J_m: float = 0.0

    def __post_init__(self):
        for name in (
            "ordering_amplitude_J_m3", "density_scale_m2", "width_ratio",
            "order_barrier_J_m3", "wall_partition_coefficient_J_m",
            "target_wall_density_m2",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.center_ratio < 0.0 or self.gradient_coefficient_J_m < 0.0:
            raise ValueError("center and gradient coefficients must be nonnegative")


def smooth_order(q):
    value = np.asarray(q, dtype=float)
    if np.any(~np.isfinite(value)) or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("wall order must lie in [0,1]")
    return value*value*(3.0-2.0*value)


def local_wall_energy_J_m3(total_density_m2, wall_density_m2, order, parameters):
    """Return ordering, bounded order barrier, and reservoir partition energy."""

    rho = np.asarray(total_density_m2, dtype=float)
    wall = np.asarray(wall_density_m2, dtype=float)
    q = np.asarray(order, dtype=float)
    rho, wall, q = np.broadcast_arrays(rho, wall, q)
    if np.any(~np.isfinite(rho)) or np.any(rho < 0.0):
        raise ValueError("total density must be finite and nonnegative")
    if np.any(~np.isfinite(wall)) or np.any(wall < 0.0) or np.any(wall > rho):
        raise ValueError("wall density must be finite and lie within total density")
    h = smooth_order(q)
    r = rho/parameters.density_scale_m2
    inherited_dip = -parameters.ordering_amplitude_J_m3*np.exp(
        -0.5*((r-parameters.center_ratio)/parameters.width_ratio)**2
    )
    barrier = parameters.order_barrier_J_m3*q*q*(1.0-q)*(1.0-q)
    target = h*parameters.target_wall_density_m2
    partition = (
        0.5*parameters.wall_partition_coefficient_J_m
        *(wall-target)**2/parameters.target_wall_density_m2
    )
    return h*inherited_dip + barrier + partition


def gradient_wall_energy_J_m3(order, spacing_m, parameters):
    """Periodic square-gradient energy density; no preferred wavenumber."""

    q = np.asarray(order, dtype=float)
    if q.ndim != 2 or spacing_m <= 0.0:
        raise ValueError("a 2-D order field and positive spacing are required")
    gx = (np.roll(q, -1, 0)-np.roll(q, 1, 0))/(2.0*spacing_m)
    gy = (np.roll(q, -1, 1)-np.roll(q, 1, 1))/(2.0*spacing_m)
    return 0.5*parameters.gradient_coefficient_J_m*(gx*gx+gy*gy)
