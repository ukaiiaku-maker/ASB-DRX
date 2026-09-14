"""Physical and numerical compatibility energies for the common DRX model.

The physical terms price unresolved line content with the dislocation line
tension.  The quadratic terms are constraint penalties: they are reported,
but must never enter a pathway decision or a heat source.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CompatibilityEnergy:
    long_range_gnd_J: float
    frank_bilby_gb_J: float
    orientation_J: float
    numerical_alpha_penalty_J: float
    numerical_gb_penalty_J: float
    alpha_residual_rms_m2: float
    gb_residual_rms_m2: float

    @property
    def physical_total_J(self):
        return self.long_range_gnd_J + self.frank_bilby_gb_J + self.orientation_J

    @property
    def numerical_total_J(self):
        return self.numerical_alpha_penalty_J + self.numerical_gb_penalty_J


def decompose_compatibility_energy(*, signed_gnd_density_m2,
                                   boundary_density_m2,
                                   orientation_gradient_m1,
                                   line_tension_J_m,
                                   cell_area_m2,
                                   represented_thickness_m,
                                   burgers_m,
                                   alpha_target_coefficient=1.0,
                                   gb_target_coefficient=1.0,
                                   alpha_penalty_coefficient=0.0,
                                   gb_penalty_coefficient=0.0,
                                   orientation_energy_density_J_m3=None):
    """Return a unit-checked physical/numerical compatibility decomposition.

    Densities and ``|grad psi|/b`` have units m^-2.  Multiplication by line
    tension [J/m] and represented volume [m^3] gives joules.  Quadratic
    penalty coefficients retain the production functional's declared units.
    """
    kappa = np.asarray(signed_gnd_density_m2, dtype=float)
    rho_gb = np.asarray(boundary_density_m2, dtype=float)
    grad = np.asarray(orientation_gradient_m1, dtype=float)
    tension = np.asarray(line_tension_J_m, dtype=float)
    if kappa.shape != rho_gb.shape or grad.shape != kappa.shape:
        raise ValueError("compatibility fields must share one grid")
    if tension.ndim and tension.shape != kappa.shape:
        raise ValueError("line tension must be scalar or grid matched")
    if any(not np.all(np.isfinite(x)) for x in (kappa, rho_gb, grad, tension)):
        raise ValueError("compatibility inputs must be finite")
    for value, name in ((cell_area_m2, "cell area"),
                        (represented_thickness_m, "represented thickness"),
                        (burgers_m, "Burgers magnitude")):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    target = grad / float(burgers_m)
    residual_alpha = np.abs(kappa) - float(alpha_target_coefficient)*target
    residual_gb = rho_gb - float(gb_target_coefficient)*target
    volume = float(cell_area_m2)*float(represented_thickness_m)
    long_range = float(np.sum(tension*np.abs(residual_alpha), dtype=np.longdouble)*volume)
    frank_bilby = float(np.sum(tension*np.abs(residual_gb), dtype=np.longdouble)*volume)
    if orientation_energy_density_J_m3 is None:
        orientation = 0.0
    else:
        orient = np.asarray(orientation_energy_density_J_m3, dtype=float)
        if orient.shape != kappa.shape or not np.all(np.isfinite(orient)):
            raise ValueError("orientation energy density must be finite and grid matched")
        orientation = float(np.sum(orient, dtype=np.longdouble)*volume)
    # Production penalty energies are per unit out-of-plane depth before the
    # represented thickness is applied.
    numerical_alpha = float(
        0.5*float(alpha_penalty_coefficient)
        * np.sum(residual_alpha**2, dtype=np.longdouble)
        * float(cell_area_m2)*float(represented_thickness_m))
    numerical_gb = float(
        0.5*float(gb_penalty_coefficient)
        * np.sum(residual_gb**2, dtype=np.longdouble)
        * float(cell_area_m2)*float(represented_thickness_m))
    return CompatibilityEnergy(
        long_range, frank_bilby, orientation, numerical_alpha, numerical_gb,
        float(np.sqrt(np.mean(residual_alpha**2))),
        float(np.sqrt(np.mean(residual_gb**2))))
