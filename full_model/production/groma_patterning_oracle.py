"""Reduced conservative Groma--Zaiser--Ispanovity patterning oracle.

This is a dimensionless, one-dimensional, single-slip mechanism oracle for
Eqs. (38) of Groma, Zaiser, and Ispanovity, Phys. Rev. B 93, 214110
(2016). Length is measured in the reference dislocation spacing and density
in the reference total density. It is deliberately not a production BCC
closure: the correlation coefficients A and D require independent DDD or
experimental constraint before promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GromaPatterningParameters:
    applied_stress: float = 1.25
    friction_coefficient: float = 1.0
    density_diffusion: float = 1.0
    gnd_backstress: float = 1.0
    minimum_density: float = 1.0e-8
    mobility_law: str = "linear"
    arrhenius_inverse_temperature: float = 10.0
    arrhenius_floor: float = 0.05
    arrhenius_a: float = 2.2
    arrhenius_n: float = 2.5
    arrhenius_critical_stress: float = 1.0
    arrhenius_speed_scale: float = 1.0

    def __post_init__(self):
        values = (self.applied_stress, self.friction_coefficient,
                  self.density_diffusion, self.gnd_backstress,
                  self.minimum_density, self.arrhenius_inverse_temperature,
                  self.arrhenius_a, self.arrhenius_n,
                  self.arrhenius_critical_stress,
                  self.arrhenius_speed_scale)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("patterning parameters must be positive finite")
        if self.mobility_law not in ("linear", "exp_floor"):
            raise ValueError("unknown patterning mobility law")
        if not 0.0 <= self.arrhenius_floor <= 1.0:
            raise ValueError("Arrhenius floor must lie in [0,1]")


def periodic_gradient(field, spacing):
    value = np.asarray(field, dtype=float)
    return (np.roll(value, -1)-np.roll(value, 1))/(2.0*float(spacing))


def glide_speed(stress, parameters):
    stress = np.asarray(stress, dtype=float)
    if parameters.mobility_law == "linear":
        return stress
    normalized = np.abs(stress)/parameters.arrhenius_critical_stress
    barrier_fraction = (
        parameters.arrhenius_floor
        +(1.0-parameters.arrhenius_floor)*np.exp(
            -parameters.arrhenius_a*normalized**parameters.arrhenius_n))
    activation = np.exp(
        -parameters.arrhenius_inverse_temperature*barrier_fraction)
    return (parameters.arrhenius_speed_scale*activation
            *np.tanh(stress/parameters.arrhenius_critical_stress))


def conservative_rhs(total_density, gnd_density, spacing, parameters):
    """Return conservative total/GND rates and right-face fluxes.

    The total density ``rho`` and signed density ``kappa`` obey ``|kappa|<=rho``.
    The friction stress is ``C sqrt(rho)``. The two correlation stresses are
    ``-D/rho d(kappa)/dx`` and ``-A/rho d(rho)/dx``; omitting the latter removes
    finite-wavelength selection in the source theory.
    """
    rho = np.asarray(total_density, dtype=float)
    kappa = np.asarray(gnd_density, dtype=float)
    dx = float(spacing)
    if (rho.ndim != 1 or kappa.shape != rho.shape or rho.size < 8
            or np.any(~np.isfinite(rho)) or np.any(~np.isfinite(kappa))
            or np.any(rho <= parameters.minimum_density)
            or np.any(np.abs(kappa) > rho)):
        raise ValueError("inadmissible Groma density state")
    # Center every constitutive factor on the same finite-volume face. This
    # avoids the odd/even null mode created by applying a centered derivative
    # twice while retaining exact telescoping content balances.
    rho_right = np.roll(rho, -1)
    kappa_right = np.roll(kappa, -1)
    rho_face = .5*(rho+rho_right)
    kappa_face = .5*(kappa+kappa_right)
    rho_x_face = (rho_right-rho)/dx
    kappa_x_face = (kappa_right-kappa)/dx
    polarization = kappa_face/rho_face
    friction = parameters.friction_coefficient*np.sqrt(rho_face)
    backstress = (-parameters.gnd_backstress
                  *kappa_x_face/rho_face)
    diffusion_stress = (-parameters.density_diffusion
                        *rho_x_face/rho_face)
    stress_plus = (parameters.applied_stress+backstress
                   -(1.0-polarization)*friction+diffusion_stress)
    stress_minus = (parameters.applied_stress+backstress
                    -(1.0+polarization)*friction-diffusion_stress)
    rho_plus = .5*(rho_face+kappa_face)
    rho_minus = .5*(rho_face-kappa_face)
    plus_flux = rho_plus*glide_speed(stress_plus, parameters)
    minus_flux = -rho_minus*glide_speed(stress_minus, parameters)
    total_flux = plus_flux+minus_flux
    gnd_flux = plus_flux-minus_flux
    return (-(total_flux-np.roll(total_flux, 1))/dx,
            -(gnd_flux-np.roll(gnd_flux, 1))/dx, total_flux, gnd_flux)


def rk4_step(total_density, gnd_density, dt, spacing, parameters):
    rho = np.asarray(total_density, dtype=float)
    kappa = np.asarray(gnd_density, dtype=float)

    def rates(a, b):
        return conservative_rhs(a, b, spacing, parameters)[:2]

    r1, k1 = rates(rho, kappa)
    r2, k2 = rates(rho+.5*dt*r1, kappa+.5*dt*k1)
    r3, k3 = rates(rho+.5*dt*r2, kappa+.5*dt*k2)
    r4, k4 = rates(rho+dt*r3, kappa+dt*k3)
    updated_rho = rho+(dt/6.0)*(r1+2*r2+2*r3+r4)
    updated_kappa = kappa+(dt/6.0)*(k1+2*k2+2*k3+k4)
    # No projection is permitted: loss of positivity or |kappa|<=rho is a
    # failed trajectory, not a state that may be silently repaired.
    conservative_rhs(updated_rho, updated_kappa, spacing, parameters)
    return updated_rho, updated_kappa


def linear_dispersion(wavenumber, parameters):
    """Positive/negative branches about rho=1, kappa=0 for axial modes."""
    k = np.asarray(wavenumber, dtype=float)
    flow = parameters.friction_coefficient
    excess = parameters.applied_stress-flow
    speed = float(glide_speed(excess, parameters))
    increment = 1e-6*parameters.arrhenius_critical_stress
    tangent = float((glide_speed(excess+increment, parameters)
                     -glide_speed(excess-increment, parameters))/(2.0*increment))
    if tangent <= 0.0:
        raise ValueError("patterning mobility tangent must be positive")
    a = parameters.density_diffusion*tangent
    d = parameters.gnd_backstress*tangent
    beta = (speed+flow*tangent)*(speed-.5*flow*tangent)
    trace = (a+d)*k*k
    determinant_term = beta+a*d*k*k
    discriminant = np.maximum(trace*trace-4.0*k*k*determinant_term, 0.0)
    return -.5*trace+.5*np.sqrt(discriminant), \
        -.5*trace-.5*np.sqrt(discriminant)


def fastest_linear_mode(parameters):
    """Return the axial fastest mode for A=D, used by the V66 oracle."""
    if not math.isclose(parameters.density_diffusion,
                        parameters.gnd_backstress, rel_tol=1e-12):
        raise ValueError("closed fastest-mode expression requires A=D")
    flow = parameters.friction_coefficient
    excess = parameters.applied_stress-flow
    speed = float(glide_speed(excess, parameters))
    increment = 1e-6*parameters.arrhenius_critical_stress
    tangent = float((glide_speed(excess+increment, parameters)
                     -glide_speed(excess-increment, parameters))/(2.0*increment))
    if tangent <= 0.0:
        raise ValueError("patterning mobility tangent must be positive")
    beta = (speed+flow*tangent)*(speed-.5*flow*tangent)
    if beta >= 0.0:
        return {"unstable": False, "beta": beta,
                "wavenumber": 0.0, "wavelength": math.inf,
                "growth_rate": 0.0, "homogeneous_speed": speed,
                "mobility_tangent": tangent}
    diffusivity = parameters.density_diffusion*tangent
    wavenumber = math.sqrt(-beta)/(2.0*diffusivity)
    return {"unstable": True, "beta": beta,
            "wavenumber": wavenumber,
            "wavelength": 2.0*math.pi/wavenumber,
            "growth_rate": -beta/(4.0*diffusivity),
            "homogeneous_speed": speed, "mobility_tangent": tangent}
