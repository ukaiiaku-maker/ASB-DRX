"""Conservative staggered correlation-flux candidates for signed CDD.

The routines return right-face fluxes: entry ``i`` crosses from cell ``i`` to
``i+1``.  Correlation stresses are formed directly at that face, avoiding the
Nyquist null created by differentiating at cell centers and subsequently
averaging velocities.  No wavelength or spectral filter enters either form.
"""

from __future__ import annotations

import math

import numpy as np


def logarithmic_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Positive logarithmic mean with a stable equal-state limit."""

    if np.any(left <= 0.0) or np.any(right <= 0.0):
        raise ValueError("logarithmic mean requires positive states")
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)
    close = np.abs(right - left) <= 1.0e-10 * scale
    result = np.empty_like(left, dtype=float)
    result[close] = 0.5 * (left[close] + right[close])
    result[~close] = (right[~close] - left[~close]) / (
        np.log(right[~close]) - np.log(left[~close])
    )
    return result


def staggered_correlation_fluxes(
    mobile_plus_m2: np.ndarray,
    mobile_minus_m2: np.ndarray,
    mobility_m_Pa_s: np.ndarray,
    shear_modulus_Pa: np.ndarray,
    burgers_m: float,
    dx_m: float,
    *,
    backstress_coefficient: float,
    diffusion_coefficient: float,
    face_density: str = "arithmetic",
    density_floor_m2: float = 1.0e8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative correlation-only positive/negative face fluxes.

    ``arithmetic`` is the direct staggered stress/density discretization.
    ``logarithmic`` is a discrete chemical-potential form: logarithmic face
    densities multiply gradients of log density and retain the exact identity
    ``L(a,b) * (log(b)-log(a)) = b-a``.
    """

    plus = np.asarray(mobile_plus_m2, dtype=float)
    minus = np.asarray(mobile_minus_m2, dtype=float)
    mobility = np.asarray(mobility_m_Pa_s, dtype=float)
    mu = np.asarray(shear_modulus_Pa, dtype=float)
    if plus.ndim != 2 or plus.shape != minus.shape or mobility.shape != plus.shape:
        raise ValueError("signed densities and mobility must share shape (families, cells)")
    if mu.shape not in ((plus.shape[1],), plus.shape):
        raise ValueError("shear modulus must be cellwise or family/cellwise")
    if np.any(plus < 0.0) or np.any(minus < 0.0):
        raise ValueError("signed populations must be nonnegative")
    for name, value in (
        ("burgers_m", burgers_m), ("dx_m", dx_m),
        ("density_floor_m2", density_floor_m2),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if backstress_coefficient < 0.0 or diffusion_coefficient < 0.0:
        raise ValueError("correlation coefficients must be nonnegative")
    right_plus = np.roll(plus, -1, axis=1)
    right_minus = np.roll(minus, -1, axis=1)
    total = plus + minus
    right_total = right_plus + right_minus
    kappa = plus - minus
    right_kappa = right_plus - right_minus
    forest = np.sum(total, axis=0)
    forest_right = np.roll(forest, -1)
    forest_face = np.maximum(0.5 * (forest + forest_right), density_floor_m2)
    total_face = np.maximum(0.5 * (total + right_total), density_floor_m2)
    mobility_face = 0.5 * (mobility + np.roll(mobility, -1, axis=1))
    mu_family = np.broadcast_to(mu, plus.shape)
    mu_face = 0.5 * (mu_family + np.roll(mu_family, -1, axis=1))
    back = (
        -backstress_coefficient * mu_face * burgers_m
        * (right_kappa - kappa) / dx_m / forest_face[None, :]
    )
    diffusion = (
        -diffusion_coefficient * mu_face * burgers_m
        * (right_total - total) / dx_m / total_face
    )
    if face_density == "arithmetic":
        plus_face = 0.5 * (plus + right_plus)
        minus_face = 0.5 * (minus + right_minus)
    elif face_density == "logarithmic":
        plus_face = logarithmic_mean(
            np.maximum(plus, density_floor_m2),
            np.maximum(right_plus, density_floor_m2),
        )
        minus_face = logarithmic_mean(
            np.maximum(minus, density_floor_m2),
            np.maximum(right_minus, density_floor_m2),
        )
    else:
        raise ValueError("face_density must be 'arithmetic' or 'logarithmic'")
    plus_flux = plus_face * mobility_face * (back + diffusion)
    minus_flux = minus_face * mobility_face * (-back + diffusion)
    return plus_flux, minus_flux


def periodic_flux_divergence(face_flux: np.ndarray, dx_m: float) -> np.ndarray:
    """Return ``-div(flux)`` with exact telescoping periodic balance."""

    return -(face_flux - np.roll(face_flux, 1, axis=1)) / dx_m


def correlation_rhs(
    mobile_plus_m2: np.ndarray,
    mobile_minus_m2: np.ndarray,
    mobility_m_Pa_s: np.ndarray,
    shear_modulus_Pa: np.ndarray,
    burgers_m: float,
    dx_m: float,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    plus_flux, minus_flux = staggered_correlation_fluxes(
        mobile_plus_m2, mobile_minus_m2, mobility_m_Pa_s,
        shear_modulus_Pa, burgers_m, dx_m, **kwargs,
    )
    return (
        periodic_flux_divergence(plus_flux, dx_m),
        periodic_flux_divergence(minus_flux, dx_m),
    )


def discrete_mode_growth_rates_s_inv(
    grid_points: int,
    domain_m: float,
    density_m2: float,
    mobility_m_Pa_s: float,
    shear_modulus_Pa: float,
    burgers_m: float,
    *,
    backstress_coefficient: float,
    diffusion_coefficient: float,
    face_density: str,
    relative_amplitude: float = 1.0e-7,
) -> dict[str, list[float]]:
    """Finite-difference total and signed eigenmodes through Nyquist."""

    if grid_points < 4 or grid_points % 2:
        raise ValueError("an even grid with at least four points is required")
    x = np.arange(grid_points)
    base_plus = np.full((1, grid_points), 0.5 * density_m2)
    base_minus = base_plus.copy()
    mobility = np.full_like(base_plus, mobility_m_Pa_s)
    mu = np.full(grid_points, shear_modulus_Pa)
    dx = domain_m / grid_points
    modes = []
    total_growth = []
    signed_growth = []
    for mode in range(1, grid_points // 2 + 1):
        wave = np.cos(2.0 * np.pi * mode * x / grid_points)
        amplitude = relative_amplitude * density_m2
        # Equal perturbations isolate total density; opposite perturbations
        # isolate signed density.  Project RHS back onto the imposed mode.
        projections = []
        for sign in (1.0, -1.0):
            plus = base_plus + 0.5 * amplitude * wave[None, :]
            minus = base_minus + sign * 0.5 * amplitude * wave[None, :]
            rhs_plus, rhs_minus = correlation_rhs(
                plus, minus, mobility, mu, burgers_m, dx,
                backstress_coefficient=backstress_coefficient,
                diffusion_coefficient=diffusion_coefficient,
                face_density=face_density,
            )
            response = rhs_plus + sign * rhs_minus
            projections.append(
                float(np.dot(response[0], wave) / np.dot(wave, wave) / amplitude)
            )
        modes.append(mode)
        total_growth.append(projections[0])
        signed_growth.append(projections[1])
    return {"mode": modes, "total_s_inv": total_growth, "signed_s_inv": signed_growth}


FLUX_PARAMETER_CLASSIFICATION = {
    "face_density": "numerical_regularization",
    "backstress_coefficient": "generic_development_parameter",
    "diffusion_coefficient": "generic_development_parameter",
}


def variational_correlation_energy_J_m3(
    mobile_plus_m2: np.ndarray,
    mobile_minus_m2: np.ndarray,
    shear_modulus_Pa: np.ndarray,
    burgers_m: float,
    *,
    backstress_coefficient: float,
    diffusion_coefficient: float,
    reference_density_m2: float,
    density_floor_m2: float = 1.0e8,
) -> np.ndarray:
    """Cellwise integrable correlation energy for the v3 candidate.

    ``rho_a`` and the forest include a fixed positive numerical background in
    this isolated candidate.  Its contribution is reported through the floor
    parameter and must converge away before production use.
    """

    plus = np.asarray(mobile_plus_m2, dtype=float)
    minus = np.asarray(mobile_minus_m2, dtype=float)
    mu = np.asarray(shear_modulus_Pa, dtype=float)
    if plus.ndim != 2 or minus.shape != plus.shape or mu.shape != (plus.shape[1],):
        raise ValueError("energy fields have inconsistent shapes")
    if np.any(plus < 0.0) or np.any(minus < 0.0):
        raise ValueError("energy requires nonnegative populations")
    if reference_density_m2 <= 0.0 or density_floor_m2 <= 0.0:
        raise ValueError("energy density scales must be positive")
    rho = plus + minus + 2.0 * density_floor_m2
    kappa = plus - minus
    forest = np.sum(rho, axis=0)
    entropy_like = np.sum(
        rho * (np.log(rho / reference_density_m2) - 1.0), axis=0
    )
    polarization = 0.5 * np.sum(kappa * kappa, axis=0) / forest
    return mu * burgers_m**2 * (
        diffusion_coefficient * entropy_like
        + backstress_coefficient * polarization
    )


def variational_chemical_potentials_J_m(
    mobile_plus_m2: np.ndarray,
    mobile_minus_m2: np.ndarray,
    shear_modulus_Pa: np.ndarray,
    burgers_m: float,
    *,
    backstress_coefficient: float,
    diffusion_coefficient: float,
    reference_density_m2: float,
    density_floor_m2: float = 1.0e8,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical derivatives of :func:`variational_correlation_energy_J_m3`."""

    plus = np.asarray(mobile_plus_m2, dtype=float)
    minus = np.asarray(mobile_minus_m2, dtype=float)
    mu = np.asarray(shear_modulus_Pa, dtype=float)
    # Reuse validation and keep the energy/derivative regularization identical.
    variational_correlation_energy_J_m3(
        plus, minus, mu, burgers_m,
        backstress_coefficient=backstress_coefficient,
        diffusion_coefficient=diffusion_coefficient,
        reference_density_m2=reference_density_m2,
        density_floor_m2=density_floor_m2,
    )
    rho = plus + minus + 2.0 * density_floor_m2
    kappa = plus - minus
    forest = np.sum(rho, axis=0)
    kappa_squared = np.sum(kappa * kappa, axis=0)
    common = (
        diffusion_coefficient * np.log(rho / reference_density_m2)
        - 0.5 * backstress_coefficient
        * kappa_squared[None, :] / forest[None, :] ** 2
    )
    signed = backstress_coefficient * kappa / forest[None, :]
    factor = mu[None, :] * burgers_m**2
    return factor * (common + signed), factor * (common - signed)


def variational_correlation_fluxes(
    mobile_plus_m2: np.ndarray,
    mobile_minus_m2: np.ndarray,
    mobility_m_Pa_s: np.ndarray,
    shear_modulus_Pa: np.ndarray,
    burgers_m: float,
    dx_m: float,
    *,
    backstress_coefficient: float,
    diffusion_coefficient: float,
    reference_density_m2: float,
    density_floor_m2: float = 1.0e8,
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete-gradient face flux with a provable unloaded energy inequality."""

    plus = np.asarray(mobile_plus_m2, dtype=float)
    minus = np.asarray(mobile_minus_m2, dtype=float)
    mobility = np.asarray(mobility_m_Pa_s, dtype=float)
    if mobility.shape != plus.shape or np.any(mobility < 0.0):
        raise ValueError("mobility must be nonnegative and match populations")
    chemical_plus, chemical_minus = variational_chemical_potentials_J_m(
        plus, minus, shear_modulus_Pa, burgers_m,
        backstress_coefficient=backstress_coefficient,
        diffusion_coefficient=diffusion_coefficient,
        reference_density_m2=reference_density_m2,
        density_floor_m2=density_floor_m2,
    )
    mobility_face = 0.5 * (mobility + np.roll(mobility, -1, axis=1))

    def population_mobility(field: np.ndarray) -> np.ndarray:
        effective = field + density_floor_m2
        face = logarithmic_mean(effective, np.roll(effective, -1, axis=1))
        return np.maximum(face - density_floor_m2, 0.0)

    plus_flux = (
        -mobility_face / burgers_m * population_mobility(plus)
        * (np.roll(chemical_plus, -1, axis=1) - chemical_plus) / dx_m
    )
    minus_flux = (
        -mobility_face / burgers_m * population_mobility(minus)
        * (np.roll(chemical_minus, -1, axis=1) - chemical_minus) / dx_m
    )
    return plus_flux, minus_flux


def variational_dissipation_J_m2_s(
    mobile_plus_m2: np.ndarray,
    mobile_minus_m2: np.ndarray,
    mobility_m_Pa_s: np.ndarray,
    shear_modulus_Pa: np.ndarray,
    burgers_m: float,
    dx_m: float,
    **kwargs,
) -> tuple[float, float]:
    """Return chain-rule energy rate and negative face-square dissipation."""

    chemical_plus, chemical_minus = variational_chemical_potentials_J_m(
        mobile_plus_m2, mobile_minus_m2, shear_modulus_Pa, burgers_m, **kwargs
    )
    flux_plus, flux_minus = variational_correlation_fluxes(
        mobile_plus_m2, mobile_minus_m2, mobility_m_Pa_s,
        shear_modulus_Pa, burgers_m, dx_m, **kwargs
    )
    rhs_plus = periodic_flux_divergence(flux_plus, dx_m)
    rhs_minus = periodic_flux_divergence(flux_minus, dx_m)
    chain_rate = float(np.sum(
        chemical_plus * rhs_plus + chemical_minus * rhs_minus
    ) * dx_m)
    delta_plus = np.roll(chemical_plus, -1, axis=1) - chemical_plus
    delta_minus = np.roll(chemical_minus, -1, axis=1) - chemical_minus
    mobility_face = 0.5 * (
        mobility_m_Pa_s + np.roll(mobility_m_Pa_s, -1, axis=1)
    )

    def population_mobility(field: np.ndarray) -> np.ndarray:
        effective = field + kwargs.get("density_floor_m2", 1.0e8)
        return np.maximum(
            logarithmic_mean(effective, np.roll(effective, -1, axis=1))
            - kwargs.get("density_floor_m2", 1.0e8), 0.0,
        )

    square_rate = -float(np.sum(
        mobility_face / burgers_m * (
            population_mobility(mobile_plus_m2) * delta_plus**2
            + population_mobility(mobile_minus_m2) * delta_minus**2
        ) / dx_m
    ))
    return chain_rate, square_rate
