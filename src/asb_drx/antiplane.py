"""Periodic antiplane elastic equilibrium with a prescribed plastic shear."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class AntiplaneEquilibrium:
    stress_x_Pa: np.ndarray
    stress_y_Pa: np.ndarray
    mean_stress_Pa: float
    elastic_energy_J_m3: float
    equilibrium_residual_Pa_m_inv: float


def solve_periodic_antiplane(
    applied_shear: float,
    plastic_shear: np.ndarray,
    shear_modulus_Pa: float,
    dx_m: float,
) -> AntiplaneEquilibrium:
    """Return the exact discrete-Fourier equilibrium for scalar antiplane shear.

    The elastic distortion before relaxation is
    ``a=(applied_shear-plastic_shear, 0)``.  Every nonzero Fourier mode is
    projected transverse to its wave vector, while the zero mode retains the
    imposed mean shear.  This is the minimum-elastic-energy periodic solution.
    """
    plastic = np.asarray(plastic_shear, dtype=float)
    if plastic.ndim != 2 or min(plastic.shape) < 2 or not np.all(np.isfinite(plastic)):
        raise ValueError("plastic_shear must be a finite two-dimensional field")
    if not math.isfinite(applied_shear):
        raise ValueError("applied_shear must be finite")
    if not math.isfinite(shear_modulus_Pa) or shear_modulus_Pa <= 0.0:
        raise ValueError("shear_modulus_Pa must be finite and positive")
    if not math.isfinite(dx_m) or dx_m <= 0.0:
        raise ValueError("dx_m must be finite and positive")

    ny, nx = plastic.shape
    ax = applied_shear - plastic
    ax_hat = np.fft.fft2(ax)
    kx_values = 2.0 * math.pi * np.fft.fftfreq(nx, d=dx_m)
    ky_values = 2.0 * math.pi * np.fft.fftfreq(ny, d=dx_m)
    # A real collocated grid has no signed derivative for its self-conjugate
    # Nyquist coefficient.  Assigning that derivative zero preserves Hermitian
    # symmetry of the vector projection and therefore the exact work identity.
    if nx % 2 == 0:
        kx_values[nx // 2] = 0.0
    if ny % 2 == 0:
        ky_values[ny // 2] = 0.0
    kx = kx_values[None, :]
    ky = ky_values[:, None]
    k2 = kx**2 + ky**2
    kx_grid = np.broadcast_to(kx, k2.shape)
    ky_grid = np.broadcast_to(ky, k2.shape)
    transverse_x = np.ones_like(k2)
    transverse_y = np.zeros_like(k2)
    nonzero = k2 > 0.0
    transverse_x[nonzero] = 1.0 - kx_grid[nonzero] ** 2 / k2[nonzero]
    transverse_y[nonzero] = -kx_grid[nonzero] * ky_grid[nonzero] / k2[nonzero]
    stress_x_hat = shear_modulus_Pa * transverse_x * ax_hat
    stress_y_hat = shear_modulus_Pa * transverse_y * ax_hat
    stress_x = np.fft.ifft2(stress_x_hat).real
    stress_y = np.fft.ifft2(stress_y_hat).real
    divergence_hat = 1j * (kx_grid * stress_x_hat + ky_grid * stress_y_hat)
    divergence = np.fft.ifft2(divergence_hat).real
    energy = 0.5 / shear_modulus_Pa * float(np.mean(stress_x**2 + stress_y**2))
    return AntiplaneEquilibrium(
        stress_x_Pa=stress_x,
        stress_y_Pa=stress_y,
        mean_stress_Pa=float(np.mean(stress_x)),
        elastic_energy_J_m3=energy,
        equilibrium_residual_Pa_m_inv=float(np.max(np.abs(divergence))),
    )


def midpoint_work_ledger_J_m3(
    old: AntiplaneEquilibrium,
    new: AntiplaneEquilibrium,
    applied_shear_increment: float,
    plastic_shear_increment: np.ndarray,
) -> tuple[float, float, float, float]:
    """Exact quadratic-energy identity for one equilibrium-to-equilibrium step."""
    increment = np.asarray(plastic_shear_increment, dtype=float)
    if increment.shape != old.stress_x_Pa.shape or increment.shape != new.stress_x_Pa.shape:
        raise ValueError("plastic_shear_increment shape mismatch")
    if not np.all(np.isfinite(increment)) or not math.isfinite(applied_shear_increment):
        raise ValueError("work increments must be finite")
    external = 0.5 * (old.mean_stress_Pa + new.mean_stress_Pa) * applied_shear_increment
    midpoint_stress_x = 0.5 * (old.stress_x_Pa + new.stress_x_Pa)
    plastic_work = float(np.mean(midpoint_stress_x * increment))
    elastic_change = new.elastic_energy_J_m3 - old.elastic_energy_J_m3
    closure = external - plastic_work - elastic_change
    return external, plastic_work, elastic_change, closure
