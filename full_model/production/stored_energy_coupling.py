"""Common variational stored-energy coupling for multiphase DRX.

The interpolation

    f_s(eta) = sum_i h(eta_i) psi_i / sum_i h(eta_i),
    h(x) = x**2 (3 - 2 x),

uses the same functional for every phase.  Phase lineage may determine the
material state ``psi_i`` (for example, a newly recovered child), but never the
sign of the evolution equation.
"""

from __future__ import annotations

import numpy as np


def smooth_phase_interpolation(eta):
    """Return ``h(eta)`` and ``dh/deta`` for bounded phase support."""
    fields = np.asarray(eta, dtype=float)
    if (fields.ndim < 1 or not np.all(np.isfinite(fields))
            or np.any(fields < 0.0) or np.any(fields > 1.0)):
        raise ValueError("phase support must be finite and in [0,1]")
    return fields*fields*(3.0-2.0*fields), 6.0*fields*(1.0-fields)


def common_variational_stored_energy(eta, phase_energy_J_m3):
    """Return common stored energy and all phase derivatives in J/m3.

    ``eta`` and ``phase_energy_J_m3`` share ``(..., nphase)`` shape.  The
    returned derivative is the exact partial derivative of the normalized
    interpolation above.  Pure phases are stationary and absent phases cannot
    nucleate from this term because ``h'(0)=h'(1)=0``.
    """
    fields = np.asarray(eta, dtype=float)
    energies = np.asarray(phase_energy_J_m3, dtype=float)
    if fields.shape != energies.shape:
        raise ValueError("phase support and phase energies must have identical shape")
    if not np.all(np.isfinite(energies)) or np.any(energies < 0.0):
        raise ValueError("phase stored energies must be finite and nonnegative")
    h, dh = smooth_phase_interpolation(fields)
    denominator = np.sum(h, axis=-1, keepdims=True)
    if np.any(denominator <= 0.0):
        raise ValueError("at least one phase must have support at every point")
    mixture = np.sum(h*energies, axis=-1, keepdims=True)/denominator
    derivative = dh*(energies-mixture)/denominator
    return mixture[..., 0], derivative


def phase_mean_stored_energy_states(eta, stored_energy_J_m3, purity_threshold=0.8):
    """Construct evolving per-phase material energies from current fields.

    A phase state is the mean current stored energy in its dominant pure
    support.  If no pure pixel exists, a support-weighted mean is used.  The
    result is broadcast over the spatial grid so all phases enter the same
    variational interpolation.
    """
    fields = np.asarray(eta, dtype=float)
    stored = np.asarray(stored_energy_J_m3, dtype=float)
    if fields.ndim != 3 or stored.shape != fields.shape[:2]:
        raise ValueError("eta must be (nx,ny,nphase) and match stored energy")
    if (not np.all(np.isfinite(stored)) or np.any(stored < 0.0)
            or not 0.0 < float(purity_threshold) <= 1.0):
        raise ValueError("stored energy and purity threshold are invalid")
    dominant = np.argmax(fields, axis=2)
    values = np.empty(fields.shape[2], dtype=float)
    for i in range(fields.shape[2]):
        pure = (dominant == i) & (fields[:, :, i] >= purity_threshold)
        if np.any(pure):
            values[i] = float(np.mean(stored[pure]))
        else:
            weight = fields[:, :, i]
            total = float(np.sum(weight))
            values[i] = (float(np.sum(weight*stored))/total
                         if total > 0.0 else float(np.mean(stored)))
    return np.broadcast_to(values, fields.shape).copy(), values
