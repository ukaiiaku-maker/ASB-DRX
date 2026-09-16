"""Diagnostics for one periodic authoritative Nye state."""

from __future__ import annotations

import numpy as np


def periodic_nye_decomposition(reservoir_nye_m1, curl_beta_nye_m1, spacing_m):
    """Decompose a 2-D Nye mismatch into zero-mode and line-continuity parts.

    Both tensors use ``(x,y,i,j)`` storage.  The divergence is taken on the
    first tensor index, ``d_x alpha[x,j] + d_y alpha[y,j]``.  The remainder is
    the nonzero, divergence-free/curl-compatible part plus discretization and
    connection terms, which require separate manufactured tests to distinguish.
    """
    reservoir = np.asarray(reservoir_nye_m1, dtype=float)
    curl_beta = np.asarray(curl_beta_nye_m1, dtype=float)
    if reservoir.shape != curl_beta.shape or reservoir.ndim != 4 \
            or reservoir.shape[-2:] != (3, 3):
        raise ValueError("Nye fields must share shape (nx,ny,3,3)")
    dx = float(spacing_m)
    if not np.isfinite(dx) or dx <= 0.0:
        raise ValueError("spacing must be positive")
    mismatch = reservoir-curl_beta
    zero = np.mean(mismatch, axis=(0, 1))
    nonzero = mismatch-zero
    divergence = ((np.roll(nonzero[:, :, 0, :], -1, axis=0)
                   -np.roll(nonzero[:, :, 0, :], 1, axis=0))/(2.0*dx)
                  +(np.roll(nonzero[:, :, 1, :], -1, axis=1)
                    -np.roll(nonzero[:, :, 1, :], 1, axis=1))/(2.0*dx))
    reference = max(float(np.sqrt(np.mean(curl_beta**2))), 1.0)
    return {
        "reservoir_zero_mode_m1": np.mean(reservoir, axis=(0, 1)),
        "curl_beta_zero_mode_m1": np.mean(curl_beta, axis=(0, 1)),
        "mismatch_zero_mode_m1": zero,
        "zero_mode_relative_rms": float(np.sqrt(np.mean(zero**2))/reference),
        "nonzero_mode_relative_rms": float(np.sqrt(np.mean(nonzero**2))/reference),
        "total_relative_rms": float(np.sqrt(np.mean(mismatch**2))/reference),
        "line_continuity_divergence_rms_m2": float(np.sqrt(np.mean(divergence**2))),
        "periodic_curl_zero_mode_relative_rms": float(
            np.sqrt(np.mean(np.mean(curl_beta, axis=(0, 1))**2))/reference),
    }
