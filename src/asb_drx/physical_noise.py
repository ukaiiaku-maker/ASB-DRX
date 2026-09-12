"""Grid-consistent periodic random fields defined by a physical spectrum."""

from __future__ import annotations

import math

import numpy as np


def periodic_physical_noise(
    grid_points: int,
    domain_m: float,
    correlation_length_m: float,
    seed: int,
    *,
    cutoff_wavenumber_m_inv: float | None = None,
) -> np.ndarray:
    """Sample a zero-mean unit-RMS Gaussian-spectrum periodic field.

    The spectral envelope is ``exp(-(k*ell)^2/4)`` in amplitude.  A physical
    cutoff, default ``4/ell``, makes refinements of the same box identical at
    shared nodes once the cutoff is resolved.  Different box lengths sample
    the same physical power spectrum rather than the same Fourier mode count.
    """

    if grid_points < 8 or grid_points % 2:
        raise ValueError("an even grid with at least eight points is required")
    for name, value in (
        ("domain_m", domain_m), ("correlation_length_m", correlation_length_m),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    cutoff = 4.0 / correlation_length_m if cutoff_wavenumber_m_inv is None else cutoff_wavenumber_m_inv
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff_wavenumber_m_inv must be finite and positive")
    maximum_mode = min(
        grid_points // 2 - 1,
        int(math.floor(cutoff * domain_m / (2.0 * math.pi))),
    )
    if maximum_mode < 1:
        raise ValueError("physical spectrum contains no resolved nonzero mode")
    rng = np.random.default_rng(seed)
    coordinate = np.arange(grid_points) * domain_m / grid_points
    field = np.zeros(grid_points)
    for mode in range(1, maximum_mode + 1):
        k = 2.0 * math.pi * mode / domain_m
        envelope = math.exp(-0.25 * (k * correlation_length_m) ** 2)
        cosine, sine = rng.normal(size=2)
        field += envelope * (
            cosine * np.cos(k * coordinate) + sine * np.sin(k * coordinate)
        )
    field -= np.mean(field)
    rms = math.sqrt(float(np.mean(field * field)))
    if rms == 0.0:
        raise RuntimeError("physical random field has zero variance")
    return field / rms
