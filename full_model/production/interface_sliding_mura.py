"""Mura-compatible upper-bound map for tangential GB Burgers fluctuations.

The explicit interface reservoir contains a lab-frame Burgers vector density
but no production GB-sliding distortion.  This module provides a deliberately
separate discriminator: the zero-integral tangential fluctuation is debited
from the interface and credited to a minimal-L2 plastic distortion whose
spectral curl is exactly the same Nye content.  The nonzero topological mean
remains on the interface.  No lattice-family transfer kernel is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .tensorial_nye import nye_from_plastic_distortion, spectral_derivatives


def _effective_wavevectors(shape, spacing):
    kx = 2*np.pi*np.fft.fftfreq(shape[0], d=spacing)
    ky = 2*np.pi*np.fft.fftfreq(shape[1], d=spacing)
    # The authoritative derivative returns the real part of the inverse FFT;
    # an even-grid Nyquist derivative is therefore the zero operator.
    if shape[0] % 2 == 0:
        kx[shape[0]//2] = 0.0
    if shape[1] % 2 == 0:
        ky[shape[1]//2] = 0.0
    return np.meshgrid(kx, ky, indexing="ij")


def _representable_zero_mean(field, spacing):
    value = np.asarray(field, dtype=float)
    kx, ky = _effective_wavevectors(value.shape[:2], spacing)
    active = kx*kx+ky*ky > 0.0
    spectrum = np.fft.fftn(value, axes=(0, 1))
    spectrum[~active] = 0.0
    return np.fft.ifftn(spectrum, axes=(0, 1)).real


@dataclass(frozen=True)
class InterfaceSlidingMuraProjection:
    candidate_state: object
    additional_plastic_distortion: np.ndarray
    sliding_nye_m1: np.ndarray
    reaction_fraction: float
    removed_interface_burgers_m1: np.ndarray
    maximum_local_burgers_balance_m1: float
    integrated_burgers_balance_m: np.ndarray
    dual_nye_relative_rms: float
    removed_carrier_line_cell_sum_m2: float
    maximum_carrier_deficit_m2: float
    feasible: bool


def minimal_plastic_distortion_for_alpha_z(alpha_vector_m1, spacing_m):
    """Invert ``alpha[:, :, i, z]`` with the zero-mean Coulomb-gauge field."""
    alpha = np.asarray(alpha_vector_m1, dtype=float)
    spacing = float(spacing_m)
    if (alpha.ndim != 3 or alpha.shape[-1] != 3
            or np.any(~np.isfinite(alpha)) or not math.isfinite(spacing)
            or spacing <= 0.0):
        raise ValueError("finite grid x 3 Nye field and positive spacing required")
    nx, ny = alpha.shape[:2]
    zero_mode = np.mean(alpha, axis=(0, 1))
    zero_tolerance = 4096*np.finfo(float).eps*max(
        float(np.max(np.abs(alpha), initial=0.0)), 1.0)
    if np.max(np.abs(zero_mode), initial=0.0) > zero_tolerance:
        raise ValueError("periodic inverse curl requires zero-integral Nye content")
    kx, ky = _effective_wavevectors((nx, ny), spacing)
    k2 = kx*kx+ky*ky
    active = k2 > 0.0
    beta = np.zeros(alpha.shape[:2]+(3, 3), dtype=float)
    for component in range(3):
        spectrum = np.fft.fftn(alpha[..., component])
        bx = np.zeros_like(spectrum)
        by = np.zeros_like(spectrum)
        bx[active] = -1j*ky[active]*spectrum[active]/k2[active]
        by[active] = 1j*kx[active]*spectrum[active]/k2[active]
        beta[..., component, 0] = np.fft.ifftn(bx).real
        beta[..., component, 1] = np.fft.ifftn(by).real
    return beta


def project_tangential_interface_burgers_to_sliding(
        state, *, spacing_m, burgers_m, reaction_fraction):
    """Debit zero-integral tangential content and return its conjugate beta."""
    state.validate()
    spacing = float(spacing_m)
    b = float(burgers_m)
    fraction = float(reaction_fraction)
    if not math.isfinite(spacing) or spacing <= 0.0 or not math.isfinite(
            b) or b <= 0.0:
        raise ValueError("positive finite spacing and Burgers magnitude required")
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("reaction fraction must lie in [0,1]")
    owner = {grain_id: index for index, grain_id in enumerate(state.grain_ids)}
    removed_total = np.zeros(state.supports.shape[1:]+(3,), dtype=float)
    next_components = []
    removed_line = maximum_deficit = 0.0
    for component in state.interfaces:
        if component.boundary_burgers_m1 is None:
            raise ValueError("interface sliding requires explicit Burgers content")
        left = owner[component.grain_a_id]
        right = owner[component.grain_b_id]
        gx, gy = spectral_derivatives(
            state.supports[right]-state.supports[left], spacing)
        magnitude = np.hypot(gx, gy)
        nx = np.divide(gx, magnitude, out=np.zeros_like(gx), where=magnitude > 0.0)
        ny = np.divide(gy, magnitude, out=np.zeros_like(gy), where=magnitude > 0.0)
        tangent = np.stack((-ny, nx, np.zeros_like(nx)), axis=-1)
        band_sum = max(float(np.sum(magnitude, dtype=np.longdouble)), 1e-300)
        burgers = np.asarray(component.boundary_burgers_m1, dtype=float)
        tangential = (np.sum(burgers*tangent[..., None, :], axis=-1)[..., None]
                      *tangent[..., None, :])
        integral = np.sum(tangential, axis=(0, 1), dtype=np.longdouble).astype(float)
        topological_shape = magnitude[..., None, None]*integral[None, None]/band_sum
        removed = _representable_zero_mean(
            fraction*(tangential-topological_shape), spacing)
        # Correct the residual sum explicitly in the same boundary-supported
        # shape; this leaves no hidden zero-mode in the spectral inversion.
        residual = np.sum(removed, axis=(0, 1), dtype=np.longdouble).astype(float)
        removed -= magnitude[..., None, None]*residual[None, None]/band_sum
        next_burgers = burgers-removed
        removed_total += np.sum(removed, axis=-2)

        plus = np.asarray(component.boundary_plus_m2, dtype=float)
        minus = np.asarray(component.boundary_minus_m2, dtype=float)
        line = np.sum(plus+minus, axis=-1)
        old_required = np.linalg.norm(burgers, axis=-1)/b
        new_required = np.linalg.norm(next_burgers, axis=-1)/b
        carrier_decrease = np.maximum(old_required-new_required, 0.0)
        target_line = line-carrier_decrease
        deficit = np.maximum(new_required-target_line, 0.0)
        maximum_deficit = max(
            maximum_deficit, float(np.max(deficit, initial=0.0)))
        target_line = np.maximum(target_line, new_required)
        retained = np.divide(
            target_line, line, out=np.ones_like(line), where=line > 0.0)
        retained = np.minimum(retained, 1.0)
        next_plus = plus*retained[..., None]
        next_minus = minus*retained[..., None]
        removed_line += float(np.sum(
            plus+minus-next_plus-next_minus, dtype=np.longdouble))
        next_components.append(replace(
            component, boundary_plus_m2=next_plus,
            boundary_minus_m2=next_minus,
            boundary_burgers_m1=next_burgers))

    beta = minimal_plastic_distortion_for_alpha_z(removed_total, spacing)
    alpha = nye_from_plastic_distortion(beta, spacing)
    target = np.zeros_like(alpha)
    target[..., :, 2] = removed_total
    residual = alpha-target
    scale = max(float(np.sqrt(np.mean(target*target))), 1.0)
    dual = float(np.sqrt(np.mean(residual*residual))/scale)
    local_balance = float(np.max(np.abs(
        target[..., :, 2]-removed_total), initial=0.0))
    integrated_balance = np.sum(
        target[..., :, 2]-removed_total, axis=(0, 1),
        dtype=np.longdouble).astype(float)*spacing**2
    candidate = replace(state, interfaces=tuple(next_components))
    candidate.validate()
    tolerance = 4096*np.finfo(float).eps*max(
        float(np.max(np.abs(removed_total), initial=0.0)), 1.0)
    return InterfaceSlidingMuraProjection(
        candidate, beta, alpha, fraction, removed_total, local_balance,
        integrated_balance, dual, removed_line, maximum_deficit,
        bool(maximum_deficit <= tolerance and dual < 2e-11))
