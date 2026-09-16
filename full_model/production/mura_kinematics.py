"""Spectral discrete complex for Mura-consistent Nye/plastic flow.

The authoritative convention is ``alpha = -Curl(beta_p)``.  A single plastic
flow tensor ``J`` therefore advances both fields through
``beta_dot = J`` and ``alpha_dot = -Curl(J)``.  No reconstruction or
post-step projection is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tensorial_nye import (
    divergence_of_nye, nye_from_plastic_distortion, rotated_system_fields)


@dataclass(frozen=True)
class MuraIncrement:
    plastic_flow_rate_s: np.ndarray
    nye_rate_m1_s: np.ndarray
    div_curl_residual_m2_s: np.ndarray


def plastic_flow_from_signed_alignment(plus_alignment_m2,
                                       minus_alignment_m2,
                                       velocity_plus_m_s,
                                       velocity_minus_m_s,
                                       systems, orientation_rad):
    """Return ``J=sum b tensor (kappa x v)`` from one signed face event."""
    kp = np.asarray(plus_alignment_m2, dtype=float)
    km = np.asarray(minus_alignment_m2, dtype=float)
    vp = np.asarray(velocity_plus_m_s, dtype=float)
    vm = np.asarray(velocity_minus_m_s, dtype=float)
    expected = np.asarray(orientation_rad).shape+(len(systems), 3)
    if (kp.shape != expected or km.shape != expected or vp.shape != expected
            or vm.shape != expected or any(np.any(~np.isfinite(x))
                                            for x in (kp, km, vp, vm))):
        raise ValueError("signed alignment and velocity fields are inconsistent")
    burgers, _, _ = rotated_system_fields(systems, orientation_rad)
    swept = np.cross(kp, vp)-np.cross(km, vm)
    return np.einsum("...ai,...al->...il", burgers, swept)


def mura_increment(plastic_flow_rate_s, spacing_m):
    flow = np.asarray(plastic_flow_rate_s, dtype=float)
    if flow.ndim != 4 or flow.shape[-2:] != (3, 3):
        raise ValueError("plastic flow must have shape (nx,ny,3,3)")
    nye_rate = nye_from_plastic_distortion(flow, spacing_m)
    div_curl = divergence_of_nye(nye_rate, spacing_m)
    return MuraIncrement(flow, nye_rate, div_curl)


def accept_mura_step(beta_p, alpha_m1, plastic_flow_rate_s, dt_s, spacing_m,
                     *, relative_tolerance=2e-11):
    """Atomically accept a flow only when the dual-Nye identity remains closed."""
    beta = np.asarray(beta_p, dtype=float)
    alpha = np.asarray(alpha_m1, dtype=float)
    flow = np.asarray(plastic_flow_rate_s, dtype=float)
    dt = float(dt_s)
    if beta.shape != alpha.shape or beta.shape != flow.shape or dt < 0.0:
        raise ValueError("Mura step fields or time increment are invalid")
    operator = mura_increment(flow, spacing_m)
    new_beta = beta+dt*flow
    new_alpha = alpha+dt*operator.nye_rate_m1_s
    curl_beta = nye_from_plastic_distortion(new_beta, spacing_m)
    residual = new_alpha-curl_beta
    scale = max(float(np.sqrt(np.mean(curl_beta**2))), 1.0)
    relative = float(np.sqrt(np.mean(residual**2))/scale)
    if relative > float(relative_tolerance):
        raise RuntimeError("accepted Mura step violates alpha=-Curl(beta_p)")
    return new_beta, new_alpha, {
        "dual_nye_relative_rms": relative,
        "div_curl_rms_m2_s": float(np.sqrt(np.mean(
            operator.div_curl_residual_m2_s**2))),
        "accepted_step_hard_invariant_passed": True,
    }
