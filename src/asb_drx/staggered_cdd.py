"""Staggered signed-flux kinematics for Mission-v3 CDD integration."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StaggeredSignedState:
    """Cell populations and right-face accumulated plastic slips."""

    mobile_plus_m2: np.ndarray
    mobile_minus_m2: np.ndarray
    face_slip: np.ndarray
    burgers_m: float
    dx_m: float

    def __post_init__(self) -> None:
        plus = np.asarray(self.mobile_plus_m2, dtype=float)
        minus = np.asarray(self.mobile_minus_m2, dtype=float)
        slip = np.asarray(self.face_slip, dtype=float)
        if plus.ndim != 2 or minus.shape != plus.shape or slip.shape != plus.shape:
            raise ValueError("populations and face slip must share shape (families, cells)")
        if any(np.any(~np.isfinite(x)) for x in (plus, minus, slip)):
            raise ValueError("staggered fields must be finite")
        if np.any(plus < 0.0) or np.any(minus < 0.0):
            raise ValueError("line populations must be nonnegative")
        if not math.isfinite(self.burgers_m) or self.burgers_m <= 0.0:
            raise ValueError("burgers_m must be finite and positive")
        if not math.isfinite(self.dx_m) or self.dx_m <= 0.0:
            raise ValueError("dx_m must be finite and positive")
        object.__setattr__(self, "mobile_plus_m2", plus.copy())
        object.__setattr__(self, "mobile_minus_m2", minus.copy())
        object.__setattr__(self, "face_slip", slip.copy())


def signed_face_orowan_rate_s_inv(
    plus_face_flux_m_inv_s: np.ndarray,
    minus_face_flux_m_inv_s: np.ndarray,
    burgers_m: float,
) -> np.ndarray:
    """Return face slip rate ``b(F+ - F-)``."""

    plus_flux = np.asarray(plus_face_flux_m_inv_s, dtype=float)
    minus_flux = np.asarray(minus_face_flux_m_inv_s, dtype=float)
    if plus_flux.shape != minus_flux.shape or plus_flux.ndim != 2:
        raise ValueError("signed face fluxes must share shape (families, cells)")
    if not math.isfinite(burgers_m) or burgers_m <= 0.0:
        raise ValueError("burgers_m must be finite and positive")
    return burgers_m * (plus_flux - minus_flux)


def advance_staggered_flux(
    state: StaggeredSignedState,
    plus_face_flux_m_inv_s: np.ndarray,
    minus_face_flux_m_inv_s: np.ndarray,
    dt_s: float,
) -> StaggeredSignedState:
    """Conservatively advance populations and compatible face slip."""

    plus_flux = np.asarray(plus_face_flux_m_inv_s, dtype=float)
    minus_flux = np.asarray(minus_face_flux_m_inv_s, dtype=float)
    if plus_flux.shape != state.mobile_plus_m2.shape or minus_flux.shape != plus_flux.shape:
        raise ValueError("face fluxes must match the staggered state")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")
    divergence_plus = (
        plus_flux - np.roll(plus_flux, 1, axis=1)
    ) / state.dx_m
    divergence_minus = (
        minus_flux - np.roll(minus_flux, 1, axis=1)
    ) / state.dx_m
    plus = state.mobile_plus_m2 - dt_s * divergence_plus
    minus = state.mobile_minus_m2 - dt_s * divergence_minus
    tolerance = 2.0e-13 * max(
        float(np.max(state.mobile_plus_m2)),
        float(np.max(state.mobile_minus_m2)), 1.0,
    )
    if np.min(plus) < -tolerance or np.min(minus) < -tolerance:
        raise RuntimeError("staggered flux step violated population nonnegativity")
    slip = state.face_slip + dt_s * signed_face_orowan_rate_s_inv(
        plus_flux, minus_flux, state.burgers_m
    )
    return StaggeredSignedState(
        np.maximum(plus, 0.0), np.maximum(minus, 0.0), slip,
        state.burgers_m, state.dx_m,
    )


def nye_compatibility_residual_m_inv(state: StaggeredSignedState) -> np.ndarray:
    """Return ``div(gamma_face) + b*kappa_cell`` family by family."""

    slip_gradient = (
        state.face_slip - np.roll(state.face_slip, 1, axis=1)
    ) / state.dx_m
    kappa = state.mobile_plus_m2 - state.mobile_minus_m2
    return slip_gradient + state.burgers_m * kappa


def cell_centered_slip(face_slip: np.ndarray) -> np.ndarray:
    """Second-order face-to-cell reconstruction for Gate-A constitutive calls."""

    slip = np.asarray(face_slip, dtype=float)
    if slip.ndim != 2:
        raise ValueError("face_slip must have shape (families, cells)")
    return 0.5 * (slip + np.roll(slip, 1, axis=1))
