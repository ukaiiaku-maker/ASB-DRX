"""Staggered signed-flux kinematics for Mission-v3 CDD integration."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StaggeredSignedState:
    """Signed cell inventories and right-face accumulated plastic slips.

    Locked and wall populations are stationary reservoirs.  They remain part of
    the signed Burgers inventory used by the discrete Nye compatibility check.
    """

    mobile_plus_m2: np.ndarray
    mobile_minus_m2: np.ndarray
    face_slip: np.ndarray
    burgers_m: float
    dx_m: float
    locked_plus_m2: np.ndarray | None = None
    locked_minus_m2: np.ndarray | None = None
    wall_plus_m2: np.ndarray | None = None
    wall_minus_m2: np.ndarray | None = None

    def __post_init__(self) -> None:
        plus = np.asarray(self.mobile_plus_m2, dtype=float)
        minus = np.asarray(self.mobile_minus_m2, dtype=float)
        slip = np.asarray(self.face_slip, dtype=float)
        if plus.ndim != 2 or minus.shape != plus.shape or slip.shape != plus.shape:
            raise ValueError("populations and face slip must share shape (families, cells)")
        reservoirs = []
        for value in (
            self.locked_plus_m2, self.locked_minus_m2,
            self.wall_plus_m2, self.wall_minus_m2,
        ):
            array = np.zeros_like(plus) if value is None else np.asarray(value, dtype=float)
            if array.shape != plus.shape:
                raise ValueError("stationary populations must match mobile populations")
            reservoirs.append(array)
        if any(np.any(~np.isfinite(x)) for x in (plus, minus, slip, *reservoirs)):
            raise ValueError("staggered fields must be finite")
        if any(np.any(x < 0.0) for x in (plus, minus, *reservoirs)):
            raise ValueError("line populations must be nonnegative")
        if not math.isfinite(self.burgers_m) or self.burgers_m <= 0.0:
            raise ValueError("burgers_m must be finite and positive")
        if not math.isfinite(self.dx_m) or self.dx_m <= 0.0:
            raise ValueError("dx_m must be finite and positive")
        object.__setattr__(self, "mobile_plus_m2", plus.copy())
        object.__setattr__(self, "mobile_minus_m2", minus.copy())
        object.__setattr__(self, "face_slip", slip.copy())
        for name, value in zip((
            "locked_plus_m2", "locked_minus_m2", "wall_plus_m2", "wall_minus_m2",
        ), reservoirs):
            object.__setattr__(self, name, value.copy())

    @property
    def total_signed_density_m2(self) -> np.ndarray:
        """Net signed density, including mobile, locked, and wall content."""

        return (
            self.mobile_plus_m2 - self.mobile_minus_m2
            + self.locked_plus_m2 - self.locked_minus_m2
            + self.wall_plus_m2 - self.wall_minus_m2
        )


@dataclass(frozen=True)
class StaggeredFluxLedger:
    """Content balance, including any roundoff-scale positivity correction."""

    mobile_content_before_m2: float
    mobile_content_after_m2: float
    clipping_added_m2: float
    balance_residual_m2: float
    signed_clipping_added_m2: float


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

    advanced, _ = advance_staggered_flux_with_ledger(
        state, plus_face_flux_m_inv_s, minus_face_flux_m_inv_s, dt_s
    )
    return advanced


def advance_staggered_flux_with_ledger(
    state: StaggeredSignedState,
    plus_face_flux_m_inv_s: np.ndarray,
    minus_face_flux_m_inv_s: np.ndarray,
    dt_s: float,
) -> tuple[StaggeredSignedState, StaggeredFluxLedger]:
    """Advance a periodic flux and expose every positivity correction."""

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
    raw_plus = state.mobile_plus_m2 - dt_s * divergence_plus
    raw_minus = state.mobile_minus_m2 - dt_s * divergence_minus
    tolerance = 2.0e-13 * max(
        float(np.max(state.mobile_plus_m2)),
        float(np.max(state.mobile_minus_m2)), 1.0,
    )
    if np.min(raw_plus) < -tolerance or np.min(raw_minus) < -tolerance:
        raise RuntimeError("staggered flux step violated population nonnegativity")
    plus = np.maximum(raw_plus, 0.0)
    minus = np.maximum(raw_minus, 0.0)
    correction_plus = plus - raw_plus
    correction_minus = minus - raw_minus
    slip = state.face_slip + dt_s * signed_face_orowan_rate_s_inv(
        plus_flux, minus_flux, state.burgers_m
    )
    advanced = StaggeredSignedState(
        plus, minus, slip, state.burgers_m, state.dx_m,
        state.locked_plus_m2, state.locked_minus_m2,
        state.wall_plus_m2, state.wall_minus_m2,
    )
    before = float(np.sum(state.mobile_plus_m2) + np.sum(state.mobile_minus_m2))
    after = float(np.sum(plus) + np.sum(minus))
    clipping = float(np.sum(correction_plus) + np.sum(correction_minus))
    return advanced, StaggeredFluxLedger(
        before, after, clipping, after - before - clipping,
        float(np.sum(correction_plus) - np.sum(correction_minus)),
    )


def nye_compatibility_residual_m_inv(state: StaggeredSignedState) -> np.ndarray:
    """Return ``div(gamma_face) + b*kappa_cell`` family by family."""

    slip_gradient = (
        state.face_slip - np.roll(state.face_slip, 1, axis=1)
    ) / state.dx_m
    kappa = state.total_signed_density_m2
    return slip_gradient + state.burgers_m * kappa


def transfer_mobile_to_stationary(
    state: StaggeredSignedState,
    locked_fraction: float | np.ndarray,
    wall_fraction: float | np.ndarray,
) -> StaggeredSignedState:
    """Transfer signed mobile content without changing local Burgers inventory."""

    locked = np.broadcast_to(np.asarray(locked_fraction, dtype=float), state.mobile_plus_m2.shape)
    wall = np.broadcast_to(np.asarray(wall_fraction, dtype=float), state.mobile_plus_m2.shape)
    if np.any(~np.isfinite(locked)) or np.any(~np.isfinite(wall)):
        raise ValueError("transfer fractions must be finite")
    if np.any(locked < 0.0) or np.any(wall < 0.0) or np.any(locked + wall > 1.0):
        raise ValueError("nonnegative locked and wall fractions must sum to at most one")
    retained = 1.0 - locked - wall
    return StaggeredSignedState(
        retained * state.mobile_plus_m2,
        retained * state.mobile_minus_m2,
        state.face_slip, state.burgers_m, state.dx_m,
        state.locked_plus_m2 + locked * state.mobile_plus_m2,
        state.locked_minus_m2 + locked * state.mobile_minus_m2,
        state.wall_plus_m2 + wall * state.mobile_plus_m2,
        state.wall_minus_m2 + wall * state.mobile_minus_m2,
    )


def cell_centered_slip(face_slip: np.ndarray) -> np.ndarray:
    """Second-order face-to-cell reconstruction for Gate-A constitutive calls."""

    slip = np.asarray(face_slip, dtype=float)
    if slip.ndim != 2:
        raise ValueError("face_slip must have shape (families, cells)")
    return 0.5 * (slip + np.roll(slip, 1, axis=1))


def face_traction_from_cells(cell_traction_Pa: np.ndarray) -> np.ndarray:
    """Adjoint cell-to-face projection for the periodic slip reconstruction."""

    traction = np.asarray(cell_traction_Pa, dtype=float)
    if traction.ndim != 2:
        raise ValueError("cell_traction_Pa must have shape (families, cells)")
    return 0.5 * (traction + np.roll(traction, -1, axis=1))


def work_conjugacy_residual_J_m3(
    cell_traction_Pa: np.ndarray,
    face_slip_increment: np.ndarray,
) -> float:
    """Difference between cell and face plastic work under adjoint projection."""

    increment = np.asarray(face_slip_increment, dtype=float)
    traction = np.asarray(cell_traction_Pa, dtype=float)
    if increment.shape != traction.shape or increment.ndim != 2:
        raise ValueError("traction and slip increment must share (families, cells)")
    cell_work = float(np.sum(traction * cell_centered_slip(increment)))
    face_work = float(np.sum(face_traction_from_cells(traction) * increment))
    return cell_work - face_work
