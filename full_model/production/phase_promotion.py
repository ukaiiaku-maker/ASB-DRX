"""Conservative dislocation-state transfer for full-v34 embryo promotion.

The operation removes only sign-neutral mobile pairs and unsigned forest/wall
content from a resolved embryo core.  Removed line length is stored on a
declared boundary shell.  Signed mobile Burgers content is unchanged exactly;
insufficient shell capacity rejects the operation instead of clipping.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PromotionTransferLedger:
    line_content_before_m: float
    line_content_after_m: float
    line_content_transferred_m: float
    line_content_closure_m: float
    maximum_signed_burgers_density_change_m2: float


def conservative_neutral_density_relief(
        rp, rm, rho_forest, rho_wall, rho_gb, core_mask, shell_mask, *,
        target_core_density_m2, cell_area_m2, maximum_density_m2):
    """Return relieved copies plus an exact line/Burgers-content ledger.

    ``rp`` and ``rm`` have shape ``(nx, ny, nslip)`` and units m^-2. Forest
    has the same shape; wall and GB densities have shape ``(nx, ny)``. The
    scalar target is a maximum total core density, but signed mobile content is
    an inviolable lower bound.
    """
    rp0 = np.asarray(rp, dtype=float)
    rm0 = np.asarray(rm, dtype=float)
    forest0 = np.asarray(rho_forest, dtype=float)
    wall0 = np.asarray(rho_wall, dtype=float)
    gb0 = np.asarray(rho_gb, dtype=float)
    core = np.asarray(core_mask, dtype=bool)
    shell = np.asarray(shell_mask, dtype=bool)
    if rp0.ndim != 3 or rm0.shape != rp0.shape or forest0.shape != rp0.shape:
        raise ValueError("mobile and forest populations must share (nx,ny,nslip)")
    if wall0.shape != rp0.shape[:2] or gb0.shape != wall0.shape:
        raise ValueError("wall and GB fields must match the population grid")
    if core.shape != wall0.shape or shell.shape != wall0.shape or np.any(core & shell):
        raise ValueError("core and shell must be disjoint masks on the population grid")
    if not np.any(core) or not np.any(shell):
        raise ValueError("promotion requires nonempty resolved core and shell")
    arrays = (rp0, rm0, forest0, wall0, gb0)
    if any(not np.all(np.isfinite(a)) or np.any(a < 0.0) for a in arrays):
        raise ValueError("all dislocation densities must be finite and nonnegative")
    for value, name in ((target_core_density_m2, "target density"),
                        (cell_area_m2, "cell area"),
                        (maximum_density_m2, "maximum density")):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    rp1, rm1 = rp0.copy(), rm0.copy()
    forest1, wall1, gb1 = forest0.copy(), wall0.copy(), gb0.copy()
    signed0 = rp0 - rm0
    pair_density = 2.0*np.minimum(rp0, rm0)
    signed_floor = np.sum(np.abs(signed0), axis=2)
    pair_total = np.sum(pair_density, axis=2)
    pair_keep_total = np.minimum(
        pair_total, np.maximum(float(target_core_density_m2) - signed_floor, 0.0))
    pair_scale = np.zeros_like(pair_total)
    np.divide(pair_keep_total, pair_total, out=pair_scale, where=pair_total > 0.0)
    pair_kept = pair_density * pair_scale[:, :, None]
    rp_core = np.maximum(signed0, 0.0) + 0.5*pair_kept
    rm_core = np.maximum(-signed0, 0.0) + 0.5*pair_kept
    rp1[core] = rp_core[core]
    rm1[core] = rm_core[core]
    forest1[core] = 0.0
    wall1[core] = 0.0

    old_bulk = np.sum(rp0 + rm0, axis=2) + np.sum(forest0, axis=2) + wall0
    new_bulk = np.sum(rp1 + rm1, axis=2) + np.sum(forest1, axis=2) + wall1
    removed_density_integral = float(np.sum((old_bulk - new_bulk)[core]))
    if removed_density_integral < -1e-8:
        raise RuntimeError("promotion relief attempted to create core line content")
    add_density = removed_density_integral / float(np.count_nonzero(shell))
    if np.any(gb1[shell] + add_density > maximum_density_m2):
        raise ValueError("insufficient boundary-shell capacity for conservative promotion")
    gb1[shell] += add_density

    line_before = float((np.sum(old_bulk) + np.sum(gb0))*cell_area_m2)
    line_after = float((np.sum(new_bulk) + np.sum(gb1))*cell_area_m2)
    transferred = removed_density_integral*cell_area_m2
    closure = line_after - line_before
    signed_change = float(np.max(np.abs((rp1-rm1) - signed0)))
    tolerance = 512.0*math.ulp(max(abs(line_before), abs(line_after), 1e-300))
    if abs(closure) > tolerance or signed_change != 0.0:
        raise RuntimeError("promotion transfer failed line/Burgers-content closure")
    return rp1, rm1, forest1, wall1, gb1, PromotionTransferLedger(
        line_before, line_after, transferred, closure, signed_change)
