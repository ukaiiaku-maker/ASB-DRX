"""Read-only wall-to-lattice compatibility reconstruction.

This module does not create orientation or phase ownership.  It asks a narrower
question: whether the signed *ordered* wall inventory already present inside a
material owner admits a common-axis lattice-rotation field.  The reconstructed
field is the least-squares integral of the Frank--Bilby gradient implied by
that inventory, with each connected owner's mean orientation preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import lsqr

from .extensive_wall import orientation_gradient_frank_bilby_target_m1
from .tensorial_nye import rotated_system_fields


@dataclass(frozen=True)
class OrientationWallCompatibility:
    ordered_wall_nye_m1: np.ndarray
    desired_orientation_gradient_m1: np.ndarray
    reconstructed_orientation_rad: np.ndarray
    before_relative_mismatch: float
    reconstructed_relative_mismatch: float
    before_orientation_span_rad: float
    reconstructed_orientation_span_rad: float
    domain_cell_count: int


def ordered_wall_nye_from_common(state, systems):
    """Return the common-axis Nye tensor carried by ordered signed wall line.

    The 2.5-D reduction uses a z-directed line, matching the independent
    Frank--Bilby circuits used for subgrain qualification.  ``wall_order`` is
    only the conservative partition of the existing signed wall reservoirs;
    no line is created by this projection.
    """
    burgers, _, _ = rotated_system_fields(systems, state.orientation_rad)
    signed_ordered = (np.asarray(state.wall_order)[..., None]
                      *(np.asarray(state.wall_plus_m2)
                        -np.asarray(state.wall_minus_m2)))
    alpha = np.zeros(np.asarray(state.orientation_rad).shape+(3, 3))
    alpha[..., :, 2] = np.einsum(
        "...a,...ai->...i", signed_ordered, burgers)
    return alpha


def desired_orientation_gradient_from_nye(alpha_m1, orientation_rad):
    """Project common-axis Nye onto the two z-rotation gradient directions."""
    alpha = np.asarray(alpha_m1, dtype=float)
    theta = np.asarray(orientation_rad, dtype=float)
    if alpha.shape != theta.shape+(3, 3):
        raise ValueError("Nye and orientation layouts are inconsistent")
    c, s = np.cos(theta), np.sin(theta)
    dR_ey = np.stack((-c, -s, np.zeros_like(theta)), axis=-1)
    dR_ex = np.stack((-s, c, np.zeros_like(theta)), axis=-1)
    closure = alpha[..., :, 2]
    gx = np.einsum("...i,...i->...", closure, dR_ey)
    gy = -np.einsum("...i,...i->...", closure, dR_ex)
    return np.stack((gx, gy), axis=-1)


def _integrate_component_gradient(current, gradient, component, spacing_m):
    coordinates = np.argwhere(component)
    count = len(coordinates)
    if count < 2:
        return np.asarray(current)[component].copy()
    shape = component.shape
    lookup = {tuple(index): number for number, index in enumerate(coordinates)}
    rows = []
    columns = []
    values = []
    rhs = []
    row = 0
    for index in coordinates:
        here = tuple(index)
        for axis in (0, 1):
            neighbor = list(here)
            neighbor[axis] = (neighbor[axis]+1) % shape[axis]
            neighbor = tuple(neighbor)
            if neighbor not in lookup:
                continue
            rows.extend((row, row))
            columns.extend((lookup[here], lookup[neighbor]))
            values.extend((-1.0/spacing_m, 1.0/spacing_m))
            rhs.append(.5*(gradient[here+(axis,)]+gradient[neighbor+(axis,)]))
            row += 1
    if row == 0:
        return np.asarray(current)[component].copy()
    matrix = sparse.coo_matrix(
        (values, (rows, columns)), shape=(row, count)).tocsr()
    solution = lsqr(matrix, np.asarray(rhs), atol=1e-12, btol=1e-12,
                    iter_lim=max(100, 4*count))[0]
    # The edge-gradient functional has an exact rigid-rotation null mode.
    # Preserve it from the accepted state rather than choosing a target angle.
    solution += float(np.mean(np.asarray(current)[component])-np.mean(solution))
    return solution


def reconstruct_orientation_from_ordered_nye(
        alpha_m1, orientation_rad, domain_mask, spacing_m):
    """Least-squares integrate the inventory-implied gradient on each owner."""
    theta = np.asarray(orientation_rad, dtype=float)
    domain = np.asarray(domain_mask, dtype=bool)
    spacing = float(spacing_m)
    if domain.shape != theta.shape or not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("compatible orientation requires a matched domain and spacing")
    gradient = desired_orientation_gradient_from_nye(alpha_m1, theta)
    result = theta.copy()
    labels, count = ndimage.label(domain)
    for label in range(1, count+1):
        component = labels == label
        result[component] = _integrate_component_gradient(
            theta, gradient, component, spacing)
    return result, gradient


def evaluate_orientation_wall_compatibility(
        state, systems, domain_mask, spacing_m):
    """Evaluate whether existing ordered wall line supports lattice rotation."""
    domain = np.asarray(domain_mask, dtype=bool)
    if domain.shape != np.asarray(state.orientation_rad).shape or not np.any(domain):
        raise ValueError("a nonempty owner domain is required")
    alpha = ordered_wall_nye_from_common(state, systems)
    reconstructed, desired = reconstruct_orientation_from_ordered_nye(
        alpha, state.orientation_rad, domain, spacing_m)
    before_target = orientation_gradient_frank_bilby_target_m1(
        state.orientation_rad, spacing_m)
    after_target = orientation_gradient_frank_bilby_target_m1(
        reconstructed, spacing_m)
    scale = max(float(np.linalg.norm(alpha[domain])), 1e-300)
    before = float(np.linalg.norm((alpha-before_target)[domain])/scale)
    after = float(np.linalg.norm((alpha-after_target)[domain])/scale)
    accepted = np.asarray(state.orientation_rad)[domain]
    candidate = reconstructed[domain]
    return OrientationWallCompatibility(
        alpha, desired, reconstructed, before, after,
        float(np.max(accepted)-np.min(accepted)),
        float(np.max(candidate)-np.min(candidate)), int(np.sum(domain)))
