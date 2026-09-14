"""Physical pair validation and boundary-displacement seeds for SIBM.

All distances are physical metres.  The routines operate on the actual
post-initialization phase fields and never allocate a label or orientation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy import ndimage


CONNECTIVITY = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


@dataclass(frozen=True)
class PairGeometryReport:
    valid: bool
    reasons: tuple[str, ...]
    parent_max_fraction: float
    child_max_fraction: float
    parent_connected_core_area_m2: float
    child_connected_core_area_m2: float
    parent_core_inradius_m: float
    child_core_inradius_m: float
    parent_available_normal_depth_m: float
    child_available_normal_depth_m: float
    connected_hagb_length_m: float
    seed_centre_to_triple_junction_m: float
    seed_tip_to_other_boundary_m: float
    seed_neck_to_other_boundary_m: float
    seed_window_clearance_m: float
    required_parent_depth_m: float
    required_child_depth_m: float
    required_core_inradius_m: float
    required_triple_junction_clearance_m: float
    required_window_clearance_m: float
    required_hagb_length_m: float

    def to_dict(self):
        return {key: (None if isinstance(value, float) and not math.isfinite(value)
                      else value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class BoundaryDisplacementSeed:
    eta: np.ndarray
    signed_distance_m: np.ndarray
    seeded_signed_distance_m: np.ndarray
    displacement_m: np.ndarray
    amplitude_m: float
    half_chord_m: float
    chord_length_m: float
    arc_length_m: float
    swept_area_m2: float
    tip_curvature_m_1: float
    maximum_simplex_error: float
    maximum_nonpair_change: float


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask, CONNECTIVITY)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, labels, np.arange(1, count + 1))
    return labels == int(np.argmax(sizes) + 1)


def pair_signed_distance(eta: np.ndarray, parent: int, child: int,
                         spacing_m: float) -> np.ndarray:
    """Signed distance, positive in the parent, from the pair equality contour."""
    phi = np.asarray(eta[..., child] - eta[..., parent], dtype=float)
    child_side = phi >= 0.0
    raw = (ndimage.distance_transform_edt(~child_side)
           - ndimage.distance_transform_edt(child_side))
    # EDT assigns one cell, rather than half a cell, to samples adjacent to a
    # between-cell contour.  Centre the signed distance on that contour so a
    # flat equilibrium profile is reconstructed without a one-cell jump.
    return spacing_m * np.sign(raw) * np.maximum(np.abs(raw) - 0.5, 0.0)


def pinned_cap_geometry(amplitude_m: float, half_chord_m: float) -> dict[str, float]:
    if amplitude_m <= 0.0 or half_chord_m <= 0.0:
        raise ValueError("pinned cap amplitude and half chord must be positive")
    a, b = float(amplitude_m), float(half_chord_m)
    radius = (a * a + b * b) / (2.0 * a)
    angle = 4.0 * math.atan(a / b)
    arc = radius * angle
    area = 0.5 * radius * radius * (angle - math.sin(angle))
    return {"radius_m": radius, "central_angle_rad": angle,
            "chord_length_m": 2.0 * b, "arc_length_m": arc,
            "swept_area_m2": area, "curvature_m-1": 1.0 / radius}


def pinned_cap_energy(amplitude_m: float, half_chord_m: float, *,
                      boundary_energy_J_m2: float, stored_pressure_Pa: float,
                      compatibility_pressure_Pa: float = 0.0,
                      drag_pressure_Pa: float = 0.0,
                      represented_thickness_m: float = 1.0) -> dict[str, float]:
    """Energy and analytical amplitude derivative for a pinned 2-D cap."""
    geometry = pinned_cap_geometry(amplitude_m, half_chord_m)
    a, b = float(amplitude_m), float(half_chord_m)
    radius = geometry["radius_m"]
    angle = geometry["central_angle_rad"]
    radius_prime = (a * a - b * b) / (2.0 * a * a)
    angle_prime = 4.0 * b / (b * b + a * a)
    arc_prime = radius_prime * angle + radius * angle_prime
    area_prime = (radius * radius_prime * (angle - math.sin(angle))
                  + 0.5 * radius * radius * angle_prime * (1.0 - math.cos(angle)))
    gamma = float(boundary_energy_J_m2)
    stored = float(stored_pressure_Pa)
    compatibility = float(compatibility_pressure_Pa)
    drag = float(drag_pressure_Pa)
    thickness = float(represented_thickness_m)
    area = geometry["swept_area_m2"]
    excess_arc = geometry["arc_length_m"] - geometry["chord_length_m"]
    result = dict(geometry)
    result.update({
        "stored_energy_work_J": -stored * area * thickness,
        "compatibility_energy_J": compatibility * area * thickness,
        "drag_energy_J": drag * area * thickness,
        "boundary_energy_J": gamma * excess_arc * thickness,
        "numerical_penalty_J": 0.0,
        "total_energy_J": thickness * (
            gamma * excess_arc - (stored - compatibility - drag) * area),
        "d_total_energy_da_J_m": thickness * (
            gamma * arc_prime - (stored - compatibility - drag) * area_prime),
        "d_arc_length_da": arc_prime,
        "d_swept_area_da_m": area_prime,
    })
    return result


def displace_pair_boundary(eta: np.ndarray, *, parent: int, child: int,
                           spacing_m: float, interface_width_m: float,
                           centre_index: tuple[int, int],
                           advance_direction_index: tuple[int, int],
                           amplitude_m: float, half_chord_m: float,
                           active_window_radius_m: float,
                           profile: str = "pinned_cap") -> BoundaryDisplacementSeed:
    """Displace an existing pair contour and rebuild its equilibrium profile."""
    original = np.asarray(eta, dtype=float)
    if amplitude_m == 0.0:
        zeros = np.zeros(original.shape[:2])
        distance = pair_signed_distance(original, parent, child, spacing_m)
        return BoundaryDisplacementSeed(original.copy(), distance, distance.copy(), zeros,
                                        0.0, half_chord_m, 2.0 * half_chord_m,
                                        2.0 * half_chord_m, 0.0, 0.0, 0.0, 0.0)
    if amplitude_m < 0.0 or half_chord_m <= 0.0:
        raise ValueError("seed dimensions are invalid")
    ci, cj = centre_index
    ai, aj = advance_direction_index
    nx, ny = original.shape[:2]
    ii = ((np.arange(nx) - ci + nx // 2) % nx - nx // 2)[:, None] * spacing_m
    jj = ((np.arange(ny) - cj + ny // 2) % ny - ny // 2)[None, :] * spacing_m
    normal = ai * ii + aj * jj
    tangent = -aj * ii + ai * jj
    if profile == "pinned_cap":
        geometry = pinned_cap_geometry(amplitude_m, half_chord_m)
        radius = geometry["radius_m"]
        displacement = np.where(
            np.abs(tangent) <= half_chord_m,
            np.sqrt(np.maximum(radius * radius - tangent * tangent, 0.0))
            - (radius - amplitude_m), 0.0)
    elif profile == "cosine":
        geometry = {"chord_length_m": 2.0 * half_chord_m,
                    "arc_length_m": float("nan"), "swept_area_m2": float("nan"),
                    "curvature_m-1": float("nan")}
        displacement = np.where(
            np.abs(tangent) <= half_chord_m,
            0.5 * amplitude_m * (1.0 + np.cos(np.pi * tangent / half_chord_m)), 0.0)
    else:
        raise ValueError("unknown boundary displacement profile")
    radial = np.hypot(normal, tangent)
    displacement = np.where(radial <= active_window_radius_m, displacement, 0.0)
    distance = pair_signed_distance(original, parent, child, spacing_m)
    seeded_distance = distance - displacement
    pair_sum = original[..., parent] + original[..., child]
    pair_neighbourhood = ((pair_sum > 0.70)
                          & (radial <= active_window_radius_m)
                          & (np.abs(distance) <= max(
                              4.0 * interface_width_m + amplitude_m,
                              amplitude_m + spacing_m)))
    child_fraction = 0.5 * (1.0 - np.tanh(
        seeded_distance / (math.sqrt(2.0) * interface_width_m)))
    result = original.copy()
    result[..., child] = np.where(
        pair_neighbourhood, pair_sum * child_fraction, original[..., child])
    result[..., parent] = np.where(
        pair_neighbourhood, pair_sum * (1.0 - child_fraction), original[..., parent])
    result /= np.maximum(np.sum(result, axis=2, keepdims=True), 1e-300)
    nonpair = [g for g in range(result.shape[2]) if g not in (parent, child)]
    nonpair_change = (float(np.max(np.abs(result[..., nonpair] - original[..., nonpair])))
                      if nonpair else 0.0)
    return BoundaryDisplacementSeed(
        result, distance, seeded_distance, displacement, amplitude_m, half_chord_m,
        geometry["chord_length_m"], geometry["arc_length_m"],
        geometry["swept_area_m2"], geometry["curvature_m-1"],
        float(np.max(np.abs(np.sum(result, axis=2) - 1.0))), nonpair_change)


def validate_post_seed_pair(eta: np.ndarray, *, parent: int, child: int,
                            spacing_m: float, interface_width_m: float,
                            centre_index: tuple[int, int],
                            advance_direction_index: tuple[int, int],
                            seed_amplitude_m: float, half_chord_m: float,
                            active_window_radius_m: float,
                            purity_threshold: float = 0.8) -> PairGeometryReport:
    """Recompute the v13 physical pair precondition from post-seed fields."""
    values = np.asarray(eta, dtype=float)
    pcore = _largest_component(values[..., parent] >= purity_threshold)
    ccore = _largest_component(values[..., child] >= purity_threshold)
    parea = float(np.sum(pcore) * spacing_m ** 2)
    carea = float(np.sum(ccore) * spacing_m ** 2)
    pin = float(np.max(ndimage.distance_transform_edt(pcore)) * spacing_m)
    cin = float(np.max(ndimage.distance_transform_edt(ccore)) * spacing_m)
    distance = pair_signed_distance(values, parent, child, spacing_m)
    pdepth = float(np.max(distance[pcore])) if np.any(pcore) else 0.0
    cdepth = float(np.max(-distance[ccore])) if np.any(ccore) else 0.0
    band = ((values[..., parent] > 0.15) & (values[..., child] > 0.15)
            & (values[..., parent] + values[..., child] > 0.70))
    hagb = _largest_component(band)
    hagb_length = float(np.sum(hagb) * spacing_m)
    nonpair_values = np.delete(values, (parent, child), axis=2)
    nonpair = (np.max(nonpair_values, axis=2) if nonpair_values.shape[2]
               else np.zeros(values.shape[:2]))
    triple = band & (nonpair > 0.15)
    other_boundary = nonpair > 0.15
    ci, cj = centre_index
    ai, aj = advance_direction_index
    ti, tj = -aj, ai
    nx, ny = values.shape[:2]
    def periodic_distance_to(mask, point):
        if not np.any(mask):
            return float("inf")
        i, j = np.argwhere(mask).T
        di = np.minimum(abs(i - point[0]), nx - abs(i - point[0]))
        dj = np.minimum(abs(j - point[1]), ny - abs(j - point[1]))
        return float(np.min(np.hypot(di, dj)) * spacing_m)
    tip = (int(round(ci + ai * seed_amplitude_m / spacing_m)) % nx,
           int(round(cj + aj * seed_amplitude_m / spacing_m)) % ny)
    neck_offset = half_chord_m / spacing_m
    neck1 = (int(round(ci + ti * neck_offset)) % nx,
             int(round(cj + tj * neck_offset)) % ny)
    neck2 = (int(round(ci - ti * neck_offset)) % nx,
             int(round(cj - tj * neck_offset)) % ny)
    triple_distance = periodic_distance_to(triple, (ci, cj))
    tip_other = periodic_distance_to(other_boundary, tip)
    neck_other = min(periodic_distance_to(other_boundary, neck1),
                     periodic_distance_to(other_boundary, neck2))
    window_clearance = active_window_radius_m - math.hypot(
        seed_amplitude_m, half_chord_m)
    required_parent = seed_amplitude_m + 2.0 * interface_width_m
    required_child = 2.0 * interface_width_m
    required_inradius = 2.0 * interface_width_m
    required_triple = 3.0 * interface_width_m
    required_window = 2.0 * interface_width_m
    required_hagb = 2.0 * half_chord_m + 6.0 * interface_width_m
    checks = {
        "parent connected pure core": parea > 0.0,
        "child connected pure core": carea > 0.0,
        "parent core inradius": pin >= required_inradius,
        "child core inradius": cin >= required_inradius,
        "parent normal depth": pdepth >= required_parent,
        "child normal depth": cdepth >= required_child,
        "triple-junction clearance": triple_distance >= required_triple,
        "seed tip other-boundary clearance": tip_other >= required_window,
        "seed neck other-boundary clearance": neck_other >= required_window,
        "active-window clearance": window_clearance >= required_window,
        "connected usable HAGB length": hagb_length >= required_hagb,
        "phase simplex": float(np.max(np.abs(np.sum(values, axis=2) - 1.0))) <= 1e-12,
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return PairGeometryReport(
        not reasons, reasons, float(np.max(values[..., parent])),
        float(np.max(values[..., child])), parea, carea, pin, cin, pdepth, cdepth,
        hagb_length, triple_distance, tip_other, neck_other, window_clearance,
        required_parent, required_child, required_inradius, required_triple,
        required_window, required_hagb)
