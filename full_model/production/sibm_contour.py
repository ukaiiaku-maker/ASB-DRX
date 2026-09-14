"""Contour-resolved diagnostics for a declared pair-local SIBM segment."""

from __future__ import annotations

from scipy import ndimage
import numpy as np


def _zero_points(phi, mask, spacing_m):
    points = []
    nx, ny = phi.shape
    for axis in (0, 1):
        shifted = np.roll(phi, -1, axis=axis)
        valid = mask & np.roll(mask, -1, axis=axis) & (phi*shifted <= 0.0)
        for i, j in np.argwhere(valid):
            denom = abs(phi[i, j])+abs(shifted[i, j])
            frac = abs(phi[i, j])/max(denom, 1e-300)
            x, y = float(i), float(j)
            if axis == 0:
                x += frac
            else:
                y += frac
            points.append((x*spacing_m, y*spacing_m, int(i), int(j)))
    return points


def measure_pair_contour(*, eta, reference_eta, active_mask, centre_index,
                         advance_direction_index, parent_label, child_label,
                         spacing_m, dt_s, local_pressure_Pa,
                         previous_excess_area_m2=None,
                         active_window_radius_m=None):
    """Measure signed pair-contour motion and guard against lateral spreading."""
    eta = np.asarray(eta, float); ref = np.asarray(reference_eta, float)
    mask = np.asarray(active_mask, bool)
    if eta.shape != ref.shape or eta.shape[:2] != mask.shape:
        raise ValueError("SIBM contour grids are inconsistent")
    p, c = int(parent_label), int(child_label)
    phi = eta[:, :, c]-eta[:, :, p]
    phi0 = ref[:, :, c]-ref[:, :, p]
    current = _zero_points(phi, mask, spacing_m)
    reference = _zero_points(phi0, mask, spacing_m)
    if not current or not reference:
        raise ValueError("pair zero contour is not resolved in the active window")
    ci, cj = centre_index; ai, aj = advance_direction_index
    ti, tj = -aj, ai
    origin = np.array([ci*spacing_m, cj*spacing_m])
    def coordinates(points):
        xy = np.asarray([[q[0], q[1]] for q in points])-origin
        return xy@np.array([ai, aj]), xy@np.array([ti, tj])
    n, s = coordinates(current); n0, s0 = coordinates(reference)
    displacement = np.asarray([n[k]-n0[np.argmin(np.abs(s0-s[k]))]
                               for k in range(len(n))])
    tip_k = int(np.argmax(displacement)); tip = current[tip_k]
    tip_displacement = float(displacement[tip_k])
    median_displacement = float(np.median(displacement))
    amplitude = tip_displacement-median_displacement
    active_displacement = displacement > max(0.1*spacing_m, 0.1*max(amplitude, 0.0))
    neck_width = (float(np.ptp(s[active_displacement])+spacing_m)
                  if np.any(active_displacement) else 0.0)

    child_now = phi > 0.0; child_ref = phi0 > 0.0
    gained = child_now & ~child_ref & mask
    lost = child_ref & ~child_now & mask
    excess_area = float((np.sum(gained)-np.sum(lost))*spacing_m**2)
    labels, count = ndimage.label(gained, np.array([[0,1,0],[1,1,1],[0,1,0]]))
    component = int(labels[int(ci), int(cj)]) if count else 0
    if count and component == 0:
        component = int(labels[tip[2], tip[3]])
    component_area = float(np.sum(labels == component)*spacing_m**2) if component else 0.0

    gx = (np.roll(phi, -1, 0)-np.roll(phi, 1, 0))/(2*spacing_m)
    gy = (np.roll(phi, -1, 1)-np.roll(phi, 1, 1))/(2*spacing_m)
    mag = np.maximum(np.hypot(gx, gy), 1e-300)
    nx, ny = gx/mag, gy/mag
    curvature = ((np.roll(nx, -1, 0)-np.roll(nx, 1, 0))/(2*spacing_m)
                 +(np.roll(ny, -1, 1)-np.roll(ny, 1, 1))/(2*spacing_m))
    neck_k = int(np.argmin(np.abs(displacement-median_displacement)))
    if active_window_radius_m is None:
        distance = ndimage.distance_transform_edt(mask)*spacing_m
        tip_window_distance = float(distance[tip[2], tip[3]])
    else:
        di = min(abs(tip[2]-ci), phi.shape[0]-abs(tip[2]-ci))*spacing_m
        dj = min(abs(tip[3]-cj), phi.shape[1]-abs(tip[3]-cj))*spacing_m
        tip_window_distance = max(float(active_window_radius_m)-np.hypot(di, dj), 0.0)
    h = eta**2*(3.0-2.0*eta); h0 = ref**2*(3.0-2.0*ref)
    area_change = np.sum(h-h0, axis=(0, 1))*spacing_m**2
    nonpair = [float(area_change[g]) for g in range(eta.shape[2]) if g not in (p, c)]
    outside = ~mask
    maximum_outside = float(np.max(np.abs(eta[outside]-ref[outside]))) if np.any(outside) else 0.0
    velocity = (0.0 if previous_excess_area_m2 is None else
                (excess_area-previous_excess_area_m2)
                / max(neck_width, spacing_m)/max(dt_s, 1e-300))
    return {
        "signed_normal_displacement_mean_m": float(np.mean(displacement)),
        "bulge_tip_displacement_m": tip_displacement,
        "bulge_amplitude_m": float(amplitude),
        "neck_width_m": neck_width,
        "excess_bulge_area_m2": excess_area,
        "pair_local_interface_length_m": float(len(current)*spacing_m),
        "tip_curvature_m-1": float(curvature[tip[2], tip[3]]),
        "neck_curvature_m-1": float(curvature[current[neck_k][2], current[neck_k][3]]),
        "local_normal_pressure_Pa": float(local_pressure_Pa),
        "area_equivalent_normal_velocity_m_s": float(velocity),
        "tip_distance_to_window_m": tip_window_distance,
        "seeded_component_id": component,
        "seeded_component_area_m2": component_area,
        "maximum_abs_nonpair_area_change_m2": max(map(abs, nonpair), default=0.0),
        "maximum_phase_change_outside_window": maximum_outside,
        "contour_points": len(current),
    }
