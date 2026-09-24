"""Read-only plateau recognition and neutral common-state phase handoff.

Recognition consumes the authoritative orientation and Nye fields.  It does
not create an orientation, line content, heat, sweep, or a grain label.  The
separate handoff routine is permitted only after qualification and copies the
already evolved common state into phase owners without changing its mixture.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .common_front_state import initialize_common_front, reconstruct_common
from .moving_front import DefectState, initialize_existing_subgrain_front
from .tensorial_nye import frank_bilby_closure_from_orientations


def _periodic_angle_delta(angle, reference):
    return np.arctan2(np.sin(angle-reference), np.cos(angle-reference))


def _window_indices(center, half_width, size):
    return np.mod(np.arange(center-half_width, center+half_width+1), size)


def _ray_closure(alpha, orientation, component, center, axis, side, spacing_m,
                 half_width):
    transverse_axis = 1-axis
    transverse = int(round(center[transverse_axis])) % component.shape[transverse_axis]
    line = component[:, transverse] if axis == 0 else component[transverse, :]
    occupied = np.flatnonzero(line)
    if occupied.size == 0:
        return None
    boundary = int(occupied.min() if side < 0 else occupied.max())
    indices = _window_indices(boundary, half_width, line.size)
    if axis == 0:
        from_nye = np.sum(alpha[indices, transverse, :, 2], axis=0)*spacing_m
        inside_index = (boundary+1 if side < 0 else boundary-1) % line.size
        outside_index = (boundary-1 if side < 0 else boundary+1) % line.size
        inside = orientation[inside_index, transverse]
        outside = orientation[outside_index, transverse]
        tangent = np.array([0.0, 1.0, 0.0])
    else:
        from_nye = np.sum(alpha[transverse, indices, :, 2], axis=0)*spacing_m
        inside_index = (boundary+1 if side < 0 else boundary-1) % line.size
        outside_index = (boundary-1 if side < 0 else boundary+1) % line.size
        inside = orientation[transverse, inside_index]
        outside = orientation[transverse, outside_index]
        tangent = np.array([1.0, 0.0, 0.0])
    if side < 0:
        from_lattice = frank_bilby_closure_from_orientations(
            outside, inside, tangent)
    else:
        from_lattice = frank_bilby_closure_from_orientations(
            inside, outside, tangent)
    # The contour orientation fixes an otherwise conventional overall sign.
    lattice_norm = max(np.linalg.norm(from_lattice), 1e-30)
    lattice_direction = from_lattice/lattice_norm
    projected = float(np.dot(from_nye, lattice_direction))
    relative = min(abs(projected-lattice_norm),
                   abs(projected+lattice_norm))/lattice_norm
    transverse = np.linalg.norm(
        from_nye-np.dot(from_nye, lattice_direction)*lattice_direction)
    return {
        "axis": axis, "side": side,
        "nye_closure": from_nye,
        "lattice_closure": from_lattice,
        "relative_residual": float(relative),
        "transverse_leakage_relative": float(transverse/lattice_norm),
    }


def recognize_common_orientation_plateau(
        orientation_rad, family_nye_m1, wall_order, total_density_m2,
        spacing_m, *, minimum_misorientation_deg=2.0,
        minimum_boundary_order=0.45, minimum_ordered_fraction=0.70,
        maximum_frank_bilby_relative_residual=0.20):
    """Recognize a resolved intragranular plateau without mutating state."""
    orientation = np.asarray(orientation_rad, dtype=float)
    family_nye = np.asarray(family_nye_m1, dtype=float)
    order = np.asarray(wall_order, dtype=float)
    density = np.asarray(total_density_m2, dtype=float)
    if orientation.ndim != 2 or family_nye.shape[:2] != orientation.shape \
            or family_nye.shape[-2:] != (3, 3) or order.shape != orientation.shape \
            or density.shape != orientation.shape:
        raise ValueError("common-state recognition fields are inconsistent")
    alpha = np.sum(family_nye, axis=2) if family_nye.ndim == 5 else family_nye
    reference = float(np.median(orientation))
    contrast = np.abs(_periodic_angle_delta(orientation, reference))
    threshold = math.radians(float(minimum_misorientation_deg))
    maximum = float(np.max(contrast))
    candidate = contrast >= max(threshold, 0.75*maximum)
    labels, count = ndimage.label(candidate)
    if count == 0:
        return {"qualified": False, "reason": "no_resolved_orientation_plateau"}
    sizes = ndimage.sum(candidate, labels, range(1, count+1))
    component = labels == 1+int(np.argmax(sizes))
    interior = ndimage.binary_erosion(component, iterations=2)
    shell = ndimage.binary_dilation(component, iterations=2) & ~interior
    if not np.any(interior) or not np.any(~component) or not np.any(shell):
        return {"qualified": False, "reason": "unresolved_interior_or_boundary"}
    theta_in = float(np.mean(orientation[interior]))
    theta_out = float(np.mean(orientation[~component]))
    misorientation = abs(float(_periodic_angle_delta(theta_in, theta_out)))
    ordered_fraction = float(np.mean(order[shell] >= minimum_boundary_order))
    center = np.mean(np.argwhere(interior), axis=0)
    half_width = max(4, int(round(math.sqrt(np.sum(component))/8.0)))
    rays = [
        _ray_closure(alpha, orientation, component, center, axis, side,
                     float(spacing_m), half_width)
        for axis in (0, 1) for side in (-1, 1)
    ]
    rays = [item for item in rays if item is not None]
    fb_residual = float(np.median(
        [item["relative_residual"] for item in rays])) if rays else math.inf
    result = {
        "qualified": bool(
            misorientation >= threshold
            and ordered_fraction >= minimum_ordered_fraction
            and fb_residual <= maximum_frank_bilby_relative_residual
            and np.mean(density[interior]) < np.mean(density[~component])),
        "read_only": True,
        "misorientation_rad": misorientation,
        "misorientation_deg": math.degrees(misorientation),
        "area_m2": float(np.sum(component)*float(spacing_m)**2),
        "equivalent_radius_m": float(math.sqrt(
            np.sum(component)*float(spacing_m)**2/math.pi)),
        "boundary_order_mean": float(np.mean(order[shell])),
        "boundary_order_closure_fraction": ordered_fraction,
        "frank_bilby_median_projected_relative_residual": fb_residual,
        "frank_bilby_rays": rays,
        "interior_total_density_m2": float(np.mean(density[interior])),
        "exterior_total_density_m2": float(np.mean(density[~component])),
        "lower_density_interior": bool(
            np.mean(density[interior]) < np.mean(density[~component])),
        "component_mask": component,
        "interior_mask": interior,
        "shell_mask": shell,
    }
    return result


def neutral_common_subgrain_handoff(common, recognition, spacing_m,
                                    parent_label=0, child_label=1):
    """Allocate phase ownership with no physical cleanup or state reset."""
    if not recognition.get("qualified", False):
        raise ValueError("neutral handoff requires qualified recognition")
    support = np.asarray(recognition["component_mask"], dtype=float)
    defect = DefectState(
        np.asarray(common.mobile_plus_m2), np.asarray(common.mobile_minus_m2),
        np.asarray(common.forest_plus_m2)+np.asarray(common.forest_minus_m2),
        np.sum(np.asarray(common.wall_plus_m2)+np.asarray(common.wall_minus_m2),
               axis=2),
    )
    sparse = initialize_existing_subgrain_front(
        defect, support, int(parent_label), int(child_label))
    handed = initialize_common_front(sparse, common)
    reconstructed, _ = reconstruct_common(handed, float(spacing_m))
    fields = (
        "mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
        "forest_minus_m2", "wall_plus_m2", "wall_minus_m2", "junction_m2",
        "wall_order", "multi_hit_coordination", "slip", "beta_p",
        "alignment_m2", "family_nye_m1", "orientation_rad", "temperature_K",
    )
    def maximum_abs(value):
        array = np.asarray(value)
        return 0.0 if array.size == 0 else float(np.max(np.abs(array)))
    errors = {name: maximum_abs(
        np.asarray(getattr(reconstructed, name))-np.asarray(getattr(common, name)))
              for name in fields}
    scales = {name: max(maximum_abs(getattr(common, name)), 1.0)
              for name in fields}
    relative = {name: errors[name]/scales[name] for name in fields}
    if max(relative.values()) > 128.0*np.finfo(float).eps:
        raise RuntimeError("neutral common-state handoff changed the physical state")
    return handed, {
        "representation_only": True,
        "physical_sweep_m3": 0.0,
        "generated_heat_J": 0.0,
        "invented_signed_line": False,
        "orientation_target_imposed": False,
        "maximum_relative_state_change": max(relative.values()),
        "field_maximum_absolute_changes": errors,
    }
