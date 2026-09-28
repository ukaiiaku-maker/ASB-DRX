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


def _largest_periodic_component(mask):
    """Return the largest four-connected component on a periodic 2-D grid."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return None
    parent = list(range(count+1))

    def root(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        if left and right:
            a, b = root(int(left)), root(int(right))
            if a != b:
                parent[b] = a

    for j in range(mask.shape[1]):
        union(labels[0, j], labels[-1, j])
    for i in range(mask.shape[0]):
        union(labels[i, 0], labels[i, -1])
    roots = np.zeros_like(labels)
    for value in range(1, count+1):
        roots[labels == value] = root(value)
    values, sizes = np.unique(roots[roots > 0], return_counts=True)
    return roots == int(values[np.argmax(sizes)])


def _periodic_binary_operation(mask, iterations, operation):
    tiled = np.tile(np.asarray(mask, dtype=bool), (3, 3))
    changed = operation(tiled, iterations=int(iterations))
    nx, ny = mask.shape
    return changed[nx:2*nx, ny:2*ny]


def _circular_mean(angle):
    values = np.asarray(angle, dtype=float)
    return float(np.arctan2(np.mean(np.sin(values)), np.mean(np.cos(values))))


def _periodic_component_center(component):
    coordinates = np.argwhere(component)
    if coordinates.size == 0:
        raise ValueError("periodic center requires nonempty support")
    center = []
    for axis, size in enumerate(component.shape):
        phase = 2.0*np.pi*coordinates[:, axis]/size
        angle = np.arctan2(np.mean(np.sin(phase)), np.mean(np.cos(phase)))
        center.append(float(np.mod(angle, 2.0*np.pi)*size/(2.0*np.pi)))
    return np.asarray(center)


def _full_vector_closure_residual(measured, expected):
    measured = np.asarray(measured, dtype=float)
    expected = np.asarray(expected, dtype=float)
    scale = max(float(np.linalg.norm(expected)), 1e-30)
    direction = expected/scale
    projected = float(np.dot(measured, direction))
    transverse = measured-projected*direction
    return {
        "full_vector_relative_residual": float(
            np.linalg.norm(measured-expected)/scale),
        "projected_relative_residual": float(abs(projected-scale)/scale),
        "transverse_leakage_relative": float(np.linalg.norm(transverse)/scale),
    }


def _ray_closure(alpha, orientation, component, center, axis, side, spacing_m,
                 half_width):
    transverse_axis = 1-axis
    transverse = int(round(center[transverse_axis])) % component.shape[transverse_axis]
    line = component[:, transverse] if axis == 0 else component[transverse, :]
    occupied = np.flatnonzero(line)
    if occupied.size == 0:
        return None
    origin = int(round(center[axis])) % line.size
    if not line[origin]:
        periodic_distance = np.minimum(
            np.mod(occupied-origin, line.size), np.mod(origin-occupied, line.size))
        origin = int(occupied[np.argmin(periodic_distance)])
    boundary = origin
    for distance in range(1, line.size+1):
        trial = (origin+side*distance) % line.size
        if not line[trial]:
            break
        boundary = trial
    else:
        return None
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
        # Curl convention used by ``nye_from_plastic_distortion`` fixes the
        # horizontal circuit tangent opposite to the array x direction.
        tangent = np.array([-1.0, 0.0, 0.0])
    if side < 0:
        from_lattice = frank_bilby_closure_from_orientations(
            outside, inside, tangent)
    else:
        from_lattice = frank_bilby_closure_from_orientations(
            inside, outside, tangent)
    # The declared ray traversal fixes the sign: negative-side circuits use
    # outside->inside and positive-side circuits use inside->outside.  Do not
    # choose a separate best sign from the measured result.
    residual = _full_vector_closure_residual(from_nye, from_lattice)
    return {
        "axis": axis, "side": side,
        "nye_closure": from_nye,
        "lattice_closure": from_lattice,
        "relative_residual": residual["full_vector_relative_residual"],
        **residual,
    }


def recognize_common_orientation_plateau(
        orientation_rad, family_nye_m1, wall_order, total_density_m2,
        spacing_m, *, minimum_misorientation_deg=2.0,
        minimum_boundary_order=0.45, minimum_ordered_fraction=0.70,
        maximum_frank_bilby_relative_residual=0.20,
        interior_distance_m=2.0e-7, boundary_shell_thickness_m=2.0e-7,
        circuit_half_width_m=1.0e-7):
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
    reference = _circular_mean(orientation)
    contrast = np.abs(_periodic_angle_delta(orientation, reference))
    threshold = math.radians(float(minimum_misorientation_deg))
    maximum = float(np.max(contrast))
    candidate = contrast >= max(threshold, 0.75*maximum)
    component = _largest_periodic_component(candidate)
    if component is None:
        return {"qualified": False, "reason": "no_resolved_orientation_plateau"}
    interior_pixels = max(1, int(round(float(interior_distance_m)/spacing_m)))
    shell_pixels = max(1, int(round(
        float(boundary_shell_thickness_m)/spacing_m)))
    interior = _periodic_binary_operation(
        component, interior_pixels, ndimage.binary_erosion)
    dilated = _periodic_binary_operation(
        component, shell_pixels, ndimage.binary_dilation)
    shell = dilated & ~interior
    if not np.any(interior) or not np.any(~component) or not np.any(shell):
        return {"qualified": False, "reason": "unresolved_interior_or_boundary"}
    theta_in = _circular_mean(orientation[interior])
    theta_out = _circular_mean(orientation[~component])
    misorientation = abs(float(_periodic_angle_delta(theta_in, theta_out)))
    ordered_fraction = float(np.mean(order[shell] >= minimum_boundary_order))
    center = _periodic_component_center(interior)
    half_width = max(0, int(round(float(circuit_half_width_m)/spacing_m)))
    rays = [
        _ray_closure(alpha, orientation, component, center, axis, side,
                     float(spacing_m), half_width)
        for axis in (0, 1) for side in (-1, 1)
    ]
    rays = [item for item in rays if item is not None]
    fb_residuals = [item["full_vector_relative_residual"] for item in rays]
    fb_residual = float(max(fb_residuals)) if rays else math.inf
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
        "periodic_component_geometry": True,
        "interior_distance_m": float(interior_pixels*spacing_m),
        "boundary_shell_thickness_m": float(shell_pixels*spacing_m),
        "circuit_half_width_m": float(half_width*spacing_m),
        "frank_bilby_maximum_full_vector_relative_residual": fb_residual,
        # Compatibility alias: its value now intentionally uses the stronger
        # full-vector maximum and must not be interpreted as the old median.
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
