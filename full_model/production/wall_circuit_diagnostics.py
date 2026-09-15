"""Local, integrated and evolution-independent wall-circuit diagnostics.

Nothing in this module enters a constitutive residual.  It postprocesses the
evolved orientation and Nye inventories, measures local two-sided plateaus,
and evaluates Frank--Bilby closure over a fixed physical normal window.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
from scipy import ndimage

try:
    from .tensorial_nye import rotation_z, spectral_derivatives
except ImportError:
    from tensorial_nye import rotation_z, spectral_derivatives


@dataclass(frozen=True)
class WallCircuitSegment:
    label: int
    centroid_index: tuple[float, float]
    normal_xy: tuple[float, float]
    tangent_xyz: tuple[float, float, float]
    left_orientation_rad: float
    right_orientation_rad: float
    local_misorientation_rad: float
    integrated_ordered_burgers: tuple[float, float, float]
    integrated_total_burgers: tuple[float, float, float]
    frank_bilby_burgers: tuple[float, float, float]
    ordered_supply_ratio: float
    total_supply_ratio: float
    ordered_frank_bilby_residual: float
    total_frank_bilby_residual: float
    segment_pixel_count: int
    estimated_segment_length_m: float

    def to_dict(self):
        return asdict(self)


def _sample_scalar(field, points):
    coordinates = np.asarray(points, dtype=float).T
    return ndimage.map_coordinates(
        np.asarray(field, dtype=float), coordinates, order=1, mode="wrap")


def _sample_vector(field, points):
    return np.stack([_sample_scalar(field[..., i], points)
                     for i in range(field.shape[-1])], axis=-1)


def _circular_mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.arctan2(np.mean(np.sin(values)), np.mean(np.cos(values))))


def local_integrated_wall_circuits(
        orientation_rad, ordered_nye_m1, total_nye_m1, spacing_m, *,
        normal_window_m, plateau_offset_m=None, line_axis=2,
        support_threshold_fraction=0.35, minimum_segment_pixels=3,
        burgers_floor=1e-8):
    """Return local circuit segments and ordered-line overlap metrics.

    Candidate support is determined only from the postprocessed orientation
    gradient.  Frank--Bilby closure is calculated from independently sampled
    left/right plateaus.  Neither quantity can feed back into evolution.
    """
    theta = np.asarray(orientation_rad, dtype=float)
    ordered = np.asarray(ordered_nye_m1, dtype=float)
    total = np.asarray(total_nye_m1, dtype=float)
    if theta.ndim != 2 or ordered.shape != theta.shape+(3, 3) or total.shape != ordered.shape:
        raise ValueError("orientation and Nye arrays have inconsistent layouts")
    if spacing_m <= 0 or normal_window_m <= 0:
        raise ValueError("physical diagnostic lengths must be positive")
    tx, ty = spectral_derivatives(theta, spacing_m)
    magnitude = np.hypot(tx, ty)
    peak = float(np.max(magnitude))
    support = (magnitude >= support_threshold_fraction*peak) if peak > 0 else np.zeros_like(theta, bool)
    labels, count = ndimage.label(support, np.ones((3, 3), dtype=int))
    half_samples = max(1, int(np.ceil(0.5*normal_window_m/spacing_m)))
    offsets = np.arange(-half_samples, half_samples+1, dtype=float)
    plateau = (0.75*normal_window_m if plateau_offset_m is None
               else float(plateau_offset_m))
    plateau_pixels = plateau/spacing_m
    segments = []
    for label in range(1, count+1):
        pixels = np.argwhere(labels == label)
        if len(pixels) < minimum_segment_pixels:
            continue
        weights = magnitude[labels == label]
        center = np.average(pixels, axis=0, weights=np.maximum(weights, 1e-300))
        normal = np.array([
            np.average(tx[labels == label], weights=np.maximum(weights, 1e-300)),
            np.average(ty[labels == label], weights=np.maximum(weights, 1e-300))])
        norm = float(np.linalg.norm(normal))
        if norm <= 0:
            continue
        normal /= norm
        tangent = np.array([-normal[1], normal[0], 0.0])
        # Average several tangential circuits along the connected segment.
        stride = max(1, len(pixels)//32)
        ordered_circuits = []; total_circuits = []
        for point in pixels[::stride]:
            path = point[None, :]+offsets[:, None]*normal[None, :]
            ordered_circuits.append(np.sum(
                _sample_vector(ordered[..., :, line_axis], path), axis=0)*spacing_m)
            total_circuits.append(np.sum(
                _sample_vector(total[..., :, line_axis], path), axis=0)*spacing_m)
        B_ordered = np.mean(ordered_circuits, axis=0)
        B_total = np.mean(total_circuits, axis=0)
        left_points = pixels-pixel_unit(normal)*plateau_pixels
        right_points = pixels+pixel_unit(normal)*plateau_pixels
        left = _circular_mean(_sample_scalar(theta, left_points))
        right = _circular_mean(_sample_scalar(theta, right_points))
        B_fb = (rotation_z(right)-rotation_z(left))@tangent
        denominator = max(float(np.linalg.norm(B_fb)), float(burgers_floor))
        segments.append(WallCircuitSegment(
            int(label), (float(center[0]), float(center[1])),
            (float(normal[0]), float(normal[1])), tuple(map(float, tangent)),
            left, right, float(abs(np.angle(np.exp(1j*(right-left))))),
            tuple(map(float, B_ordered)), tuple(map(float, B_total)),
            tuple(map(float, B_fb)), float(np.linalg.norm(B_ordered)/denominator),
            float(np.linalg.norm(B_total)/denominator),
            float(np.linalg.norm(B_ordered-B_fb)/denominator),
            float(np.linalg.norm(B_total-B_fb)/denominator),
            int(len(pixels)), float(len(pixels)*spacing_m)))
    # Coverage uses a fixed physical neighbourhood of the detected interface,
    # not the raw gradient amplitude; it therefore does not change merely
    # because the diffuse profile sharpens at fixed integrated closure.
    distance_to_support_m = ndimage.distance_transform_edt(~support)*spacing_m
    wall_weight = (distance_to_support_m <= 0.5*normal_window_m).astype(float)
    ordered_scalar = np.linalg.norm(ordered, axis=(-2, -1))
    total_ordered = float(np.sum(ordered_scalar))
    overlap = (float(np.sum(ordered_scalar*wall_weight))/total_ordered
               if total_ordered > 0 else 0.0)
    return {
        "nye_integration_convention": (
            "2.5D line-column integral: integral alpha[:,line_axis] dn; "
            "line_axis is the retained out-of-plane line coordinate"),
        "line_axis": int(line_axis),
        "candidate_support": support,
        "candidate_weight": wall_weight,
        "segments": segments,
        "ordered_line_overlap": overlap,
        "ordered_line_outside_support": 1.0-overlap if total_ordered > 0 else 0.0,
        "global_orientation_span_rad_diagnostic_only": float(np.ptp(theta)),
    }


def classify_persistent_wall_history(
        snapshots, *, minimum_misorientation_rad=np.deg2rad(1.0),
        maximum_frank_bilby_residual=0.25, minimum_supply_ratio=0.75,
        maximum_supply_ratio=1.25, minimum_ordered_overlap=0.7,
        required_release_persistence_s=0.0):
    """Classify a time sequence using only local, integrated wall measures.

    Each snapshot must contain ``time_s``, ``orientation_rad``,
    ``ordered_nye_m1``, ``total_nye_m1``, ``spacing_m``,
    ``normal_window_m``, and a boolean ``mechanical_loading_active``.  This
    postprocessor has no phase/grain state and cannot allocate a grain label.
    """
    records = []
    for snapshot in snapshots:
        audit = local_integrated_wall_circuits(
            snapshot["orientation_rad"], snapshot["ordered_nye_m1"],
            snapshot["total_nye_m1"], snapshot["spacing_m"],
            normal_window_m=snapshot["normal_window_m"],
            plateau_offset_m=snapshot.get("plateau_offset_m"))
        qualifying = []
        for segment in audit["segments"]:
            qualifies = bool(
                segment.local_misorientation_rad >= minimum_misorientation_rad
                and segment.ordered_frank_bilby_residual
                <= maximum_frank_bilby_residual
                and minimum_supply_ratio <= segment.ordered_supply_ratio
                <= maximum_supply_ratio)
            if qualifies:
                qualifying.append(segment.label)
        audit_record = {
            key: value for key, value in audit.items()
            if key not in ("candidate_support", "candidate_weight", "segments")
        }
        audit_record["segments"] = [segment.to_dict()
                                    for segment in audit["segments"]]
        record = {
            "time_s": float(snapshot["time_s"]),
            "mechanical_loading_active": bool(
                snapshot["mechanical_loading_active"]),
            "qualifying_segment_labels": qualifying,
            "ordered_line_overlap": audit["ordered_line_overlap"],
            "local_integrated_audit": audit_record,
        }
        record["snapshot_qualified"] = bool(
            qualifying and audit["ordered_line_overlap"] >= minimum_ordered_overlap)
        records.append(record)
    records.sort(key=lambda item: item["time_s"])
    loaded = [item for item in records if item["mechanical_loading_active"]]
    released = [item for item in records if not item["mechanical_loading_active"]]
    release_duration = 0.0
    if released and all(item["snapshot_qualified"] for item in released):
        release_duration = released[-1]["time_s"]-released[0]["time_s"]
    passed = bool(
        loaded and any(item["snapshot_qualified"] for item in loaded)
        and released and all(item["snapshot_qualified"] for item in released)
        and release_duration >= required_release_persistence_s)
    return {
        "records": records,
        "loaded_wall_observed": bool(
            any(item["snapshot_qualified"] for item in loaded)),
        "release_duration_s": float(release_duration),
        "required_release_persistence_s": float(required_release_persistence_s),
        "fixture_passed": bool(records),
        "scientific_gate_passed": passed,
        "classification": (
            "PERSISTENT_LOCAL_INTEGRATED_LAGB_PRECURSOR" if passed else
            "NO_PERSISTENT_LOCAL_INTEGRATED_LAGB_PRECURSOR"),
        "grain_labels_allocated": 0,
    }


def pixel_unit(vector):
    """Named helper preventing accidental metre/index mixing in sampling."""
    value = np.asarray(vector, dtype=float)
    return value/max(float(np.linalg.norm(value)), 1e-300)
