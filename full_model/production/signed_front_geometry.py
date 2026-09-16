"""Authoritative signed geometry for a resolved two-phase front.

The convention is deliberately independent of phase-profile width and of any
kinetic law.  ``phi = eta_receiver - eta_donor``; therefore ``phi > 0`` is
receiver material.  A positive reported area means that receiver material has
grown.  Crossings are located by linear interpolation and matched by their
orientation on each normal ray.  Periodic displacement uses the minimum image.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np


@dataclass(frozen=True)
class SignedFrontMeasure:
    signed_receiver_area_m2: float
    receiver_gain_area_m2: float
    donor_gain_area_m2: float
    signed_receiver_volume_m3: float
    receiver_gain_volume_m3: float
    donor_gain_volume_m3: float
    crossing_count: int
    maximum_abs_displacement_m: float
    displacements_m: tuple[float, ...]
    crossing_orientations: tuple[int, ...]


def _wrapped_delta(value: float, period: float) -> float:
    """Minimum-image delta in ``[-period/2, period/2)``."""
    return float((value + 0.5 * period) % period - 0.5 * period)


def _ray_crossings(values: np.ndarray, valid: np.ndarray, periodic: bool):
    n = values.size
    stop = n if periodic else n - 1
    result = []
    for i in range(stop):
        j = (i + 1) % n
        if not (valid[i] and valid[j]):
            continue
        a, b = float(values[i]), float(values[j])
        # A sampled zero belongs to its outgoing edge only.  This avoids
        # double counting without moving the zero contour.
        if a == 0.0:
            if b == 0.0:
                continue
            before = float(values[(i - 1) % n]) if (periodic or i > 0) else -b
            orientation = 1 if b > before else -1
            result.append((float(i), orientation))
        elif a * b < 0.0:
            # The equilibrium Allen--Cahn interface is tanh-shaped.  Its
            # inverse-profile coordinate is affine through the interface, so
            # interpolation there makes a fixed zero invariant to diffuse
            # width.  This is also sign preserving for arbitrary bounded
            # phase differences.  Values outside the phase simplex are not
            # accepted by the public routine.
            eps = 8.0 * np.finfo(float).eps
            qa = float(np.arctanh(np.clip(a, -1.0 + eps, 1.0 - eps)))
            qb = float(np.arctanh(np.clip(b, -1.0 + eps, 1.0 - eps)))
            fraction = abs(qa) / (abs(qa) + abs(qb))
            result.append(((i + fraction) % n, 1 if b > a else -1))
    return result


def _match(reference, current, period):
    if len(reference) != len(current):
        raise ValueError("front topology changed: crossing counts differ")
    if not reference:
        return []
    best = None
    for order in permutations(range(len(current))):
        deltas = tuple(_wrapped_delta(current[j] - reference[i], period)
                       for i, j in enumerate(order))
        cost = sum(d * d for d in deltas)
        candidate = (cost, deltas)
        if best is None or candidate < best:
            best = candidate
    return list(best[1])


def measure_signed_front_motion(reference_phi, current_phi, *, normal_axis,
                                spacing_m, represented_thickness_m=1.0,
                                active_mask=None, periodic=True):
    """Measure signed donor/receiver passage from matched zero contours.

    ``normal_axis`` selects the ray direction; the other grid spacing supplies
    the tributary width.  The active mask must resolve the same crossing
    topology in both states.  Width relaxation at fixed zero contours is
    exactly invisible to this measure.
    """
    ref = np.asarray(reference_phi, dtype=float)
    cur = np.asarray(current_phi, dtype=float)
    axis = int(normal_axis)
    dx = float(spacing_m)
    thickness = float(represented_thickness_m)
    if (ref.ndim != 2 or cur.shape != ref.shape or axis not in (0, 1)
            or not np.all(np.isfinite(ref)) or not np.all(np.isfinite(cur))
            or np.max(np.abs(ref)) > 1.0 + 32.0*np.finfo(float).eps
            or np.max(np.abs(cur)) > 1.0 + 32.0*np.finfo(float).eps
            or not np.isfinite(dx) or dx <= 0.0
            or not np.isfinite(thickness) or thickness <= 0.0):
        raise ValueError("signed-front inputs are invalid")
    mask = (np.ones_like(ref, dtype=bool) if active_mask is None
            else np.asarray(active_mask, dtype=bool))
    if mask.shape != ref.shape:
        raise ValueError("active mask shape differs from phase fields")
    ray_ref = np.moveaxis(ref, axis, 0)
    ray_cur = np.moveaxis(cur, axis, 0)
    ray_mask = np.moveaxis(mask, axis, 0)
    period = float(ref.shape[axis])
    signed_displacements = []
    orientations = []
    raw_displacements = []
    for j in range(ray_ref.shape[1]):
        r = _ray_crossings(ray_ref[:, j], ray_mask[:, j], bool(periodic))
        c = _ray_crossings(ray_cur[:, j], ray_mask[:, j], bool(periodic))
        for orientation in (-1, 1):
            rp = [x for x, sign in r if sign == orientation]
            cp = [x for x, sign in c if sign == orientation]
            for displacement in _match(rp, cp, period):
                raw_displacements.append(displacement * dx)
                # phi>0 is receiver.  For a -to-+ crossing, motion in the
                # positive coordinate direction consumes receiver material.
                signed_displacements.append(-orientation * displacement * dx)
                orientations.append(orientation)
    signed_area = float(np.sum(signed_displacements, dtype=np.longdouble) * dx)
    gain = max(signed_area, 0.0)
    loss = max(-signed_area, 0.0)
    return SignedFrontMeasure(
        signed_receiver_area_m2=signed_area,
        receiver_gain_area_m2=gain,
        donor_gain_area_m2=loss,
        signed_receiver_volume_m3=signed_area * thickness,
        receiver_gain_volume_m3=gain * thickness,
        donor_gain_volume_m3=loss * thickness,
        crossing_count=len(raw_displacements),
        maximum_abs_displacement_m=max(map(abs, raw_displacements), default=0.0),
        displacements_m=tuple(float(x) for x in raw_displacements),
        crossing_orientations=tuple(orientations))
