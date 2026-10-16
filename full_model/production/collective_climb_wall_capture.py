"""Quarantined finite collective-climb capture path for V67 discrimination.

The local closed-ring event can only relabel mobile line already on its ring.
This module tests the distinct physical hypothesis that polarized mobile line
inside a prospective cell collectively climbs/glides to its boundary.  Each
sign and Burgers-family inventory is transported conservatively; no line,
orientation target, phase support, or grain label is created.

This is not enabled by the production interval.  Promotion requires a
downhill complete-energy path, an Arrhenius kinetic law, and the usual balance
and restart qualification.
"""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from .orientation_wall_compatibility import (
    ordered_wall_nye_from_common, reconstruct_orientation_from_ordered_nye,
)


def _transport_to_receiver(source, opposite, support, donor, receiver,
                           coordinate):
    """Move polarized source content to a receiver at fixed inventory."""
    excess = np.maximum(np.asarray(source)-np.asarray(opposite), 0.0)
    removed = float(coordinate)*np.asarray(donor)[..., None]*excess
    weight = np.asarray(support, dtype=float)
    receiver_weight = np.asarray(receiver, dtype=float)
    denominator = float(np.sum(weight*receiver_weight, dtype=np.longdouble))
    if denominator <= 0.0:
        return np.zeros_like(source), np.zeros_like(source)
    inventory = np.sum(
        weight[..., None]*removed, axis=(0, 1), dtype=np.longdouble)
    added = (receiver_weight[..., None]
             *np.asarray(inventory, dtype=float)[None, None, :]/denominator)
    return removed, added


def collective_climb_wall_capture_candidate(
        state, owner_index, donor, receiver, coordinate, order_increment):
    """Transport core mobile excess to an ordered wall receiver.

    ``donor`` and ``receiver`` are bounded dimensionless spatial weights.  The
    returned scalar is the support-weighted line-cell inventory moved into the
    wall reservoir.
    """
    lam = float(coordinate)
    if not math.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("collective climb coordinate must lie in [0,1]")
    donor = np.asarray(donor, dtype=float)
    receiver = np.asarray(receiver, dtype=float)
    if (donor.shape != state.supports.shape[1:]
            or receiver.shape != donor.shape
            or np.any(~np.isfinite(donor)) or np.any(~np.isfinite(receiver))
            or np.any(donor < 0.0) or np.any(donor > 1.0)
            or np.any(receiver < 0.0)):
        raise ValueError("collective climb masks are invalid")
    owner = state.owners[int(owner_index)]
    support = np.asarray(state.supports[int(owner_index)], dtype=float)
    remove_plus, add_plus = _transport_to_receiver(
        owner.mobile_plus_m2, owner.mobile_minus_m2, support, donor, receiver,
        lam)
    remove_minus, add_minus = _transport_to_receiver(
        owner.mobile_minus_m2, owner.mobile_plus_m2, support, donor, receiver,
        lam)
    normalized_receiver = receiver/max(float(np.max(receiver)), 1e-300)
    order = np.asarray(owner.wall_order)+(
        lam*float(order_increment)*normalized_receiver
        *(1.0-np.asarray(owner.wall_order)))
    changed = replace(
        owner,
        mobile_plus_m2=np.asarray(owner.mobile_plus_m2)-remove_plus,
        mobile_minus_m2=np.asarray(owner.mobile_minus_m2)-remove_minus,
        wall_plus_m2=np.asarray(owner.wall_plus_m2)+add_plus,
        wall_minus_m2=np.asarray(owner.wall_minus_m2)+add_minus,
        wall_order=order)
    owners = list(state.owners)
    owners[int(owner_index)] = changed
    candidate = replace(state, owners=tuple(owners))
    candidate.validate()
    moved = float(np.sum(
        support[..., None]*(add_plus+add_minus), dtype=np.longdouble))
    return candidate, moved


def collective_climb_wall_orientation_candidate(
        state, owner_index, donor, receiver, coordinate, order_increment,
        systems, spacing_m, domain):
    """Add only the Nye-conjugate orientation increment of captured line."""
    wall, moved = collective_climb_wall_capture_candidate(
        state, owner_index, donor, receiver, coordinate, order_increment)
    old = state.owners[int(owner_index)]
    changed = wall.owners[int(owner_index)]
    active = np.asarray(domain, dtype=bool)
    if np.count_nonzero(active) < 2:
        return wall, moved, 0.0
    old_reconstructed, _ = reconstruct_orientation_from_ordered_nye(
        ordered_wall_nye_from_common(old, systems), old.orientation_rad,
        active, spacing_m)
    new_reconstructed, _ = reconstruct_orientation_from_ordered_nye(
        ordered_wall_nye_from_common(changed, systems), old.orientation_rad,
        active, spacing_m)
    increment = np.arctan2(
        np.sin(new_reconstructed-old_reconstructed),
        np.cos(new_reconstructed-old_reconstructed))
    orientation = np.asarray(old.orientation_rad).copy()
    orientation[active] += increment[active]
    orientation += float(np.mean(old.orientation_rad)-np.mean(orientation))
    owners = list(wall.owners)
    owners[int(owner_index)] = replace(changed, orientation_rad=orientation)
    candidate = replace(wall, owners=tuple(owners))
    candidate.validate()
    rotation = float(np.max(np.abs(
        orientation-np.asarray(old.orientation_rad))))
    return candidate, moved, rotation
