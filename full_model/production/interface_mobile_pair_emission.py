"""Conservative neutral-pair emission from explicit GB mobile reservoirs.

This atomic projection is default-off.  It transfers only equal positive and
negative family content from the mobile-channel row of an existing interface
reservoir into its two incident lattice owners.  The emitted pair therefore
has zero net Burgers vector.  Its conjugate grain-spanning slip updates beta_p
and Nye through the authoritative kinematic map.  Complete-energy pricing is
required before a caller may publish the candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .tensorial_nye import (
    TensorialKinematicState, accept_slip_increment,
    nye_from_plastic_distortion,
)


@dataclass(frozen=True)
class InterfaceMobilePairEmissionProjection:
    candidate_state: object
    removed_interface_pair_cell_sum_m2: float
    created_lattice_pair_cell_sum_m2: float
    line_balance_residual_cell_sum_m2: float
    signed_content_change_cell_sum_m2: float
    maximum_mura_relative_residual: float
    emitted_pair_density_m2_by_cell: np.ndarray


def project_interface_mobile_pair_emission(
        state, *, fraction, resolved_stress_by_owner, systems, spacing_m,
        burgers_m, source_grain_size_m, emission_weight_by_owner=None,
        preserve_owner_mean_orientation=True):
    """Transfer neutral interface pairs to incident mobile populations.

    Interface row zero is the declared mobile-line channel.  At every cell and
    family, at most ``min(boundary_plus, boundary_minus)`` is transferable.
    The intensive owner increment is chosen so its support-weighted extensive
    sum equals the interface debit exactly.
    """
    state.validate()
    spacing = float(spacing_m)
    burgers = float(burgers_m)
    grain_size = float(source_grain_size_m)
    if any(not math.isfinite(x) or x <= 0.0
           for x in (spacing, burgers, grain_size)):
        raise ValueError("spacing, Burgers magnitude, and grain size must be positive")
    stresses = tuple(np.asarray(value, dtype=float)
                     for value in resolved_stress_by_owner)
    if (len(stresses) != len(state.owners)
            or any(value.shape != state.owners[0].mobile_plus_m2.shape
                   or np.any(~np.isfinite(value)) for value in stresses)):
        raise ValueError("resolved stresses require owner x grid x family")
    raw_extent = np.asarray(fraction, dtype=float)
    target_shape = ((len(state.interfaces),)
                    +state.owners[0].mobile_plus_m2.shape)
    if raw_extent.ndim == 0:
        extents = np.full(target_shape, float(raw_extent))
    else:
        try:
            extents = np.broadcast_to(raw_extent, target_shape)
        except ValueError as exc:
            raise ValueError(
                "emission fraction must broadcast to interface x grid x family") from exc
    if np.any(~np.isfinite(extents)) or np.any((extents < 0.0)|(extents > 1.0)):
        raise ValueError("emission fraction must lie in [0,1]")
    if emission_weight_by_owner is None:
        weights = tuple(np.ones_like(value) for value in stresses)
    else:
        weights = tuple(np.asarray(value, dtype=float)
                        for value in emission_weight_by_owner)
        if (len(weights) != len(state.owners)
                or any(value.shape != stresses[0].shape
                       or np.any(~np.isfinite(value)) or np.any(value < 0.0)
                       for value in weights)):
            raise ValueError("emission weights require nonnegative owner x grid x family")

    owner_index = {grain: i for i, grain in enumerate(state.grain_ids)}
    additions = [np.zeros_like(owner.mobile_plus_m2)
                 for owner in state.owners]
    next_interfaces = []
    emitted_by_cell = np.zeros(state.supports.shape[1:])
    removed_pairs = 0.0
    for edge_index, component in enumerate(state.interfaces):
        ia = owner_index[component.grain_a_id]
        ib = owner_index[component.grain_b_id]
        plus = np.asarray(component.boundary_plus_m2, dtype=float)
        minus = np.asarray(component.boundary_minus_m2, dtype=float)
        eta_a = np.asarray(state.supports[ia])[..., None]
        eta_b = np.asarray(state.supports[ib])[..., None]
        weighted_a = eta_a*weights[ia]
        weighted_b = eta_b*weights[ib]
        weight_sum = weighted_a+weighted_b
        active = weight_sum > 256.0*np.finfo(float).eps
        neutral = extents[edge_index]*np.minimum(
            plus[..., 0, :], minus[..., 0, :])
        neutral = np.where(active, neutral, 0.0)
        extensive_a = np.divide(
            neutral*weighted_a, weight_sum, out=np.zeros_like(neutral),
            where=active)
        extensive_b = neutral-extensive_a
        additions[ia] += np.divide(
            extensive_a, eta_a, out=np.zeros_like(neutral), where=eta_a > 0.0)
        additions[ib] += np.divide(
            extensive_b, eta_b, out=np.zeros_like(neutral), where=eta_b > 0.0)
        next_plus = plus.copy()
        next_minus = minus.copy()
        next_plus[..., 0, :] -= neutral
        next_minus[..., 0, :] -= neutral
        next_interfaces.append(replace(
            component, boundary_plus_m2=next_plus,
            boundary_minus_m2=next_minus))
        removed_pairs += float(np.sum(neutral, dtype=np.longdouble))
        emitted_by_cell += np.sum(neutral, axis=2)

    if removed_pairs == 0.0:
        return InterfaceMobilePairEmissionProjection(
            state, 0.0, 0.0, 0.0, 0.0, 0.0, emitted_by_cell)

    owners = []
    created_pairs = 0.0
    signed_change = 0.0
    maximum_mura = 0.0
    for support, owner, stress, addition in zip(
            state.supports, state.owners, stresses, additions):
        created_pairs += float(np.sum(
            np.asarray(support)[..., None]*addition,
            dtype=np.longdouble))
        # A pair density 1/d^2 corresponds to one grain-spanning source event;
        # the conjugate shear per event is b/d.
        slip_increment = np.sign(stress)*addition*burgers*grain_size
        kinematics = TensorialKinematicState(
            np.asarray(owner.slip), np.asarray(owner.beta_p),
            np.asarray(owner.alignment_m2), np.asarray(owner.family_nye_m1))
        updated = accept_slip_increment(
            kinematics, slip_increment, systems,
            np.asarray(owner.orientation_rad), spacing)
        dbeta = updated.beta_p-np.asarray(owner.beta_p)
        orientation = (np.asarray(owner.orientation_rad)
                       +.5*(dbeta[..., 1, 0]-dbeta[..., 0, 1]))
        if preserve_owner_mean_orientation:
            orientation += (float(np.mean(owner.orientation_rad))
                            -float(np.mean(orientation)))
        next_owner = replace(
            owner,
            mobile_plus_m2=np.asarray(owner.mobile_plus_m2)+addition,
            mobile_minus_m2=np.asarray(owner.mobile_minus_m2)+addition,
            slip=updated.slip, beta_p=updated.beta_p,
            family_nye_m1=updated.family_nye_m1,
            alignment_m2=updated.alignment_m2,
            orientation_rad=orientation)
        owners.append(next_owner)
        signed_change += float(np.sum(
            np.asarray(support)[..., None]*(
                (next_owner.mobile_plus_m2-next_owner.mobile_minus_m2)
                -(owner.mobile_plus_m2-owner.mobile_minus_m2)),
            dtype=np.longdouble))
        beta_nye_increment = nye_from_plastic_distortion(dbeta, spacing)
        family_nye_increment = np.sum(
            updated.family_nye_m1-np.asarray(owner.family_nye_m1), axis=2)
        mura_residual = family_nye_increment-beta_nye_increment
        mura_scale = max(float(np.sqrt(np.mean(
            beta_nye_increment*beta_nye_increment))), 1.0)
        maximum_mura = max(
            maximum_mura,
            float(np.sqrt(np.mean(mura_residual*mura_residual))/mura_scale))

    candidate = replace(
        state, owners=tuple(owners), interfaces=tuple(next_interfaces))
    candidate.validate()
    line_balance = 2.0*(created_pairs-removed_pairs)
    return InterfaceMobilePairEmissionProjection(
        candidate, removed_pairs, created_pairs, line_balance, signed_change,
        maximum_mura, emitted_by_cell)
