"""One-owner adapter for common Mura state and a moving material front.

The legacy sparse-front object is retained as a geometry and compatibility
view.  It is not an independent constitutive state when this adapter is
active.  All phase-supported dislocation and kinematic fields live here and
are committed together after an accepted Mura/front proposal.

Quantities ending in ``_m2`` are line density (m^-2), ``beta_p`` and slip are
dimensionless, Nye is m^-1, and temperature is K.  Owner fields are intensive;
their support-weighted products are the extensive cell inventories.  A phase
with zero support therefore owns no hidden extensive line content.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import json

import numpy as np

try:
    from .common_tensorial_wall import CommonWallState
    from .density_state_map import DensityInventory, SIGNED_RESERVOIRS
    from .moving_front import DefectState, SparseFrontState
    from .tensorial_nye import nye_from_plastic_distortion
    from .wall_topology_supply import (
        ReservoirAlignmentState, TOPOLOGY_MOMENT_FIELDS,
    )
except ImportError:  # pragma: no cover - production script execution
    from common_tensorial_wall import CommonWallState
    from density_state_map import DensityInventory, SIGNED_RESERVOIRS
    from moving_front import DefectState, SparseFrontState
    from tensorial_nye import nye_from_plastic_distortion
    from wall_topology_supply import (
        ReservoirAlignmentState, TOPOLOGY_MOMENT_FIELDS,
    )


LEGACY_SCHEMA = "full-v34-common-front-authoritative-state/v1"
SCHEMA = "full-v35-common-front-authoritative-state/v2"

LINE_FIELDS = (
    "mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
    "forest_minus_m2", "wall_plus_m2", "wall_minus_m2", "junction_m2")
BOUNDED_FIELDS = ("wall_order", "multi_hit_coordination")
KINEMATIC_FIELDS = ("slip", "beta_p", "alignment_m2")


@dataclass(frozen=True)
class CommonFrontLedger:
    attempted_commits: int = 0
    accepted_commits: int = 0
    rejected_commits: int = 0
    processed_line_m: float = 0.0
    transmitted_line_m: float = 0.0
    boundary_line_m: float = 0.0
    annihilated_line_m: float = 0.0
    sink_line_m: float = 0.0
    maximum_line_closure_m: float = 0.0
    maximum_signed_closure_m2: float = 0.0
    interface_nye_norm_m1: float = 0.0
    complete_energy_accepted: int = 0
    generated_heat_J: float = 0.0
    thermostat_export_J: float = 0.0
    material_sink_export_J: float = 0.0
    maximum_abs_first_law_residual_J: float = 0.0


@dataclass(frozen=True)
class CommonFrontState:
    front: SparseFrontState
    parent: CommonWallState
    child: CommonWallState
    wake: CommonWallState
    boundary_plus_m2: np.ndarray
    boundary_minus_m2: np.ndarray
    boundary_junction_m2: np.ndarray
    interface_nye_m1: np.ndarray
    ledger: CommonFrontLedger = CommonFrontLedger()
    parent_density: DensityInventory | None = None
    child_density: DensityInventory | None = None
    wake_density: DensityInventory | None = None
    parent_alignment: ReservoirAlignmentState | None = None
    child_alignment: ReservoirAlignmentState | None = None
    wake_alignment: ReservoirAlignmentState | None = None


def has_reservoir_moment_owners(state):
    values = (
        state.parent_density, state.child_density, state.wake_density,
        state.parent_alignment, state.child_alignment, state.wake_alignment)
    if any(value is None for value in values):
        if not all(value is None for value in values):
            raise ValueError("partial reservoir-moment owner state")
        return False
    return True


def attach_reservoir_moment_owners(
        state, parent_density, child_density, wake_density,
        parent_alignment, child_alignment, wake_alignment,
        systems, topologies):
    """Attach a declared initialization map for all V24 reservoir moments."""
    owners = ((parent_density, parent_alignment),
              (child_density, child_alignment),
              (wake_density, wake_alignment))
    for density, alignment in owners:
        density.validate(len(systems), len(topologies))
        alignment.validate(density, len(systems))
    candidate = replace(
        state, parent_density=parent_density, child_density=child_density,
        wake_density=wake_density, parent_alignment=parent_alignment,
        child_alignment=child_alignment, wake_alignment=wake_alignment)
    _validate_owner_density_common(candidate)
    return candidate


def _validate_owner_density_common(state, tolerance=64*np.finfo(float).eps):
    if not has_reservoir_moment_owners(state):
        return True
    for owner_name, common, density in zip(
            ("parent", "child", "wake"),
            (state.parent, state.child, state.wake),
            (state.parent_density, state.child_density, state.wake_density)):
        pairs = (
            ("mobile_plus_m2", common.mobile_plus_m2, density.mobile_plus_m2),
            ("mobile_minus_m2", common.mobile_minus_m2, density.mobile_minus_m2),
            ("forest_plus_m2", common.forest_plus_m2, density.forest_plus_m2),
            ("forest_minus_m2", common.forest_minus_m2, density.forest_minus_m2),
            ("wall_plus_m2", common.wall_plus_m2,
             density.wall_tangle_plus_m2+density.wall_ordered_plus_m2),
            ("wall_minus_m2", common.wall_minus_m2,
             density.wall_tangle_minus_m2+density.wall_ordered_minus_m2),
            ("junction_m2", common.junction_m2, density.junction_m2))
        for field_name, left, right in pairs:
            scale = max(float(np.max(np.abs(left))),
                        float(np.max(np.abs(right))), 1.0)
            maximum = float(np.max(np.abs(left-right)))
            if maximum > tolerance*scale:
                raise RuntimeError(
                    "common and reservoir owner densities diverged: "
                    f"owner={owner_name}, field={field_name}, "
                    f"maximum_abs={maximum:.17g}, scale={scale:.17g}, "
                    f"relative={maximum/scale:.17g}, tolerance={tolerance:.17g}")
    return True


def _common_with_authoritative_density_views(common, density):
    """Rebuild scalar compatibility views in the density ledger's sum order.

    The split tangle/ordered reservoirs are authoritative.  Repeatedly
    advancing their already-summed wall view independently accumulates a
    cancellation-sized difference, so publish the exact compatibility view
    rather than loosening the owner-consistency invariant.
    """
    return replace(
        common,
        mobile_plus_m2=np.asarray(density.mobile_plus_m2).copy(),
        mobile_minus_m2=np.asarray(density.mobile_minus_m2).copy(),
        forest_plus_m2=np.asarray(density.forest_plus_m2).copy(),
        forest_minus_m2=np.asarray(density.forest_minus_m2).copy(),
        wall_plus_m2=(np.asarray(density.wall_tangle_plus_m2)
                      +np.asarray(density.wall_ordered_plus_m2)),
        wall_minus_m2=(np.asarray(density.wall_tangle_minus_m2)
                       +np.asarray(density.wall_ordered_minus_m2)),
        junction_m2=np.asarray(density.junction_m2).copy())


def _copy_state(state):
    return CommonWallState(**{
        item.name: np.array(getattr(state, item.name), copy=True)
        for item in fields(CommonWallState)})


def _zero_line_fields(state, support):
    values = {}
    for item in fields(CommonWallState):
        value = np.array(getattr(state, item.name), copy=True)
        if item.name in LINE_FIELDS:
            weight = support[(...,)+(None,)*(value.ndim-support.ndim)]
            value = np.where(weight > 0.0, value, 0.0)
        values[item.name] = value
    return CommonWallState(**values)


def initialize_common_front(front, common):
    """Map one common state to phase owners without changing its mixture."""
    parent_w, child_w, wake_w = front.material_support_weights()
    grid = parent_w.shape
    nf = common.mobile_plus_m2.shape[2]
    nj = common.junction_m2.shape[2]
    def owner_from_defect(defect, support):
        forest_common = common.forest_plus_m2+common.forest_minus_m2
        wall_common = np.sum(
            common.wall_plus_m2+common.wall_minus_m2, axis=2)
        fp_fraction = np.divide(
            common.forest_plus_m2,
            forest_common,
            out=np.full_like(common.forest_plus_m2, .5),
            where=(common.forest_plus_m2+common.forest_minus_m2) > 0.0)
        wall_total = np.sum(
            common.wall_plus_m2+common.wall_minus_m2, axis=2, keepdims=True)
        wp_fraction = np.divide(common.wall_plus_m2, wall_total,
                                out=np.full_like(common.wall_plus_m2, .5/nf),
                                where=wall_total > 0.0)
        wm_fraction = np.divide(common.wall_minus_m2, wall_total,
                                out=np.full_like(common.wall_minus_m2, .5/nf),
                                where=wall_total > 0.0)
        owner = _copy_state(common)
        forest_plus = np.asarray(defect.forest)*fp_fraction
        forest_minus = np.asarray(defect.forest)-forest_plus
        if np.array_equal(np.asarray(defect.forest), forest_common):
            forest_plus = common.forest_plus_m2.copy()
            forest_minus = common.forest_minus_m2.copy()
        wall_plus = np.asarray(defect.wall)[..., None]*wp_fraction
        wall_minus = np.asarray(defect.wall)[..., None]*wm_fraction
        if np.array_equal(np.asarray(defect.wall), wall_common):
            wall_plus = common.wall_plus_m2.copy()
            wall_minus = common.wall_minus_m2.copy()
        owner = CommonWallState(
            np.asarray(defect.rp).copy(), np.asarray(defect.rm).copy(),
            forest_plus, forest_minus, wall_plus, wall_minus,
            owner.junction_m2, owner.wall_order,
            owner.multi_hit_coordination, owner.slip, owner.beta_p,
            owner.alignment_m2, owner.family_nye_m1,
            owner.orientation_rad, owner.temperature_K)
        return _zero_line_fields(owner, support)
    return CommonFrontState(
        front,
        owner_from_defect(front.parent, parent_w),
        owner_from_defect(front.child, child_w),
        owner_from_defect(front.recovered_wake, wake_w),
        np.zeros(grid+(3, nf)), np.zeros(grid+(3, nf)),
        np.zeros(grid+(nj,)), np.zeros(grid+(3, 3)), CommonFrontLedger())


def _weighted(values, weights):
    result = np.zeros_like(np.asarray(values[0]), dtype=float)
    for value, weight2 in zip(values, weights):
        weight = weight2[(...,)+(None,)*(np.asarray(value).ndim-weight2.ndim)]
        result += weight*np.asarray(value)
    return result


def reconstruct_common(state, spacing_m):
    """Return the common view; Nye is the curl of the reconstructed beta.

    Computing the curl after support weighting includes the discrete product
    rule/interface term.  ``interface_nye_m1`` records the difference from a
    support-weighted sum of owner Nye tensors, so intrinsic HAGB energy need
    not be charged as plastic excess a second time.
    """
    weights = state.front.material_support_weights()
    owners = (state.parent, state.child, state.wake)
    values = {}
    for item in fields(CommonWallState):
        if item.name == "family_nye_m1":
            continue
        values[item.name] = _weighted(
            [getattr(owner, item.name) for owner in owners], weights)
    total_nye = nye_from_plastic_distortion(values["beta_p"], spacing_m)
    bulk_family = _weighted([owner.family_nye_m1 for owner in owners], weights)
    bulk_total = np.sum(bulk_family, axis=2)
    correction = total_nye-bulk_total
    magnitude = np.abs(values["slip"])
    fraction = np.divide(magnitude, np.sum(magnitude, axis=2, keepdims=True),
                         out=np.full_like(magnitude, 1.0/magnitude.shape[2]),
                         where=np.sum(magnitude, axis=2, keepdims=True) > 0.0)
    values["family_nye_m1"] = bulk_family+fraction[..., None, None]*correction[..., None, :, :]
    return CommonWallState(**values), correction


def _weighted_dataclass(values, weights, cls):
    return cls(**{
        item.name: _weighted(
            [np.asarray(getattr(value, item.name)) for value in values],
            weights)
        for item in fields(cls)})


def reconstruct_reservoir_moments(state):
    """Support-weight the owned scalar reservoirs and first moments exactly."""
    if not has_reservoir_moment_owners(state):
        raise RuntimeError("reservoir-resolved moment owners are not initialized")
    weights = state.front.material_support_weights()
    density = _weighted_dataclass(
        (state.parent_density, state.child_density, state.wake_density),
        weights, DensityInventory)
    alignment = _weighted_dataclass(
        (state.parent_alignment, state.child_alignment, state.wake_alignment),
        weights, ReservoirAlignmentState)
    return density, alignment


def reconstruct_mechanical_state(state, spacing_m, systems, topologies):
    """Return the next Mura state without Nye inversion or moment reset."""
    try:
        from .v24_mechanical_wall import V24MechanicalWallState
    except ImportError:  # pragma: no cover - direct script execution
        from v24_mechanical_wall import V24MechanicalWallState
    common, _ = reconstruct_common(state, spacing_m)
    density, alignment = reconstruct_reservoir_moments(state)
    # The scalar common reservoirs are compatibility views of the authoritative
    # split inventory.  Build those views in the same floating-point order as
    # V24 validation rather than relying on distributivity across two weighted
    # sums (which can differ by one ulp after repeated cycles).
    common = replace(
        common,
        mobile_plus_m2=density.mobile_plus_m2,
        mobile_minus_m2=density.mobile_minus_m2,
        forest_plus_m2=density.forest_plus_m2,
        forest_minus_m2=density.forest_minus_m2,
        wall_plus_m2=(density.wall_tangle_plus_m2
                      +density.wall_ordered_plus_m2),
        wall_minus_m2=(density.wall_tangle_minus_m2
                       +density.wall_ordered_minus_m2),
        junction_m2=density.junction_m2)
    result = V24MechanicalWallState(common, density, alignment)
    result.validate(systems, topologies)
    return result


def _support_delta(owner, old_mix, new_mix, support, name):
    value = np.asarray(getattr(owner, name))
    old = np.asarray(getattr(old_mix, name)); new = np.asarray(getattr(new_mix, name))
    weight = support[(...,)+(None,)*(value.ndim-support.ndim)]
    active = weight > 64.0*np.finfo(float).eps
    if name in LINE_FIELDS:
        ratio = np.divide(new, old, out=np.zeros_like(new), where=old > 0.0)
        candidate = np.where(new < old, value*ratio, value+np.maximum(new-old, 0.0))
        # Once initialized, a temporarily vanishing support retains its
        # intensive owner history for a later retreat/revisit.  Its extensive
        # contribution remains exactly zero because its support weight is zero.
        return np.where(active, np.maximum(candidate, 0.0), value)
    candidate = value+(new-old)
    if name in BOUNDED_FIELDS:
        candidate = np.clip(candidate, 0.0, 1.0)
    if name == "temperature_K":
        candidate = np.maximum(candidate, 1.0)
    return np.where(active, candidate, value)


def apply_common_increment(state, updated, spacing_m):
    """Project one accepted Mura increment into supported material owners."""
    old, _ = reconstruct_common(state, spacing_m)
    weights = state.front.material_support_weights()
    owners = []
    for owner, support in zip((state.parent, state.child, state.wake), weights):
        arrays = {}
        for item in fields(CommonWallState):
            if item.name == "family_nye_m1":
                arrays[item.name] = np.asarray(owner.family_nye_m1).copy()
            else:
                arrays[item.name] = _support_delta(
                    owner, old, updated, support, item.name)
        owners.append(CommonWallState(**arrays))
    candidate = replace(state, parent=owners[0], child=owners[1], wake=owners[2])
    # The accepted operator's family decomposition is distributed as an
    # increment; reconstruction subsequently adds the phase product rule.
    old_family = old.family_nye_m1
    increment = np.asarray(updated.family_nye_m1)-old_family
    fixed = []
    for owner, support in zip(owners, weights):
        active = support[..., None, None, None] > 64.0*np.finfo(float).eps
        fixed.append(replace(owner, family_nye_m1=np.where(
            active, owner.family_nye_m1+increment, owner.family_nye_m1)))
    candidate = replace(candidate, parent=fixed[0], child=fixed[1], wake=fixed[2])
    check, correction = reconstruct_common(candidate, spacing_m)
    # Every field except Nye (which intentionally gains the interface curl)
    # must reconstruct the accepted state to floating-point tolerance.
    for item in fields(CommonWallState):
        if item.name == "family_nye_m1":
            continue
        actual = np.asarray(getattr(check, item.name))
        expected = np.asarray(getattr(updated, item.name))
        scale = max(float(np.max(np.abs(expected))), 1.0)
        if float(np.max(np.abs(actual-expected))) > 1024*np.finfo(float).eps*scale:
            raise RuntimeError(f"common-front projection failed for {item.name}")
    return _sync_front_views(replace(candidate, interface_nye_m1=correction))


def _density_support_delta(owner, old_mix, new_mix, support, name):
    value = np.asarray(getattr(owner, name))
    old = np.asarray(getattr(old_mix, name))
    new = np.asarray(getattr(new_mix, name))
    weight = support[(...,)+(None,)*(value.ndim-support.ndim)]
    active = weight > 64.0*np.finfo(float).eps
    ratio = np.divide(new, old, out=np.zeros_like(new), where=old > 0.0)
    candidate = np.where(new < old, value*ratio,
                         value+np.maximum(new-old, 0.0))
    return np.where(active, np.maximum(candidate, 0.0), value)


def _moment_support_delta(owner, old_mix, new_mix, support, name,
                          nonnegative=False):
    value = np.asarray(getattr(owner, name))
    old = np.asarray(getattr(old_mix, name))
    new = np.asarray(getattr(new_mix, name))
    weight = support[(...,)+(None,)*(value.ndim-support.ndim)]
    active = weight > 64.0*np.finfo(float).eps
    candidate = value+(new-old)
    if nonnegative:
        candidate = np.maximum(candidate, 0.0)
    return np.where(active, candidate, value)


def _reference_minimum_change_bounded_moments(baselines, bounds, weights,
                                               target):
    """Project owner moments onto exact mixture and line-length constraints.

    This alternating projection starts from each owner's density-scaled prior
    moment.  It therefore preserves owner history whenever the accepted common
    moment leaves freedom, while handling the unique fully polarized limit
    without clipping or manufacturing a direction from Nye.
    """
    kappa = np.stack(baselines, axis=0)
    radius = np.stack(bounds, axis=0)[..., None]
    weight = np.stack(weights, axis=0)
    w = weight[(...,)+(None,)*(kappa.ndim-weight.ndim)]
    active = w > 64.0*np.finfo(float).eps
    denominator = np.sum(w*w, axis=0)
    scale = max(float(np.max(np.abs(target))), 1.0)
    for _ in range(256):
        residual = target-np.sum(w*kappa, axis=0)
        if float(np.max(np.abs(residual))) <= 256*np.finfo(float).eps*scale:
            break
        correction = np.divide(
            w*residual[None, ...], denominator[None, ...],
            out=np.zeros_like(kappa), where=denominator[None, ...] > 0.0)
        trial = np.where(active, kappa+correction, kappa)
        norm = np.linalg.norm(trial, axis=-1, keepdims=True)
        factor = np.minimum(
            1.0, np.divide(radius, norm, out=np.ones_like(norm),
                           where=norm > 0.0))
        kappa = np.where(active, trial*factor, kappa)
    residual = target-np.sum(w*kappa, axis=0)
    unresolved = np.max(np.abs(residual), axis=-1, keepdims=True) > (
        2048*np.finfo(float).eps*scale)
    if np.any(unresolved):
        # At full polarization the feasible owner moments are unique and an
        # alternating projection approaches them only asymptotically.  Use
        # that analytical limit locally: every active owner carries the
        # accepted common direction at its own line-length bound.  The target
        # is the accepted reservoir moment itself, never a Nye-derived field.
        total_radius = np.sum(w*radius, axis=0)
        direction = np.divide(
            target, total_radius,
            out=np.zeros_like(target), where=total_radius > 0.0)
        analytical = radius*direction[None, ...]
        mask = unresolved[None, ...]
        kappa = np.where(mask&active, analytical, kappa)
        residual = target-np.sum(w*kappa, axis=0)
    if float(np.max(np.abs(residual))) > 2048*np.finfo(float).eps*scale:
        raise RuntimeError("bounded owner-moment projection did not converge")
    return tuple(kappa[index] for index in range(len(baselines)))


def _minimum_change_bounded_moments(baselines, bounds, weights, target):
    """Equivalent compact solve for exact mixture and line-length constraints.

    The V35 reference map above updates every grid/reservoir point until the
    slowest active constraint converges.  This implementation applies the same
    affine correction and Euclidean-ball projection, but removes converged
    points from subsequent iterations.  It changes neither the admissible set
    nor the full-polarization analytical limit.
    """
    kappa = np.stack(baselines, axis=0).copy()
    radius = np.stack(bounds, axis=0)[..., None]
    weight = np.stack(weights, axis=0)
    w = weight[(...,)+(None,)*(kappa.ndim-weight.ndim)]
    active = w > 64.0*np.finfo(float).eps
    scalar_shape = kappa.shape[:-1]
    w_scalar = np.broadcast_to(w[..., 0], scalar_shape)
    radius_scalar = np.broadcast_to(radius[..., 0], scalar_shape)
    active_scalar = np.broadcast_to(active[..., 0], scalar_shape)
    owners = kappa.shape[0]
    vector_size = kappa.shape[-1]
    flat = kappa.reshape(owners, -1, vector_size)
    flat_w = w_scalar.reshape(owners, -1)
    flat_radius = radius_scalar.reshape(owners, -1)
    flat_active = active_scalar.reshape(owners, -1)
    flat_target = np.asarray(target).reshape(-1, vector_size)
    denominator = np.sum(flat_w*flat_w, axis=0)
    scale = max(float(np.max(np.abs(target))), 1.0)
    tolerance = 256*np.finfo(float).eps*scale
    unresolved = np.arange(flat_target.shape[0])
    for _ in range(256):
        if unresolved.size == 0:
            break
        index = unresolved
        residual = (flat_target[index]
                    -np.sum(flat_w[:, index, None]*flat[:, index, :], axis=0))
        keep = np.max(np.abs(residual), axis=-1) > tolerance
        if not np.any(keep):
            unresolved = np.empty(0, dtype=int)
            break
        index = index[keep]
        residual = residual[keep]
        correction = np.divide(
            flat_w[:, index, None]*residual[None, ...],
            denominator[index][None, :, None],
            out=np.zeros((owners, index.size, vector_size), dtype=flat.dtype),
            where=denominator[index][None, :, None] > 0.0)
        trial = np.where(flat_active[:, index, None],
                         flat[:, index, :]+correction, flat[:, index, :])
        norm = np.linalg.norm(trial, axis=-1)
        factor = np.minimum(
            1.0, np.divide(flat_radius[:, index], norm,
                           out=np.ones_like(norm), where=norm > 0.0))
        flat[:, index, :] = np.where(
            flat_active[:, index, None], trial*factor[..., None],
            flat[:, index, :])
        unresolved = index
    residual = flat_target-np.sum(flat_w[..., None]*flat, axis=0)
    analytical_mask = np.max(np.abs(residual), axis=-1) > (
        2048*np.finfo(float).eps*scale)
    if np.any(analytical_mask):
        index = np.flatnonzero(analytical_mask)
        total_radius = np.sum(
            flat_w[:, index]*flat_radius[:, index], axis=0)
        direction = np.divide(
            flat_target[index], total_radius[:, None],
            out=np.zeros_like(flat_target[index]),
            where=total_radius[:, None] > 0.0)
        analytical = flat_radius[:, index, None]*direction[None, ...]
        flat[:, index, :] = np.where(
            flat_active[:, index, None], analytical, flat[:, index, :])
        residual = flat_target-np.sum(flat_w[..., None]*flat, axis=0)
    if float(np.max(np.abs(residual))) > 2048*np.finfo(float).eps*scale:
        raise RuntimeError("bounded owner-moment projection did not converge")
    result = flat.reshape(kappa.shape)
    return tuple(result[index] for index in range(len(baselines)))


def apply_mechanical_increment(state, updated, spacing_m, systems, topologies):
    """Project one accepted V24/Mura transaction into every active owner.

    Scalar density and every reservoir-resolved first moment are advanced from
    the accepted mechanical state.  Inactive owners retain their intensive
    history for later retreat/revisit; they contribute zero extensive content
    while their support is zero.
    """
    old_mechanical = reconstruct_mechanical_state(
        state, spacing_m, systems, topologies)
    common_candidate = apply_common_increment(state, updated.common, spacing_m)
    weights = state.front.material_support_weights()
    density_owners = []
    old_alignment_owners = (
        state.parent_alignment, state.child_alignment, state.wake_alignment)
    for density_owner, support in zip(
            (state.parent_density, state.child_density, state.wake_density),
            weights):
        density_owners.append(DensityInventory(**{
            item.name: _density_support_delta(
                density_owner, old_mechanical.density, updated.density,
                support, item.name)
            for item in fields(DensityInventory)}))
    # Preserve each owner's polarization while matching the accepted common
    # first moment exactly.  Scaling by its own accepted scalar-density change
    # respects |kappa|<=rho; the small common residual carries genuine line
    # turning/advection from the accepted Mura face flux.  It is applied to
    # every active co-located owner, whose support weights sum to one.
    alignment_arrays = [dict() for _ in range(3)]
    density_by_alignment = {
        name: name for name in SIGNED_RESERVOIRS}
    density_by_alignment["junction_alignment_m2"] = "junction_m2"
    for item in fields(ReservoirAlignmentState):
        name = item.name
        if name in TOPOLOGY_MOMENT_FIELDS:
            for index, (owner, support) in enumerate(zip(
                    old_alignment_owners, weights)):
                alignment_arrays[index][name] = _moment_support_delta(
                    owner, old_mechanical.reservoir_alignment,
                    updated.reservoir_alignment, support, name,
                    nonnegative=True)
            continue
        density_name = density_by_alignment[name]
        baselines = []
        for old_owner, old_density, new_density in zip(
                old_alignment_owners,
                (state.parent_density, state.child_density,
                 state.wake_density), density_owners):
            old_moment = np.asarray(getattr(old_owner, name))
            old_rho = np.asarray(getattr(old_density, density_name))
            new_rho = np.asarray(getattr(new_density, density_name))
            ratio = np.divide(new_rho, old_rho, out=np.zeros_like(new_rho),
                              where=old_rho > 0.0)
            baselines.append(
                old_moment*ratio[(...,)+(None,)*(old_moment.ndim-ratio.ndim)])
        target = np.asarray(getattr(updated.reservoir_alignment, name))
        if name == "junction_alignment_m2":
            multiplicity = np.asarray([
                topology.product_line_multiplicity for topology in topologies])
            bounds = [density.junction_m2*multiplicity
                      for density in density_owners]
        else:
            bounds = [np.asarray(getattr(density, density_name))
                      for density in density_owners]
        projected = _minimum_change_bounded_moments(
            baselines, bounds, weights, target)
        for index, value in enumerate(projected):
            alignment_arrays[index][name] = value
    alignment_owners = [
        ReservoirAlignmentState(**arrays) for arrays in alignment_arrays]
    for density, alignment in zip(density_owners, alignment_owners):
        alignment.validate(density, len(systems))
    common_candidate = replace(
        common_candidate,
        parent=_common_with_authoritative_density_views(
            common_candidate.parent, density_owners[0]),
        child=_common_with_authoritative_density_views(
            common_candidate.child, density_owners[1]),
        wake=_common_with_authoritative_density_views(
            common_candidate.wake, density_owners[2]))
    candidate = replace(
        common_candidate,
        parent_density=density_owners[0], child_density=density_owners[1],
        wake_density=density_owners[2],
        parent_alignment=alignment_owners[0],
        child_alignment=alignment_owners[1],
        wake_alignment=alignment_owners[2])
    check = reconstruct_mechanical_state(
        candidate, spacing_m, systems, topologies)
    for group in ("density", "reservoir_alignment"):
        actual_group = getattr(check, group)
        expected_group = getattr(updated, group)
        for item in fields(type(actual_group)):
            actual = np.asarray(getattr(actual_group, item.name))
            expected = np.asarray(getattr(expected_group, item.name))
            scale = max(float(np.max(np.abs(expected))), 1.0)
            if float(np.max(np.abs(actual-expected))) > (
                    2048*np.finfo(float).eps*scale):
                raise RuntimeError(
                    f"reservoir owner projection failed for {item.name}")
    _validate_owner_density_common(candidate)
    return candidate


def _line_total(owner):
    return (np.sum(owner.mobile_plus_m2+owner.mobile_minus_m2
                   +owner.forest_plus_m2+owner.forest_minus_m2
                   +owner.wall_plus_m2+owner.wall_minus_m2, axis=2)
            +np.sum(owner.junction_m2, axis=2))


def _blend(old, old_weight, incoming, increment):
    w = old_weight[(...,)+(None,)*(old.ndim-old_weight.ndim)]
    q = increment[(...,)+(None,)*(old.ndim-old_weight.ndim)]
    return np.divide(w*old+q*incoming, w+q, out=np.asarray(old).copy(), where=w+q > 0.0)


def _blend_common(old, old_weight, incoming, increment):
    return CommonWallState(**{
        item.name: _blend(np.asarray(getattr(old, item.name)), old_weight,
                          np.asarray(getattr(incoming, item.name)), increment)
        for item in fields(CommonWallState)})


def _mixed_donor(a, wa, b, wb, total):
    arrays = {}
    for item in fields(CommonWallState):
        av = np.asarray(getattr(a, item.name)); bv = np.asarray(getattr(b, item.name))
        aw = wa[(...,)+(None,)*(av.ndim-wa.ndim)]
        bw = wb[(...,)+(None,)*(av.ndim-wb.ndim)]
        tw = total[(...,)+(None,)*(av.ndim-total.ndim)]
        arrays[item.name] = np.divide(aw*av+bw*bv, tw, out=av.copy(), where=tw > 0.0)
    return CommonWallState(**arrays)


def _transmitted_state(donor, transmission):
    arrays = {}
    scalar = np.mean(transmission, axis=2)
    for item in fields(CommonWallState):
        value = np.asarray(getattr(donor, item.name))
        if item.name in LINE_FIELDS or item.name in KINEMATIC_FIELDS or item.name == "family_nye_m1":
            if item.name in ("mobile_plus_m2", "mobile_minus_m2",
                             "forest_plus_m2", "forest_minus_m2",
                             "wall_plus_m2", "wall_minus_m2", "slip"):
                factor = transmission
            elif item.name in ("alignment_m2", "family_nye_m1"):
                factor = transmission[(...,)+(None,)*(value.ndim-transmission.ndim)]
            else:
                factor = scalar[(...,)+(None,)*(value.ndim-scalar.ndim)]
            arrays[item.name] = factor*value
        else:
            arrays[item.name] = value.copy()
    return CommonWallState(**arrays)


def _mixed_dataclass(a, wa, b, wb, total, cls):
    arrays = {}
    for item in fields(cls):
        av = np.asarray(getattr(a, item.name))
        bv = np.asarray(getattr(b, item.name))
        aw = wa[(...,)+(None,)*(av.ndim-wa.ndim)]
        bw = wb[(...,)+(None,)*(av.ndim-wb.ndim)]
        tw = total[(...,)+(None,)*(av.ndim-total.ndim)]
        arrays[item.name] = np.divide(
            aw*av+bw*bv, tw, out=av.copy(), where=tw > 0.0)
    return cls(**arrays)


def _blend_dataclass(old, old_weight, incoming, increment, cls):
    return cls(**{
        item.name: _blend(np.asarray(getattr(old, item.name)), old_weight,
                          np.asarray(getattr(incoming, item.name)), increment)
        for item in fields(cls)})


def _transmitted_density(donor, transmission):
    tf = np.mean(transmission, axis=2)
    return DensityInventory(**{
        item.name: ((tf[..., None]*np.asarray(getattr(donor, item.name)))
                    if item.name == "junction_m2" else
                    transmission*np.asarray(getattr(donor, item.name)))
        for item in fields(DensityInventory)})


def _transmitted_alignment(donor, transmission):
    tf = np.mean(transmission, axis=2)
    arrays = {}
    for item in fields(ReservoirAlignmentState):
        value = np.asarray(getattr(donor, item.name))
        if item.name == "junction_alignment_m2":
            factor = tf[..., None, None]
        elif item.name in TOPOLOGY_MOMENT_FIELDS:
            factor = transmission
        else:
            factor = transmission[..., None]
        arrays[item.name] = factor*value
    return ReservoirAlignmentState(**arrays)


def _pair_partition(plus, minus, transmission, boundary_fraction,
                    sink_fraction, signed_sink_fraction):
    blocked_p = (1.0-transmission)*plus
    blocked_m = (1.0-transmission)*minus
    neutral = np.minimum(blocked_p, blocked_m)
    excess_p = blocked_p-neutral; excess_m = blocked_m-neutral
    bp = excess_p*(1.0-signed_sink_fraction)+boundary_fraction*neutral
    bm = excess_m*(1.0-signed_sink_fraction)+boundary_fraction*neutral
    annihilated = 2.0*(1.0-boundary_fraction-sink_fraction)*neutral
    sink = 2.0*sink_fraction*neutral+signed_sink_fraction*(excess_p+excess_m)
    return bp, bm, annihilated, sink


def _defect_view(owner):
    return DefectState(
        owner.mobile_plus_m2.copy(), owner.mobile_minus_m2.copy(),
        (owner.forest_plus_m2+owner.forest_minus_m2).copy(),
        np.sum(owner.wall_plus_m2+owner.wall_minus_m2, axis=2))


def _sync_front_views(state):
    front = replace(
        state.front, parent=_defect_view(state.parent),
        child=_defect_view(state.child),
        recovered_wake=_defect_view(state.wake))
    return replace(state, front=front)


def commit_front_result(state, accepted_front, *, spacing_m, cell_volume_m3,
                        transmission_fraction, boundary_storage_fraction,
                        neutral_sink_fraction, signed_sink_fraction=0.0):
    """Commit accepted front geometry to every common reservoir atomically.

    ``accepted_front`` supplies only accepted support/history/topology data.
    Its legacy four-reservoir material fields are discarded and rebuilt as
    read-only views from the authoritative owners below.
    """
    old_front = state.front
    delta = np.asarray(accepted_front.chi)-np.asarray(old_front.chi)
    positive = np.maximum(delta, 0.0); negative = np.maximum(-delta, 0.0)
    if np.any(positive*negative):
        raise RuntimeError("front sweep signs overlap")
    transmission = np.broadcast_to(np.asarray(transmission_fraction, dtype=float),
                                   old_front.parent.rp.shape)
    tf = np.mean(transmission, axis=2)
    parent, child, wake = state.parent, state.child, state.wake
    moments_active = has_reservoir_moment_owners(state)
    parent_density, child_density, wake_density = (
        state.parent_density, state.child_density, state.wake_density)
    parent_alignment, child_alignment, wake_alignment = (
        state.parent_alignment, state.child_alignment, state.wake_alignment)
    boundary_p = state.boundary_plus_m2.copy(); boundary_m = state.boundary_minus_m2.copy()
    boundary_j = state.boundary_junction_m2.copy()
    processed = transmitted = annihilated = sink = 0.0
    signed_closure = 0.0

    def transfer(donor, sweep, recipient, recipient_weight,
                 donor_density=None, donor_alignment=None,
                 recipient_density=None, recipient_alignment=None):
        nonlocal boundary_p, boundary_m, boundary_j
        nonlocal processed, transmitted, annihilated, sink, signed_closure
        incoming = _transmitted_state(donor, transmission)
        pairs = ((donor.mobile_plus_m2, donor.mobile_minus_m2),
                 (donor.forest_plus_m2, donor.forest_minus_m2),
                 (donor.wall_plus_m2, donor.wall_minus_m2))
        for pair_index, (plus, minus) in enumerate(pairs):
            bp, bm, ann, snk = _pair_partition(
                plus, minus, transmission, boundary_storage_fraction,
                neutral_sink_fraction, signed_sink_fraction)
            boundary_p[..., pair_index, :] += sweep[..., None]*bp
            boundary_m[..., pair_index, :] += sweep[..., None]*bm
            annihilated += float(np.sum(sweep[..., None]*ann, dtype=np.longdouble)*cell_volume_m3)
            sink += float(np.sum(sweep[..., None]*snk, dtype=np.longdouble)*cell_volume_m3)
            incoming_plus = getattr(incoming, LINE_FIELDS[pair_index*2])
            incoming_minus = getattr(incoming, LINE_FIELDS[pair_index*2+1])
            signed_sink = signed_sink_fraction*((1.0-transmission)*(plus-minus))
            residual = ((plus-minus)-(incoming_plus-incoming_minus)
                        -(bp-bm)-signed_sink)
            signed_closure = max(
                signed_closure, float(np.max(np.abs(residual))))
        blocked_j = (1.0-tf[..., None])*donor.junction_m2
        boundary_j += sweep[..., None]*boundary_storage_fraction*blocked_j
        annihilated += float(np.sum(
            sweep[..., None]*(1.0-boundary_storage_fraction-neutral_sink_fraction)*blocked_j,
            dtype=np.longdouble)*cell_volume_m3)
        sink += float(np.sum(sweep[..., None]*neutral_sink_fraction*blocked_j,
                             dtype=np.longdouble)*cell_volume_m3)
        processed += float(np.sum(sweep*_line_total(donor), dtype=np.longdouble)*cell_volume_m3)
        transmitted += float(np.sum(sweep*_line_total(incoming), dtype=np.longdouble)*cell_volume_m3)
        common_result = _blend_common(
            recipient, recipient_weight, incoming, sweep)
        if donor_density is None:
            return common_result, None, None
        incoming_density = _transmitted_density(donor_density, transmission)
        incoming_alignment = _transmitted_alignment(
            donor_alignment, transmission)
        return (
            common_result,
            _blend_dataclass(recipient_density, recipient_weight,
                             incoming_density, sweep, DensityInventory),
            _blend_dataclass(recipient_alignment, recipient_weight,
                             incoming_alignment, sweep,
                             ReservoirAlignmentState))

    if np.any(positive):
        revisit = np.minimum(positive, old_front.processed_max-old_front.chi)
        virgin = positive-revisit
        donor = _mixed_donor(wake, revisit, parent, virgin, positive)
        donor_density = donor_alignment = None
        if moments_active:
            donor_density = _mixed_dataclass(
                wake_density, revisit, parent_density, virgin, positive,
                DensityInventory)
            donor_alignment = _mixed_dataclass(
                wake_alignment, revisit, parent_alignment, virgin, positive,
                ReservoirAlignmentState)
        child, child_density, child_alignment = transfer(
            donor, positive, child, old_front.chi,
            donor_density, donor_alignment, child_density, child_alignment)
    if np.any(negative):
        wake_weight = old_front.processed_max-old_front.chi
        wake, wake_density, wake_alignment = transfer(
            child, negative, wake, wake_weight,
            child_density if moments_active else None,
            child_alignment if moments_active else None,
            wake_density, wake_alignment)

    common_boundary = np.sum(boundary_p+boundary_m, axis=(2, 3))+np.sum(boundary_j, axis=2)
    common_signed = np.sum(boundary_p-boundary_m, axis=2)
    rebuilt_front = replace(
        accepted_front, parent=_defect_view(parent), child=_defect_view(child),
        recovered_wake=_defect_view(wake),
        boundary_line_density_m2=common_boundary,
        boundary_signed_density_m2=common_signed)
    candidate = replace(
        state, front=rebuilt_front, parent=parent, child=child, wake=wake,
        boundary_plus_m2=boundary_p, boundary_minus_m2=boundary_m,
        boundary_junction_m2=boundary_j,
        parent_density=parent_density, child_density=child_density,
        wake_density=wake_density, parent_alignment=parent_alignment,
        child_alignment=child_alignment, wake_alignment=wake_alignment)
    mixture, correction = reconstruct_common(candidate, spacing_m)
    boundary_added = float(np.sum(
        common_boundary-(np.sum(state.boundary_plus_m2+state.boundary_minus_m2, axis=(2, 3))
                         +np.sum(state.boundary_junction_m2, axis=2)),
        dtype=np.longdouble)*cell_volume_m3)
    # Close the transaction from its terms rather than subtracting two large
    # whole-domain inventories.  The latter loses all useful digits for a
    # subcell sweep and can falsely report a percent-level relative residual.
    closure = processed-transmitted-boundary_added-annihilated-sink
    scale = max(abs(processed), abs(transmitted), abs(boundary_added), 1e-300)
    if abs(closure) > 32768*np.finfo(float).eps*scale:
        raise RuntimeError(f"common-front line balance failed: {closure:.17g} m")
    _validate_owner_density_common(candidate)
    old = state.ledger
    ledger = replace(
        old, attempted_commits=old.attempted_commits+1,
        accepted_commits=old.accepted_commits+1,
        processed_line_m=old.processed_line_m+processed,
        transmitted_line_m=old.transmitted_line_m+transmitted,
        boundary_line_m=old.boundary_line_m+boundary_added,
        annihilated_line_m=old.annihilated_line_m+annihilated,
        sink_line_m=old.sink_line_m+sink,
        maximum_line_closure_m=max(old.maximum_line_closure_m, abs(closure)),
        maximum_signed_closure_m2=max(old.maximum_signed_closure_m2, signed_closure),
        interface_nye_norm_m1=float(np.linalg.norm(correction)))
    return replace(candidate, interface_nye_m1=correction, ledger=ledger), mixture


def state_arrays(state):
    payload = {}
    for owner_name in ("parent", "child", "wake"):
        owner = getattr(state, owner_name)
        for item in fields(CommonWallState):
            payload[f"{owner_name}__{item.name}"] = np.asarray(getattr(owner, item.name))
    payload.update(boundary_plus_m2=state.boundary_plus_m2,
                   boundary_minus_m2=state.boundary_minus_m2,
                   boundary_junction_m2=state.boundary_junction_m2,
                   interface_nye_m1=state.interface_nye_m1)
    if has_reservoir_moment_owners(state):
        for owner_name in ("parent", "child", "wake"):
            density = getattr(state, owner_name+"_density")
            alignment = getattr(state, owner_name+"_alignment")
            for item in fields(DensityInventory):
                payload[f"{owner_name}__density__{item.name}"] = np.asarray(
                    getattr(density, item.name))
            for item in fields(ReservoirAlignmentState):
                payload[f"{owner_name}__alignment__{item.name}"] = np.asarray(
                    getattr(alignment, item.name))
    return payload


def state_metadata_json(state):
    moments = has_reservoir_moment_owners(state)
    return json.dumps({
        "schema": SCHEMA if moments else LEGACY_SCHEMA,
        "ledger": state.ledger.__dict__,
        "reservoir_moment_owners": moments},
        sort_keys=True)


def state_from_checkpoint(metadata_json, arrays, front):
    metadata = json.loads(str(metadata_json))
    if metadata.get("schema") not in (LEGACY_SCHEMA, SCHEMA):
        raise ValueError("unsupported common-front checkpoint schema")
    owners = []
    for owner_name in ("parent", "child", "wake"):
        owners.append(CommonWallState(**{
            item.name: np.asarray(arrays[f"{owner_name}__{item.name}"]).copy()
            for item in fields(CommonWallState)}))
    kwargs = {}
    if metadata.get("reservoir_moment_owners", False):
        for owner_name in ("parent", "child", "wake"):
            kwargs[owner_name+"_density"] = DensityInventory(**{
                item.name: np.asarray(arrays[
                    f"{owner_name}__density__{item.name}"]).copy()
                for item in fields(DensityInventory)})
            kwargs[owner_name+"_alignment"] = ReservoirAlignmentState(**{
                item.name: np.asarray(arrays[
                    f"{owner_name}__alignment__{item.name}"]).copy()
                for item in fields(ReservoirAlignmentState)})
    result = CommonFrontState(
        front=front, parent=owners[0], child=owners[1], wake=owners[2],
        boundary_plus_m2=np.asarray(arrays["boundary_plus_m2"]).copy(),
        boundary_minus_m2=np.asarray(arrays["boundary_minus_m2"]).copy(),
        boundary_junction_m2=np.asarray(
            arrays["boundary_junction_m2"]).copy(),
        interface_nye_m1=np.asarray(arrays["interface_nye_m1"]).copy(),
        ledger=CommonFrontLedger(**metadata.get("ledger", {})), **kwargs)
    _validate_owner_density_common(result)
    return result
