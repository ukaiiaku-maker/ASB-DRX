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
    from .moving_front import DefectState, SparseFrontState
    from .tensorial_nye import nye_from_plastic_distortion
except ImportError:  # pragma: no cover - production script execution
    from common_tensorial_wall import CommonWallState
    from moving_front import DefectState, SparseFrontState
    from tensorial_nye import nye_from_plastic_distortion


SCHEMA = "full-v34-common-front-authoritative-state/v1"

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


def _support_delta(owner, old_mix, new_mix, support, name):
    value = np.asarray(getattr(owner, name))
    old = np.asarray(getattr(old_mix, name)); new = np.asarray(getattr(new_mix, name))
    weight = support[(...,)+(None,)*(value.ndim-support.ndim)]
    active = weight > 64.0*np.finfo(float).eps
    if name in LINE_FIELDS:
        ratio = np.divide(new, old, out=np.zeros_like(new), where=old > 0.0)
        candidate = np.where(new < old, value*ratio, value+np.maximum(new-old, 0.0))
        return np.where(active, np.maximum(candidate, 0.0), 0.0)
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
    boundary_p = state.boundary_plus_m2.copy(); boundary_m = state.boundary_minus_m2.copy()
    boundary_j = state.boundary_junction_m2.copy()
    processed = transmitted = annihilated = sink = 0.0
    signed_closure = 0.0

    def transfer(donor, sweep, recipient, recipient_weight):
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
        return _blend_common(recipient, recipient_weight, incoming, sweep)

    if np.any(positive):
        revisit = np.minimum(positive, old_front.processed_max-old_front.chi)
        virgin = positive-revisit
        donor = _mixed_donor(wake, revisit, parent, virgin, positive)
        child = transfer(donor, positive, child, old_front.chi)
    if np.any(negative):
        wake_weight = old_front.processed_max-old_front.chi
        wake = transfer(child, negative, wake, wake_weight)

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
        boundary_junction_m2=boundary_j)
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
    return payload


def state_metadata_json(state):
    return json.dumps({"schema": SCHEMA, "ledger": state.ledger.__dict__}, sort_keys=True)


def state_from_checkpoint(metadata_json, arrays, front):
    metadata = json.loads(str(metadata_json))
    if metadata.get("schema") != SCHEMA:
        raise ValueError("unsupported common-front checkpoint schema")
    owners = []
    for owner_name in ("parent", "child", "wake"):
        owners.append(CommonWallState(**{
            item.name: np.asarray(arrays[f"{owner_name}__{item.name}"]).copy()
            for item in fields(CommonWallState)}))
    return CommonFrontState(
        front, *owners, np.asarray(arrays["boundary_plus_m2"]).copy(),
        np.asarray(arrays["boundary_minus_m2"]).copy(),
        np.asarray(arrays["boundary_junction_m2"]).copy(),
        np.asarray(arrays["interface_nye_m1"]).copy(),
        CommonFrontLedger(**metadata.get("ledger", {})))
