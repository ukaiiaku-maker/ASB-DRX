"""Authoritative multi-grain owner state and joint material transaction.

This module generalizes the support-weighted reconstruction used by
``CommonFrontState``.  Every proposal in a joint transaction is evaluated from
one immutable accepted state; donor capacity is shared cellwise, so incident
edges cannot spend the same material twice.  The module does not price a
complete physical event yet.  Publication by a production driver must remain
disabled until the combined energy transaction accepts the returned candidate.

Owner fields are intensive.  Their support-weighted products are cell
inventories.  Supports are dimensionless, signed line reservoirs are m^-2,
plastic distortion/slip are dimensionless, Nye is m^-1, and temperature is K.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math

import numpy as np

from .common_front_state import CommonFrontState, LINE_FIELDS
from .common_tensorial_wall import CommonWallState
from .tensorial_nye import (
    nye_from_plastic_distortion, spectral_derivatives,
)


SCHEMA = "asb-drx/multigrain-common-owner/v1"
SUPPORT_TOLERANCE = 256.0*np.finfo(float).eps


@dataclass(frozen=True)
class MultiGrainLedger:
    attempted_transactions: int = 0
    capacity_feasible_candidates: int = 0
    energy_accepted_transactions: int = 0
    rejected_transactions: int = 0
    proposed_material_fraction: float = 0.0
    accepted_material_fraction: float = 0.0
    capacity_limited_material_fraction: float = 0.0
    maximum_support_closure: float = 0.0
    maximum_donor_overdraft: float = 0.0
    maximum_line_export_closure_m2: float = 0.0
    fresh_sweep_fraction: float = 0.0
    revisit_sweep_fraction: float = 0.0
    annihilated_line_cell_sum_m2: float = 0.0
    external_sink_line_cell_sum_m2: float = 0.0


@dataclass(frozen=True)
class InterfaceComponentState:
    component_id: str
    grain_a_id: int
    grain_b_id: int
    first_passage_fraction: np.ndarray
    revisit_fraction: np.ndarray
    boundary_plus_m2: np.ndarray
    boundary_minus_m2: np.ndarray
    boundary_junction_m2: np.ndarray
    exposure_s: float = 0.0
    cumulative_work_J: float = 0.0
    cumulative_heat_J: float = 0.0
    cumulative_signed_sweep_fraction: float = 0.0
    cumulative_absolute_sweep_fraction: float = 0.0
    periodic_winding_x: int = 0
    periodic_winding_y: int = 0


@dataclass(frozen=True)
class PhysicalTransferChannels:
    """Per-unit-swept-support products of a declared transfer law."""

    boundary_plus_m2: np.ndarray
    boundary_minus_m2: np.ndarray
    boundary_junction_m2: np.ndarray
    annihilated_line_m2: np.ndarray
    external_sink_line_m2: np.ndarray


@dataclass(frozen=True)
class PhysicalTransferLaw:
    transmission_fraction: object
    boundary_storage_fraction: float
    neutral_sink_fraction: float
    signed_sink_fraction: float = 0.0

    def validate(self):
        transmission = np.asarray(self.transmission_fraction, dtype=float)
        if np.any(~np.isfinite(transmission)) or np.any(
                (transmission < 0.0)|(transmission > 1.0)):
            raise ValueError("transmission fraction must lie in [0,1]")
        values = (self.boundary_storage_fraction,
                  self.neutral_sink_fraction, self.signed_sink_fraction)
        if any(not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
               for value in values):
            raise ValueError("transfer channel fractions must lie in [0,1]")
        if self.boundary_storage_fraction+self.neutral_sink_fraction > 1.0:
            raise ValueError("neutral boundary and sink fractions exceed unity")


@dataclass(frozen=True)
class MultiGrainCommonState:
    grain_ids: tuple[int, ...]
    supports: np.ndarray
    owners: tuple[CommonWallState, ...]
    interfaces: tuple[InterfaceComponentState, ...] = ()
    ledger: MultiGrainLedger = MultiGrainLedger()
    schema: str = SCHEMA

    def validate(self):
        support = np.asarray(self.supports, dtype=float)
        if support.ndim != 3 or support.shape[0] != len(self.grain_ids):
            raise ValueError("supports require grain x grid layout")
        if len(self.grain_ids) < 2 or len(set(self.grain_ids)) != len(
                self.grain_ids):
            raise ValueError("grain IDs must be unique and contain at least two grains")
        if len(self.owners) != len(self.grain_ids):
            raise ValueError("every physical grain requires exactly one owner")
        if np.any(~np.isfinite(support)) or np.any(support < -SUPPORT_TOLERANCE):
            raise ValueError("supports must be finite and nonnegative")
        closure = np.max(np.abs(np.sum(support, axis=0)-1.0))
        if closure > SUPPORT_TOLERANCE:
            raise ValueError(f"supports are not normalized: maximum={closure:.17g}")
        grid = support.shape[1:]
        reference = self.owners[0]
        for owner in self.owners:
            for item in fields(CommonWallState):
                value = np.asarray(getattr(owner, item.name))
                expected = np.asarray(getattr(reference, item.name)).shape
                if value.shape != expected or value.shape[:2] != grid:
                    raise ValueError(f"owner field {item.name} has inconsistent layout")
                if np.any(~np.isfinite(value)):
                    raise ValueError(f"owner field {item.name} contains nonfinite values")
            for name in LINE_FIELDS:
                if np.any(np.asarray(getattr(owner, name)) < 0.0):
                    raise ValueError(f"owner field {name} must be nonnegative")
        return grid


@dataclass(frozen=True)
class JointTransferProposal:
    interface_id: str
    donor_id: int
    receiver_id: int
    requested_fraction: np.ndarray
    product_owner: CommonWallState
    # Explicit nonnegative line exported from the represented state per unit
    # transferred support.  Keys must cover every LINE_FIELDS member.
    line_export_m2: dict[str, np.ndarray]
    physical_channels: PhysicalTransferChannels | None = None
    physical_interval_s: float = 0.0


@dataclass(frozen=True)
class JointTransferResult:
    accepted: bool
    capacity_feasible: bool
    classification: str
    candidate: MultiGrainCommonState
    accepted_fraction_by_interface: dict[str, np.ndarray]
    capacity_scale_by_donor: dict[int, np.ndarray]
    line_export_m2_by_field: dict[str, float]
    physical_channel_cell_sums_m2: dict[str, float]


def _scaled_product_owner(donor, receiver, transmission):
    """Derive the swept product without a caller-selected low-density state."""
    scalar = np.mean(transmission, axis=2)
    arrays = {}
    for item in fields(CommonWallState):
        value = np.asarray(getattr(donor, item.name), dtype=float)
        if item.name in LINE_FIELDS or item.name in (
                "slip", "beta_p", "alignment_m2", "family_nye_m1"):
            if item.name in ("mobile_plus_m2", "mobile_minus_m2",
                             "forest_plus_m2", "forest_minus_m2",
                             "wall_plus_m2", "wall_minus_m2", "slip"):
                factor = transmission
            elif item.name in ("alignment_m2", "family_nye_m1"):
                factor = transmission[(...,)+(None,)*(value.ndim-transmission.ndim)]
            else:
                factor = scalar[(...,)+(None,)*(value.ndim-scalar.ndim)]
            arrays[item.name] = factor*value
        elif item.name == "orientation_rad":
            # Crystal identity belongs to the receiving grain, not to an
            # arithmetic average or to the consumed donor support.
            arrays[item.name] = np.asarray(receiver.orientation_rad).copy()
        else:
            arrays[item.name] = value.copy()
    return CommonWallState(**arrays)


def derive_physical_transfer_proposal(
        state: MultiGrainCommonState, *, interface_id: str, donor_id: int,
        receiver_id: int, requested_fraction, law: PhysicalTransferLaw,
        interval_s: float):
    """Build all product/export channels from the current physical owners."""
    state.validate()
    law.validate()
    if not math.isfinite(float(interval_s)) or float(interval_s) <= 0.0:
        raise ValueError("physical transfer requires a positive finite interval")
    index = {grain_id: position for position, grain_id in enumerate(state.grain_ids)}
    if donor_id not in index or receiver_id not in index or donor_id == receiver_id:
        raise ValueError("physical transfer requires distinct known grains")
    donor = state.owners[index[donor_id]]
    receiver = state.owners[index[receiver_id]]
    shape = donor.mobile_plus_m2.shape
    transmission = np.broadcast_to(
        np.asarray(law.transmission_fraction, dtype=float), shape)
    product = _scaled_product_owner(donor, receiver, transmission)
    exports = {name: np.asarray(getattr(donor, name))
               -np.asarray(getattr(product, name)) for name in LINE_FIELDS}
    boundary_plus = np.zeros(shape[:2]+(3, shape[2]))
    boundary_minus = np.zeros_like(boundary_plus)
    annihilated = np.zeros(shape[:2])
    sink = np.zeros(shape[:2])
    for pair_index, names in enumerate((
            ("mobile_plus_m2", "mobile_minus_m2"),
            ("forest_plus_m2", "forest_minus_m2"),
            ("wall_plus_m2", "wall_minus_m2"))):
        plus = np.asarray(getattr(donor, names[0]))
        minus = np.asarray(getattr(donor, names[1]))
        blocked_plus = (1.0-transmission)*plus
        blocked_minus = (1.0-transmission)*minus
        neutral = np.minimum(blocked_plus, blocked_minus)
        excess_plus = blocked_plus-neutral
        excess_minus = blocked_minus-neutral
        boundary_plus[..., pair_index, :] = (
            excess_plus*(1.0-law.signed_sink_fraction)
            +law.boundary_storage_fraction*neutral)
        boundary_minus[..., pair_index, :] = (
            excess_minus*(1.0-law.signed_sink_fraction)
            +law.boundary_storage_fraction*neutral)
        annihilated += np.sum(
            2.0*(1.0-law.boundary_storage_fraction
                 -law.neutral_sink_fraction)*neutral, axis=2)
        sink += np.sum(
            2.0*law.neutral_sink_fraction*neutral
            +law.signed_sink_fraction*(excess_plus+excess_minus), axis=2)
    tf = np.mean(transmission, axis=2)
    blocked_junction = (1.0-tf[..., None])*donor.junction_m2
    boundary_junction = law.boundary_storage_fraction*blocked_junction
    annihilated += np.sum(
        (1.0-law.boundary_storage_fraction-law.neutral_sink_fraction)
        *blocked_junction, axis=2)
    sink += np.sum(law.neutral_sink_fraction*blocked_junction, axis=2)
    channels = PhysicalTransferChannels(
        boundary_plus, boundary_minus, boundary_junction, annihilated, sink)
    return JointTransferProposal(
        str(interface_id), int(donor_id), int(receiver_id),
        np.asarray(requested_fraction, dtype=float), product, exports, channels,
        float(interval_s))


@dataclass(frozen=True)
class MultiGrainNyeAudit:
    """Independent ownership audit for the one authoritative Nye tensor.

    ``support_gradient_m1`` is the physical product-rule term generated by
    spatially varying material support. ``owner_reservoir_mismatch_m1`` is a
    pre-existing inconsistency between each owner's Curl(beta_p) and supplied
    family reservoirs.  They must never be merged and renamed as interface
    content.
    """

    exact_reconstructed_m1: np.ndarray
    owner_curl_weighted_m1: np.ndarray
    owner_reservoir_weighted_m1: np.ndarray
    support_gradient_m1: np.ndarray
    owner_reservoir_mismatch_m1: np.ndarray
    discrete_representation_residual_m1: np.ndarray


def _curl_from_derivatives(dx, dy):
    alpha = np.zeros_like(dx)
    alpha[..., :, 0] = -dy[..., :, 2]
    alpha[..., :, 1] = dx[..., :, 2]
    alpha[..., :, 2] = dy[..., :, 0]-dx[..., :, 1]
    return alpha


def audit_multigrain_nye(state: MultiGrainCommonState, spacing_m: float):
    """Separate support-gradient Nye from inconsistent owner reservoirs."""
    state.validate()
    beta_bar = np.zeros_like(state.owners[0].beta_p, dtype=float)
    owner_curl = np.zeros(beta_bar.shape, dtype=float)
    owner_reservoir = np.zeros(beta_bar.shape, dtype=float)
    support_gradient = np.zeros(beta_bar.shape, dtype=float)
    for index, owner in enumerate(state.owners):
        weight = np.asarray(state.supports[index], dtype=float)
        beta = np.asarray(owner.beta_p, dtype=float)
        beta_bar += weight[..., None, None]*beta
        bx, by = spectral_derivatives(beta, spacing_m)
        curl_beta = _curl_from_derivatives(bx, by)
        owner_curl += weight[..., None, None]*curl_beta
        owner_reservoir += weight[..., None, None]*np.sum(
            np.asarray(owner.family_nye_m1, dtype=float), axis=2)
        wx, wy = spectral_derivatives(weight, spacing_m)
        support_gradient += _curl_from_derivatives(
            wx[..., None, None]*beta, wy[..., None, None]*beta)
    exact = nye_from_plastic_distortion(beta_bar, spacing_m)
    assembled = owner_curl+support_gradient
    return MultiGrainNyeAudit(
        exact_reconstructed_m1=exact,
        owner_curl_weighted_m1=owner_curl,
        owner_reservoir_weighted_m1=owner_reservoir,
        support_gradient_m1=support_gradient,
        owner_reservoir_mismatch_m1=owner_curl-owner_reservoir,
        discrete_representation_residual_m1=exact-assembled)


def _weight(value, support):
    array = np.asarray(value, dtype=float)
    return support[(...,)+(None,)*(array.ndim-support.ndim)]*array


def reconstruct_multigrain_common(state: MultiGrainCommonState, spacing_m: float):
    """Reconstruct one mixture without feeding interface Nye into bulk kinetics.

    The returned second value remains the exact-minus-owner-reservoir tensor
    for API compatibility.  Production callers requiring attribution must use
    :func:`audit_multigrain_nye`, which separates its physical and inconsistent
    contributions.
    """
    state.validate()
    values = {}
    for item in fields(CommonWallState):
        if item.name == "family_nye_m1":
            continue
        values[item.name] = sum(
            (_weight(getattr(owner, item.name), state.supports[index])
             for index, owner in enumerate(state.owners)),
            np.zeros_like(np.asarray(getattr(state.owners[0], item.name)),
                          dtype=float))
    bulk_family = sum(
        (_weight(owner.family_nye_m1, state.supports[index])
         for index, owner in enumerate(state.owners)),
        np.zeros_like(np.asarray(state.owners[0].family_nye_m1), dtype=float))
    audit = audit_multigrain_nye(state, float(spacing_m))
    correction = (audit.exact_reconstructed_m1
                  -audit.owner_reservoir_weighted_m1)
    # Interface excess has its own owner and energy.  Arbitrarily distributing
    # it according to |slip| would make a diagnostic partition affect kinetics.
    values["family_nye_m1"] = bulk_family
    return CommonWallState(**values), correction


def multigrain_state_digest(state: MultiGrainCommonState):
    """Content identity used to prevent stale energy decisions."""
    digest = hashlib.sha256()
    digest.update(multigrain_checkpoint_metadata(state).encode("utf-8"))
    for name, value in sorted(multigrain_checkpoint_arrays(state).items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def from_common_front_pair(state: CommonFrontState, parent_id=0, child_id=1):
    """Pair-limit adapter; nonzero recovered wake requires a history model."""
    parent, child, wake = state.front.material_support_weights()
    if float(np.max(np.abs(wake))) > SUPPORT_TOLERANCE:
        raise ValueError("pair adapter cannot discard nonzero recovered-wake support")
    supports = np.stack((parent, child), axis=0)
    candidate = MultiGrainCommonState(
        (int(parent_id), int(child_id)), supports,
        (state.parent, state.child), (), MultiGrainLedger())
    candidate.validate()
    return candidate


def _proposal_arrays(state, proposals):
    grid = state.supports.shape[1:]
    index = {grain_id: position for position, grain_id in enumerate(state.grain_ids)}
    prepared = []
    seen = set()
    for proposal in proposals:
        if proposal.interface_id in seen:
            raise ValueError("interface IDs must be unique within a transaction")
        seen.add(proposal.interface_id)
        if proposal.donor_id == proposal.receiver_id:
            raise ValueError("self-transfer is not a boundary event")
        if proposal.donor_id not in index or proposal.receiver_id not in index:
            raise ValueError("proposal references an unknown grain ID")
        requested = np.asarray(proposal.requested_fraction, dtype=float)
        if requested.shape != grid or np.any(~np.isfinite(requested)) or np.any(
                requested < 0.0):
            raise ValueError("requested transfer must be finite and nonnegative")
        for name in LINE_FIELDS:
            if name not in proposal.line_export_m2:
                raise ValueError(f"proposal omits explicit {name} export")
            exported = np.asarray(proposal.line_export_m2[name], dtype=float)
            shape = np.asarray(getattr(proposal.product_owner, name)).shape
            if exported.shape != shape or np.any(~np.isfinite(exported)) or np.any(
                    exported < 0.0):
                raise ValueError(f"invalid explicit {name} export")
        if proposal.physical_channels is not None:
            channels = proposal.physical_channels
            expected = grid+(3, np.asarray(
                proposal.product_owner.mobile_plus_m2).shape[2])
            if (np.asarray(channels.boundary_plus_m2).shape != expected
                    or np.asarray(channels.boundary_minus_m2).shape != expected):
                raise ValueError("physical boundary reservoirs are not grid matched")
            if np.asarray(channels.boundary_junction_m2).shape != np.asarray(
                    proposal.product_owner.junction_m2).shape:
                raise ValueError("physical boundary junction is not grid matched")
            for value in (channels.boundary_plus_m2,
                          channels.boundary_minus_m2,
                          channels.boundary_junction_m2,
                          channels.annihilated_line_m2,
                          channels.external_sink_line_m2):
                if np.any(~np.isfinite(value)) or np.any(np.asarray(value) < 0.0):
                    raise ValueError("physical transfer channels must be finite and nonnegative")
            if not math.isfinite(float(proposal.physical_interval_s)) or float(
                    proposal.physical_interval_s) <= 0.0:
                raise ValueError("physical channel proposal lacks a valid interval")
        prepared.append((proposal, requested, index[proposal.donor_id],
                         index[proposal.receiver_id]))
    return prepared


def joint_material_transaction(state: MultiGrainCommonState, proposals):
    """Apply simultaneous competing transfers from one immutable state.

    This function enforces material and explicit line-inventory feasibility.
    It returns an unpublished candidate.  A future complete combined-energy
    transaction must accept that candidate before a production driver may
    publish it.
    """
    state.validate()
    prepared = _proposal_arrays(state, tuple(proposals))
    if not prepared:
        return JointTransferResult(
            False, False, "NO_PROPOSALS", state, {}, {}, {}, {})
    outgoing = np.zeros_like(state.supports)
    for _, requested, donor, _ in prepared:
        outgoing[donor] += requested
    scales = np.ones_like(state.supports)
    positive = outgoing > 0.0
    scales[positive] = np.minimum(
        1.0, state.supports[positive]/outgoing[positive])
    accepted = {}
    next_support = np.asarray(state.supports, dtype=float).copy()
    for proposal, requested, donor, receiver in prepared:
        extent = requested*scales[donor]
        accepted[proposal.interface_id] = extent
        next_support[donor] -= extent
        next_support[receiver] += extent
    next_support = np.maximum(next_support, 0.0)
    support_closure = float(np.max(np.abs(np.sum(next_support, axis=0)-1.0)))
    if support_closure > SUPPORT_TOLERANCE:
        raise RuntimeError("joint support transaction failed material closure")

    owner_values = [{item.name: _weight(getattr(owner, item.name),
                                        state.supports[index])
                     for item in fields(CommonWallState)}
                    for index, owner in enumerate(state.owners)]
    line_export_totals = {name: 0.0 for name in LINE_FIELDS}
    maximum_line_closure = 0.0
    interfaces = {item.component_id: item for item in state.interfaces}
    interface_order = [item.component_id for item in state.interfaces]
    fresh_total = revisit_total = annihilated_total = sink_total = 0.0
    for proposal, _, donor, receiver in prepared:
        extent = accepted[proposal.interface_id]
        for item in fields(CommonWallState):
            donor_value = np.asarray(getattr(state.owners[donor], item.name))
            product_value = np.asarray(getattr(proposal.product_owner, item.name))
            moved_out = _weight(donor_value, extent)
            moved_in = _weight(product_value, extent)
            if item.name in LINE_FIELDS:
                exported = _weight(proposal.line_export_m2[item.name], extent)
                residual = moved_out-moved_in-exported
                if residual.size:
                    scale = max(float(np.max(np.abs(moved_out))), 1.0)
                    maximum_line_closure = max(
                        maximum_line_closure,
                        float(np.max(np.abs(residual))/scale))
                    if float(np.max(np.abs(residual))) > SUPPORT_TOLERANCE*scale:
                        raise ValueError(
                            f"{proposal.interface_id} {item.name} line export does not close")
                line_export_totals[item.name] += float(np.sum(exported))
            owner_values[donor][item.name] -= moved_out
            owner_values[receiver][item.name] += moved_in
        channels = proposal.physical_channels
        if channels is not None:
            component = interfaces.get(proposal.interface_id)
            if component is None:
                grid = extent.shape
                component = InterfaceComponentState(
                    proposal.interface_id, proposal.donor_id,
                    proposal.receiver_id, np.zeros(grid), np.zeros(grid),
                    np.zeros_like(channels.boundary_plus_m2),
                    np.zeros_like(channels.boundary_minus_m2),
                    np.zeros_like(channels.boundary_junction_m2))
                interface_order.append(proposal.interface_id)
            if {component.grain_a_id, component.grain_b_id} != {
                    proposal.donor_id, proposal.receiver_id}:
                raise ValueError("interface identity was reused for another grain pair")
            sign = (1.0 if (proposal.donor_id == component.grain_a_id
                            and proposal.receiver_id == component.grain_b_id)
                    else -1.0)
            prior_sign = np.sign(component.cumulative_signed_sweep_fraction)
            if prior_sign != 0.0 and sign != prior_sign:
                # Retreat first traverses material swept by the previous
                # direction; only any excess enters previously unseen support.
                revisit = np.minimum(extent, component.first_passage_fraction)
                fresh = extent-revisit
            else:
                fresh = np.minimum(
                    extent, np.maximum(
                        1.0-component.first_passage_fraction, 0.0))
                revisit = extent-fresh
            multiplier = extent[(...,)+(None,)*(channels.boundary_plus_m2.ndim
                                                   -extent.ndim)]
            junction_multiplier = extent[(...,)+(None,)*(
                channels.boundary_junction_m2.ndim-extent.ndim)]
            component = replace(
                component,
                first_passage_fraction=component.first_passage_fraction+fresh,
                revisit_fraction=component.revisit_fraction+revisit,
                boundary_plus_m2=(component.boundary_plus_m2
                                  +multiplier*channels.boundary_plus_m2),
                boundary_minus_m2=(component.boundary_minus_m2
                                   +multiplier*channels.boundary_minus_m2),
                boundary_junction_m2=(component.boundary_junction_m2
                                      +junction_multiplier
                                      *channels.boundary_junction_m2),
                exposure_s=component.exposure_s+proposal.physical_interval_s,
                cumulative_signed_sweep_fraction=(
                    component.cumulative_signed_sweep_fraction
                    +sign*float(np.sum(extent, dtype=np.longdouble))),
                cumulative_absolute_sweep_fraction=(
                    component.cumulative_absolute_sweep_fraction
                    +float(np.sum(extent, dtype=np.longdouble))))
            interfaces[proposal.interface_id] = component
            fresh_total += float(np.sum(fresh, dtype=np.longdouble))
            revisit_total += float(np.sum(revisit, dtype=np.longdouble))
            annihilated_total += float(np.sum(
                extent*channels.annihilated_line_m2, dtype=np.longdouble))
            sink_total += float(np.sum(
                extent*channels.external_sink_line_m2, dtype=np.longdouble))

    next_owners = []
    for grain, old_owner in enumerate(state.owners):
        arrays = {}
        support = next_support[grain]
        for item in fields(CommonWallState):
            extensive = owner_values[grain][item.name]
            weight = support[(...,)+(None,)*(extensive.ndim-support.ndim)]
            old = np.asarray(getattr(old_owner, item.name), dtype=float)
            arrays[item.name] = np.divide(
                extensive, weight, out=old.copy(), where=weight > SUPPORT_TOLERANCE)
        next_owners.append(CommonWallState(**arrays))

    proposed_total = float(sum(np.sum(item[1]) for item in prepared))
    accepted_total = float(sum(np.sum(value) for value in accepted.values()))
    overdraft = float(np.max(np.maximum(outgoing-state.supports, 0.0)))
    ledger = MultiGrainLedger(
        attempted_transactions=state.ledger.attempted_transactions+1,
        capacity_feasible_candidates=state.ledger.capacity_feasible_candidates+1,
        energy_accepted_transactions=state.ledger.energy_accepted_transactions,
        rejected_transactions=state.ledger.rejected_transactions,
        proposed_material_fraction=(state.ledger.proposed_material_fraction
                                    +proposed_total),
        accepted_material_fraction=(state.ledger.accepted_material_fraction
                                    +accepted_total),
        capacity_limited_material_fraction=(
            state.ledger.capacity_limited_material_fraction
            +proposed_total-accepted_total),
        maximum_support_closure=max(state.ledger.maximum_support_closure,
                                    support_closure),
        maximum_donor_overdraft=max(state.ledger.maximum_donor_overdraft,
                                    overdraft),
        maximum_line_export_closure_m2=max(
            state.ledger.maximum_line_export_closure_m2,
            maximum_line_closure),
        fresh_sweep_fraction=state.ledger.fresh_sweep_fraction+fresh_total,
        revisit_sweep_fraction=state.ledger.revisit_sweep_fraction+revisit_total,
        annihilated_line_cell_sum_m2=(
            state.ledger.annihilated_line_cell_sum_m2+annihilated_total),
        external_sink_line_cell_sum_m2=(
            state.ledger.external_sink_line_cell_sum_m2+sink_total))
    candidate = MultiGrainCommonState(
        state.grain_ids, next_support, tuple(next_owners),
        tuple(interfaces[key] for key in interface_order),
        ledger, state.schema)
    candidate.validate()
    return JointTransferResult(
        False, True, "CAPACITY_FEASIBLE_UNPRICED_CANDIDATE", candidate, accepted,
        {grain_id: scales[index] for index, grain_id in enumerate(state.grain_ids)
         if np.any(outgoing[index] > 0.0)}, line_export_totals,
        {"annihilated": annihilated_total, "external_sink": sink_total})


def publish_energy_accepted_candidate(result, energy_decision):
    """Publish only a candidate accepted by the complete combined functional.

    ``energy_decision`` is deliberately supplied by the complete production
    energy transaction.  This capacity operator may not infer dissipation or
    heat as the residual needed to close a balance.
    """
    if not result.capacity_feasible:
        raise ValueError("joint candidate is not capacity feasible")
    required = ("accepted", "complete_functional", "independent_dissipation",
                "first_law_residual_J", "tolerance_J")
    missing = [name for name in required if not hasattr(energy_decision, name)]
    if missing:
        raise ValueError(f"complete energy decision omits {missing}")
    if not bool(energy_decision.complete_functional):
        raise ValueError("partial pairwise energy cannot publish a joint candidate")
    if not bool(energy_decision.independent_dissipation):
        raise ValueError("joint dissipation must be computed independently")
    if not hasattr(energy_decision, "candidate_state_digest"):
        raise ValueError("energy decision is not bound to a candidate state")
    if energy_decision.candidate_state_digest != multigrain_state_digest(
            result.candidate):
        raise ValueError("stale or unrelated joint energy decision")
    for name in ("first_law_residual_J", "tolerance_J"):
        if not math.isfinite(float(getattr(energy_decision, name))):
            raise ValueError("joint energy decision contains nonfinite values")
    if float(energy_decision.tolerance_J) < 0.0:
        raise ValueError("joint energy tolerance must be nonnegative")
    if abs(float(energy_decision.first_law_residual_J)) > float(
            energy_decision.tolerance_J):
        raise ValueError("joint candidate fails complete first-law closure")
    if not bool(energy_decision.accepted):
        return None
    ledger = replace(
        result.candidate.ledger,
        energy_accepted_transactions=(
            result.candidate.ledger.energy_accepted_transactions+1))
    return replace(result.candidate, ledger=ledger)


def multigrain_checkpoint_arrays(state):
    state.validate()
    arrays = {"supports": np.asarray(state.supports).copy()}
    for index, owner in enumerate(state.owners):
        for item in fields(CommonWallState):
            arrays[f"owner__{index}__{item.name}"] = np.asarray(
                getattr(owner, item.name)).copy()
    for index, interface in enumerate(state.interfaces):
        for name in ("first_passage_fraction", "revisit_fraction",
                     "boundary_plus_m2", "boundary_minus_m2",
                     "boundary_junction_m2"):
            arrays[f"interface__{index}__{name}"] = np.asarray(
                getattr(interface, name)).copy()
    return arrays


def multigrain_checkpoint_metadata(state):
    state.validate()
    metadata = {
        "schema": state.schema,
        "grain_ids": list(state.grain_ids),
        "ledger": asdict(state.ledger),
        "interfaces": [{
            "component_id": item.component_id,
            "grain_a_id": item.grain_a_id,
            "grain_b_id": item.grain_b_id,
            "exposure_s": item.exposure_s,
            "cumulative_work_J": item.cumulative_work_J,
            "cumulative_heat_J": item.cumulative_heat_J,
            "cumulative_signed_sweep_fraction": (
                item.cumulative_signed_sweep_fraction),
            "cumulative_absolute_sweep_fraction": (
                item.cumulative_absolute_sweep_fraction),
            "periodic_winding_x": item.periodic_winding_x,
            "periodic_winding_y": item.periodic_winding_y,
        } for item in state.interfaces],
    }
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def multigrain_from_checkpoint(metadata_json, arrays):
    metadata = json.loads(str(metadata_json))
    if metadata.get("schema") != SCHEMA:
        raise ValueError("unsupported multi-grain checkpoint schema")
    supports = np.asarray(arrays["supports"]).copy()
    owners = []
    for index, _ in enumerate(metadata["grain_ids"]):
        owners.append(CommonWallState(**{
            item.name: np.asarray(arrays[
                f"owner__{index}__{item.name}"]).copy()
            for item in fields(CommonWallState)}))
    interfaces = []
    for index, item in enumerate(metadata["interfaces"]):
        interfaces.append(InterfaceComponentState(
            component_id=str(item["component_id"]),
            grain_a_id=int(item["grain_a_id"]),
            grain_b_id=int(item["grain_b_id"]),
            first_passage_fraction=np.asarray(arrays[
                f"interface__{index}__first_passage_fraction"]).copy(),
            revisit_fraction=np.asarray(arrays[
                f"interface__{index}__revisit_fraction"]).copy(),
            boundary_plus_m2=np.asarray(arrays[
                f"interface__{index}__boundary_plus_m2"]).copy(),
            boundary_minus_m2=np.asarray(arrays[
                f"interface__{index}__boundary_minus_m2"]).copy(),
            boundary_junction_m2=np.asarray(arrays[
                f"interface__{index}__boundary_junction_m2"]).copy(),
            exposure_s=float(item["exposure_s"]),
            cumulative_work_J=float(item["cumulative_work_J"]),
            cumulative_heat_J=float(item["cumulative_heat_J"]),
            cumulative_signed_sweep_fraction=float(item.get(
                "cumulative_signed_sweep_fraction", 0.0)),
            cumulative_absolute_sweep_fraction=float(item.get(
                "cumulative_absolute_sweep_fraction", 0.0)),
            periodic_winding_x=int(item.get("periodic_winding_x", 0)),
            periodic_winding_y=int(item.get("periodic_winding_y", 0))))
    state = MultiGrainCommonState(
        tuple(int(value) for value in metadata["grain_ids"]), supports,
        tuple(owners), tuple(interfaces),
        MultiGrainLedger(**metadata["ledger"]), metadata["schema"])
    state.validate()
    return state
