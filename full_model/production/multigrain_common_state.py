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
import json

import numpy as np

from .common_front_state import CommonFrontState, LINE_FIELDS
from .common_tensorial_wall import CommonWallState
from .tensorial_nye import nye_from_plastic_distortion


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


@dataclass(frozen=True)
class JointTransferResult:
    accepted: bool
    capacity_feasible: bool
    classification: str
    candidate: MultiGrainCommonState
    accepted_fraction_by_interface: dict[str, np.ndarray]
    capacity_scale_by_donor: dict[int, np.ndarray]
    line_export_m2_by_field: dict[str, float]


def _weight(value, support):
    array = np.asarray(value, dtype=float)
    return support[(...,)+(None,)*(array.ndim-support.ndim)]*array


def reconstruct_multigrain_common(state: MultiGrainCommonState, spacing_m: float):
    """Reconstruct one physical mixture; take Curl after beta reconstruction."""
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
    total_nye = nye_from_plastic_distortion(values["beta_p"], float(spacing_m))
    bulk_family = sum(
        (_weight(owner.family_nye_m1, state.supports[index])
         for index, owner in enumerate(state.owners)),
        np.zeros_like(np.asarray(state.owners[0].family_nye_m1), dtype=float))
    correction = total_nye-np.sum(bulk_family, axis=2)
    magnitude = np.abs(values["slip"])
    total = np.sum(magnitude, axis=2, keepdims=True)
    family_fraction = np.divide(
        magnitude, total, out=np.full_like(magnitude, 1.0/magnitude.shape[2]),
        where=total > 0.0)
    values["family_nye_m1"] = (
        bulk_family+family_fraction[..., None, None]*correction[..., None, :, :])
    return CommonWallState(**values), correction


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
            False, False, "NO_PROPOSALS", state, {}, {}, {})
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
            maximum_line_closure))
    candidate = MultiGrainCommonState(
        state.grain_ids, next_support, tuple(next_owners), state.interfaces,
        ledger, state.schema)
    candidate.validate()
    return JointTransferResult(
        False, True, "CAPACITY_FEASIBLE_UNPRICED_CANDIDATE", candidate, accepted,
        {grain_id: scales[index] for index, grain_id in enumerate(state.grain_ids)
         if np.any(outgoing[index] > 0.0)}, line_export_totals)


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
            cumulative_heat_J=float(item["cumulative_heat_J"])))
    state = MultiGrainCommonState(
        tuple(int(value) for value in metadata["grain_ids"]), supports,
        tuple(owners), tuple(interfaces),
        MultiGrainLedger(**metadata["ledger"]), metadata["schema"])
    state.validate()
    return state
