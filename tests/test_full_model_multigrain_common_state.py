from dataclasses import fields
from types import SimpleNamespace

import numpy as np

from full_model.production.common_front_state import (
    LINE_FIELDS, initialize_common_front, reconstruct_common,
)
from full_model.production.common_tensorial_wall import CommonWallState
from full_model.production.moving_front import (
    DefectState, initialize_existing_subgrain_front,
)
from full_model.production.multigrain_common_state import (
    InterfaceComponentState, JointTransferProposal, MultiGrainCommonState,
    from_common_front_pair, joint_material_transaction,
    multigrain_checkpoint_arrays, multigrain_checkpoint_metadata,
    multigrain_from_checkpoint, publish_energy_accepted_candidate,
    reconstruct_multigrain_common,
)


def owner(grid=(8, 8), families=4, value=1.0, orientation=0.0):
    family = np.full(grid+(families,), value)
    return CommonWallState(
        family.copy(), family.copy(), family.copy(), family.copy(),
        family.copy(), family.copy(), np.zeros(grid+(0,)),
        np.full(grid, 0.25), np.zeros(grid), np.zeros_like(family),
        np.zeros(grid+(3, 3)), np.zeros(grid+(families, 3)),
        np.zeros(grid+(families, 3, 3)), np.full(grid, orientation),
        np.full(grid, 900.0))


def zero_exports(product):
    return {name: np.zeros_like(getattr(product, name)) for name in LINE_FIELDS}


def test_unsupported_owner_cannot_contribute_to_reconstructed_state():
    support = np.zeros((3, 8, 8)); support[0] = 0.6; support[1] = 0.4
    state = MultiGrainCommonState(
        (11, 22, 33), support,
        (owner(value=2.0), owner(value=4.0), owner(value=1.0e12)))
    common, correction = reconstruct_multigrain_common(state, 1.0e-7)
    np.testing.assert_allclose(common.mobile_plus_m2, 2.8)
    np.testing.assert_array_equal(correction, 0.0)


def test_pair_adapter_reconstructs_qualified_pair_limit():
    common = owner(value=3.0)
    grid = common.orientation_rad.shape
    child = np.zeros(grid); child[2:6, 2:6] = 1.0
    defect = DefectState(
        common.mobile_plus_m2, common.mobile_minus_m2,
        common.forest_plus_m2+common.forest_minus_m2,
        np.sum(common.wall_plus_m2+common.wall_minus_m2, axis=2))
    front = initialize_existing_subgrain_front(defect, child, 0, 1)
    pair = initialize_common_front(front, common)
    multi = from_common_front_pair(pair, parent_id=17, child_id=29)
    pair_mix, pair_correction = reconstruct_common(pair, 1.0e-7)
    multi_mix, multi_correction = reconstruct_multigrain_common(multi, 1.0e-7)
    for item in fields(CommonWallState):
        np.testing.assert_allclose(
            getattr(multi_mix, item.name), getattr(pair_mix, item.name))
    np.testing.assert_allclose(multi_correction, pair_correction)


def test_two_incident_edges_share_donor_capacity_without_double_spending():
    support = np.zeros((3, 8, 8)); support[0] = 0.8
    support[1] = 0.1; support[2] = 0.1
    owners = (owner(value=2.0), owner(value=2.0), owner(value=2.0))
    state = MultiGrainCommonState((101, 7, 55), support, owners)
    requested = np.full((8, 8), 0.6)
    proposals = (
        JointTransferProposal("A-B", 101, 7, requested, owners[0],
                              zero_exports(owners[0])),
        JointTransferProposal("A-C", 101, 55, requested, owners[0],
                              zero_exports(owners[0])),
    )
    result = joint_material_transaction(state, proposals)
    assert not result.accepted
    assert result.capacity_feasible
    assert result.classification == "CAPACITY_FEASIBLE_UNPRICED_CANDIDATE"
    np.testing.assert_allclose(result.accepted_fraction_by_interface["A-B"], .4)
    np.testing.assert_allclose(result.accepted_fraction_by_interface["A-C"], .4)
    np.testing.assert_allclose(result.candidate.supports[0], 0.0, atol=1e-15)
    np.testing.assert_allclose(result.candidate.supports[1], 0.5)
    np.testing.assert_allclose(result.candidate.supports[2], 0.5)
    np.testing.assert_allclose(np.sum(result.candidate.supports, axis=0), 1.0)
    assert result.candidate.ledger.capacity_limited_material_fraction > 0.0
    assert result.candidate.ledger.maximum_support_closure < 1e-14


def test_joint_transaction_is_label_and_proposal_order_neutral():
    support = np.zeros((3, 8, 8)); support[:] = np.array([.8, .1, .1])[:, None, None]
    owners = (owner(value=2.0), owner(value=2.0), owner(value=2.0))
    state = MultiGrainCommonState((3, 8, 21), support, owners)
    requested = np.full((8, 8), .6)
    ab = JointTransferProposal("3-8", 3, 8, requested, owners[0],
                               zero_exports(owners[0]))
    ac = JointTransferProposal("3-21", 3, 21, requested, owners[0],
                               zero_exports(owners[0]))
    forward = joint_material_transaction(state, (ab, ac)).candidate
    reverse = joint_material_transaction(state, (ac, ab)).candidate
    np.testing.assert_array_equal(forward.supports, reverse.supports)
    for left, right in zip(forward.owners, reverse.owners):
        for item in fields(CommonWallState):
            np.testing.assert_array_equal(
                getattr(left, item.name), getattr(right, item.name))

    permuted_state = MultiGrainCommonState(
        (21, 3, 8), support[[2, 0, 1]], (owners[2], owners[0], owners[1]))
    permuted = joint_material_transaction(permuted_state, (
        JointTransferProposal("3-21", 3, 21, requested, owners[0],
                              zero_exports(owners[0])),
        JointTransferProposal("3-8", 3, 8, requested, owners[0],
                              zero_exports(owners[0])),
    )).candidate
    physical = {grain_id: forward.supports[index]
                for index, grain_id in enumerate(forward.grain_ids)}
    permuted_physical = {grain_id: permuted.supports[index]
                         for index, grain_id in enumerate(permuted.grain_ids)}
    for grain_id in physical:
        np.testing.assert_array_equal(physical[grain_id],
                                      permuted_physical[grain_id])


def test_line_reduction_requires_explicit_export_and_rolls_back_input():
    support = np.zeros((2, 8, 8)); support[0] = .75; support[1] = .25
    donor = owner(value=4.0); product = owner(value=2.0)
    state = MultiGrainCommonState((1, 2), support, (donor, product))
    bad = JointTransferProposal(
        "bad", 1, 2, np.full((8, 8), .1), product, zero_exports(product))
    try:
        joint_material_transaction(state, (bad,))
    except ValueError as error:
        assert "line export does not close" in str(error)
    else:
        raise AssertionError("unledgered line deletion was accepted")
    np.testing.assert_array_equal(state.supports[0], .75)
    exports = {}
    for name in LINE_FIELDS:
        exports[name] = np.asarray(getattr(donor, name))-np.asarray(
            getattr(product, name))
    good = JointTransferProposal(
        "good", 1, 2, np.full((8, 8), .1), product, exports)
    result = joint_material_transaction(state, (good,))
    assert result.capacity_feasible
    assert all(value > 0.0 for name, value in
               result.line_export_m2_by_field.items()
               if name != "junction_m2")


def test_unpriced_candidate_cannot_be_published_and_rejection_is_atomic():
    support = np.zeros((2, 8, 8)); support[0] = .75; support[1] = .25
    owners = (owner(value=2.0), owner(value=2.0))
    state = MultiGrainCommonState((1, 2), support, owners)
    proposal = JointTransferProposal(
        "edge", 1, 2, np.full((8, 8), .1), owners[0],
        zero_exports(owners[0]))
    result = joint_material_transaction(state, (proposal,))
    assert not result.accepted and result.capacity_feasible
    incomplete = SimpleNamespace(
        accepted=True, complete_functional=False,
        independent_dissipation=True, first_law_residual_J=0.0,
        tolerance_J=1e-20)
    try:
        publish_energy_accepted_candidate(result, incomplete)
    except ValueError as error:
        assert "partial pairwise energy" in str(error)
    else:
        raise AssertionError("unpriced joint candidate was published")
    rejected = SimpleNamespace(
        accepted=False, complete_functional=True,
        independent_dissipation=True, first_law_residual_J=0.0,
        tolerance_J=1e-20)
    assert publish_energy_accepted_candidate(result, rejected) is None
    np.testing.assert_array_equal(state.supports[0], .75)
    accepted = SimpleNamespace(
        accepted=True, complete_functional=True,
        independent_dissipation=True, first_law_residual_J=1e-24,
        tolerance_J=1e-20)
    published = publish_energy_accepted_candidate(result, accepted)
    assert published.ledger.capacity_feasible_candidates == 1
    assert published.ledger.energy_accepted_transactions == 1


def test_temporal_subdivision_converges_exactly_for_constant_joint_products():
    support = np.zeros((3, 8, 8)); support[:] = np.array([.7, .2, .1])[:, None, None]
    owners = (owner(value=2.0), owner(value=2.0), owner(value=2.0))
    state = MultiGrainCommonState((1, 2, 3), support, owners)
    def proposals(amount):
        return (
            JointTransferProposal(
                "1-2", 1, 2, np.full((8, 8), amount), owners[0],
                zero_exports(owners[0])),
            JointTransferProposal(
                "1-3", 1, 3, np.full((8, 8), amount), owners[0],
                zero_exports(owners[0])),
        )
    full = joint_material_transaction(state, proposals(.2)).candidate
    half = joint_material_transaction(state, proposals(.1)).candidate
    half = joint_material_transaction(half, proposals(.1)).candidate
    np.testing.assert_allclose(full.supports, half.supports, atol=1e-15)
    full_mix, _ = reconstruct_multigrain_common(full, 1e-7)
    half_mix, _ = reconstruct_multigrain_common(half, 1e-7)
    for item in fields(CommonWallState):
        np.testing.assert_allclose(
            getattr(full_mix, item.name), getattr(half_mix, item.name),
            atol=1e-15)


def test_exact_checkpoint_round_trip_includes_interfaces_and_ledger():
    support = np.zeros((3, 8, 8)); support[:] = np.array([.6, .3, .1])[:, None, None]
    grid = support.shape[1:]
    interface = InterfaceComponentState(
        "junction-edge-17", 4, 9, np.full(grid, .2), np.full(grid, .03),
        np.full(grid+(3, 4), 2.0), np.full(grid+(3, 4), 1.0),
        np.zeros(grid+(0,)), exposure_s=2e-9, cumulative_work_J=3e-18,
        cumulative_heat_J=2e-18)
    state = MultiGrainCommonState(
        (4, 9, 15), support,
        (owner(value=1.0), owner(value=2.0), owner(value=3.0)),
        (interface,))
    proposal = JointTransferProposal(
        "4-9", 4, 9, np.full(grid, .1), state.owners[0],
        zero_exports(state.owners[0]))
    evolved = joint_material_transaction(state, (proposal,)).candidate
    arrays = multigrain_checkpoint_arrays(evolved)
    restored = multigrain_from_checkpoint(
        multigrain_checkpoint_metadata(evolved), arrays)
    assert restored.grain_ids == evolved.grain_ids
    assert restored.ledger == evolved.ledger
    assert restored.interfaces[0].component_id == "junction-edge-17"
    for key, value in arrays.items():
        replay = multigrain_checkpoint_arrays(restored)[key]
        np.testing.assert_array_equal(value, replay)
