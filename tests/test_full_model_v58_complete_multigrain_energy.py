from dataclasses import replace

import numpy as np
import pytest

from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.common_tensorial_wall import (
    wall_total_free_energy_density_J_m3,
)
from full_model.production.complete_multigrain_energy import (
    MultiGrainDissipation, evaluate_complete_multigrain_energy,
    evaluate_joint_multigrain_transaction,
)
from full_model.production.multigrain_common_state import (
    JointTransferProposal, MultiGrainCommonState, PhysicalTransferLaw,
    audit_multigrain_nye, derive_physical_transfer_proposal,
    joint_material_transaction, multigrain_state_digest,
    publish_energy_accepted_candidate,
)
from tests.test_full_model_multigrain_common_state import owner, zero_exports
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, rotated_system_fields,
)


def _state(n=12):
    supports = np.zeros((3, n, n))
    supports[0, :n//2] = 1.0
    supports[1, n//2:, :n//2] = 1.0
    supports[2, n//2:, n//2:] = 1.0
    return MultiGrainCommonState(
        (10, 20, 30), supports,
        (owner((n, n), value=2e14, orientation=0.0),
         owner((n, n), value=1e14, orientation=.2),
         owner((n, n), value=1.5e14, orientation=-.15)))


def test_nye_audit_separates_support_gradient_and_owner_mismatch():
    state = _state(16)
    beta = state.owners[1].beta_p.copy()
    beta[..., 1, 2] = .02
    family_nye = state.owners[2].family_nye_m1.copy()
    family_nye[..., 0, 0, 1] = 3e4
    state = replace(state, owners=(
        state.owners[0], replace(state.owners[1], beta_p=beta),
        replace(state.owners[2], family_nye_m1=family_nye)))
    audit = audit_multigrain_nye(state, 2e-8)
    assert np.linalg.norm(audit.support_gradient_m1) > 0.0
    assert np.linalg.norm(audit.owner_reservoir_mismatch_m1) > 0.0
    np.testing.assert_allclose(
        audit.exact_reconstructed_m1,
        audit.owner_curl_weighted_m1+audit.support_gradient_m1
        +audit.discrete_representation_residual_m1,
        rtol=2e-13, atol=2e-8)


def test_complete_joint_identity_is_bound_and_accepted():
    spacing = 2e-8
    state = _state()
    zero = np.zeros(state.supports.shape[1:])
    proposal = JointTransferProposal(
        "10-20", 10, 20, zero, state.owners[0],
        zero_exports(state.owners[0]))
    capacity = joint_material_transaction(state, (proposal,))
    parameters = CommonWallParameters(spacing_m=spacing)
    result = evaluate_joint_multigrain_transaction(
        state, capacity, spacing_m=spacing,
        represented_thickness_m=5e-10, wall_parameters=parameters,
        interval_s=1e-9, dissipation=MultiGrainDissipation())
    assert result.decision.accepted
    assert result.decision.complete_functional
    assert result.decision.independent_dissipation
    assert result.decision.before_state_digest == multigrain_state_digest(state)
    assert result.decision.candidate_state_digest == multigrain_state_digest(
        capacity.candidate)
    assert abs(result.decision.first_law_residual_J) <= result.decision.tolerance_J


def test_nonzero_event_cannot_invent_heat_to_close_energy():
    spacing = 2e-8
    state = _state()
    request = np.zeros(state.supports.shape[1:])
    request[4:6, :] = .1
    proposal = JointTransferProposal(
        "10-20", 10, 20, request, state.owners[0],
        zero_exports(state.owners[0]))
    capacity = joint_material_transaction(state, (proposal,))
    result = evaluate_joint_multigrain_transaction(
        state, capacity, spacing_m=spacing,
        represented_thickness_m=5e-10,
        wall_parameters=CommonWallParameters(spacing_m=spacing),
        interval_s=1e-9, dissipation=MultiGrainDissipation())
    # The event is admitted only if the independently supplied zero
    # dissipation closes both availability and the first law; no residual heat
    # is manufactured internally.
    assert result.decision.generated_heat_J == 0.0
    if not result.decision.accepted:
        assert result.published_state is state
        assert result.decision.classification in {
            "REJECTED_UPHILL_COMPLETE_PHYSICAL_ENERGY",
            "REJECTED_INDEPENDENT_DISSIPATION_MISMATCH",
            "REJECTED_FIRST_LAW_MISMATCH",
        }


def test_stale_decision_is_rejected_by_publication_guard():
    state = _state(8)
    request = np.zeros((8, 8)); request[:2] = .05
    proposal = JointTransferProposal(
        "10-20", 10, 20, request, state.owners[0],
        zero_exports(state.owners[0]))
    capacity = joint_material_transaction(state, (proposal,))
    result = evaluate_joint_multigrain_transaction(
        state, capacity, spacing_m=2e-8,
        represented_thickness_m=5e-10,
        wall_parameters=CommonWallParameters(spacing_m=2e-8),
        interval_s=1e-9, dissipation=MultiGrainDissipation())
    other = replace(capacity, candidate=replace(
        capacity.candidate, supports=capacity.candidate.supports.copy()))
    other.candidate.supports[0, 0, 0] -= .01
    other.candidate.supports[1, 0, 0] += .01
    with pytest.raises(ValueError, match="stale or unrelated"):
        publish_energy_accepted_candidate(other, result.decision)


def test_complete_energy_counts_multiphase_interface_once():
    spacing = 2e-8
    state = _state()
    energy = evaluate_complete_multigrain_energy(
        state, spacing_m=spacing, represented_thickness_m=5e-10,
        wall_parameters=CommonWallParameters(spacing_m=spacing),
        phase_barrier_J_m3=5e6, phase_gradient_J_m=5e-7,
        reference_temperature_K=900.0)
    assert energy.phase_gradient_J > 0.0
    assert energy.phase_local_J == 0.0
    assert energy.numerical_constraint_J == 0.0


def test_nonlinear_defect_storage_is_integrated_by_material_owner():
    spacing = 2e-8
    state = _state(8)
    supports = np.empty_like(state.supports)
    supports[0] = .2; supports[1] = .3; supports[2] = .5
    state = replace(state, supports=supports)
    parameters = CommonWallParameters(spacing_m=spacing)
    energy = evaluate_complete_multigrain_energy(
        state, spacing_m=spacing, represented_thickness_m=5e-10,
        wall_parameters=parameters, reference_temperature_K=900.0)
    expected_density = sum(
        state.supports[index]*wall_total_free_energy_density_J_m3(
            item, parameters, (), bcc_four_family_systems())
        for index, item in enumerate(state.owners))
    expected = float(np.sum(expected_density)*spacing**2*5e-10)
    assert energy.defect_storage_J == pytest.approx(expected, rel=2e-15)


def test_physical_transfer_derives_product_channels_and_recurrent_history():
    state = _state(8)
    donor = replace(
        state.owners[0], slip=np.full_like(state.owners[0].slip, .03),
        beta_p=np.full_like(state.owners[0].beta_p, .02),
        alignment_m2=np.full_like(state.owners[0].alignment_m2, 4e6),
        family_nye_m1=np.full_like(state.owners[0].family_nye_m1, 3e5))
    state = replace(state, owners=(donor, state.owners[1], state.owners[2]))
    request = np.zeros((8, 8)); request[3:5, :] = .2
    law = PhysicalTransferLaw(
        transmission_fraction=.5, boundary_storage_fraction=.1,
        neutral_sink_fraction=.05)
    forward = derive_physical_transfer_proposal(
        state, interface_id="junction-arm-a", donor_id=10, receiver_id=20,
        requested_fraction=request, law=law, interval_s=2e-9,
        systems=bcc_four_family_systems())
    # Line transmission is derived from the donor law. Kinematic history is
    # the persistent receiver owner's dormant history; migration must not
    # inject a scaled donor plastic distortion into that crystal.
    np.testing.assert_allclose(
        forward.product_owner.mobile_plus_m2,
        .5*state.owners[0].mobile_plus_m2)
    np.testing.assert_array_equal(
        forward.product_owner.orientation_rad,
        state.owners[1].orientation_rad)
    for name in ("slip", "beta_p", "alignment_m2", "family_nye_m1"):
        np.testing.assert_array_equal(
            getattr(forward.product_owner, name),
            getattr(state.owners[1], name))
    first = joint_material_transaction(state, (forward,))
    interface = first.candidate.interfaces[0]
    assert interface.exposure_s == 2e-9
    assert np.sum(interface.first_passage_fraction) > 0.0
    assert np.sum(interface.revisit_fraction) == 0.0
    assert np.sum(interface.boundary_plus_m2) > 0.0
    exported = sum(first.line_export_m2_by_field.values())
    stored = float(np.sum(interface.boundary_plus_m2+interface.boundary_minus_m2)
                   +np.sum(interface.boundary_junction_m2))
    channels = first.physical_channel_cell_sums_m2
    np.testing.assert_allclose(
        exported, stored+channels["annihilated"]+channels["external_sink"],
        rtol=2e-15)

    reverse = derive_physical_transfer_proposal(
        first.candidate, interface_id="junction-arm-a", donor_id=20,
        receiver_id=10, requested_fraction=request, law=law,
        interval_s=3e-9, systems=bcc_four_family_systems())
    second = joint_material_transaction(first.candidate, (reverse,))
    recurrent = second.candidate.interfaces[0]
    assert recurrent.exposure_s == 5e-9
    assert np.sum(recurrent.revisit_fraction) > 0.0
    assert recurrent.cumulative_absolute_sweep_fraction > abs(
        recurrent.cumulative_signed_sweep_fraction)


def test_two_physical_incident_edges_share_capacity_and_keep_histories():
    state = _state(8)
    request = np.zeros((8, 8)); request[:4] = .8
    law = PhysicalTransferLaw(.7, .1, .05)
    proposals = tuple(derive_physical_transfer_proposal(
        state, interface_id=name, donor_id=10, receiver_id=receiver,
        requested_fraction=request, law=law, interval_s=1e-9,
        systems=bcc_four_family_systems())
        for name, receiver in (("10-20", 20), ("10-30", 30)))
    result = joint_material_transaction(state, proposals)
    np.testing.assert_allclose(np.sum(result.candidate.supports, axis=0), 1.0)
    assert len(result.candidate.interfaces) == 2
    assert result.candidate.ledger.capacity_limited_material_fraction > 0.0
    assert result.candidate.ledger.maximum_line_export_closure_m2 < 1e-13


def test_misoriented_transfer_closes_vector_burgers_content_at_interface():
    state = _state(8)
    biased = state.owners[0].mobile_plus_m2.copy()
    biased[..., 0] *= 1.7
    state = replace(state, owners=(
        replace(state.owners[0], mobile_plus_m2=biased),
        state.owners[1], state.owners[2]))
    systems = bcc_four_family_systems()
    request = np.zeros((8, 8)); request[:2] = .1
    proposal = derive_physical_transfer_proposal(
        state, interface_id="10-20", donor_id=10, receiver_id=20,
        requested_fraction=request,
        law=PhysicalTransferLaw(.6, .1, .0), interval_s=1e-9,
        systems=systems)
    bd, _, _ = rotated_system_fields(systems, state.owners[0].orientation_rad)
    br, _, _ = rotated_system_fields(systems, state.owners[1].orientation_rad)
    pairs = (("mobile_plus_m2", "mobile_minus_m2"),
             ("forest_plus_m2", "forest_minus_m2"),
             ("wall_plus_m2", "wall_minus_m2"))
    for reservoir, (plus, minus) in enumerate(pairs):
        donor_b = np.einsum(
            "...a,...ai->...i",
            getattr(state.owners[0], plus)-getattr(state.owners[0], minus), bd)
        product_b = np.einsum(
            "...a,...ai->...i",
            getattr(proposal.product_owner, plus)
            -getattr(proposal.product_owner, minus), br)
        np.testing.assert_allclose(
            donor_b-product_b,
            proposal.physical_channels.boundary_burgers_m1[..., reservoir, :],
            rtol=5e-14, atol=5e-10)
    result = joint_material_transaction(state, (proposal,))
    assert np.linalg.norm(result.candidate.interfaces[0].boundary_burgers_m1) > 0.0
    component = result.candidate.interfaces[0]
    stored = float(np.sum(component.boundary_plus_m2
                          +component.boundary_minus_m2)
                   +np.sum(component.boundary_junction_m2))
    exported = sum(result.line_export_m2_by_field.values())
    np.testing.assert_allclose(
        exported,
        stored+result.physical_channel_cell_sums_m2["annihilated"]
        +result.physical_channel_cell_sums_m2["external_sink"], rtol=2e-14)
