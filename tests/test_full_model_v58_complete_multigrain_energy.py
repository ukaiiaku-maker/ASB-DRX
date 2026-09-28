from dataclasses import replace

import numpy as np
import pytest

from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.complete_multigrain_energy import (
    MultiGrainDissipation, evaluate_complete_multigrain_energy,
    evaluate_joint_multigrain_transaction,
)
from full_model.production.multigrain_common_state import (
    JointTransferProposal, MultiGrainCommonState, audit_multigrain_nye,
    joint_material_transaction, multigrain_state_digest,
    publish_energy_accepted_candidate,
)
from tests.test_full_model_multigrain_common_state import owner, zero_exports


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
