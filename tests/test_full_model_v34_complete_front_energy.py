from dataclasses import replace

import numpy as np

from full_model.production.common_front_state import (
    initialize_common_front, state_arrays)
from full_model.production.common_tensorial_wall import (
    CommonWallParameters, CommonWallState)
from full_model.production.complete_front_energy import (
    evaluate_common_front_transaction, evaluate_complete_front_energy,
    independently_assemble_product_rule_nye)
from full_model.production.moving_front import (
    DefectState, initialize_declared_boundary_front)


def _common(n=12, nj=2):
    shape = (n, n, 4)
    beta = np.zeros((n, n, 3, 3))
    return CommonWallState(
        np.full(shape, 2.0e14), np.full(shape, 1.5e14),
        np.full(shape, 1.2e14), np.full(shape, .8e14),
        np.full(shape, .5e14), np.full(shape, .3e14),
        np.full((n, n, nj), .4e14), np.full((n, n), .2),
        np.full((n, n), .1), np.zeros(shape), beta,
        np.zeros(shape+(3,)), np.zeros(shape+(3, 3)),
        np.zeros((n, n)), np.full((n, n), 900.0))


def _state(n=12):
    common = _common(n)
    defect = DefectState(
        common.mobile_plus_m2.copy(), common.mobile_minus_m2.copy(),
        common.forest_plus_m2+common.forest_minus_m2,
        np.sum(common.wall_plus_m2+common.wall_minus_m2, axis=2))
    chi = np.zeros((n, n))
    front = initialize_declared_boundary_front(defect, defect, chi, 0, 1)
    return initialize_common_front(front, common)


def _eta(n, shift=0.0):
    x = np.arange(n)[:, None]
    b = .5*(1.0+np.tanh((x-(.45*n+shift))/1.5))*np.ones((1, n))
    return np.stack((1.0-b, b), axis=2)


def _accepted_front(state, amount=.25):
    chi = np.zeros_like(state.front.chi)
    chi[:, :3] = amount
    return replace(state.front, chi=chi, processed_max=chi.copy(),
                   cleanup_max=chi.copy())


def _options(spacing=2e-8, boundary_line=1e-9):
    return dict(
        wall_parameters=CommonWallParameters(spacing_m=spacing),
        mean_strain=np.zeros((2, 2)), phase_barrier_J_m3=5e6,
        phase_gradient_J_m=5e-7,
        boundary_line_energy_J_m=boundary_line,
        reference_temperature_K=900.0)


def test_identical_owner_beta_has_no_spurious_support_interface_nye():
    spacing = 2e-8
    state = _state(16)
    x = np.arange(16)[:, None]
    chi = .5+.3*np.sin(2*np.pi*x/16)*np.ones((1, 16))
    front = replace(state.front, chi=chi, processed_max=chi.copy(),
                    cleanup_max=chi.copy())
    state = replace(state, front=front)
    beta = state.parent.beta_p.copy()
    beta[..., 0, 2] = .035
    state = replace(state, parent=replace(state.parent, beta_p=beta),
                    child=replace(state.child, beta_p=beta),
                    wake=replace(state.wake, beta_p=beta))
    audit = independently_assemble_product_rule_nye(state, spacing)
    assert np.max(np.abs(audit.support_interface_m1)) < 1e-9
    assert np.max(np.abs(audit.exact_reconstructed_m1)) < 1e-9


def test_beta_jump_support_gradient_has_signed_independent_product_rule():
    spacing = 2e-8
    state = _state(16)
    x = np.arange(16)[:, None]
    chi = .5+.25*np.sin(2*np.pi*x/16)*np.ones((1, 16))
    front = replace(state.front, chi=chi, processed_max=chi.copy(),
                    cleanup_max=chi.copy())
    beta = state.child.beta_p.copy()
    beta[..., 1, 2] = .02
    state = replace(state, front=front,
                    child=replace(state.child, beta_p=beta))
    audit = independently_assemble_product_rule_nye(state, spacing)
    assert np.linalg.norm(audit.support_interface_m1) > 0.0
    np.testing.assert_allclose(
        audit.independently_assembled_m1,
        audit.exact_reconstructed_m1, rtol=2e-12, atol=2e-9)


def test_complete_trial_accepts_downhill_and_prices_junction_and_beta():
    spacing = 2e-8
    volume = spacing*spacing*5e-10
    state = _state()
    eta = _eta(12)
    accepted = _accepted_front(state)
    result = evaluate_common_front_transaction(
        state, accepted, eta, eta, spacing_m=spacing,
        cell_volume_m3=volume, represented_thickness_m=5e-10,
        transmission_fraction=.6, boundary_storage_fraction=.05,
        neutral_sink_fraction=.05, energy_kwargs=_options(spacing))
    assert result.decision.accepted
    assert abs(result.decision.first_law_residual_J) < 1e-27
    assert result.published_state is result.candidate_state
    assert result.published_state.ledger.complete_energy_accepted == 1
    assert result.published_state.ledger.generated_heat_J > 0.0
    assert result.published_state.ledger.maximum_abs_first_law_residual_J < 1e-27
    assert result.decision.before.signed_junction_storage_J != 0.0
    assert (result.decision.candidate.signed_junction_storage_J
            != result.decision.before.signed_junction_storage_J)

    parent_beta = state.parent.beta_p.copy()
    parent_beta[..., 0, 0] = .02
    beta_state = replace(state, parent=replace(state.parent, beta_p=parent_beta))
    priced = evaluate_complete_front_energy(
        beta_state, eta, spacing_m=spacing,
        represented_thickness_m=5e-10, **_options(spacing))
    unpriced = evaluate_complete_front_energy(
        state, eta, spacing_m=spacing,
        represented_thickness_m=5e-10, **_options(spacing))
    assert priced.recoverable_elastic_J > unpriced.recoverable_elastic_J


def test_uphill_complete_trial_rejects_without_mutating_any_owner_or_ledger():
    spacing = 2e-8
    volume = spacing*spacing*5e-10
    state = _state()
    before = state_arrays(state)
    result = evaluate_common_front_transaction(
        state, _accepted_front(state), _eta(12), _eta(12),
        spacing_m=spacing, cell_volume_m3=volume,
        represented_thickness_m=5e-10, transmission_fraction=.0,
        boundary_storage_fraction=1.0, neutral_sink_fraction=.0,
        energy_kwargs=_options(spacing, boundary_line=1e-4))
    assert not result.decision.accepted
    assert result.decision.classification == "REJECTED_UPHILL_COMPLETE_PHYSICAL_ENERGY"
    assert result.published_state is state
    assert result.decision.first_law_residual_J == 0.0
    for name, value in before.items():
        np.testing.assert_array_equal(
            value, state_arrays(result.published_state)[name])
    assert result.candidate_state.ledger.accepted_commits == 1
    assert result.published_state.ledger.accepted_commits == 0


def test_exact_zero_event_is_bitwise_identity_including_ledger():
    spacing = 2e-8
    state = _state()
    eta = _eta(12)
    result = evaluate_common_front_transaction(
        state, state.front, eta, eta, spacing_m=spacing,
        cell_volume_m3=spacing*spacing*5e-10,
        represented_thickness_m=5e-10, transmission_fraction=.5,
        boundary_storage_fraction=.1, neutral_sink_fraction=.05,
        energy_kwargs=_options(spacing))
    assert result.decision.classification == "EXACT_ZERO_EVENT_IDENTITY"
    assert result.published_state is state
    assert result.published_state.ledger == state.ledger
    for name, value in state_arrays(state).items():
        np.testing.assert_array_equal(
            value, state_arrays(result.published_state)[name])
