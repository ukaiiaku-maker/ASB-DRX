from dataclasses import replace

import numpy as np
import pytest

from full_model.production.common_front_state import (
    initialize_common_front, state_arrays)
from full_model.production.common_tensorial_wall import (
    CommonWallParameters, CommonWallState)
from full_model.production.complete_front_energy import (
    evaluate_common_front_transaction, evaluate_complete_directional_kinetics,
    evaluate_complete_front_energy,
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


def test_directional_kinetics_prices_actual_opposite_not_negated_forward():
    spacing = 2e-8
    volume = spacing*spacing*5e-10
    state = _state()
    # Give both directions finite support so each actual trial can move.
    chi = np.full_like(state.front.chi, .4)
    state = replace(state, front=replace(
        state.front, chi=chi, processed_max=np.full_like(chi, .7),
        cleanup_max=np.full_like(chi, .7)))
    forward_chi = chi.copy(); forward_chi[:, :3] += .1
    forward = replace(state.front, chi=forward_chi)
    eta0 = _eta(12); eta1 = _eta(12, shift=-.1)
    kinetics = evaluate_complete_directional_kinetics(
        state, forward, eta0, eta1, event_volume_m3=volume,
        spacing_m=spacing, cell_volume_m3=volume,
        represented_thickness_m=5e-10, transmission_fraction=.6,
        boundary_storage_fraction=.05, neutral_sink_fraction=.05,
        energy_kwargs=_options(spacing))
    assert kinetics.forward_signed_volume_m3 > 0.0
    assert kinetics.opposite_signed_volume_m3 < 0.0
    assert np.isfinite(kinetics.a_to_b_event_J)
    assert np.isfinite(kinetics.b_to_a_event_J)
    assert not kinetics.actual_reverse_edge
    assert kinetics.reverse_edge_status == (
        "DISTINCT_OUTGOING_ENDPOINTS_FROM_ONE_ACCEPTED_STATE")
    assert kinetics.a_to_b_endpoint.before_helmholtz_J == (
        kinetics.b_to_a_endpoint.before_helmholtz_J)
    assert kinetics.a_to_b_endpoint.endpoint_helmholtz_J == (
        kinetics.forward.decision.candidate.helmholtz_J)
    assert kinetics.b_to_a_endpoint.endpoint_helmholtz_J == (
        kinetics.opposite.decision.candidate.helmholtz_J)
    assert kinetics.a_to_b_endpoint.processed_line_increment_m >= 0.0
    assert kinetics.b_to_a_endpoint.processed_line_increment_m >= 0.0
    assert kinetics.a_to_b_endpoint.generated_heat_J >= 0.0
    assert kinetics.b_to_a_endpoint.generated_heat_J >= 0.0
    assert kinetics.a_to_b_endpoint.maximum_abs_line_density_increment_m2 >= 0.0
    assert kinetics.b_to_a_endpoint.maximum_abs_line_density_increment_m2 >= 0.0
    assert kinetics.a_to_b_endpoint.maximum_abs_temperature_increment_K >= 0.0
    assert kinetics.b_to_a_endpoint.maximum_abs_temperature_increment_K >= 0.0
    # Irreversible processing means the independently evaluated opposite need
    # not be the algebraic negative of the forward trial.
    assert kinetics.a_to_b_event_J != -kinetics.b_to_a_event_J


def test_directional_event_normalization_rejects_roundoff_sweep():
    spacing = 2e-8
    volume = spacing*spacing*5e-10
    state = _state()
    tiny = state.front.chi.copy(); tiny[0, 0] = 1e-12
    front = replace(state.front, chi=tiny, processed_max=tiny.copy(),
                    cleanup_max=tiny.copy())
    eta = _eta(12)
    with pytest.raises(ValueError, match="below resolved volume"):
        evaluate_complete_directional_kinetics(
            state, front, eta, eta, event_volume_m3=volume,
            spacing_m=spacing, cell_volume_m3=volume,
            represented_thickness_m=5e-10, transmission_fraction=.6,
            boundary_storage_fraction=.05, neutral_sink_fraction=.05,
            energy_kwargs=_options(spacing))
