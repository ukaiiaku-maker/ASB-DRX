from dataclasses import replace

import numpy as np

from full_model.production.multigrain_common_state import MultiGrainCommonState
from full_model.production.wall_neutral_pair_recovery import (
    WallNeutralPairRecoveryParameters,
    evaluate_wall_neutral_pair_recovery,
)
from tests.test_full_model_common_tensorial_wall import fixture


def _state(two_owners=False):
    systems, topologies, parameters, owner, _ = fixture(8)
    plus = np.full_like(owner.wall_plus_m2, 2.5e14)
    minus = np.full_like(owner.wall_minus_m2, 1.5e14)
    order = np.ones_like(owner.wall_order)
    owner = replace(
        owner, wall_plus_m2=plus, wall_minus_m2=minus,
        wall_order=order, temperature_K=np.full_like(owner.temperature_K, 1100.0))
    if two_owners:
        support = (.76+.24*np.arange(8)[:, None]/7.0)*np.ones((1, 8))
        state = MultiGrainCommonState(
            (1, 2), np.stack((support, 1.0-support)), (owner, owner))
    else:
        state = MultiGrainCommonState(
            (1,), np.ones((1, 8, 8)), (owner,))
    return systems, topologies, parameters, state


def _evaluate(state, systems, topologies, parameters):
    return evaluate_wall_neutral_pair_recovery(
        state, interval_s=5e-7, mean_strain=np.zeros((2, 2)),
        spacing_m=parameters.spacing_m, represented_thickness_m=5e-10,
        systems=systems, wall_parameters=parameters,
        energy_kwargs={"topologies": topologies, "systems": systems},
        parameters=WallNeutralPairRecoveryParameters())


def test_wall_recovery_preserves_signed_content_and_closes_line_and_energy():
    systems, topologies, parameters, state = _state()
    decision = _evaluate(state, systems, topologies, parameters)
    assert decision.accepted
    assert decision.annihilated_pair_cell_sum_m2 > 0.0
    scale = float(np.sum(
        state.owners[0].wall_plus_m2+state.owners[0].wall_minus_m2))
    assert decision.signed_content_residual_m2 <= 1e-13*scale
    assert abs(decision.line_balance_residual_m2) <= 1e-12*scale
    assert decision.delta_helmholtz_J < 0.0
    assert decision.generated_heat_J > 0.0
    assert decision.relative_first_law_residual < .05
    changed = decision.committed_state.owners[0]
    np.testing.assert_allclose(
        changed.wall_plus_m2-changed.wall_minus_m2,
        state.owners[0].wall_plus_m2-state.owners[0].wall_minus_m2,
        rtol=0.0, atol=.125)


def test_wall_recovery_is_sign_exchange_symmetric():
    systems, topologies, parameters, state = _state()
    reversed_owner = replace(
        state.owners[0], wall_plus_m2=state.owners[0].wall_minus_m2.copy(),
        wall_minus_m2=state.owners[0].wall_plus_m2.copy())
    reversed_state = replace(state, owners=(reversed_owner,))
    forward = _evaluate(state, systems, topologies, parameters)
    backward = _evaluate(reversed_state, systems, topologies, parameters)
    assert forward.classification == backward.classification
    assert backward.annihilated_pair_cell_sum_m2 == forward.annihilated_pair_cell_sum_m2
    np.testing.assert_allclose(
        backward.committed_state.owners[0].wall_plus_m2,
        forward.committed_state.owners[0].wall_minus_m2, rtol=1e-14)
    np.testing.assert_allclose(
        backward.committed_state.owners[0].wall_minus_m2,
        forward.committed_state.owners[0].wall_plus_m2, rtol=1e-14)


def test_wall_recovery_uses_support_weighted_multigrain_inventory():
    systems, topologies, parameters, state = _state(two_owners=True)
    decision = _evaluate(state, systems, topologies, parameters)
    assert decision.accepted
    scale = sum(float(np.sum(
        support[..., None]*(owner.wall_plus_m2+owner.wall_minus_m2)))
        for support, owner in zip(state.supports, state.owners))
    assert decision.signed_content_residual_m2 <= 1e-13*scale
    assert abs(decision.line_balance_residual_m2) <= 1e-12*scale


def test_zero_gnd_neutral_wall_recovers_without_creating_signed_content():
    systems, topologies, parameters, state = _state()
    owner = state.owners[0]
    neutral = np.full_like(owner.wall_plus_m2, 2e14)
    state = replace(state, owners=(replace(
        owner, wall_plus_m2=neutral, wall_minus_m2=neutral),))
    decision = _evaluate(state, systems, topologies, parameters)
    assert decision.accepted
    changed = decision.committed_state.owners[0]
    np.testing.assert_array_equal(
        changed.wall_plus_m2-changed.wall_minus_m2,
        np.zeros_like(neutral))


def test_wall_recovery_is_burgers_family_permutation_symmetric():
    systems, topologies, parameters, state = _state()
    permutation = np.asarray((2, 0, 3, 1))
    owner = state.owners[0]
    permuted = replace(
        owner,
        wall_plus_m2=owner.wall_plus_m2[..., permutation],
        wall_minus_m2=owner.wall_minus_m2[..., permutation])
    permuted_state = replace(state, owners=(permuted,))
    forward = _evaluate(state, systems, topologies, parameters)
    changed_systems = tuple(systems[index] for index in permutation)
    changed = _evaluate(
        permuted_state, changed_systems, topologies, parameters)
    assert forward.classification == changed.classification
    inverse = np.argsort(permutation)
    np.testing.assert_allclose(
        changed.committed_state.owners[0].wall_plus_m2[..., inverse],
        forward.committed_state.owners[0].wall_plus_m2, rtol=1e-14)
    np.testing.assert_allclose(
        changed.committed_state.owners[0].wall_minus_m2[..., inverse],
        forward.committed_state.owners[0].wall_minus_m2, rtol=1e-14)


def test_unordered_neutral_wall_is_not_annihilated_by_structural_recovery():
    systems, topologies, parameters, state = _state()
    owner = replace(state.owners[0], wall_order=np.zeros((8, 8)))
    decision = _evaluate(
        replace(state, owners=(owner,)), systems, topologies, parameters)
    assert not decision.accepted
    assert decision.classification == "INACTIVE_NO_ACCESSIBLE_NEUTRAL_WALL_PAIR"
    np.testing.assert_array_equal(decision.committed_state.owners[0].wall_plus_m2,
                                  owner.wall_plus_m2)


def test_wall_recovery_full_and_two_half_intervals_agree_below_five_percent():
    systems, topologies, parameters, state = _state()
    kwargs = dict(
        mean_strain=np.zeros((2, 2)), spacing_m=parameters.spacing_m,
        represented_thickness_m=5e-10, systems=systems,
        wall_parameters=parameters,
        energy_kwargs={"topologies": topologies, "systems": systems},
        parameters=WallNeutralPairRecoveryParameters())
    full = evaluate_wall_neutral_pair_recovery(
        state, interval_s=5e-7, **kwargs)
    half_1 = evaluate_wall_neutral_pair_recovery(
        state, interval_s=2.5e-7, **kwargs)
    half_2 = evaluate_wall_neutral_pair_recovery(
        half_1.committed_state, interval_s=2.5e-7, **kwargs)
    for name in ("wall_plus_m2", "wall_minus_m2", "temperature_K"):
        one = np.asarray(getattr(full.committed_state.owners[0], name))
        two = np.asarray(getattr(half_2.committed_state.owners[0], name))
        relative = np.max(np.abs(one-two))/max(float(np.max(np.abs(two))), 1.0)
        assert relative < .05
    pair_two_half = (half_1.annihilated_pair_cell_sum_m2
                     +half_2.annihilated_pair_cell_sum_m2)
    assert abs(full.annihilated_pair_cell_sum_m2-pair_two_half)/pair_two_half < .05
