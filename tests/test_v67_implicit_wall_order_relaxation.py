from dataclasses import replace

import numpy as np

from full_model.production.implicit_wall_order_relaxation import (
    evaluate_implicit_wall_order_relaxation,
)
from full_model.production.multigrain_common_state import MultiGrainCommonState
from tests.test_full_model_common_tensorial_wall import fixture


def _one_owner(polarized=True):
    systems, topologies, parameters, owner, _ = fixture(6)
    if polarized:
        plus = np.zeros_like(owner.wall_plus_m2)
        minus = np.zeros_like(owner.wall_minus_m2)
        plus[..., 0] = 5.0e14
        q = np.full(owner.wall_order.shape, .02)
    else:
        plus = np.full_like(owner.wall_plus_m2, 2.0e14)
        minus = plus.copy()
        q = np.zeros_like(owner.wall_order)
    owner = replace(
        owner, wall_plus_m2=plus, wall_minus_m2=minus, wall_order=q,
        temperature_K=np.full_like(owner.temperature_K, 1000.0))
    state = MultiGrainCommonState(
        (1,), np.ones((1,)+q.shape), (owner,))
    return systems, topologies, parameters, state


def _advance(polarized=True):
    systems, topologies, parameters, state = _one_owner(polarized)
    decision = evaluate_implicit_wall_order_relaxation(
        state, interval_s=2.0e-4, mean_strain=np.zeros((2, 2)),
        spacing_m=parameters.spacing_m,
        represented_thickness_m=5.0e-10, topologies=topologies,
        systems=systems, wall_parameters=parameters,
        energy_kwargs={"topologies": topologies, "systems": systems})
    return state, decision


def test_implicit_wall_order_is_arrhenius_bounded_downhill_and_first_law_closed():
    state, decision = _advance(True)
    assert decision.accepted
    assert decision.classification == (
        "ACCEPTED_IMPLICIT_ARRHENIUS_WALL_ORDER_RELAXATION")
    # The zero-stress fixture has a slower EXP-floor clock, but the implicit
    # path is no longer clipped by the former 0.15 explicit-step cap.
    assert .15 < decision.owner_fraction_by_id[1] <= 1.0
    assert decision.owner_rate_s_by_id[1] > 0.0
    assert decision.delta_helmholtz_J < 0.0
    assert decision.generated_heat_J > 0.0
    assert decision.relative_first_law_residual < 1.0e-10
    assert np.mean(decision.committed_state.owners[0].wall_order) > np.mean(
        state.owners[0].wall_order)
    # Ordering changes no signed or total line reservoir.
    for name in ("mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
                 "forest_minus_m2", "wall_plus_m2", "wall_minus_m2"):
        np.testing.assert_array_equal(
            getattr(decision.committed_state.owners[0], name),
            getattr(state.owners[0], name))


def test_unpolarized_zero_order_control_remains_inactive():
    state, decision = _advance(False)
    assert not decision.accepted
    assert decision.classification == "INACTIVE_NO_DOWNHILL_WALL_ORDER_STATE"
    np.testing.assert_array_equal(
        decision.committed_state.owners[0].wall_order,
        state.owners[0].wall_order)


def test_wall_disordering_uses_slower_stress_independent_recovery_clock():
    systems, topologies, parameters, state = _one_owner(False)
    owner = replace(
        state.owners[0],
        wall_order=np.full(state.owners[0].wall_order.shape, .8))
    state = replace(state, owners=(owner,))
    decision = evaluate_implicit_wall_order_relaxation(
        state, interval_s=2.0e-4, mean_strain=np.zeros((2, 2)),
        spacing_m=parameters.spacing_m,
        represented_thickness_m=5.0e-10, topologies=topologies,
        systems=systems, wall_parameters=parameters,
        energy_kwargs={"topologies": topologies, "systems": systems})
    assert decision.accepted
    assert decision.owner_disordering_rate_s_by_id[1] < (
        decision.owner_rate_s_by_id[1])
    assert 0.0 < decision.owner_disordering_fraction_by_id[1] < 1.0e-3
    assert np.mean(decision.committed_state.owners[0].wall_order) < .8
    assert np.mean(decision.committed_state.owners[0].wall_order) > .79
    assert decision.delta_helmholtz_J < 0.0
    assert decision.relative_first_law_residual < 1.0e-10
