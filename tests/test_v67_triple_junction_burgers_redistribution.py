from dataclasses import replace

import numpy as np

from full_model.analysis.run_v66_structural_bicrystal_front import _state
from full_model.production.multigrain_common_state import (
    InterfaceComponentState, MultiGrainCommonState,
)
from full_model.production.triple_junction_burgers_redistribution import (
    project_triple_junction_burgers,
)


def _fixture(sign=1.0, order=(0, 1, 2)):
    n = 6
    two, _, _ = _state(n, 4.8e-6, 1000.0, 2e7)
    supports = np.full((3, n, n), 1.0/3.0)
    owners = (two.owners[0], two.owners[1], two.owners[0])
    pairs = ((10, 20), (10, 30), (20, 30))
    vectors = ((3.0e4, 0.0, 0.0), (-1.0e4, 2.0e4, 0.0),
               (0.0, -1.0e4, 0.0))
    components = []
    for index, (a, b) in enumerate(pairs):
        plus = np.full((n, n, 3, 4), 1.0e14)
        minus = np.full_like(plus, .8e14)
        burgers = np.zeros((n, n, 3, 3))
        burgers[..., 0, :] = sign*np.asarray(vectors[index])
        components.append(InterfaceComponentState(
            f"edge-{a}-{b}", a, b, np.zeros((n, n)), np.zeros((n, n)),
            plus, minus, np.zeros((n, n, 3, 0)),
            boundary_burgers_m1=burgers))
    state = MultiGrainCommonState(
        (10, 20, 30), supports, owners,
        tuple(components[index] for index in order))
    state.validate()
    return state


def _by_id(state):
    return {item.component_id: item for item in state.interfaces}


def test_triple_junction_projection_conserves_pointwise_vector_and_labels():
    state = _fixture()
    result = project_triple_junction_burgers(
        state, burgers_m=2.48e-10, reaction_fraction=.4)
    assert result.feasible
    assert result.maximum_pointwise_vector_balance_m1 < 1e-10
    np.testing.assert_allclose(result.integrated_vector_balance_m, 0.0,
                               atol=1e-8, rtol=0.0)
    assert result.carrier_capacity_after_cell_sum_m2 < (
        result.carrier_capacity_before_cell_sum_m2)
    assert result.removed_line_cell_sum_m2 > 0.0
    assert result.candidate_state.grain_ids == state.grain_ids
    np.testing.assert_array_equal(result.candidate_state.supports, state.supports)
    for before, after in zip(state.owners, result.candidate_state.owners):
        assert before is after


def test_triple_junction_projection_has_sign_exchange_symmetry():
    positive = project_triple_junction_burgers(
        _fixture(1.0), burgers_m=2.48e-10, reaction_fraction=.7)
    negative = project_triple_junction_burgers(
        _fixture(-1.0), burgers_m=2.48e-10, reaction_fraction=.7)
    pos = _by_id(positive.candidate_state)
    neg = _by_id(negative.candidate_state)
    for key in pos:
        np.testing.assert_allclose(
            neg[key].boundary_burgers_m1,
            -pos[key].boundary_burgers_m1, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            neg[key].boundary_plus_m2, pos[key].boundary_plus_m2)
        np.testing.assert_allclose(
            neg[key].boundary_minus_m2, pos[key].boundary_minus_m2)


def test_triple_junction_projection_is_interface_permutation_symmetric():
    direct = project_triple_junction_burgers(
        _fixture(order=(0, 1, 2)), burgers_m=2.48e-10,
        reaction_fraction=.55)
    permuted = project_triple_junction_burgers(
        _fixture(order=(2, 0, 1)), burgers_m=2.48e-10,
        reaction_fraction=.55)
    left = _by_id(direct.candidate_state)
    right = _by_id(permuted.candidate_state)
    for key in left:
        np.testing.assert_allclose(
            left[key].boundary_burgers_m1,
            right[key].boundary_burgers_m1, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            left[key].boundary_plus_m2, right[key].boundary_plus_m2)


def test_zero_coordinate_changes_only_declared_capacity_relaxation():
    state = _fixture()
    retained = project_triple_junction_burgers(
        state, burgers_m=2.48e-10, reaction_fraction=0.0,
        relax_scalar_line_to_capacity=False)
    for before, after in zip(state.interfaces, retained.candidate_state.interfaces):
        np.testing.assert_array_equal(
            after.boundary_burgers_m1, before.boundary_burgers_m1)
        np.testing.assert_array_equal(after.boundary_plus_m2,
                                      before.boundary_plus_m2)
        np.testing.assert_array_equal(after.boundary_minus_m2,
                                      before.boundary_minus_m2)
    assert retained.removed_line_cell_sum_m2 == 0.0


def test_shared_coordinate_backtracks_before_any_local_carrier_deficit():
    state = _fixture()
    components = []
    for index, item in enumerate(state.interfaces):
        burgers = (np.zeros_like(item.boundary_burgers_m1) if index == 0
                   else item.boundary_burgers_m1)
        required = np.linalg.norm(burgers, axis=-1)/2.48e-10
        plus = np.broadcast_to(
            required[..., None]/4.0, item.boundary_plus_m2.shape).copy()
        components.append(replace(
            item, boundary_plus_m2=plus,
            boundary_minus_m2=np.zeros_like(plus),
            boundary_burgers_m1=burgers))
    state = replace(state, interfaces=tuple(components))
    state.validate()
    result = project_triple_junction_burgers(
        state, burgers_m=2.48e-10, reaction_fraction=1.0)
    assert result.feasible
    assert result.capacity_limited_cell_count > 0
    assert result.maximum_carrier_deficit_m2 == 0.0
    assert result.maximum_pointwise_vector_balance_m1 < 1e-10
