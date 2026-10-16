from dataclasses import replace

import numpy as np

from full_model.production.collective_climb_wall_capture import (
    collective_climb_wall_capture_candidate,
    collective_climb_wall_orientation_candidate,
)
from full_model.production.multigrain_common_state import MultiGrainCommonState
from tests.test_full_model_common_tensorial_wall import fixture


def _state():
    systems, topologies, parameters, owner, _ = fixture(8)
    plus = np.asarray(owner.mobile_plus_m2).copy()
    minus = np.asarray(owner.mobile_minus_m2).copy()
    plus[2:6, 2:6, 0] += 2e14
    minus[1:4, 1:4, 1] += 1e14
    owner = replace(
        owner, mobile_plus_m2=plus, mobile_minus_m2=minus,
        wall_order=np.full_like(owner.wall_order, .35))
    return (MultiGrainCommonState(
        (1,), np.ones((1, 8, 8)), (owner,)), systems, parameters)


def _masks():
    x, y = np.indices((8, 8))
    radius = np.sqrt((x-4)**2+(y-4)**2)
    return (radius < 2.5).astype(float), np.exp(-.5*((radius-3.0)/.6)**2)


def _inventory(owner):
    return (
        np.sum(owner.mobile_plus_m2+owner.wall_plus_m2, axis=(0, 1)),
        np.sum(owner.mobile_minus_m2+owner.wall_minus_m2, axis=(0, 1)),
    )


def test_collective_climb_capture_conserves_each_sign_and_family():
    state, systems, parameters = _state()
    donor, receiver = _masks()
    before = _inventory(state.owners[0])
    candidate, moved = collective_climb_wall_capture_candidate(
        state, 0, donor, receiver, .25, .01)
    after = _inventory(candidate.owners[0])
    assert moved > 0.0
    np.testing.assert_allclose(after[0], before[0], rtol=2e-15, atol=.25)
    np.testing.assert_allclose(after[1], before[1], rtol=2e-15, atol=.25)
    assert np.all(candidate.owners[0].mobile_plus_m2 >= 0.0)
    assert np.all(candidate.owners[0].mobile_minus_m2 >= 0.0)
    np.testing.assert_array_equal(candidate.supports, state.supports)
    assert candidate.grain_ids == state.grain_ids


def test_collective_climb_capture_has_exact_sign_exchange_symmetry():
    state, systems, parameters = _state()
    donor, receiver = _masks()
    owner = state.owners[0]
    swapped_owner = replace(
        owner,
        mobile_plus_m2=np.asarray(owner.mobile_minus_m2),
        mobile_minus_m2=np.asarray(owner.mobile_plus_m2),
        wall_plus_m2=np.asarray(owner.wall_minus_m2),
        wall_minus_m2=np.asarray(owner.wall_plus_m2))
    swapped = replace(state, owners=(swapped_owner,))
    direct, moved = collective_climb_wall_capture_candidate(
        state, 0, donor, receiver, .25, .01)
    reverse, moved_swapped = collective_climb_wall_capture_candidate(
        swapped, 0, donor, receiver, .25, .01)
    np.testing.assert_allclose(
        direct.owners[0].mobile_plus_m2,
        reverse.owners[0].mobile_minus_m2, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        direct.owners[0].wall_plus_m2,
        reverse.owners[0].wall_minus_m2, rtol=0.0, atol=0.0)
    assert moved == moved_swapped


def test_collective_climb_orientation_is_incremental_and_mean_neutral():
    state, systems, parameters = _state()
    donor, receiver = _masks()
    candidate, moved, rotation = collective_climb_wall_orientation_candidate(
        state, 0, donor, receiver, .25, .01, systems,
        parameters.spacing_m, np.ones((8, 8), dtype=bool))
    assert moved > 0.0
    assert rotation > 0.0
    np.testing.assert_allclose(
        np.mean(candidate.owners[0].orientation_rad),
        np.mean(state.owners[0].orientation_rad), rtol=0.0, atol=1e-18)
