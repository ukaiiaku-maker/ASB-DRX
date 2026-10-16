from dataclasses import replace

import numpy as np

from full_model.production.joint_wall_topology_orientation import (
    joint_wall_orientation_candidate,
)
from full_model.production.multigrain_common_state import MultiGrainCommonState
from tests.test_full_model_common_tensorial_wall import fixture


def test_joint_topology_probe_preserves_line_reservoir_and_owner_mean():
    systems, topologies, parameters, owner, _ = fixture(8)
    mobile_plus = np.asarray(owner.mobile_plus_m2).copy()
    mobile_minus = np.asarray(owner.mobile_minus_m2).copy()
    mobile_plus[2:6, 2:6, 0] += 2e14
    owner = replace(
        owner, mobile_plus_m2=mobile_plus, mobile_minus_m2=mobile_minus,
        wall_order=np.full_like(owner.wall_order, .5))
    state = MultiGrainCommonState(
        (1,), np.ones((1, 8, 8)), (owner,))
    x, y = np.indices((8, 8))
    ring = np.exp(-.5*((np.sqrt((x-4)**2+(y-4)**2)-2.0)/.75)**2)
    candidate, transferred, rotation = joint_wall_orientation_candidate(
        state, 0, ring, .2, systems, parameters.spacing_m, 0.0, .01)
    changed = candidate.owners[0]
    assert transferred > 0.0
    assert rotation >= 0.0
    np.testing.assert_allclose(
        changed.mobile_plus_m2+changed.wall_plus_m2,
        owner.mobile_plus_m2+owner.wall_plus_m2, rtol=0.0, atol=.125)
    np.testing.assert_allclose(
        changed.mobile_minus_m2+changed.wall_minus_m2,
        owner.mobile_minus_m2+owner.wall_minus_m2, rtol=0.0, atol=.125)
    assert np.mean(changed.orientation_rad) == np.mean(owner.orientation_rad)
    np.testing.assert_array_equal(candidate.supports, state.supports)
    assert candidate.grain_ids == state.grain_ids
