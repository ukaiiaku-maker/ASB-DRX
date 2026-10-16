from dataclasses import replace

import numpy as np

from full_model.production.joint_wall_topology_orientation import (
    joint_wall_orientation_candidate,
)
from full_model.production.collective_wall_nucleation import (
    _collective_wall_candidate,
)
from full_model.production.multigrain_common_state import MultiGrainCommonState
from full_model.production.orientation_wall_compatibility import (
    ordered_wall_nye_from_common, reconstruct_orientation_from_ordered_nye,
)
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
    np.testing.assert_allclose(
        np.mean(changed.orientation_rad), np.mean(owner.orientation_rad),
        rtol=0.0, atol=1e-18)
    np.testing.assert_array_equal(candidate.supports, state.supports)
    assert candidate.grain_ids == state.grain_ids


def test_joint_orientation_is_incremental_not_baseline_relaxation():
    systems, topologies, parameters, owner, _ = fixture(8)
    mobile_plus = np.asarray(owner.mobile_plus_m2).copy()
    mobile_plus[2:6, 2:6, 0] += 2e14
    # Deliberately retain an orientation that is not the reconstruction of the
    # existing ordered wall.  A new wall event must not consume that old
    # mismatch as free energy.
    retained = .02*np.sin(2*np.pi*np.arange(8)/8)[:, None]
    owner = replace(
        owner, mobile_plus_m2=mobile_plus,
        orientation_rad=np.asarray(owner.orientation_rad)+retained,
        wall_order=np.full_like(owner.wall_order, .5))
    state = MultiGrainCommonState((1,), np.ones((1, 8, 8)), (owner,))
    x, y = np.indices((8, 8))
    ring = np.exp(-.5*((np.sqrt((x-4)**2+(y-4)**2)-2.0)/.75)**2)
    coordinate = .2
    candidate, transferred, _ = joint_wall_orientation_candidate(
        state, 0, ring, coordinate, systems, parameters.spacing_m, 0.0, .01)
    changed = candidate.owners[0]
    wall_only, _ = _collective_wall_candidate(
        state, 0, ring, .01, coordinate)
    domain = np.ones((8, 8), dtype=bool)
    old_reconstructed, _ = reconstruct_orientation_from_ordered_nye(
        ordered_wall_nye_from_common(owner, systems), owner.orientation_rad,
        domain, parameters.spacing_m)
    new_reconstructed, _ = reconstruct_orientation_from_ordered_nye(
        ordered_wall_nye_from_common(wall_only.owners[0], systems),
        owner.orientation_rad,
        domain, parameters.spacing_m)
    expected_increment = np.arctan2(
        np.sin(new_reconstructed-old_reconstructed),
        np.cos(new_reconstructed-old_reconstructed))
    observed_increment = np.arctan2(
        np.sin(changed.orientation_rad-owner.orientation_rad),
        np.cos(changed.orientation_rad-owner.orientation_rad))
    expected_increment -= np.mean(expected_increment)
    np.testing.assert_allclose(
        observed_increment, expected_increment, rtol=0.0, atol=2e-15)
    assert transferred > 0.0
