from dataclasses import replace

import numpy as np

from full_model.production.common_front_state import (
    apply_common_increment, commit_front_result, initialize_common_front,
    reconstruct_common, state_arrays, state_from_checkpoint,
    state_metadata_json)
from full_model.production.common_tensorial_wall import CommonWallState
from full_model.production.moving_front import (
    DefectState, initialize_declared_boundary_front)
from full_model.production.tensorial_nye import nye_from_plastic_distortion


def _common(n=8, nf=4, nj=2):
    shape = (n, n, nf)
    zeros = np.zeros(shape)
    beta = np.zeros((n, n, 3, 3))
    return CommonWallState(
        np.full(shape, 4e14), np.full(shape, 3e14),
        np.full(shape, 2e14), np.full(shape, 1e14),
        np.full(shape, 8e13), np.full(shape, 5e13),
        np.full((n, n, nj), 2e13), np.full((n, n), .25),
        np.full((n, n), .1), zeros.copy(), beta,
        np.zeros(shape+(3,)), np.zeros(shape+(3, 3)),
        np.zeros((n, n)), np.full((n, n), 900.0))


def _front(common, chi=None):
    n = common.mobile_plus_m2.shape[0]
    if chi is None:
        chi = np.zeros((n, n))
    defect = DefectState(
        common.mobile_plus_m2.copy(), common.mobile_minus_m2.copy(),
        (common.forest_plus_m2+common.forest_minus_m2).copy(),
        np.sum(common.wall_plus_m2+common.wall_minus_m2, axis=2))
    return initialize_declared_boundary_front(defect, defect, chi, 0, 1)


def test_i0_initial_map_and_front_off_common_increment_are_exact():
    common = _common(); state = initialize_common_front(_front(common), common)
    rebuilt, interface = reconstruct_common(state, 1e-7)
    for name in common.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(rebuilt, name), getattr(common, name))
    np.testing.assert_array_equal(interface, 0.0)

    updated = replace(
        common,
        mobile_plus_m2=common.mobile_plus_m2+7e10,
        forest_minus_m2=common.forest_minus_m2-3e10,
        slip=common.slip+2e-4,
        beta_p=common.beta_p+3e-5,
        temperature_K=common.temperature_K+2.0)
    # A spatially uniform beta increment has zero Nye.
    updated = replace(updated, family_nye_m1=np.zeros_like(updated.family_nye_m1))
    projected = apply_common_increment(state, updated, 1e-7)
    check, interface = reconstruct_common(projected, 1e-7)
    for name in updated.__dataclass_fields__:
        np.testing.assert_allclose(getattr(check, name), getattr(updated, name),
                                   rtol=0.0, atol=0.25)
    np.testing.assert_array_equal(interface, 0.0)


def test_phase_weighted_beta_curl_contains_interface_product_rule():
    n = 16; spacing = 2e-8
    common = _common(n=n)
    x = np.arange(n)[:, None]
    chi = .5+.25*np.sin(2*np.pi*x/n)*np.ones((1, n))
    state = initialize_common_front(_front(common, chi), common)
    beta_child = state.child.beta_p.copy(); beta_child[..., 0, 2] = .03
    state = replace(state, child=replace(state.child, beta_p=beta_child))
    rebuilt, interface = reconstruct_common(state, spacing)
    expected = nye_from_plastic_distortion(rebuilt.beta_p, spacing)
    np.testing.assert_allclose(np.sum(rebuilt.family_nye_m1, axis=2), expected,
                               rtol=1e-12, atol=1e-10)
    assert np.linalg.norm(interface) > 0.0


def test_i2_signed_forest_wall_junction_transfer_retreat_and_restart():
    common = _common(n=6)
    state = initialize_common_front(_front(common), common)
    chi = np.zeros((6, 6)); chi[:, :2] = .4
    accepted = replace(state.front, chi=chi, processed_max=chi.copy(),
                       cleanup_max=chi.copy())
    advanced, mixture = commit_front_result(
        state, accepted, spacing_m=1e-7, cell_volume_m3=1e-20,
        transmission_fraction=.6, boundary_storage_fraction=.1,
        neutral_sink_fraction=.05)
    assert advanced.ledger.processed_line_m > 0.0
    assert advanced.ledger.transmitted_line_m > 0.0
    assert advanced.ledger.boundary_line_m > 0.0
    assert (advanced.ledger.maximum_line_closure_m
            / advanced.ledger.processed_line_m) < 1e-12
    assert advanced.ledger.maximum_signed_closure_m2 <= 1.0
    assert np.any(advanced.boundary_plus_m2[..., 1, :] > 0.0)
    assert np.any(advanced.boundary_minus_m2[..., 2, :] > 0.0)
    assert np.any(advanced.boundary_junction_m2 > 0.0)
    np.testing.assert_array_equal(
        advanced.front.child.forest,
        advanced.child.forest_plus_m2+advanced.child.forest_minus_m2)

    restored = state_from_checkpoint(
        state_metadata_json(advanced), state_arrays(advanced), advanced.front)
    for key, value in state_arrays(advanced).items():
        np.testing.assert_array_equal(value, state_arrays(restored)[key])
    assert restored.ledger == advanced.ledger

    retreated_front = replace(restored.front, chi=np.zeros_like(chi))
    retreated, _ = commit_front_result(
        restored, retreated_front, spacing_m=1e-7, cell_volume_m3=1e-20,
        transmission_fraction=.6, boundary_storage_fraction=.1,
        neutral_sink_fraction=.05)
    assert np.any(retreated.wake.forest_plus_m2 > 0.0)
    assert np.any(retreated.wake.wall_minus_m2 > 0.0)
    assert np.any(retreated.wake.junction_m2 > 0.0)
    assert (retreated.ledger.maximum_line_closure_m
            / retreated.ledger.processed_line_m) < 1e-12


def test_rejected_trial_is_nonmutating():
    common = _common(); state = initialize_common_front(_front(common), common)
    before = state_arrays(state)
    # No commit call models rejection: the immutable snapshot remains exact.
    for key, value in before.items():
        np.testing.assert_array_equal(value, state_arrays(state)[key])
    assert state.ledger.accepted_commits == 0
