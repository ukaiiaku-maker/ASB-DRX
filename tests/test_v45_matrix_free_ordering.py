from dataclasses import replace

import numpy as np
import pytest

from full_model.production.extensive_wall import (
    ExtensiveWallParameters, _ordering_affinity_linear_action,
    accepted_ordering_step, extensive_wall_chemical_potentials_J_m,
)
from tests.test_v40_ordering_finite_time import _compact_active_case


def run_backend(backend, exposure=20.0):
    case = _compact_active_case()
    state, density, systems, topologies, parameters, target, stress, attempt = case
    parameters = replace(
        parameters, ordering_integration_method="finite_time_bdf",
        ordering_finite_time_backend=backend)
    return accepted_ordering_step(
        density, systems, topologies, state.common.orientation_rad, target,
        stress, state.common.temperature_K, parameters, exposure/attempt)


def test_matrix_free_finite_time_matches_dense_oracle_and_closes_clock():
    # The exponential backend is a transient oracle below the independently
    # tested finite/asymptotic overlap, not the production stiff dispatch.
    dense = run_backend("dense_bdf_oracle", exposure=1e-6)
    matrix_free = run_backend("matrix_free_exponential_rosenbrock", exposure=1e-6)
    for name in ("wall_tangle_plus_m2", "wall_tangle_minus_m2",
                 "wall_ordered_plus_m2", "wall_ordered_minus_m2"):
        np.testing.assert_allclose(getattr(matrix_free[0], name),
                                   getattr(dense[0], name),
                                   atol=1e-2, rtol=2e-8)
    ledger = matrix_free[1]
    assert ledger["integration_method"] == "bounded_finite_time_matrix_free_exprb1"
    assert ledger["complete_elapsed_time_s"] > 0.0
    assert ledger["discarded_reaction_time_s"] == 0.0
    assert ledger["dense_jacobian_bytes_avoided"] > 0
    assert ledger["linear_iterations"] > 0
    assert ledger["active_degrees_of_freedom"] > 0


def test_matrix_free_ordering_conserves_each_signed_family_pool():
    case = _compact_active_case()
    initial = case[1]
    updated = run_backend("matrix_free_exponential_rosenbrock", exposure=1e-6)[0]
    for sign in ("plus", "minus"):
        before = (getattr(initial, f"wall_tangle_{sign}_m2")
                  +getattr(initial, f"wall_ordered_{sign}_m2"))
        after = (getattr(updated, f"wall_tangle_{sign}_m2")
                 +getattr(updated, f"wall_ordered_{sign}_m2"))
        np.testing.assert_array_equal(after, before)
        assert np.min(getattr(updated, f"wall_tangle_{sign}_m2")) >= 0.0
        assert np.min(getattr(updated, f"wall_ordered_{sign}_m2")) >= 0.0


def test_matrix_free_attempt_exposure_must_be_positive():
    with pytest.raises(ValueError, match="matrix-free attempt exposure"):
        ExtensiveWallParameters(
            spacing_m=1e-8,
            ordering_matrix_free_max_attempt_exposure=0.0)


def test_ordering_fft_action_matches_directional_difference_and_is_adjoint_symmetric():
    state, density, systems, topologies, parameters, target, _, _ = (
        _compact_active_case())
    total_plus = density.wall_tangle_plus_m2+density.wall_ordered_plus_m2
    total_minus = density.wall_tangle_minus_m2+density.wall_ordered_minus_m2
    centered = replace(
        density,
        wall_tangle_plus_m2=.5*total_plus,
        wall_ordered_plus_m2=.5*total_plus,
        wall_tangle_minus_m2=.5*total_minus,
        wall_ordered_minus_m2=.5*total_minus)
    rng = np.random.default_rng(45)
    dp = rng.normal(size=total_plus.shape)*np.minimum(total_plus, 1e12)
    dm = rng.normal(size=total_minus.shape)*np.minimum(total_minus, 1e12)
    ep, em = _ordering_affinity_linear_action(
        dp, dm, systems, state.common.orientation_rad, parameters)
    h = 1e-4
    def affinity(sign, inventory):
        mu = extensive_wall_chemical_potentials_J_m(
            inventory, systems, topologies, state.common.orientation_rad,
            target, parameters)
        return mu[f"ordered_{sign}"]-mu[f"tangle_{sign}"]
    plus = replace(
        centered,
        wall_tangle_plus_m2=centered.wall_tangle_plus_m2-h*dp,
        wall_ordered_plus_m2=centered.wall_ordered_plus_m2+h*dp,
        wall_tangle_minus_m2=centered.wall_tangle_minus_m2-h*dm,
        wall_ordered_minus_m2=centered.wall_ordered_minus_m2+h*dm)
    minus = replace(
        centered,
        wall_tangle_plus_m2=centered.wall_tangle_plus_m2+h*dp,
        wall_ordered_plus_m2=centered.wall_ordered_plus_m2-h*dp,
        wall_tangle_minus_m2=centered.wall_tangle_minus_m2+h*dm,
        wall_ordered_minus_m2=centered.wall_ordered_minus_m2-h*dm)
    np.testing.assert_allclose(
        (affinity("plus", plus)-affinity("plus", minus))/(2*h), ep,
        rtol=2e-8, atol=2e-14)
    np.testing.assert_allclose(
        (affinity("minus", plus)-affinity("minus", minus))/(2*h), em,
        rtol=2e-8, atol=2e-14)
    vp = rng.normal(size=dp.shape); vm = rng.normal(size=dm.shape)
    hp, hm = _ordering_affinity_linear_action(
        vp, vm, systems, state.common.orientation_rad, parameters)
    left = np.sum(vp*ep)+np.sum(vm*em)
    right = np.sum(dp*hp)+np.sum(dm*hm)
    np.testing.assert_allclose(left, right, rtol=2e-12, atol=1e-18)


def test_finite_rate_and_convex_asymptotic_overlap_before_production_switch():
    case = _compact_active_case()
    state, density, systems, topologies, parameters, target, stress, attempt = case
    dense_parameters = replace(
        parameters, ordering_integration_method="finite_time_bdf",
        ordering_finite_time_backend="dense_bdf_oracle")
    finite = accepted_ordering_step(
        density, systems, topologies, state.common.orientation_rad, target,
        stress, state.common.temperature_K, dense_parameters, .5/attempt)[0]
    asymptotic_parameters = replace(
        parameters, ordering_integration_method="implicit_backward_euler",
        ordering_asymptotic_minimum_attempt_exposure=1.000001)
    asymptotic_result = accepted_ordering_step(
        density, systems, topologies, state.common.orientation_rad, target,
        stress, state.common.temperature_K, asymptotic_parameters, 2.0/attempt)
    asymptotic, ledger = asymptotic_result[0], asymptotic_result[1]
    assert ledger["integration_method"] == "bounded_convex_asymptotic"
    assert not ledger["finite_time_kinetic_accuracy_certified_by_this_solve"]
    for name in ("wall_tangle_plus_m2", "wall_tangle_minus_m2",
                 "wall_ordered_plus_m2", "wall_ordered_minus_m2"):
        np.testing.assert_allclose(getattr(asymptotic, name),
                                   getattr(finite, name),
                                   atol=1e-2, rtol=2e-8)
