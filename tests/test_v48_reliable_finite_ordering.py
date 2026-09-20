from dataclasses import replace

import numpy as np

from full_model.production.extensive_wall import accepted_ordering_step
from tests.test_v40_ordering_finite_time import _compact_active_case


def _solve(backend, exposure, *, rtol=2e-4, atol=2e-8):
    state, density, systems, topologies, parameters, target, stress, attempt = (
        _compact_active_case())
    parameters = replace(
        parameters,
        ordering_integration_method="finite_time_bdf",
        ordering_finite_time_backend=backend,
        ordering_finite_relative_tolerance=rtol,
        ordering_finite_absolute_tolerance=atol)
    return accepted_ordering_step(
        density, systems, topologies, state.common.orientation_rad, target,
        stress, state.common.temperature_K, parameters, exposure/attempt)


def _ordered(result):
    state = result[0]
    return np.concatenate((state.wall_ordered_plus_m2.ravel(),
                           state.wall_ordered_minus_m2.ravel()))


def test_adaptive_rosenbrock_is_error_controlled_and_conservative():
    result = _solve("matrix_free_adaptive_rosenbrock_euler", 1e-4)
    ledger = result[1]
    assert ledger["finite_time_error_control_assessed"]
    assert ledger["finite_time_error_tolerance_satisfied"]
    assert ledger["finite_time_local_error_control_passed"]
    assert not ledger["finite_time_kinetic_accuracy_certified_by_this_solve"]
    assert ledger["finite_time_maximum_accepted_error_norm"] <= 1.0
    assert ledger["complete_elapsed_time_s"] > 0.0
    assert ledger["discarded_reaction_time_s"] == 0.0
    initial = _compact_active_case()[1]
    for sign in ("plus", "minus"):
        before = (getattr(initial, f"wall_tangle_{sign}_m2")
                  +getattr(initial, f"wall_ordered_{sign}_m2"))
        after = (getattr(result[0], f"wall_tangle_{sign}_m2")
                 +getattr(result[0], f"wall_ordered_{sign}_m2"))
        np.testing.assert_array_equal(after, before)


def test_adaptive_rosenbrock_matches_independent_stiff_bdf_oracle():
    adaptive = _solve(
        "matrix_free_adaptive_rosenbrock_euler", 1e-4,
        rtol=5e-5, atol=5e-9)
    oracle = _solve("dense_bdf_oracle", 1e-4, rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(
        _ordered(adaptive), _ordered(oracle), rtol=3e-4, atol=5e-2)


def test_projected_rk2_is_not_certified_by_its_method_name():
    result = _solve("matrix_free_projected_rk2", 1e-4)
    ledger = result[1]
    assert not ledger["finite_time_error_control_assessed"]
    assert not ledger["finite_time_kinetic_accuracy_certified_by_this_solve"]


def test_actual_spectral_backend_exposes_false_rk2_stationarity():
    initial = _compact_active_case()[1]
    initial_ordered = np.concatenate((
        initial.wall_ordered_plus_m2.ravel(),
        initial.wall_ordered_minus_m2.ravel()))
    rk2 = _solve("matrix_free_projected_rk2", 1e-2)
    adaptive = _solve("matrix_free_adaptive_rosenbrock_euler", 1e-2)
    oracle = _solve("dense_bdf_oracle", 1e-2, rtol=1e-8, atol=1e-11)
    # This is the actual production residual with nonzero spectral coupling,
    # not a standalone scalar formula: saturated Heun stages cancel exactly.
    np.testing.assert_array_equal(_ordered(rk2), initial_ordered)
    assert rk2[1]["rk2_maximum_stage_norm_s-1"] > 0.0
    assert rk2[1]["rk2_minimum_stage_cancellation_ratio"] < 1e-12
    assert np.linalg.norm(_ordered(adaptive)-initial_ordered) > 1e6
    np.testing.assert_allclose(
        _ordered(adaptive), _ordered(oracle), rtol=3e-4, atol=5e-2)


def test_production_disables_unproved_euclidean_endpoint_handoff():
    from full_model.analysis.run_v34_finite_coupled_response import (
        resolved_bicrystal,
    )
    parameters = resolved_bicrystal(
        grid=16, length_m=3.2e-6,
        interface_width_m=4e-7)["extensive_parameters"]
    assert parameters.ordering_asymptotic_certificate_mode == "disabled"
    assert parameters.ordering_finite_time_backend == (
        "matrix_free_adaptive_rosenbrock_euler")
