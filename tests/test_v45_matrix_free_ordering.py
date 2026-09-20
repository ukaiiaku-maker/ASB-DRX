from dataclasses import replace

import numpy as np
import pytest

from full_model.production.extensive_wall import (
    ExtensiveWallParameters, accepted_ordering_step,
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
    dense = run_backend("dense_bdf_oracle")
    matrix_free = run_backend("matrix_free_backward_euler")
    for name in ("wall_tangle_plus_m2", "wall_tangle_minus_m2",
                 "wall_ordered_plus_m2", "wall_ordered_minus_m2"):
        np.testing.assert_allclose(getattr(matrix_free[0], name),
                                   getattr(dense[0], name),
                                   atol=1e-2, rtol=2e-8)
    ledger = matrix_free[1]
    assert ledger["integration_method"] == "bounded_finite_time_matrix_free_be"
    assert ledger["complete_elapsed_time_s"] > 0.0
    assert ledger["discarded_reaction_time_s"] == 0.0
    assert ledger["dense_jacobian_bytes_avoided"] > 0
    assert ledger["linear_iterations"] > 0
    assert ledger["active_degrees_of_freedom"] > 0


def test_matrix_free_ordering_conserves_each_signed_family_pool():
    case = _compact_active_case()
    initial = case[1]
    updated = run_backend("matrix_free_backward_euler", exposure=5.0)[0]
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
