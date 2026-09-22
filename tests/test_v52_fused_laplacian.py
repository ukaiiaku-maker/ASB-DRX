import time

import numpy as np
import pytest

from full_model.production.extensive_wall import (
    _laplacian, _laplacian_composed_oracle,
)


@pytest.mark.parametrize("shape", [(15, 17), (16, 18), (15, 18), (16, 17)])
@pytest.mark.parametrize("trailing", [(), (4, 2)])
def test_fused_laplacian_matches_composed_real_derivative(shape, trailing):
    rng = np.random.default_rng(521)
    field = rng.normal(size=shape+trailing)
    fused = _laplacian(field, 2.5e-8)
    reference = _laplacian_composed_oracle(field, 2.5e-8)
    relative = np.linalg.norm(fused-reference)/np.linalg.norm(reference)
    assert relative < 8e-16
    assert np.max(np.abs(fused-reference))/np.max(np.abs(reference)) < 8e-16


def test_fused_laplacian_zero_and_even_nyquist_modes_match_oracle():
    n = 16
    i, j = np.indices((n, n))
    for field in (np.ones((n, n)), (-1.0)**i, (-1.0)**j,
                  (-1.0)**(i+j)):
        np.testing.assert_allclose(
            _laplacian(field, 1e-7),
            _laplacian_composed_oracle(field, 1e-7), atol=1e-15)


def test_fused_laplacian_is_self_adjoint_and_has_expected_quadratic_form():
    rng = np.random.default_rng(522)
    left = rng.normal(size=(18, 15, 3))
    right = rng.normal(size=(18, 15, 3))
    lap_left = _laplacian(left, 4e-8)
    lap_right = _laplacian(right, 4e-8)
    np.testing.assert_allclose(
        np.sum(left*lap_right), np.sum(lap_left*right), rtol=2e-14)
    assert np.sum(left*lap_left) <= 1e-12*np.sum(np.abs(left*lap_left))


def test_fused_laplacian_compact_speed_benchmark():
    rng = np.random.default_rng(523)
    field = rng.normal(size=(128, 128, 4, 2))
    _laplacian(field, 2.5e-8); _laplacian_composed_oracle(field, 2.5e-8)
    timings = {}
    for name, operator in (("fused", _laplacian),
                           ("composed", _laplacian_composed_oracle)):
        start = time.perf_counter()
        for _ in range(5):
            operator(field, 2.5e-8)
        timings[name] = time.perf_counter()-start
    assert timings["fused"] < timings["composed"]
