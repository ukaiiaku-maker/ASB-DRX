import numpy as np

from full_model.analysis.postprocess_v66_band_sampling_audit import (
    _periodic_bilinear,
)


def test_periodic_bilinear_identity_is_exact():
    field = np.arange(64, dtype=float).reshape(8, 8)
    np.testing.assert_array_equal(_periodic_bilinear(field, 8), field)


def test_periodic_bilinear_preserves_constant_and_mean_on_integer_refinement():
    constant = np.full((8, 8), 3.25)
    np.testing.assert_array_equal(
        _periodic_bilinear(constant, 32), np.full((32, 32), 3.25))
    field = np.arange(64, dtype=float).reshape(8, 8)
    refined = _periodic_bilinear(field, 32)
    assert np.isclose(np.mean(refined), np.mean(field), rtol=0.0, atol=1e-14)
