import numpy as np

from full_model.analysis.postprocess_v66_band_carrier_audit import _correlation


def test_correlation_does_not_mutate_inputs():
    a = np.arange(9, dtype=float).reshape(3, 3)
    b = np.flipud(a).copy()
    a_before = a.copy(); b_before = b.copy()
    value = _correlation(a, b)
    assert np.isfinite(value)
    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)


def test_correlation_handles_constant_field():
    assert _correlation(np.ones((2, 2)), np.arange(4).reshape(2, 2)) == 0.0
