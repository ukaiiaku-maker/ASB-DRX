import numpy as np

from full_model.production.nye_consistency import periodic_nye_decomposition


def test_identical_periodic_nye_fields_close_all_components():
    rng = np.random.default_rng(28)
    alpha = rng.normal(size=(12, 12, 3, 3))
    result = periodic_nye_decomposition(alpha, alpha.copy(), 1e-7)
    assert result["total_relative_rms"] == 0.0
    assert result["zero_mode_relative_rms"] == 0.0
    assert result["nonzero_mode_relative_rms"] == 0.0


def test_uniform_reservoir_bias_is_isolated_as_zero_mode():
    curl_beta = np.zeros((8, 8, 3, 3))
    reservoir = curl_beta.copy(); reservoir[..., 0, 2] = 4.0
    result = periodic_nye_decomposition(reservoir, curl_beta, 2e-7)
    assert result["zero_mode_relative_rms"] > 0.0
    assert result["nonzero_mode_relative_rms"] == 0.0
    assert result["line_continuity_divergence_rms_m2"] == 0.0
