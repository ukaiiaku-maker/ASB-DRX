import numpy as np

from full_model.analysis.postprocess_v28_asb_energy import localization_metrics


def test_localization_metrics_distinguish_uniform_and_band_fields():
    uniform = localization_metrics(np.ones((32, 32)), 1e-7)
    band = np.zeros((32, 32)); band[15:17] = 1.0
    localized = localization_metrics(band, 1e-7)
    assert uniform["inverse_participation_effective_fraction"] == 1.0
    assert localized["inverse_participation_effective_fraction"] < 0.1
    assert localized["entropy_effective_fraction"] < 0.1
    assert localized["minor_second_moment_m2"] < uniform["minor_second_moment_m2"]


def test_zero_power_has_finite_empty_metrics():
    result = localization_metrics(np.zeros((8, 8)), 1e-7)
    assert result["connected_band_count"] == 0
    assert all(np.isfinite(value) for value in result.values())
