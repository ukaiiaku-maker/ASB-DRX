import numpy as np

from full_model.analysis.run_v53_spatial_comparison import spectral_ledger


def sampled_mode(n, mx, my):
    x = np.arange(n)[:, None]/n
    y = np.arange(n)[None, :]/n
    return np.sin(2*np.pi*mx*x)*np.cos(2*np.pi*my*y)


def test_common_spectral_modes_match_without_fitted_alignment():
    result = spectral_ledger(sampled_mode(128, 47, 11),
                             sampled_mode(192, 47, 11))
    assert result["whole_common_non_nyquist_band_relative_l2"] < 2e-13
    intermediate = result[
        "remaining_common_band_max_abs_mode_33_to_63"]
    assert intermediate["relative_measure_resolved"]
    assert intermediate["relative_l2"] < 2e-13
    assert result[
        "fine_only_beyond_abs_mode_63_rms_norm_fraction"] < 2e-13
    assert result["alignment"] == "none"


def test_empty_reference_band_is_not_reported_as_order_one_relative_error():
    result = spectral_ledger(sampled_mode(128, 7, 11),
                             sampled_mode(192, 7, 11))
    intermediate = result[
        "remaining_common_band_max_abs_mode_33_to_63"]
    assert not intermediate["relative_measure_resolved"]
    assert intermediate["relative_l2"] is None
    assert intermediate["classification"] == (
        "REFERENCE_BAND_NUMERICALLY_NEGLIGIBLE")
