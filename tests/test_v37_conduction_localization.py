import numpy as np

from full_model.analysis.run_v37_conduction_localization import (
    field_metrics,
    weighted_width,
)


def test_weighted_width_separates_narrow_band_from_broad_field():
    n = 128; dx = 1e-5/n
    yy, _ = np.indices((n, n), dtype=float)
    narrow = np.exp(-0.5*((yy-(n-1)/2)*dx/(0.25e-6))**2)
    broad = np.ones((n, n))
    narrow_width = weighted_width(narrow, dx)["minor_rms_width_m"]
    broad_width = weighted_width(broad, dx)["minor_rms_width_m"]
    assert abs(narrow_width-0.25e-6) < 0.01e-6
    assert broad_width > 10*narrow_width


def test_field_metrics_connected_band_and_uniform_support():
    broad = np.ones((32, 32))
    band = np.zeros((32, 32)); band[14:18, :] = 1.0
    broad_metrics = field_metrics(broad, 1.0)
    band_metrics = field_metrics(band, 1.0)
    assert broad_metrics["inverse_participation_fraction"] == 1.0
    assert band_metrics["connected_component_count"] == 1
    assert band_metrics["largest_connected_area_fraction"] == 0.125
    assert band_metrics["inverse_participation_fraction"] < 0.2
