import numpy as np

from full_model.analysis.plot_v67_poly_rx_asb_long_horizon import series


def test_plot_series_uses_physical_chunk_denominator_and_percent_growth():
    row = {
        "load": .1,
        "mean_shear_stress_Pa": -2e9,
        "temperature_range_K": 40.0,
        "active_physical_band_intervals": 25,
        "target_steps": 100,
        "final_power_localization": {"maximum_to_mean": 3.0},
        "owner_results": {"20": {"relative_area_change": .002}},
    }
    payload = {"cases": {"adiabatic": [row]}}
    values = series(payload, "adiabatic")
    assert np.isclose(values["stress_GPa"][0], 2.0)
    assert np.isclose(values["band_fraction"][0], .25)
    assert np.isclose(values["grain20_increment_percent"][0], .2)
