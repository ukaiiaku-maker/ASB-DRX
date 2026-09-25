import numpy as np

from full_model.analysis.postprocess_v55_integrated_campaign import concentration


def test_concentration_has_exact_uniform_and_single_cell_limits():
    uniform = concentration(np.ones((8, 8)))
    localized_field = np.zeros((8, 8)); localized_field[2, 3] = 4.0
    localized = concentration(localized_field)
    assert uniform["participation_fraction"] == 1.0
    assert localized["participation_fraction"] == 1.0/64.0
    assert localized["maximum_to_mean"] == 64.0


def test_concentration_ignores_negative_signed_power_for_support_metrics():
    field = np.array([[3.0, -10.0], [0.0, 1.0]])
    result = concentration(field)
    np.testing.assert_allclose(result["participation_fraction"], .4)
    assert 0.0 < result["top_five_percent_fraction"] <= 1.0
