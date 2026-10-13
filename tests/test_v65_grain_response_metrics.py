import numpy as np

from full_model.analysis.grain_response_metrics import grain_response


def test_endpoint_average_perimeter_is_the_registered_measure():
    x = np.arange(16)[:, None]
    initial = ((x >= 4) & (x < 10))*np.ones((1, 16))
    final = ((x >= 3) & (x < 11))*np.ones((1, 16))
    result = grain_response(
        np.stack((initial, 1.0-initial)),
        np.stack((final, 1.0-final)),
        spacing_m=2.0, represented_thickness_m=3.0)
    expected = ((result["final_volume_m3"]-result["initial_volume_m3"])
                /(3.0*.5*(result["initial_perimeter_m"]
                           +result["final_perimeter_m"])))
    assert result["convention"] == "endpoint_average_spectral_perimeter"
    assert np.allclose(result["equivalent_contour_displacement_m"], expected)


def test_identical_support_has_zero_growth_and_displacement():
    rng = np.random.default_rng(9)
    support = rng.random((4, 8, 8))
    support /= np.sum(support, axis=0)
    result = grain_response(
        support, support.copy(), spacing_m=1e-7,
        represented_thickness_m=4.96e-10)
    assert np.array_equal(result["volume_growth_fraction"], np.zeros(4))
    assert np.array_equal(
        result["equivalent_contour_displacement_m"], np.zeros(4))
