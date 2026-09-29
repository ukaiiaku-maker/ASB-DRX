import numpy as np

from full_model.analysis.postprocess_v59_front_locality import weighted_temperature


def test_weighted_temperature_resolves_periodic_segment_statistics():
    temperature = np.arange(16, dtype=float).reshape(4, 4)+900.0
    weight = np.zeros((4, 4)); weight[0, 0] = 1.0; weight[-1, 0] = 1.0
    result = weighted_temperature(temperature, weight)
    assert result["weighted_mean_K"] == 906.0
    assert result["active_minimum_K"] == 900.0
    assert result["active_maximum_K"] == 912.0
    assert result["segment_count"] == 1
    assert result["segments"][0]["weight_fraction"] == 1.0
