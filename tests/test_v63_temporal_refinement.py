import numpy as np

from full_model.analysis.audit_v63_temporal_refinement import scaled_error


def test_scaled_error_reports_signal_and_increment_separately():
    parent = np.array([100.0, 100.0])
    candidate = np.array([101.0, 99.0])
    reference = np.array([102.0, 98.0])
    result = scaled_error(candidate, reference, parent)
    assert result["signal_relative_l2"] < .02
    assert np.isclose(result["increment_relative_l2"], .5)
    assert not result["increment_effectively_vanishing"]


def test_scaled_error_marks_exactly_vanishing_increment():
    parent = np.ones((4, 4))
    result = scaled_error(parent, parent, parent)
    assert result["absolute_l2"] == 0.0
    assert result["increment_effectively_vanishing"]
