import numpy as np

from full_model.analysis.audit_v63_exp_floor_attribution import (
    effective_stress, weighted_relative,
)


def test_effective_stress_is_signed_and_resistance_reduces_magnitude():
    raw = np.array([-3.0, 3.0])
    value = effective_stress(raw, np.zeros(2), np.ones(2), .1)
    assert value[0] < 0.0 < value[1]
    assert np.all(np.abs(value) < np.abs(raw))


def test_weighted_relative_ignores_zero_support():
    a = np.array([1.0, 100.0])
    b = np.array([1.0, -100.0])
    assert weighted_relative(a, b, np.array([1.0, 0.0])) == 0.0
