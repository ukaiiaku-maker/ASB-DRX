import numpy as np

from full_model.analysis.render_v67_poly_rx_asb_fields import fixed_limits


def test_fixed_limits_are_shared_and_expand_constant_fields():
    frames = [
        {"fields": (np.zeros((2, 2)), np.ones((2, 2)))},
        {"fields": (np.full((2, 2), 2.0), np.ones((2, 2)))},
    ]
    assert fixed_limits(frames) == [(0.0, 2.0), (1.0, 2.0)]
