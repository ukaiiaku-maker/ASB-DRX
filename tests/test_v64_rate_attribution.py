from pathlib import Path

import numpy as np

from full_model.analysis.audit_v63_exp_floor_attribution import (
    speed_from_inputs, weighted_l2, weighted_relative,
)
from full_model.production.common_tensorial_wall import CommonWallParameters


def test_independent_rate_recompute_is_not_a_self_comparison():
    wall = CommonWallParameters(spacing_m=1e-7)
    raw = np.array([1.2e9, -8e8]); chemical = np.array([2e7, -1e7])
    resistance = np.array([3e8, 4e8]); temperature = np.array([900., 950.])
    recorded, _ = speed_from_inputs(raw, chemical, resistance, temperature, wall)
    perturbed = recorded.copy(); perturbed[0] *= 1.01
    weight = np.ones_like(raw)
    assert weighted_relative(recorded, recorded, weight) == 0.0
    assert weighted_relative(recorded, perturbed, weight) > 0.0


def test_fixed_weighted_norm_uses_declared_denominator():
    weight = np.array([1.0, 2.0])
    left = np.array([1.0, 0.0]); right = np.array([0.0, 2.0])
    scale = max(weighted_l2(left, weight), weighted_l2(right, weight))
    assert np.isclose(weighted_l2(left-right, weight)/scale,
                      np.sqrt(9.0)/np.sqrt(8.0))
