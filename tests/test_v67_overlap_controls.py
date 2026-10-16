from dataclasses import fields

import numpy as np

from full_model.analysis.run_v58_three_grain_production import initialize_state
from full_model.analysis.run_v67_drx_asb_overlap import _temperature_seed
from full_model.analysis.run_v67_interface_structural_bridge import (
    _args_from_configuration,
)


def test_checkpoint_bridge_preserves_nonzero_bath_rate():
    args = _args_from_configuration({
        "n": 16, "length_m": 4.8e-6, "interface_width_m": 6e-7,
        "temperature_K": 1000.0, "dt_s": 2e-9,
        "maximum_mechanical_subdivisions": 10,
        "thermal_control": "strong_bath", "thermal_bath_rate_s": 5e9,
    })
    assert args.thermal_control == "strong_bath"
    assert args.bath_rate == 5e9


def test_polycrystal_temperature_seed_is_mean_zero_and_owner_independent():
    state = initialize_state(
        16, 1000.0, spacing_m=3e-7, interface_width_m=6e-7)
    seeded = _temperature_seed(state, 20.0, 6e-7, 3e-7, "diagonal")
    deltas = [owner.temperature_K-before.temperature_K
              for owner, before in zip(seeded.owners, state.owners)]
    assert abs(float(np.mean(deltas[0]))) < 1e-12
    assert np.ptp(deltas[0]) > 20.0
    for delta in deltas[1:]:
        np.testing.assert_array_equal(delta, deltas[0])
    for current, before in zip(seeded.owners, state.owners):
        for item in fields(current):
            if item.name != "temperature_K":
                np.testing.assert_array_equal(
                    getattr(current, item.name), getattr(before, item.name))
