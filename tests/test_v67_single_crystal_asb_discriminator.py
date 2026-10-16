import argparse

import numpy as np

from full_model.analysis.run_v67_single_crystal_asb_discriminator import (
    initial_state,
)
from full_model.production.multigrain_common_state import multigrain_state_digest


def _args(**updates):
    values = dict(
        n=32, length=4.8e-6, interface_width=6e-7,
        temperature=1000.0, temperature_seed=.25, seed_width=6e-7,
        band_normal="x")
    values.update(updates)
    return argparse.Namespace(**values)


def test_matched_heat_controls_start_from_identical_mean_zero_seed():
    adiabatic = initial_state(_args())
    bath = initial_state(_args())
    assert multigrain_state_digest(adiabatic) == multigrain_state_digest(bath)
    temperature = adiabatic.owners[0].temperature_K
    assert np.mean(temperature) == 1000.0
    assert np.ptp(temperature) > .25


def test_temperature_seed_does_not_add_signed_or_plastic_content():
    seeded = initial_state(_args())
    unseeded = initial_state(_args(temperature_seed=0.0))
    a = seeded.owners[0]
    b = unseeded.owners[0]
    for name in (
            "mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
            "forest_minus_m2", "wall_plus_m2", "wall_minus_m2",
            "family_nye_m1", "beta_p"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert seeded.grain_ids == unseeded.grain_ids == (10,)
    np.testing.assert_array_equal(seeded.supports, unseeded.supports)
