import numpy as np

from full_model.analysis.analyze_v67_hall_petch_capacity import (
    hall_petch_capacity_ceiling_s,
)
from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters,
)


def test_capacity_ceiling_follows_d_inverse_cubed_at_fixed_floor():
    small = hall_petch_capacity_ceiling_s(
        1000.0, ArrheniusSourceParameters(grain_size_m=2.0e-6))
    large = hall_petch_capacity_ceiling_s(
        1000.0, ArrheniusSourceParameters(grain_size_m=4.0e-6))
    np.testing.assert_allclose(
        small["maximum_all_family_shear_rate_s"]
        /large["maximum_all_family_shear_rate_s"], 8.0, rtol=1e-14)


def test_capacity_ceiling_increases_with_temperature_and_family_count():
    parameters = ArrheniusSourceParameters(grain_size_m=2.4e-6)
    cold = hall_petch_capacity_ceiling_s(900.0, parameters, families=1)
    hot = hall_petch_capacity_ceiling_s(1100.0, parameters, families=4)
    assert hot["maximum_hazard_per_family_s"] > cold[
        "maximum_hazard_per_family_s"]
    assert hot["maximum_all_family_shear_rate_s"] > 4.0*cold[
        "maximum_all_family_shear_rate_s"]
