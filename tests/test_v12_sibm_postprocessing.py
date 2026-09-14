import unittest

from full_model.analysis.postprocess_v12_sibm_campaign import (
    criticality_classification,
    morphology_classification,
)


def row(*, tip=0.0, mean=0.0, area=0.0, amplitude=0.0,
        distance=2e-6, pressure=1e6):
    return {
        "bulge_tip_displacement_m": tip,
        "signed_normal_displacement_mean_m": mean,
        "excess_bulge_area_m2": area,
        "bulge_amplitude_m": amplitude,
        "tip_distance_to_window_m": distance,
        "local_normal_pressure_Pa": pressure,
    }


class V12SIBMPostprocessingTest(unittest.TestCase):
    def test_resolved_growth_requires_tip_and_mean_advance(self):
        record = {"grid": 128, "radius_um": 0.75}
        rows = [row(), row(tip=2e-7, mean=1e-7, area=1e-12)]
        self.assertEqual(morphology_classification(record, rows),
                         "SIBM_RESOLVED_NORMAL_GROWTH")

    def test_area_growth_with_flattening_is_lateral(self):
        record = {"grid": 128, "radius_um": 0.75}
        rows = [row(amplitude=5e-7),
                row(tip=5e-8, mean=2e-7, area=1e-12, amplitude=2e-7)]
        self.assertEqual(morphology_classification(record, rows),
                         "SIBM_LATERAL_SPREADING_WITHOUT_NORMAL_ADVANCE")

    def test_zero_seed_is_flat_boundary_control(self):
        record = {"grid": 128, "radius_um": 0.0}
        self.assertEqual(morphology_classification(record, [row(), row(tip=1e-6)]),
                         "SIBM_FLAT_BOUNDARY_MIGRATION_ONLY")

    def test_criticality_uses_mean_area_and_pressure_not_extreme_tip(self):
        sub = {"radius_um": 0.42}
        rows = [row(), row(tip=1e-7, mean=-2e-7, area=-1e-12, pressure=-2e6)]
        self.assertEqual(criticality_classification(sub, rows, True),
                         "SIBM_CRITICALITY_CONTROL_PASSED")
        self.assertEqual(criticality_classification(sub, rows, False),
                         "SIBM_CRITICALITY_CONTROL_FAILED")

    def test_supercritical_fixture_requires_positive_sign_triplet(self):
        sup = {"radius_um": 0.78}
        positive = [row(), row(tip=2e-7, mean=2e-7, area=1e-12, pressure=2e6)]
        negative = [row(), row(tip=2e-7, mean=-2e-7, area=-1e-12, pressure=-2e6)]
        self.assertEqual(criticality_classification(sup, positive, True),
                         "SIBM_CRITICALITY_CONTROL_PASSED")
        self.assertEqual(criticality_classification(sup, negative, True),
                         "SIBM_CRITICALITY_CONTROL_FAILED")


if __name__ == "__main__":
    unittest.main()
