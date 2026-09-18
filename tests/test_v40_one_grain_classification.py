import unittest

from full_model.analysis.postprocess_v40_one_grain import (
    qualified_lagb_candidate,
)


class V40OneGrainClassificationTest(unittest.TestCase):
    def test_angle_without_frank_bilby_closure_is_not_a_candidate(self):
        row = {
            "persistent_last_three_records": True,
            "independent_frank_bilby": {
                "candidate_wall_present": True,
                "relative_residual": 2.0,
            },
        }
        self.assertFalse(qualified_lagb_candidate(row))

    def test_persistent_angle_with_closure_is_a_candidate(self):
        row = {
            "persistent_last_three_records": True,
            "independent_frank_bilby": {
                "candidate_wall_present": True,
                "relative_residual": 0.19,
            },
        }
        self.assertTrue(qualified_lagb_candidate(row))


if __name__ == "__main__":
    unittest.main()
