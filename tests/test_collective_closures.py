from __future__ import annotations

import math
import unittest

from asb_drx.collective_closures import (
    ExponentialShotNoiseClosure,
    RearmingContactClosure,
    SequentialHitClosure,
)


class ContinuousCollectiveClosureTests(unittest.TestCase):
    def test_one_hit_and_zero_rearm_recover_poisson_baseline(self) -> None:
        rate = 17.0
        sequential = SequentialHitClosure(1)
        rearming = RearmingContactClosure(0.0)
        self.assertEqual(sequential.mean_completion_rate_s_inv(rate), rate)
        self.assertEqual(sequential.completion_wait_cv, 1.0)
        self.assertEqual(rearming.stationary_completion_rate_s_inv(rate), rate)
        self.assertEqual(rearming.completion_wait_cv(rate), 1.0)

    def test_sequential_hits_and_rearming_are_not_cluster_generators(self) -> None:
        self.assertLess(SequentialHitClosure(4).completion_wait_cv, 1.0)
        self.assertLess(RearmingContactClosure(0.1).completion_wait_cv(10.0), 1.0)

    def test_shot_noise_has_exact_independent_and_branching_limits(self) -> None:
        rate = 20.0
        independent = ExponentialShotNoiseClosure(0.0, 0.1)
        self.assertEqual(independent.stationary_mean_rate_s_inv(rate), rate)
        subcritical = ExponentialShotNoiseClosure(0.2, 0.1)
        self.assertAlmostEqual(subcritical.branching_ratio(rate), 0.4)
        self.assertAlmostEqual(
            subcritical.stationary_mean_rate_s_inv(rate), rate / 0.6
        )
        self.assertLess(subcritical.linear_memory_growth_rate_s_inv(rate), 0.0)
        with self.assertRaisesRegex(ValueError, "no stationary mean"):
            ExponentialShotNoiseClosure(0.5, 0.1).stationary_mean_rate_s_inv(rate)

    def test_invalid_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SequentialHitClosure(0)
        with self.assertRaises(ValueError):
            RearmingContactClosure(-1.0)
        with self.assertRaises(ValueError):
            ExponentialShotNoiseClosure(-1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
