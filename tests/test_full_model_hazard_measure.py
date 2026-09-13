import unittest

import numpy as np

from full_model.production.hazard_measure import (
    AreaHazardState, advance_area_hazard, exposure_statistics,
    initialize_area_hazard,
)


class AreaHazardMeasureTest(unittest.TestCase):
    def test_uniform_exposure_is_mesh_invariant(self):
        totals = []
        for n in (16, 32, 64):
            rng = np.random.default_rng(4)
            state = initialize_area_hazard((n, n), rng)
            result = advance_area_hazard(
                state, np.full((n, n), 2.0), site_density_m2=3e12,
                cell_area_m2=(4e-6/n)**2, dt_s=5e-4, rng=rng)
            totals.append(result.state.total_exposure)
        self.assertTrue(np.allclose(totals, totals[0], rtol=2e-16, atol=0.0))

    def test_translating_support_does_not_erase_exposure(self):
        rng = np.random.default_rng(5)
        state = initialize_area_hazard((16, 16), rng)
        rate = np.zeros((16, 16)); rate[2:4, :] = 1.0
        first = advance_area_hazard(
            state, rate, site_density_m2=1e10, cell_area_m2=1e-12,
            dt_s=1e-3, rng=rng).state
        second = advance_area_hazard(
            first, np.roll(rate, 7, axis=0), site_density_m2=1e10,
            cell_area_m2=1e-12, dt_s=1e-3, rng=rng).state
        self.assertEqual(second.total_exposure, 2.0*first.total_exposure)
        self.assertEqual(second.discarded_exposure, 0.0)
        self.assertEqual(second.threshold_redraws_other, 0)

    def test_multiple_crossings_are_not_silently_discarded(self):
        rng = np.random.default_rng(6)
        state = AreaHazardState(np.zeros((2, 2)), 0.0, 0.0, 0.0, 0.1)
        step = advance_area_hazard(
            state, np.ones((2, 2)), site_density_m2=1.0,
            cell_area_m2=1.0, dt_s=10.0, rng=rng, maximum_events=3)
        self.assertEqual(len(step.event_indices), 3)
        self.assertTrue(step.deferred_event_present)
        self.assertEqual(step.state.completed_events, 3)
        self.assertEqual(step.state.threshold_redraws_after_event, 3)
        self.assertGreater(step.state.residual_exposure, step.state.next_threshold)
        with self.assertRaisesRegex(RuntimeError, "deferred Poisson event"):
            advance_area_hazard(
                step.state, np.ones((2, 2)), site_density_m2=1.0,
                cell_area_m2=1.0, dt_s=1.0, rng=rng, maximum_events=3)

    def test_statistics_report_poisson_probability_and_quantiles(self):
        state = AreaHazardState(np.array([[0.1, 0.2], [0.3, 0.4]]),
                                1.0, 1.0, 1.0, 2.0)
        stats = exposure_statistics(state)
        self.assertAlmostEqual(stats["probability_at_least_one_raw_trigger"],
                               1.0-np.exp(-1.0))
        self.assertEqual(len(stats["hazard_exposure_site_quantiles"]), 7)


if __name__ == "__main__":
    unittest.main()
