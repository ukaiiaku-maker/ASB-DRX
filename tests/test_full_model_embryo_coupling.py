import unittest

import numpy as np

from full_model.production.embryo_coupling import (
    FullFieldEnvironment, advance_population, local_environment,
)
from tests.test_full_model_stateful_embryos import environment, parameters, record
from full_model.production.stateful_embryos import EmbryoPopulation


class EmbryoCouplingTest(unittest.TestCase):
    def fields(self, n=64):
        shape = (n, n)
        return FullFieldEnvironment(
            temperature_K=np.full(shape, 1100.0),
            stored_relief_J_m3=np.full(shape, 2e8),
            orientation_penalty_J_m3=np.full(shape, 1e7),
            compatibility_penalty_J_m3=np.full(shape, 1e7),
            gb_contact=np.full(shape, 0.2), wall_contact=np.full(shape, 0.3),
            gnd_contact=np.full(shape, 0.4))

    def test_uniform_fields_reduce_to_isolated_environment(self):
        local = local_environment(record(radius=2e-7), self.fields(), (4e-6, 4e-6), 2e-8, 0.8)
        expected = environment(relief=2e8)
        self.assertAlmostEqual(local.temperature_K, expected.temperature_K)
        self.assertAlmostEqual(local.stored_relief_J_m3, expected.stored_relief_J_m3)
        self.assertAlmostEqual(local.gb_contact_fraction, expected.gb_contact_fraction)
        self.assertGreater(local.phase_support_area_m2, 0.0)

    def test_population_step_closes_energy_and_allocates_no_label(self):
        population = EmbryoPopulation(1, (record(radius=2e-7),))
        updated, ledger = advance_population(
            population, fields=self.fields(), step=12, time_s=1e-6,
            proposed_dt_s=1e-7, domain_lengths_m=(4e-6, 4e-6),
            interface_width_m=2e-8, purity_threshold=0.8,
            parameters=parameters())
        self.assertEqual(ledger.embryo_count, 1)
        self.assertAlmostEqual(ledger.closure_error_J, 0.0)
        self.assertFalse(hasattr(updated.records[0], "grain_label"))


if __name__ == "__main__":
    unittest.main()
