from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from asb_drx.embryos import (
    EmbryoAttempt,
    EmbryoEvolutionParameters,
    EmbryoPopulation,
    create_embryo,
    evolve_embryo,
    load_embryo_population,
    promoted_embryo_grain_record,
    save_embryo_population,
)


class PhysicalEmbryoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = EmbryoEvolutionParameters(
            boundary_energy_J_m2=0.1,
            represented_thickness_m=1.0e-9,
            radial_mobility_m4_J_s=1.0e-14,
            minimum_resolved_radius_m=0.5e-9,
            minimum_survival_time_s=0.02,
            minimum_support_steps=2,
            minimum_phase_purity=0.8,
            minimum_misorientation_rad=math.radians(5.0),
            symmetry_order=4,
        )
        self.attempt = EmbryoAttempt(0.0, 0.1, 1000.0, 1.0e-20, 0.5, 0.1, True)

    def embryo(self, radius_m: float, orientation_rad: float = math.radians(12.0)):
        return create_embryo(
            embryo_id="embryo-7",
            position_m=(2.0e-6, 3.0e-6),
            radius_m=radius_m,
            orientation_rad=orientation_rad,
            parent_label=0,
            parent_orientation_rad=0.0,
            parent_lineage_id="root-0",
            birth_time_s=0.0,
            birth_applied_shear=0.1,
            rng_lineage="embryo-stream:7",
            attempt=self.attempt,
            parameters=self.parameters,
        )

    def test_subcritical_embryo_shrinks_and_retires(self) -> None:
        driving = 1.0e8
        record = self.embryo(0.8e-9)
        step = evolve_embryo(record, 0.1, driving, 0.0, 0.0, self.parameters)
        self.assertLess(step.record.radius_m, record.radius_m)
        self.assertEqual(step.record.status, "retired")
        self.assertGreater(step.ledger.released_heat_J, 0.0)
        self.assertAlmostEqual(step.ledger.closure_error_J, 0.0)

    def test_supercritical_growth_does_not_promote_without_phase_support(self) -> None:
        driving = 1.0e8
        record = self.embryo(2.1e-9)
        step = evolve_embryo(record, 0.01, driving, 0.0, 0.0, self.parameters)
        self.assertGreater(step.record.radius_m, record.radius_m)
        self.assertEqual(step.record.status, "active")
        self.assertEqual(step.record.support_steps, 0)

    def test_promotion_requires_escape_survival_and_persistent_pure_support(self) -> None:
        driving = 1.0e8
        record = self.embryo(2.1e-9)
        support = math.pi * self.parameters.minimum_resolved_radius_m**2
        first = evolve_embryo(record, 0.01, driving, support, 0.9, self.parameters)
        self.assertEqual(first.record.status, "active")
        second = evolve_embryo(first.record, 0.01, driving, support, 0.9, self.parameters)
        self.assertEqual(second.record.status, "promoted")
        self.assertGreater(second.record.integrated_positive_driving_J_s_m3, 0.0)
        self.assertEqual(len(second.record.history), 2)
        grain = promoted_embryo_grain_record(second.record, 1)
        self.assertTrue(grain.embryo_gate_passed)
        self.assertEqual(grain.source_embryo_id, second.record.embryo_id)

    def test_active_embryo_cannot_allocate_grain_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "promoted embryo"):
            promoted_embryo_grain_record(self.embryo(2.1e-9), 1)

    def test_crystallographically_equivalent_orientation_is_rejected(self) -> None:
        record = self.embryo(2.1e-9, 0.5 * math.pi + math.radians(2.0))
        self.assertEqual(record.status, "rejected")

    def test_population_checkpoint_is_exact_and_ids_are_unique(self) -> None:
        population = EmbryoPopulation((self.embryo(2.1e-9),))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embryos.json"
            save_embryo_population(path, population)
            restored = load_embryo_population(path)
        self.assertEqual(restored, population)
        with self.assertRaisesRegex(ValueError, "unique"):
            EmbryoPopulation((population.records[0], population.records[0]))


if __name__ == "__main__":
    unittest.main()
