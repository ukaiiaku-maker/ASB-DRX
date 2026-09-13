import math
import unittest

from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.stateful_embryos import (
    EmbryoEnvironment, EmbryoEvent, EmbryoParameters, EmbryoPopulation,
    EmbryoRecord, create_embryo, evolve_embryo, mark_promoted,
    population_from_json, population_to_json,
)


def parameters(entropy=0.0):
    return EmbryoParameters(
        represented_thickness_m=2.86e-10,
        interface_energy_J_m2=0.5,
        mobility_prefactor_m4_J_s=1e-18,
        mobility_process=ActivatedProcess("embryo_growth", 1e10, entropy, 1e10),
        mobility_enthalpy_0_J=0.35 * EV_J,
        mobility_critical_pressure_Pa=2e8,
        mobility_exp_a=1.0,
        mobility_exp_n=1.0,
        mobility_enthalpy_floor=0.05,
        minimum_resolved_radius_m=1e-9,
        minimum_survival_time_s=0.0,
        minimum_support_time_s=0.0,
        minimum_phase_purity=0.8,
        minimum_misorientation_rad=math.radians(2.0),
        orientation_symmetry_order=4,
    )


def record(radius=5e-9):
    event = EmbryoEvent(10, 1e-6, "hazard_trigger", 1.2, 1.0, 0.5*EV_J, 2e3)
    return EmbryoRecord(
        embryo_id=0, parent_grain=2, parent_lineage="initial/2",
        position_m=(2e-6, 3e-6), orientation_rad=math.radians(5),
        parent_orientation_rad=0.0, radius_m=radius, birth_step=10,
        birth_time_s=1e-6, birth_strain=0.02, rng_stream="nucleation",
        rng_state_json='{"state":1}', cumulative_hazard=1.2,
        maximum_radius_m=radius, events=(event,),
    )


def environment(relief=2e8, support=1e-12):
    return EmbryoEnvironment(
        temperature_K=1100.0, stored_relief_J_m3=relief,
        orientation_penalty_J_m3=1e7, compatibility_penalty_J_m3=1e7,
        phase_support_area_m2=support, phase_purity=0.95,
        gb_contact_fraction=0.2, wall_contact_fraction=0.3,
        gnd_contact_fraction=0.4,
    )


class StatefulEmbryoTest(unittest.TestCase):
    def test_growth_is_energy_dissipating_and_not_a_label_allocation(self):
        result = evolve_embryo(record(), 11, 1e-6, 1e-7, environment(), parameters())
        self.assertGreater(result.record.radius_m, 5e-9)
        self.assertLessEqual(result.ledger.free_energy_change_J, 0.0)
        self.assertAlmostEqual(result.ledger.closure_error_J, 0.0)
        self.assertFalse(hasattr(result.record, "grain_label"))

    def test_unfavorable_embryo_shrinks_and_retires(self):
        result = evolve_embryo(
            record(radius=1.000001e-9), 11, 1e-6, 1e-2,
            environment(relief=0.0), parameters())
        self.assertLess(result.record.radius_m, 1.000001e-9)
        self.assertEqual(result.record.status, "retired")

    def test_signed_entropy_changes_growth_timescale_once(self):
        slow = evolve_embryo(record(), 11, 1e-6, 1e-7, environment(), parameters(-1.0))
        fast = evolve_embryo(record(), 11, 1e-6, 1e-7, environment(), parameters(1.0))
        self.assertGreater(fast.record.radius_m, slow.record.radius_m)

    def test_population_restart_is_exact(self):
        evolved = evolve_embryo(record(), 11, 1e-6, 1e-7, environment(), parameters()).record
        population = EmbryoPopulation(1, (evolved,))
        encoded = population_to_json(population)
        restored = population_from_json(encoded)
        self.assertEqual(restored, population)
        self.assertEqual(population_to_json(restored), encoded)

    def test_creation_allocates_unique_embryo_not_grain_identity(self):
        initial = EmbryoPopulation(0)
        event = EmbryoEvent(10, 1e-6, "hazard_trigger", 1.2, 1.0, 0.5*EV_J, 2e3)
        population, embryo = create_embryo(
            initial, parent_grain=2, parent_lineage="initial/2",
            position_m=(2e-6, 3e-6), orientation_rad=math.radians(5),
            parent_orientation_rad=0.0, radius_m=5e-9, birth_step=10,
            birth_time_s=1e-6, birth_strain=0.02, rng_stream="nucleation",
            rng_state_json='{"state":1}', cumulative_hazard=1.2, event=event,
            parameters=parameters())
        self.assertEqual(population.next_id, 1)
        self.assertEqual(embryo.embryo_id, 0)
        self.assertFalse(hasattr(embryo, "grain_label"))

    def test_only_promotable_embryo_can_be_finalized(self):
        with self.assertRaises(ValueError):
            mark_promoted(record(), 20, 2e-6)
        evolved = evolve_embryo(record(), 11, 1e-6, 1e-7, environment(), parameters()).record
        self.assertEqual(evolved.status, "promotable")
        self.assertEqual(mark_promoted(evolved, 20, 2e-6).status, "promoted")


if __name__ == "__main__":
    unittest.main()
