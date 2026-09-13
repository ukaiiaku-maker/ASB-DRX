import unittest
from dataclasses import replace

import numpy as np

from full_model.production.phase_promotion import (
    PromotionEnergyIncreaseError, atomic_phase_promotion,
    conservative_neutral_density_relief,
    recrystallized_child_stored_energy_derivative,
)
from full_model.production.physical_grains import GrainRecord, GrainTracker
from full_model.production.stateful_embryos import (
    EmbryoPopulation, circular_phase_support, evolve_embryo,
)
from tests.test_full_model_stateful_embryos import environment, parameters, record


class PhasePromotionTransferTest(unittest.TestCase):
    def state(self):
        rp = np.full((8, 8, 2), 4.0e14)
        rm = np.full((8, 8, 2), 3.0e14)
        forest = np.full((8, 8, 2), 2.0e14)
        wall = np.full((8, 8), 1.0e14)
        gb = np.full((8, 8), 1.0e13)
        core = np.zeros((8, 8), bool); core[3:5, 3:5] = True
        shell = np.zeros((8, 8), bool); shell[2:6, 2:6] = True; shell[core] = False
        return rp, rm, forest, wall, gb, core, shell

    def test_relief_conserves_line_and_signed_burgers_content(self):
        state = self.state()
        result = conservative_neutral_density_relief(
            *state, target_core_density_m2=5e14, cell_area_m2=1e-14,
            represented_thickness_m=1e-6,
            maximum_density_m2=1e18)
        rp, rm, forest, wall, gb, ledger = result
        self.assertEqual(ledger.maximum_signed_burgers_density_change_m2, 0.0)
        self.assertLessEqual(abs(ledger.line_content_closure_m), 1e-15)
        self.assertGreater(ledger.line_content_shell_transfer_m, 0.0)
        self.assertEqual(
            ledger.line_content_removed_from_core_m,
            ledger.line_content_shell_transfer_m
            + ledger.line_content_pair_annihilation_m
            + ledger.line_content_declared_sink_m)
        self.assertTrue(np.all(forest[state[-2]] == 0.0))
        self.assertTrue(np.all(wall[state[-2]] == 0.0))
        self.assertGreater(np.mean(gb[state[-1]]), 1e13)
        self.assertTrue(np.array_equal(rp-rm, state[0]-state[1]))

    def test_promoted_child_stored_energy_drive_has_physical_sign_and_units(self):
        eta = np.array([0.0, 0.5, 1.0])
        derivative = recrystallized_child_stored_energy_derivative(
            eta, np.full(3, 3.0e8), np.full(3, 2.0e-9), 1.0e16)
        self.assertTrue(np.array_equal(
            derivative, np.array([0.0, -4.2e8, 0.0])))
        no_relief = recrystallized_child_stored_energy_derivative(
            eta, np.full(3, 1.0e7), np.full(3, 2.0e-9), 1.0e16)
        self.assertTrue(np.array_equal(no_relief, np.zeros(3)))

    def test_capacity_failure_rejects_without_hidden_clipping(self):
        with self.assertRaisesRegex(ValueError, "insufficient boundary-shell capacity"):
            conservative_neutral_density_relief(
                *self.state(), target_core_density_m2=5e14, cell_area_m2=1e-14,
                represented_thickness_m=1e-6,
                maximum_density_m2=1.1e13)

    def test_physical_shell_target_ledgers_excess_as_recovery(self):
        state = self.state()
        target = np.full((8, 8), 2.0e13)
        result = conservative_neutral_density_relief(
            *state, target_core_density_m2=5e14, cell_area_m2=1e-14,
            represented_thickness_m=1e-6, maximum_density_m2=1e18,
            target_shell_density_m2=target)
        rp, rm, forest, wall, gb, ledger = result
        self.assertTrue(np.allclose(gb[state[-1]], target[state[-1]]))
        self.assertGreater(ledger.line_content_pair_annihilation_m, 0.0)
        self.assertGreater(ledger.line_content_declared_sink_m, 0.0)
        self.assertEqual(
            ledger.line_content_removed_from_core_m,
            ledger.line_content_shell_transfer_m
            + ledger.line_content_pair_annihilation_m
            + ledger.line_content_declared_sink_m)
        self.assertLessEqual(abs(ledger.line_content_closure_m), 1e-15)
        self.assertTrue(np.array_equal(rp-rm, state[0]-state[1]))

    def test_overlapping_or_empty_masks_are_rejected(self):
        state = list(self.state())
        state[-1] = state[-2].copy()
        with self.assertRaises(ValueError):
            conservative_neutral_density_relief(
                *state, target_core_density_m2=5e14, cell_area_m2=1e-14,
                represented_thickness_m=1e-6,
                maximum_density_m2=1e18)


class AtomicPromotionTest(unittest.TestCase):
    def setup_transaction(self):
        state = PhasePromotionTransferTest().state()
        eta = np.zeros((8, 8, 3)); eta[:, :, 0] = 1.0
        psi = np.zeros(3)
        promoted = evolve_embryo(
            record(radius=1.5e-6), 11, 1e-6, 1e-7,
            environment(support=1e-10), parameters()).record
        promoted = replace(promoted, parent_grain=0, parent_lineage="grain-0")
        self.assertEqual(promoted.status, "promotable")
        embryos = EmbryoPopulation(1, (promoted,))
        tracker = GrainTracker((GrainRecord(
            0, 0.0, None, "grain-0", 0.0, None, False),))
        support = circular_phase_support(promoted, (8, 8), (8e-6, 8e-6), 2e-7)
        return eta, psi, state, embryos, tracker, support

    @staticmethod
    def favorable_energy(**trial):
        # Explicit synthetic event fixture: the allocated child has a lower
        # bulk/stored term. Other required terms remain separately reported.
        return dict(elastic=2.0, bulk_stored=10.0-trial["Ng"], line=3.0,
                    interface_order=0.25*trial["Ng"], compatibility=1.0)

    def promote(self, **overrides):
        eta, psi, state, embryos, tracker, support = self.setup_transaction()
        args = dict(
            eta=eta, psi_gv=psi, Ng=1, rp=state[0], rm=state[1],
            rho_forest=state[2], rho_wall=state[3], rho_gb=state[4],
            embryos=embryos, embryo_id=0, tracker=tracker,
            phase_support=support, purity_threshold=0.8,
            target_core_density_m2=5e14, cell_area_m2=1e-12,
            represented_thickness_m=1e-6,
            maximum_density_m2=1e18, step=12, time_s=2e-6,
            energy_evaluator=self.favorable_energy)
        args.update(overrides)
        return args, atomic_phase_promotion(**args)

    def test_supercritical_embryo_commits_once_with_exact_ledgers(self):
        _, result = self.promote()
        self.assertEqual(result.Ng, 2)
        self.assertEqual(result.embryos.records[0].status, "promoted")
        self.assertEqual(result.tracker.records[1].source_embryo_id, 0)
        self.assertLessEqual(abs(result.ledger.energy_closure_J), 1e-14)
        self.assertLessEqual(abs(result.ledger.transfer.line_content_closure_m), 1e-14)
        self.assertEqual(result.ledger.transfer.maximum_signed_burgers_density_change_m2, 0.0)
        self.assertLess(np.max(np.abs(np.sum(result.eta[:, :, :2], axis=2)-1.0)), 2e-15)
        with self.assertRaises(ValueError):
            atomic_phase_promotion(**{**self.promote()[0], "embryos": result.embryos})

    def test_capacity_and_slot_failures_leave_inputs_unchanged(self):
        args, _ = self.promote()
        snapshots = {k: v.copy() for k, v in args.items() if isinstance(v, np.ndarray)}
        for changes in ({"maximum_density_m2": 1.1e13}, {"Ng": args["eta"].shape[2]}):
            with self.assertRaises(ValueError):
                atomic_phase_promotion(**{**args, **changes})
            for key, value in snapshots.items():
                self.assertTrue(np.array_equal(args[key], value))

    def test_unresolved_pure_core_cannot_allocate_a_label(self):
        args, _ = self.promote()
        snapshots = {k: v.copy() for k, v in args.items()
                     if isinstance(v, np.ndarray)}
        with self.assertRaisesRegex(ValueError, "pure core is below"):
            atomic_phase_promotion(**{
                **args, "minimum_core_area_m2": 1.0e-8})
        for key, value in snapshots.items():
            self.assertTrue(np.array_equal(args[key], value))

    def test_label_permutation_gives_permuted_equivalent_child(self):
        _, base = self.promote()
        args, _ = self.promote()
        eta = args["eta"].copy(); eta[:, :, 1] = eta[:, :, 0]; eta[:, :, 0] = 0.0
        tracker = GrainTracker((
            GrainRecord(0, 0.2, None, "grain-0", 0.0, None, False),
            GrainRecord(1, 0.0, None, "grain-1", 0.0, None, False)))
        embryo = args["embryos"].records[0]
        embryos = EmbryoPopulation(1, (replace(embryo, parent_grain=1),))
        perm = atomic_phase_promotion(**{
            **args, "eta": eta, "Ng": 2, "tracker": tracker,
            "embryos": embryos, "psi_gv": np.zeros(3)})
        self.assertEqual(perm.tracker.records[-1].parent_label, 1)
        self.assertTrue(np.allclose(perm.eta[:, :, -1], base.eta[:, :, 1]))

    def test_energy_rejection_exposes_complete_term_ledger(self):
        args, _ = self.promote()

        def unfavorable_energy(**trial):
            return dict(elastic=2.0, bulk_stored=10.0,
                        line=3.0, interface_order=float(trial["Ng"]),
                        compatibility=1.0)

        with self.assertRaises(PromotionEnergyIncreaseError) as caught:
            atomic_phase_promotion(**{**args, "energy_evaluator": unfavorable_energy})
        error = caught.exception
        self.assertEqual(set(error.delta), {
            "elastic", "bulk_stored", "line", "interface_order", "compatibility"})
        self.assertEqual(error.delta["interface_order"], 1.0)
        self.assertEqual(error.free_energy_change_J, 1.0)
        self.assertIn("interface_order=1 J", str(error))


if __name__ == "__main__":
    unittest.main()
