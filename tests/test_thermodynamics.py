from __future__ import annotations

import math
import unittest

import numpy as np

from asb_drx.thermodynamics import (
    CircularNucleusLimit,
    DislocationReservoirs,
    GrainEnergyParameters,
    chemical_potential_J_m3,
    close_work_ledger,
    energy_checked_allen_cahn_step,
    free_energy_1d_J_m2,
)


class ThermodynamicKernelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = GrainEnergyParameters(
            well_height_J_m3=2.0e6,
            gradient_coefficient_J_m=2.0e-6,
            bulk_driving_J_m3=2.0e5,
            mobility_m3_J_s=5.0e-7,
        )

    def test_chemical_potential_is_discrete_energy_derivative(self) -> None:
        eta = np.array([0.08, 0.18, 0.31, 0.55, 0.72, 0.43, 0.21])
        direction = np.array([0.3, -0.2, 0.1, -0.4, 0.2, 0.1, -0.1])
        dx = 2.0e-6
        epsilon = 1.0e-7
        finite_difference = (
            free_energy_1d_J_m2(eta + epsilon * direction, dx, self.parameters)
            - free_energy_1d_J_m2(eta - epsilon * direction, dx, self.parameters)
        ) / (2.0 * epsilon)
        variational = dx * float(np.dot(chemical_potential_J_m3(eta, dx, self.parameters), direction))
        self.assertAlmostEqual(finite_difference / variational, 1.0, places=8)

    def test_unloaded_relaxation_never_increases_discrete_energy(self) -> None:
        coordinate = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
        eta = 0.45 + 0.08 * np.sin(coordinate) + 0.03 * np.cos(3.0 * coordinate)
        dx = 1.0e-7
        energies = [free_energy_1d_J_m2(eta, dx, self.parameters)]
        for _ in range(100):
            step = energy_checked_allen_cahn_step(eta, dx, 1.0e-4, self.parameters)
            eta = step.eta
            energies.append(step.free_energy_J_m2)
        self.assertTrue(all(after <= before for before, after in zip(energies, energies[1:])))
        self.assertLess(energies[-1], energies[0])

    def test_conservative_reservoir_transfer_and_overdraw_rejection(self) -> None:
        state = DislocationReservoirs(2.0e13, 5.0e13, 1.0e13, 0.5e13)
        updated = state.conservative_transfer("mobile_m2", "forest_m2", 0.4e13)
        self.assertEqual(updated.total_m2, state.total_m2)
        self.assertEqual(updated.mobile_m2, 1.6e13)
        self.assertEqual(updated.forest_m2, 5.4e13)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            state.conservative_transfer("wall_m2", "gb_m2", 2.0e13)

    def test_work_ledger_closes_and_rejects_excess_allocation(self) -> None:
        ledger = close_work_ledger(
            1.0e6,
            stored_dislocation_J_m3=2.0e5,
            interface_J_m3=1.0e5,
            residual_gb_J_m3=0.5e5,
            accommodation_J_m3=1.5e5,
        )
        self.assertEqual(ledger.heat_J_m3, 5.0e5)
        self.assertAlmostEqual(ledger.closure_error_J_m3, 0.0, places=12)
        with self.assertRaisesRegex(ValueError, "exceed"):
            close_work_ledger(1.0, stored_dislocation_J_m3=1.1)

    def test_circular_nucleus_limit(self) -> None:
        nucleus = CircularNucleusLimit(0.5, 2.0e6, 3.0e-10)
        critical = nucleus.critical_radius_m
        self.assertEqual(critical, 2.5e-7)
        self.assertLess(nucleus.radius_rate_m_s(0.8 * critical), 0.0)
        self.assertAlmostEqual(nucleus.radius_rate_m_s(critical), 0.0, places=18)
        self.assertGreater(nucleus.radius_rate_m_s(1.2 * critical), 0.0)
        delta = critical * 1.0e-6
        derivative = (
            nucleus.excess_energy_J_m(critical + delta)
            - nucleus.excess_energy_J_m(critical - delta)
        ) / (2.0 * delta)
        self.assertAlmostEqual(derivative, 0.0, delta=1.0e-8)

    def test_invalid_dimensions_and_ranges_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "mobility"):
            GrainEnergyParameters(1.0, 1.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "0 <= eta <= 1"):
            free_energy_1d_J_m2(np.array([0.0, 0.5, 1.1]), 1.0, self.parameters)


if __name__ == "__main__":
    unittest.main()
