from __future__ import annotations

import unittest

import numpy as np

from asb_drx.arrhenius_v3 import (
    ArrheniusMechanism,
    BoundedActivationEntropy,
    ExpFloorEnthalpy,
    KB_J_PER_K,
)
from asb_drx.vector_topology_cdd_v3 import (
    JunctionReaction,
    VectorTopologyNetwork,
    VectorTopologyState,
    advance_reactions,
    arrhenius_forest_linearization,
    reaction_rates_m2_s,
    reaction_transport_symbol_s_inv,
)


class VectorTopologyCDDV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temperature = 900.0
        mechanism = ArrheniusMechanism(
            ExpFloorEnthalpy(
                0.45 * 1.602176634e-19, 0.8e9, 900.0, 0.25, 1.2, 2.0
            ),
            BoundedActivationEntropy(reference_kB=0.2),
            2.0e7,
            validity_temperature_K=(500.0, 1400.0),
            validity_stress_Pa=(0.0, 2.0e9),
        )
        self.mechanism = mechanism
        reference = 5.0e14
        self.network = VectorTopologyNetwork(
            2.86e-10,
            (
                JunctionReaction(
                    (0, 1), "glissile", mechanism,
                    0.02 * 1.602176634e-19, reference,
                    product_velocity_m_s=2.0e-5,
                    product_diffusivity_m2_s=2.0e-12,
                ),
                JunctionReaction(
                    (2, 3), "sessile", mechanism,
                    -0.01 * 1.602176634e-19, reference,
                ),
            ),
        )
        self.mobile = np.full((8, 6, 8), 2.5e14)
        junction = np.empty((2, 6, 8))
        for index, reaction in enumerate(self.network.reactions):
            forward, reverse = reaction.rate_constants_s_inv(self.temperature)
            first, second = reaction.parent_species
            junction[index] = (
                forward / reverse * self.mobile[first] * self.mobile[second]
                / reaction.reference_density_m2
            )
        self.state = VectorTopologyState(
            self.mobile, junction, np.asarray([0.6, 0.8]), 0.2e-6, 0.2e-6
        )

    def test_incidence_obeys_frank_rule_for_every_product(self) -> None:
        np.testing.assert_allclose(
            self.network.frank_residuals_m, 0.0, atol=2.0e-25
        )
        self.assertEqual(self.network.incidence.shape, (10, 2))

    def test_detailed_balance_ratio_and_equilibrium_rate(self) -> None:
        for reaction in self.network.reactions:
            forward, reverse = reaction.rate_constants_s_inv(self.temperature)
            expected = np.exp(
                -reaction.reaction_free_energy_J
                / (KB_J_PER_K * self.temperature)
            )
            self.assertAlmostEqual(forward / reverse / expected, 1.0, places=14)
        rates = reaction_rates_m2_s(self.state, self.network, self.temperature)
        forward, _ = self.network.reactions[0].rate_constants_s_inv(self.temperature)
        self.assertLess(
            float(np.max(np.abs(rates))),
            2.0e-13 * forward * float(np.max(self.mobile)),
        )

    def test_reaction_step_preserves_nye_tensor_pointwise(self) -> None:
        perturbed_junction = self.state.junction_m2.copy()
        perturbed_junction[0] *= 0.9
        state = VectorTopologyState(
            self.state.mobile_m2, perturbed_junction,
            self.state.line_tangent_2d, self.state.dx_m, self.state.dy_m,
        )
        old_nye = state.nye_tensor_m_inv(self.network)
        advanced, ledger = advance_reactions(
            state, self.network, self.temperature, 1.0e-8
        )
        np.testing.assert_allclose(
            advanced.nye_tensor_m_inv(self.network), old_nye,
            rtol=2.0e-15, atol=2.0e-10,
        )
        self.assertLess(ledger.maximum_frank_source_residual_m_inv_s, 1.0e-8)
        self.assertGreaterEqual(ledger.minimum_population_m2, 0.0)

    def test_junction_forest_resistance_changes_barrier_not_mobility_sign(self) -> None:
        stresses = np.asarray([0.5e9, 0.4e9, 0.35e9, 0.3e9])
        coefficients = np.full((4, 2), 2.0e-7)
        junction = np.asarray([2.0e14, 1.0e14])
        velocities, derivative = arrhenius_forest_linearization(
            (self.mechanism,) * 4, np.full(4, 1.0e-9), stresses,
            junction, coefficients, self.temperature,
        )
        self.assertTrue(np.all(velocities[:4] > 0.0))
        np.testing.assert_allclose(velocities[4:], -velocities[:4])
        self.assertTrue(np.all(derivative[:4] < 0.0))
        np.testing.assert_allclose(derivative[4:], -derivative[:4])

    def test_complete_zero_mode_operator_preserves_burgers_content(self) -> None:
        base_mobile = self.state.mobile_m2[:, 0, 0]
        base_junction = self.state.junction_m2[:, 0, 0]
        operator = reaction_transport_symbol_s_inv(
            np.zeros(2), base_mobile, base_junction, self.network,
            self.temperature, self.state.line_tangent_2d,
            np.zeros(8), np.zeros((8, 2)), np.full(8, 1.0e-12),
        )
        burgers = self.network.species_burgers_vectors_m.T
        np.testing.assert_allclose(burgers @ operator, 0.0, atol=2.0e-14)
        self.assertLessEqual(float(np.max(np.linalg.eigvals(operator).real)), 2.0e-9)

    def test_positive_diffusion_damps_all_nonconserved_finite_modes_without_friction(self) -> None:
        base_mobile = self.state.mobile_m2[:, 0, 0]
        base_junction = self.state.junction_m2[:, 0, 0]
        maximum = -np.inf
        for mode in range(1, 17):
            k = np.asarray([2.0 * np.pi * mode / 16.0e-6, 0.0])
            operator = reaction_transport_symbol_s_inv(
                k, base_mobile, base_junction, self.network,
                self.temperature, self.state.line_tangent_2d,
                np.linspace(-2.0e-4, 2.0e-4, 8),
                np.zeros((8, 2)), np.full(8, 1.0e-12),
            )
            maximum = max(maximum, float(np.max(np.linalg.eigvals(operator).real)))
        self.assertLess(maximum, 0.0)


if __name__ == "__main__":
    unittest.main()
