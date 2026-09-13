from __future__ import annotations

import unittest

import numpy as np

from asb_drx.arrhenius_v3 import (
    ArrheniusMechanism, BoundedActivationEntropy, ExpFloorEnthalpy,
)
from asb_drx.nonlinear_memory_cdd_v3 import (
    FrozenMemoryParameters, FrozenMemoryState, advance_frozen_memory,
    frozen_memory_rhs, homogeneous_equilibrium_state,
)
from asb_drx.vector_topology_cdd_v3 import (
    JunctionReaction, VectorTopologyNetwork, arrhenius_forest_linearization,
    reaction_transport_memory_symbol_s_inv,
)


class NonlinearMemoryCDDV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        ev = 1.602176634e-19
        mechanism = ArrheniusMechanism(
            ExpFloorEnthalpy(0.45 * ev, 0.8e9, 900.0, 0.25, 1.2, 2.0),
            BoundedActivationEntropy(reference_kB=0.2), 2.0e7,
            validity_temperature_K=(500.0, 1400.0),
            validity_stress_Pa=(0.0, 2.0e9),
        )
        network = VectorTopologyNetwork(
            2.86e-10,
            (JunctionReaction((0, 1), "sessile", mechanism, 0.02 * ev, 5.0e14),),
        )
        self.parameters = FrozenMemoryParameters(
            network, (mechanism,) * 4, np.full(4, 1.0e-9),
            np.full(4, 0.5e9), np.full((4, 1), 1.0e-6),
            np.full(8, 1.0e-12), np.asarray([3.0e-3]), 1000.0, 4.0e-6,
        )
        self.state = homogeneous_equilibrium_state(32, 2.5e14, self.parameters)

    def test_nonlinear_rhs_linearizes_to_declared_memory_symbol(self) -> None:
        mode = 4
        k = 2.0 * np.pi * mode / self.parameters.domain_m
        base_mobile = self.state.mobile_m2[:, 0]
        base_junction = self.state.junction_m2[:, 0]
        velocity, derivative = arrhenius_forest_linearization(
            self.parameters.glide, self.parameters.event_lengths_m,
            self.parameters.resolved_stress_Pa, base_junction,
            self.parameters.forest_coefficients_Pa_m2,
            self.parameters.temperature_K,
        )
        symbol = reaction_transport_memory_symbol_s_inv(
            np.asarray([k, 0.0]), base_mobile, base_junction,
            self.parameters.network, self.parameters.temperature_K,
            np.asarray([1.0, 0.0]), velocity, derivative,
            self.parameters.mobile_diffusivity_m2_s,
            self.parameters.memory_relaxation_s,
        )
        eigenvalues, eigenvectors = np.linalg.eig(symbol)
        selected = int(np.argmax(eigenvalues.real))
        vector = eigenvectors[:, selected]
        vector /= np.max(np.abs(vector))
        x = np.arange(32)
        wave = np.exp(2j * np.pi * mode * x / 32)
        amplitude = 1.0e-7 * 2.5e14
        perturbation = amplitude * np.real(vector[:, None] * wave[None, :])
        perturbed = FrozenMemoryState(
            self.state.mobile_m2 + perturbation[:8],
            self.state.junction_m2 + perturbation[8:9],
            self.state.memory_m2 + perturbation[9:10],
        )
        base_rate = frozen_memory_rhs(self.state, self.parameters)
        rate = frozen_memory_rhs(perturbed, self.parameters)
        rate_fields = np.concatenate(rate, axis=0) - np.concatenate(base_rate, axis=0)
        coefficient = np.fft.fft(rate_fields, axis=1)[:, mode] / 32
        input_coefficient = 0.5 * amplitude * vector
        measured = np.vdot(input_coefficient, coefficient) / np.vdot(
            input_coefficient, input_coefficient
        )
        self.assertAlmostEqual(measured.real / eigenvalues[selected].real, 1.0, places=5)
        self.assertAlmostEqual(measured.imag / eigenvalues[selected].imag, 1.0, places=5)

    def test_short_implicit_interval_preserves_means_and_nonnegativity(self) -> None:
        x = np.arange(32)
        perturbation = 1.0e-4 * 2.5e14 * np.cos(2.0 * np.pi * 3.0 * x / 32)
        mobile = self.state.mobile_m2.copy()
        mobile[0] += perturbation
        state = FrozenMemoryState(mobile, self.state.junction_m2, self.state.memory_m2)
        species_before = np.concatenate((state.mobile_m2, state.junction_m2), axis=0)
        burgers = self.parameters.network.species_burgers_vectors_m
        alpha_before = burgers.T @ np.mean(species_before, axis=1)
        advanced, diagnostics = advance_frozen_memory(
            state, self.parameters, 2.0e-5, rtol=1.0e-8
        )
        species_after = np.concatenate((advanced.mobile_m2, advanced.junction_m2), axis=0)
        alpha_after = burgers.T @ np.mean(species_after, axis=1)
        np.testing.assert_allclose(alpha_after, alpha_before, rtol=2.0e-12, atol=2.0e-7)
        self.assertGreaterEqual(float(np.min(advanced.mobile_m2)), 0.0)
        self.assertGreater(diagnostics["rhs_evaluations"], 0)


if __name__ == "__main__":
    unittest.main()
