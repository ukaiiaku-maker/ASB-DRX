from __future__ import annotations

import math
import unittest

from asb_drx.analytical import KB_J_PER_K
from asb_drx.fixtures import EV_J, SingleGliderDDDParameterization


class SingleGliderDDDParameterizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture=SingleGliderDDDParameterization(); self.law=self.fixture.law()

    def test_linear_entropy_barrier_matches_ddd_equation(self) -> None:
        for temperature in (850.,1000.,1050.):
            expected=(self.fixture.enthalpy_eV*EV_J-KB_J_PER_K*temperature*self.fixture.entropy_kB)
            self.assertAlmostEqual(self.law.barrier_scale_J(temperature)/expected,1.,places=15)

    def test_taylor_geometry_and_prefactor_match_ddd_analytical_limit(self) -> None:
        rho=3e15; temperature=950.; local_stress=2e9
        self.assertEqual(self.law.taylor_ratio(rho),2*self.fixture.burgers_m*math.sqrt(rho))
        expected_prefactor=self.fixture.attempt_frequency_s_inv*16*rho**2*self.fixture.burgers_m**4
        expected=expected_prefactor*math.exp(-self.law.barrier_J(local_stress,temperature)/(KB_J_PER_K*temperature))
        self.assertAlmostEqual(self.law.shear_rate_s_inv(local_stress,rho,temperature)/expected,1.,places=14)

    def test_peak_closes_ddd_fixture_rate(self) -> None:
        peak=self.law.peak(1000.,self.fixture.source_strain_rate_s_inv)
        recovered=self.law.shear_rate_s_inv(peak.local_activation_stress_Pa,peak.density_m2,1000.)
        self.assertAlmostEqual(recovered/self.fixture.source_strain_rate_s_inv,1.,places=11)

    def test_line_energy_is_ddd_line_tension(self) -> None:
        expected=.5*self.fixture.shear_modulus_Pa*self.fixture.burgers_m**2
        self.assertEqual(self.fixture.spatial_parameters().stored_line_energy_J_m,expected)


if __name__=="__main__": unittest.main()
