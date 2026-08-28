from __future__ import annotations

import math
import unittest
import numpy as np

from asb_drx.analytical import ExpFloorLaw
from asb_drx.stability import (
    StabilityParameters, common_stress_rate_tangents,
    local_thermal_storage_rhs, thermal_storage_mode,
)

EV_J=1.602176634e-19


class StabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.law=ExpFloorLaw(1.5*EV_J,1.2e9,1000.,.2,2.,2.5,1e12,4.,2.5e-10,.3,.1)
        self.p=StabilityParameters(3.5e6,5.,5e-9,1e14)
        self.stress=3e8; self.rho=5e13; self.T=1000.

    def test_analytical_rate_tangents_match_centered_differences(self) -> None:
        a=common_stress_rate_tangents(self.law,self.stress,self.rho,self.T)
        dT=1e-3; drho=self.rho*1e-6
        qT=(common_stress_rate_tangents(self.law,self.stress,self.rho,self.T+dT).plastic_rate_s_inv-common_stress_rate_tangents(self.law,self.stress,self.rho,self.T-dT).plastic_rate_s_inv)/(2*dT)
        qr=(common_stress_rate_tangents(self.law,self.stress,self.rho+drho,self.T).plastic_rate_s_inv-common_stress_rate_tangents(self.law,self.stress,self.rho-drho,self.T).plastic_rate_s_inv)/(2*drho)
        self.assertAlmostEqual(a.temperature_tangent_s_inv_K/qT,1.,places=7)
        self.assertAlmostEqual(a.density_tangent_m2_s_inv/qr,1.,places=7)

    def test_jacobian_matches_local_rhs_finite_difference(self) -> None:
        mode=thermal_storage_mode(self.law,self.stress,self.rho,self.T,1e-12,self.p)
        dT=1e-3; drho=self.rho*1e-6
        colT=(local_thermal_storage_rhs(self.law,self.stress,self.rho,self.T+dT,self.p)-local_thermal_storage_rhs(self.law,self.stress,self.rho,self.T-dT,self.p))/(2*dT)
        colR=(local_thermal_storage_rhs(self.law,self.stress,self.rho+drho,self.T,self.p)-local_thermal_storage_rhs(self.law,self.stress,self.rho-drho,self.T,self.p))/(2*drho)
        self.assertTrue(np.allclose(mode.jacobian_s_inv[:,0],colT,rtol=1e-7,atol=1e-12))
        self.assertTrue(np.allclose(mode.jacobian_s_inv[:,1],colR,rtol=1e-7,atol=1e-20))

    def test_conduction_shifts_only_temperature_diagonal(self) -> None:
        k1=1e4; k2=2e4
        a=thermal_storage_mode(self.law,self.stress,self.rho,self.T,k1,self.p)
        b=thermal_storage_mode(self.law,self.stress,self.rho,self.T,k2,self.p)
        expected=-(self.p.thermal_conductivity_W_m_K/self.p.volumetric_heat_capacity_J_m3_K)*(k2*k2-k1*k1)
        self.assertAlmostEqual(b.jacobian_s_inv[0,0]-a.jacobian_s_inv[0,0],expected)
        self.assertTrue(np.array_equal(b.jacobian_s_inv[1],a.jacobian_s_inv[1]))

    def test_reported_growth_is_largest_real_eigenvalue(self) -> None:
        mode=thermal_storage_mode(self.law,self.stress,self.rho,self.T,2*math.pi/1.6e-5,self.p)
        self.assertEqual(mode.maximum_growth_rate_s_inv,float(np.max(np.real(mode.eigenvalues_s_inv))))

    def test_invalid_storage_partition_is_rejected(self) -> None:
        bad=StabilityParameters(3.5e6,5.,5e-6,1e14)
        with self.assertRaisesRegex(ValueError,"more energy"):
            thermal_storage_mode(self.law,self.stress,self.rho,self.T,1e4,bad)


if __name__=="__main__": unittest.main()
