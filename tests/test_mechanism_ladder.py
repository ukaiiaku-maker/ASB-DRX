from __future__ import annotations

import math
import unittest
import numpy as np

from asb_drx.analytical import ExpFloorLaw
from asb_drx.localization import LocalizationCriteria
from asb_drx.mechanism_ladder import (
    classify_mechanism_trace, matched_isothermal_case, run_mechanism_trace,
    standard_mechanism_ladder,
)
from asb_drx.multi_order import BinaryCircularLimit, diffuse_binary_circle
from asb_drx.spatial_coupled import SpatialCoupledParameters, SpatialCoupledState

EV_J = 1.602176634e-19


class MechanismLadderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.law = ExpFloorLaw(1.5*EV_J, 1.2e9, 1000.0, .2, 2.0, 2.5, 1e12, 4.0, 2.5e-10)
        self.p = SpatialCoupledParameters(8e10, 3.5e6, 5.0, 5e-9, 1e14, 2e6, 1e-6, 5e-7)
        points = 16; self.dx = 1.6e-5/points
        boundary = math.sqrt(self.p.gradient_coefficient_J_m*self.p.pair_penalty_J_m3)/3
        limit = BinaryCircularLimit(boundary, self.p.stored_line_energy_J_m*4e13, 1.0)
        self.interface = 2*math.sqrt(self.p.gradient_coefficient_J_m/self.p.pair_penalty_J_m3)
        eta = diffuse_binary_circle(points, self.dx, 1.35*limit.critical_radius_m, self.interface)
        rho=np.empty_like(eta); rho[0]=5e13; rho[1]=1e13
        self.initial=SpatialCoupledState(1e8,0,np.zeros((points,points)),np.full((points,points),1000.),rho,eta)

    def test_standard_ladder_has_six_distinct_declared_cases(self) -> None:
        ladder = standard_mechanism_ladder(10.0, 1000.0)
        self.assertEqual(len(ladder), 6)
        self.assertEqual(len({item.name for item in ladder}), 6)
        self.assertTrue(ladder[0].unload_initial_stress)
        self.assertFalse(ladder[0].controls.evolve_temperature)
        self.assertFalse(ladder[1].controls.evolve_phase)
        self.assertTrue(ladder[-1].controls.evolve_temperature and ladder[-1].controls.evolve_phase)

    def test_matched_control_changes_only_thermal_retention(self) -> None:
        case = standard_mechanism_ladder(10.0,1000.0)[-1]
        control = matched_isothermal_case(case)
        self.assertEqual(control.applied_shear_rate_s_inv, case.applied_shear_rate_s_inv)
        self.assertEqual(control.controls.evolve_phase, case.controls.evolve_phase)
        self.assertFalse(control.controls.evolve_temperature)

    def test_generic_coupled_fixture_is_not_classified_as_localized(self) -> None:
        case = standard_mechanism_ladder(10.0,1000.0)[4]
        control_case = matched_isothermal_case(case)
        trace = run_mechanism_trace(self.initial,case,self.dx,1e-5,5,self.law,self.p)
        control = run_mechanism_trace(self.initial,control_case,self.dx,1e-5,5,self.law,self.p)
        criteria = LocalizationCriteria(.4,20.,.1,3.,3,.05)
        decision = classify_mechanism_trace(trace,control,self.dx,self.interface,criteria)
        self.assertFalse(decision.localized)


if __name__ == "__main__": unittest.main()
