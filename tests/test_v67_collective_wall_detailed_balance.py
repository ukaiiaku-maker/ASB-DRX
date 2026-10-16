from dataclasses import replace
import math

import numpy as np
import pytest

from full_model.production.collective_wall_nucleation import (
    _collective_wall_candidate,
    collective_wall_detailed_balance,
)
from full_model.production.multigrain_common_state import MultiGrainCommonState
from full_model.production.structural_boundary import StructuralBoundaryParameters
from tests.test_full_model_multigrain_common_state import owner


EV_J = 1.602176634e-19
KB_J_K = 1.380649e-23


def test_one_transition_state_enforces_detailed_balance_for_downhill_path():
    activation = StructuralBoundaryParameters(
        activation_enthalpy_eV=.8, barrier_floor_eV=.05,
        attempt_frequency_Hz=1e7, activation_segment_m=2.5e-9)
    path_eV = np.array([0.0, -.5, -2.0, -4.96])
    audit = collective_wall_detailed_balance(
        path_eV*EV_J, 1000.0, activation)
    assert audit.resolved_forward_barrier_J == 0.0
    assert audit.forward_barrier_eV == pytest.approx(.8)
    assert audit.reverse_barrier_eV == pytest.approx(5.76)
    assert audit.detailed_balance_log_residual == pytest.approx(0.0, abs=2e-13)
    expected = math.exp(-path_eV[-1]*EV_J/(KB_J_K*1000.0))
    assert audit.forward_rate_Hz/audit.reverse_rate_Hz == pytest.approx(
        expected, rel=2e-13)


def test_common_saddle_preserves_floor_for_uphill_endpoint():
    activation = StructuralBoundaryParameters(
        activation_enthalpy_eV=.2, barrier_floor_eV=.05,
        attempt_frequency_Hz=1e7, activation_segment_m=2.5e-9)
    audit = collective_wall_detailed_balance(
        np.array([0.0, .1, .4])*EV_J, 900.0, activation)
    assert audit.forward_barrier_eV == pytest.approx(.6)
    assert audit.reverse_barrier_eV == pytest.approx(.2)
    assert audit.reverse_barrier_eV >= activation.barrier_floor_eV
    assert audit.detailed_balance_log_residual == pytest.approx(0.0, abs=2e-13)


def test_reaction_coordinate_conserves_signed_and_total_line_content():
    original = owner((8, 8), value=2e13)
    plus = original.mobile_plus_m2.copy()
    plus[..., 0] += 3e13
    original = replace(original, mobile_plus_m2=plus)
    state = MultiGrainCommonState(
        (7,), np.ones((1, 8, 8)), (original,))
    ring = np.zeros((8, 8)); ring[2:6, 2:6] = .75
    candidate, transferred = _collective_wall_candidate(
        state, 0, ring, .1, coordinate=.4)
    changed = candidate.owners[0]
    before_total = np.sum(
        original.mobile_plus_m2+original.mobile_minus_m2
        +original.wall_plus_m2+original.wall_minus_m2,
        dtype=np.longdouble)
    after_total = np.sum(
        changed.mobile_plus_m2+changed.mobile_minus_m2
        +changed.wall_plus_m2+changed.wall_minus_m2,
        dtype=np.longdouble)
    before_signed = np.sum(
        original.mobile_plus_m2-original.mobile_minus_m2
        +original.wall_plus_m2-original.wall_minus_m2,
        dtype=np.longdouble)
    after_signed = np.sum(
        changed.mobile_plus_m2-changed.mobile_minus_m2
        +changed.wall_plus_m2-changed.wall_minus_m2,
        dtype=np.longdouble)
    assert transferred > 0.0
    assert float(after_total-before_total) == 0.0
    assert float(after_signed-before_signed) == 0.0
    np.testing.assert_allclose(
        changed.wall_order,
        original.wall_order+.4*.1*ring*(1.0-original.wall_order))


def test_energy_path_requires_zero_origin():
    with pytest.raises(ValueError, match="start at zero"):
        collective_wall_detailed_balance(
            np.array([.1, 0.0])*EV_J, 1000.0,
            StructuralBoundaryParameters())
