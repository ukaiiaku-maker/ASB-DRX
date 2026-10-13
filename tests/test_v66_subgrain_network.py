import math

import numpy as np
import pytest

from full_model.production.arrhenius_kinetics import ActivatedProcess
from full_model.production.structural_boundary import StructuralBoundaryParameters
from full_model.production.subgrain_network import (
    SubgrainNetworkParameters,
    SubgrainNetworkState,
    advance_subgrain_network,
    boundary_velocities_m_s,
    network_energy_J_m2,
)


def _parameters():
    return SubgrainNetworkParameters(
        structural=StructuralBoundaryParameters(),
        process=ActivatedProcess("subgrain migration", 1e9),
        mobility_prefactor_m4_J_s=2e-11,
        activation_enthalpy_0_J=0.7*1.602176634e-19,
        critical_pressure_Pa=2e6, exp_a=2.0, exp_n=2.0,
        exp_floor=.05)


def test_selective_growth_direction_and_equal_energy_control():
    p = _parameters()
    state = SubgrainNetworkState(
        np.array((0.0, 1e-6, 2e-6)),
        np.radians((0.0, 2.0)), np.array((1e6, 3e6)))
    assert boundary_velocities_m_s(state, 900.0, p)[0] > 0.0
    control = SubgrainNetworkState(
        state.edges_m, state.orientation_rad, np.array((2e6, 2e6)))
    assert boundary_velocities_m_s(control, 900.0, p)[0] == 0.0


def test_orientation_gradient_concentrates_when_middle_cell_disappears():
    p = _parameters()
    # The high-energy narrow middle cell shrinks from both sides.
    state = SubgrainNetworkState(
        np.array((0.0, 1.0e-6, 1.08e-6, 2.0e-6)),
        np.radians((0.0, 2.5, 5.0)),
        np.array((1e6, 8e6, 1e6)))
    before_orientations = set(state.orientation_rad.tolist())
    result = advance_subgrain_network(state, dt_s=1e-2,
                                      temperature_K=900.0, parameters=p)
    assert result.classification == "TOPOLOGY_COARSENING"
    assert result.delta_helmholtz_J_m2 < 0.0
    assert result.orientation_sum_residual_rad == pytest.approx(0.0, abs=1e-15)
    assert result.state.orientation_rad.size == 2
    assert set(result.state.orientation_rad.tolist()).issubset(before_orientations)
    assert np.degrees(np.diff(result.state.orientation_rad))[0] == pytest.approx(5.0)
    assert result.state.eliminated_cells == 1


def test_network_energy_counts_bulk_and_each_boundary_once():
    p = _parameters()
    state = SubgrainNetworkState(
        np.array((0.0, 1e-6, 2e-6)),
        np.radians((0.0, 2.0)), np.array((1e6, 3e6)))
    bulk = 4.0
    gamma = p.structural.high_angle_energy_J_m2*(2/15)*(1-math.log(2/15))
    assert network_energy_J_m2(state, p) == pytest.approx(bulk+gamma)
