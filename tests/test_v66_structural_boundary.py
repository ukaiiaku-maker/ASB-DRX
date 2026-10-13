import math

import numpy as np
import pytest

from full_model.production.structural_boundary import (
    NetworkState,
    StructuralBoundaryParameters,
    StructuralBoundaryState,
    absorb_lattice_content,
    boundary_misorientation_rad,
    exp_floor_event_rate_Hz,
    network_structural_content_m1,
    read_shockley_energy_J_m2,
    read_shockley_slope_J_m2_rad,
    transfer_network_structural_content,
)


def test_read_shockley_value_slope_and_cap():
    p = StructuralBoundaryParameters()
    theta = math.radians(3.0)
    h = 1e-7
    numerical = (read_shockley_energy_J_m2(theta+h, p)
                 -read_shockley_energy_J_m2(theta-h, p))/(2*h)
    assert numerical == pytest.approx(
        read_shockley_slope_J_m2_rad(theta, p), rel=2e-9)
    assert read_shockley_energy_J_m2(0.0, p) == 0.0
    assert read_shockley_energy_J_m2(math.radians(20.0), p) == pytest.approx(
        p.high_angle_energy_J_m2)


def test_compatible_absorption_rotates_and_closes_line_balance():
    p = StructuralBoundaryParameters()
    initial_theta = math.radians(2.0)
    state = StructuralBoundaryState(
        2.0e14, 0.0, initial_theta/p.burgers_m)
    amount = math.radians(0.1)/p.burgers_m
    result = absorb_lattice_content(
        state, absorbed_plus_m1=amount, absorbed_minus_m1=0.0,
        compatible=True, parameters=p)
    assert result.classification == "compatible_absorption_rotation"
    assert boundary_misorientation_rad(result.candidate, p) == pytest.approx(
        initial_theta+math.radians(0.1))
    assert result.line_balance_residual_m1 == pytest.approx(0.0, abs=1e-6)
    assert result.delta_helmholtz_J_m2 < 0.0
    assert result.accepted


def test_sign_balanced_and_incompatible_controls_do_not_rotate():
    p = StructuralBoundaryParameters()
    state = StructuralBoundaryState(2.0e14, 2.0e14,
                                    math.radians(2.0)/p.burgers_m)
    amount = 1.0e6
    balanced = absorb_lattice_content(
        state, absorbed_plus_m1=amount, absorbed_minus_m1=amount,
        compatible=True, parameters=p)
    incompatible = absorb_lattice_content(
        state, absorbed_plus_m1=amount, absorbed_minus_m1=0.0,
        compatible=False, parameters=p)
    assert balanced.classification == "sign_balanced_no_rotation"
    assert incompatible.classification == "incompatible_extrinsic_storage"
    assert balanced.structural_change_m1 == 0.0
    assert incompatible.structural_change_m1 == 0.0
    assert boundary_misorientation_rad(balanced.candidate, p) == pytest.approx(
        boundary_misorientation_rad(state, p))
    assert boundary_misorientation_rad(incompatible.candidate, p) == pytest.approx(
        boundary_misorientation_rad(state, p))
    assert balanced.line_balance_residual_m1 == pytest.approx(0.0, abs=1e-8)
    assert incompatible.line_balance_residual_m1 == pytest.approx(0.0, abs=1e-8)


def test_zero_absorption_is_explicit_no_event():
    p = StructuralBoundaryParameters()
    state = StructuralBoundaryState(2e14, 2e14,
                                    math.radians(2.0)/p.burgers_m)
    result = absorb_lattice_content(
        state, absorbed_plus_m1=0.0, absorbed_minus_m1=0.0,
        compatible=True, parameters=p)
    assert result.classification == "no_incoming_content"
    assert result.candidate == state
    assert result.delta_helmholtz_J_m2 == 0.0


def test_opposite_signed_absorption_reports_annihilation():
    p = StructuralBoundaryParameters()
    state = StructuralBoundaryState(
        0.0, 2.0e14, math.radians(2.0)/p.burgers_m)
    amount = math.radians(0.25)/p.burgers_m
    result = absorb_lattice_content(
        state, absorbed_plus_m1=0.0, absorbed_minus_m1=amount,
        compatible=True, parameters=p)
    assert result.classification == "compatible_structural_annihilation"
    assert result.annihilated_line_m1 == pytest.approx(2.0*amount)
    assert result.line_balance_residual_m1 == pytest.approx(0.0, abs=1e-6)


def test_network_transfer_is_compatible_conservative_and_energy_lowering():
    p = StructuralBoundaryParameters()
    state = NetworkState(tuple(map(math.radians, (0.0, 3.0, 5.0))))
    transfer = math.radians(0.1)/p.burgers_m
    result = transfer_network_structural_content(state, transfer, p)
    assert result.balance_residual_m1 == pytest.approx(0.0, abs=1e-6)
    assert result.candidate.orientation_rad[1] == pytest.approx(
        math.radians(3.1))
    assert result.delta_helmholtz_J_m2 < 0.0
    assert result.accepted
    before = network_structural_content_m1(state, p)
    after = network_structural_content_m1(result.candidate, p)
    np.testing.assert_allclose(after-before, (transfer, -transfer), rtol=1e-13)


def test_equal_boundary_network_has_symmetric_coarsening_instability():
    p = StructuralBoundaryParameters()
    state = NetworkState(tuple(map(math.radians, (0.0, 2.5, 5.0))))
    transfer = math.radians(0.02)/p.burgers_m
    positive = transfer_network_structural_content(state, transfer, p)
    negative = transfer_network_structural_content(state, -transfer, p)
    assert positive.delta_helmholtz_J_m2 == pytest.approx(
        negative.delta_helmholtz_J_m2, rel=1e-12)
    # Read--Shockley is concave below theta_m: the symmetric state has no
    # directional bias, but either finite perturbation lowers boundary energy.
    assert positive.delta_helmholtz_J_m2 < 0.0


def test_exp_floor_rate_is_monotone_and_saturates_at_floor():
    p = StructuralBoundaryParameters()
    affinity = np.array((0.0, 1e-9, 1e3))
    rate = exp_floor_event_rate_Hz(affinity, 900.0, p)
    assert np.all(np.diff(rate) >= 0.0)
    assert rate[1] > rate[0]
    expected_ceiling = p.attempt_frequency_Hz*math.exp(
        -p.barrier_floor_eV*1.602176634e-19/(1.380649e-23*900.0))
    assert rate[-1] == pytest.approx(expected_ceiling)
