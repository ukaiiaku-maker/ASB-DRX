import math

import numpy as np
import pytest

from full_model.production.common_axis_rx import (
    CommonAxisBoundaryState,
    advance_common_axis_absorption,
    advance_common_axis_rx_interval,
)
from full_model.production.common_tensorial_wall import (
    CommonWallDriving,
    CommonWallParameters,
    CommonWallState,
)
from full_model.production.structural_boundary import StructuralBoundaryParameters
from full_model.production.tensorial_nye import bcc_four_family_systems


def _fixture(n=16):
    p = StructuralBoundaryParameters()
    theta = math.radians(2.0)
    family = (n, n, 4)
    mp = np.full(family, 1e10)
    mm = np.full(family, 1e10)
    mp[n//2-1:n//2+1, :, 0] = 2e14
    orientation = np.zeros((n, n)); orientation[n//2:] = theta
    common = CommonWallState(
        mp, mm, np.zeros(family), np.zeros(family), np.zeros(family),
        np.zeros(family), np.zeros((n, n, 0)), np.zeros((n, n)),
        np.zeros((n, n)), np.zeros(family), np.zeros((n, n, 3, 3)),
        np.zeros((n, n, 4, 3)), np.zeros((n, n, 4, 3, 3)),
        orientation, np.full((n, n), 900.0))
    boundary = CommonAxisBoundaryState(0.0, theta, theta/p.burgers_m)
    capture = np.zeros((n, n), bool); capture[n//2-1:n//2+1] = True
    rotating = np.zeros((n, n), bool); rotating[n//2:] = True
    return p, common, boundary, capture, rotating


def test_common_axis_operator_closes_line_energy_and_orientation():
    p, common, boundary, capture, rotating = _fixture()
    wall = CommonWallParameters(spacing_m=1e-7)
    result = advance_common_axis_absorption(
        common, boundary, capture_mask=capture, rotating_mask=rotating,
        family=0, compatible=True, parameters=p, wall_parameters=wall,
        systems=bcc_four_family_systems(p.burgers_m), dt_s=1e-6)
    assert result.accepted
    assert result.classification == "COMPATIBLE_ABSORPTION_ROTATION"
    assert result.delta_helmholtz_J_m2 <= 0.0
    assert result.line_balance_residual_m1 == pytest.approx(0.0, abs=1e-5)
    theta = p.burgers_m*result.boundary_state.structural_signed_m1
    assert np.all(result.common_state.orientation_rad[rotating]
                  == pytest.approx(theta))
    assert np.all(result.common_state.orientation_rad[~rotating] == 0.0)
    assert result.boundary_state.cumulative_dissipation_J_m2 == pytest.approx(
        -result.delta_helmholtz_J_m2)


def test_accommodation_can_reject_rotation_without_changing_parent():
    p, common, boundary, capture, rotating = _fixture()
    wall = CommonWallParameters(spacing_m=1e-7)
    result = advance_common_axis_absorption(
        common, boundary, capture_mask=capture, rotating_mask=rotating,
        family=0, compatible=True, parameters=p, wall_parameters=wall,
        systems=bcc_four_family_systems(p.burgers_m), dt_s=1e-6,
        accommodation_stiffness_J_m2_rad2=1e8)
    # Backtracking may accept a vanishingly small downhill move; it must never
    # publish an uphill state or silently change a rejected parent.
    assert result.delta_helmholtz_J_m2 <= 1e-13
    if not result.accepted:
        assert result.common_state is common
        assert result.boundary_state is boundary


def test_zero_rate_is_not_required_for_thermal_absorption():
    p, common, boundary, capture, rotating = _fixture()
    wall = CommonWallParameters(spacing_m=1e-7)
    # The operator has no imposed-strain-rate argument: inherited content and
    # temperature alone can drive a stationary annealing reaction.
    result = advance_common_axis_absorption(
        common, boundary, capture_mask=capture, rotating_mask=rotating,
        family=0, compatible=True, parameters=p, wall_parameters=wall,
        systems=bcc_four_family_systems(p.burgers_m), dt_s=1e-3)
    assert result.accepted
    assert result.boundary_state.exposure_s == pytest.approx(1e-3)


def test_integrated_interval_executes_source_mechanics_and_absorption():
    p, common, boundary, capture, rotating = _fixture()
    wall = CommonWallParameters(spacing_m=1e-7, transport_scheme="upwind")
    source = np.zeros(common.orientation_rad.shape)
    source[capture] = 1e12
    result = advance_common_axis_rx_interval(
        common, boundary,
        driving=CommonWallDriving(glide_speed_m_s=np.zeros_like(
            common.mobile_plus_m2), resolved_stress_Pa=np.zeros_like(
                common.mobile_plus_m2)),
        systems=bcc_four_family_systems(p.burgers_m), topologies=(),
        wall_parameters=wall, structural_parameters=p,
        capture_mask=capture, rotating_mask=rotating, family=0,
        compatible=True, dt_s=1e-6, pair_source_rate_m2_s=source,
        source_mode="test")
    assert result.source_ledger is not None
    assert result.source_ledger.line_balance_residual_m2 == pytest.approx(
        0.0, abs=1e-6)
    assert result.mechanical_acceptance_scale > 0.0
    assert result.absorption_decision.accepted
