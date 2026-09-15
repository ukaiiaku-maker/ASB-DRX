from dataclasses import replace

import numpy as np

from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallParameters, CommonWallState,
    accepted_euler_step, _periodic_upwind_rate,
)
from full_model.production.density_state_map import DensityInventory
from full_model.production.extensive_wall import ExtensiveWallParameters
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology,
)
from full_model.production.v23_coupled_wall import (
    accepted_coupled_step, make_coupled_state,
    mechanical_organization_diagnostics,
)


def fixture(n=8):
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    shape = (n, n, 4)
    mp = np.full(shape, 8e13); mm = np.full(shape, 7e13)
    fp = np.full(shape, 5e13); fm = np.full(shape, 4e13)
    wp = np.full(shape, 2e13); wm = np.full(shape, 1.5e13)
    junction = np.full((n, n, 1), 1e12)
    common = CommonWallState(
        mp, mm, fp, fm, wp, wm, junction, np.zeros((n, n)),
        np.zeros((n, n)), np.zeros(shape), np.zeros((n, n, 3, 3)),
        np.zeros(shape+(3,)), np.zeros(shape+(3, 3)),
        np.zeros((n, n)), np.full((n, n), 1100.0))
    density = DensityInventory(
        mp, mm, fp, fm, wp, wm, np.zeros(shape), np.zeros(shape), junction)
    state = make_coupled_state(common, density, topologies)
    cp = CommonWallParameters(spacing_m=2e-7, wall_order_enabled=False)
    ep = ExtensiveWallParameters(spacing_m=2e-7)
    return systems, topologies, state, cp, ep


def transport_parameters(cp):
    return replace(cp, wall_order_enabled=False, wall_order_amplitude_J_m3=0.0,
                   wall_absent_penalty_J_m3=1e-300,
                   wall_partition_J_m=1e-300, transport_scheme="upwind",
                   mobile_correlation_diffusivity_m2_s=0.0,
                   extensive_wall_partition_enabled=True)


def test_homogeneous_zero_target_reduces_exactly_to_transport_state():
    systems, topologies, state, cp, ep = fixture()
    shape = state.common.mobile_plus_m2.shape
    driving = CommonWallDriving(
        glide_speed_m_s=np.full(shape, 2e-4),
        resolved_stress_Pa=np.full(shape, 5e8))
    dt = 1e-10
    reference, _, scale = accepted_euler_step(
        state.common,
        driving, systems, topologies, transport_parameters(cp), dt)
    result, _, scales = accepted_coupled_step(
        state, driving, systems, topologies, cp, ep, dt)
    assert scales["transport_scale"] == scale
    for name in ("mobile_plus_m2", "mobile_minus_m2", "forest_plus_m2",
                 "forest_minus_m2", "wall_plus_m2", "wall_minus_m2",
                 "junction_m2", "slip", "beta_p", "alignment_m2",
                 "family_nye_m1", "orientation_rad", "temperature_K"):
        np.testing.assert_array_equal(getattr(result.common, name),
                                      getattr(reference, name))
    assert np.count_nonzero(result.density.wall_ordered_plus_m2) == 0
    assert np.count_nonzero(result.density.wall_ordered_minus_m2) == 0


def test_coupled_map_closes_split_and_exposes_mechanical_route_fields():
    systems, topologies, state, cp, ep = fixture(n=12)
    shape = state.common.mobile_plus_m2.shape
    x = np.arange(12)[:, None, None]
    stress = np.broadcast_to(4e8 + 2e8*np.cos(2*np.pi*x/12), shape).copy()
    driving = CommonWallDriving(resolved_stress_Pa=stress)
    result, ledger, _ = accepted_coupled_step(
        state, driving, systems, topologies, cp, ep, 2e-10)
    result.validate(systems, topologies)
    diagnostics = mechanical_organization_diagnostics(
        state, result, ledger, cp.spacing_m)
    assert np.ptp(diagnostics["rss_Pa"]) > 0
    assert np.max(diagnostics["slip_gradient_m1"]) > 0
    assert np.max(diagnostics["nye_norm_m1"]) > 0
    assert np.max(np.abs(diagnostics["orientation_increment_rad"])) > 0
    # The module owns no phase or label state and therefore cannot allocate a grain.
    assert not hasattr(result, "eta") and not hasattr(result, "grain_labels")


def test_v23_upwind_transport_is_periodic_conservative_and_nonnegative():
    _, _, state, cp, _ = fixture(n=16)
    shape = state.common.mobile_plus_m2.shape
    packet = np.zeros(shape)
    packet[3:6, 5:9, :] = 8e13
    velocity = np.zeros(shape+(2,)); velocity[..., 0] = 5e-3
    rate = _periodic_upwind_rate(packet, velocity, cp.spacing_m)
    np.testing.assert_allclose(np.sum(rate), 0.0, atol=1.0)
    dt = .5*cp.spacing_m/5e-3
    updated = packet+dt*rate
    assert np.min(updated) >= 0
