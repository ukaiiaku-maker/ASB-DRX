import numpy as np
import pytest

from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.production.mura_kinematics import (
    accept_family_mura_step, family_plastic_flow_from_signed_alignment,
)
from full_model.production.tensorial_nye import (
    divergence_of_nye, nye_from_plastic_distortion, rotated_system_fields,
)
from full_model.production.v24_mechanical_wall import (
    accepted_v24_mechanical_step, mechanical_checkpoint_arrays,
    mechanical_from_checkpoint_arrays,
)
from full_model.production.wall_topology_supply import (
    apply_signed_reservoir_exchange, reservoir_nye_m1,
)


def _manufactured_family_event(n, rotating=False, diffusion=False):
    state, _, _, _, systems, _, common, _, _, _ = build_case(
        n, periodic_nye_consistent=True)
    x = np.arange(n)[:, None]*2*np.pi/n
    y = np.arange(n)[None, :]*2*np.pi/n
    angle = (0.08*np.sin(x)*np.cos(2*y) if rotating
             else np.zeros((n, n)))
    _, slip, normal = rotated_system_fields(systems, angle)
    line = np.cross(normal, slip)
    profile = 1.0+0.15*np.cos(2*x-y)
    plus = 6e13*profile[..., None, None]*line
    minus = 4e13*(2.0-profile[..., None, None])*line
    if diffusion:
        # A declared physical diffusive velocity.  It enters the same Mura
        # event rather than smoothing the accepted authoritative state.
        gx = -0.3*np.sin(2*x-y)
        gy = 0.15*np.sin(2*x-y)
        velocity = np.zeros_like(plus)
        velocity[..., 0] = 2e-4*gx[..., None]
        velocity[..., 1] = 2e-4*gy[..., None]
    else:
        velocity = (1e-4*(1.0+0.1*np.sin(x+y))[..., None, None]
                    *slip)
    family_flow = family_plastic_flow_from_signed_alignment(
        plus, minus, velocity, -velocity, systems, angle)
    beta0 = np.zeros((n, n, 3, 3))
    alpha0 = np.zeros((n, n, len(systems), 3, 3))
    return accept_family_mura_step(
        beta0, alpha0, family_flow, 2e-8, common.spacing_m)


@pytest.mark.parametrize("n", (16, 32, 64, 128))
@pytest.mark.parametrize("rotating,diffusion", ((False, False),
                                                   (True, False),
                                                   (False, True)))
def test_manufactured_one_flux_complex_closes_at_all_required_grids(
        n, rotating, diffusion):
    beta, family_alpha, audit = _manufactured_family_event(
        n, rotating=rotating, diffusion=diffusion)
    alpha = np.sum(family_alpha, axis=2)
    np.testing.assert_allclose(
        alpha, nye_from_plastic_distortion(beta, 3.2e-6/n),
        rtol=2e-11, atol=2e-7)
    scale = max(float(np.sqrt(np.mean(alpha**2)))/(3.2e-6/n), 1.0)
    assert np.sqrt(np.mean(divergence_of_nye(alpha, 3.2e-6/n)**2))/scale < 2e-13
    assert audit["accepted_step_hard_invariant_passed"]
    assert not audit["post_step_projection_used"]


def test_actual_transition_band_has_no_first_step_jump_and_stays_below_gate():
    (state, driving, _, support, systems, topologies, common, extensive,
     kinetics, spacing) = build_case(32, periodic_nye_consistent=True)
    maximum_dual = 0.0
    for _ in range(6):
        state, ledger = accepted_v24_mechanical_step(
            state, driving, support, systems, topologies, common, extensive,
            kinetics, 2e-9, topology_route_enabled=False)
        reservoir = reservoir_nye_m1(
            state.reservoir_alignment, systems,
            state.common.orientation_rad, topologies)["total"]
        curl_beta = nye_from_plastic_distortion(state.common.beta_p, spacing)
        scale = max(float(np.sqrt(np.mean(curl_beta**2))), 1.0)
        maximum_dual = max(maximum_dual, float(
            np.sqrt(np.mean((reservoir-curl_beta)**2))/scale))
        assert ledger["nye_suboperator_audit"][
            "accepted_step_hard_invariant_passed"]
        assert ledger["nye_suboperator_audit"][
            "first_violating_suboperator"] is None
        assert not ledger["legacy_independent_beta_nye_rates_accepted"]
        assert not ledger["nye_suboperator_audit"]["post_step_projection_used"]
    assert maximum_dual < 0.05


def test_scalar_slip_is_owned_by_the_accepted_family_mura_event():
    args = build_case(16, periodic_nye_consistent=True)
    initial = args[0]
    updated, ledger = accepted_v24_mechanical_step(
        initial, args[1], args[3], *args[4:9], 2e-9,
        topology_route_enabled=False)
    np.testing.assert_allclose(
        updated.common.slip-initial.common.slip,
        ledger["accepted_dt_s"]*ledger["mura_slip_rate_s"],
        rtol=0.0, atol=1e-18)
    assert not ledger["legacy_independent_slip_rate_accepted"]


def test_mura_production_restart_is_bitwise_exact():
    args = build_case(16, periodic_nye_consistent=True)
    state, driving, _, support, systems, topologies, common, extensive, kinetics, _ = args
    first, _ = accepted_v24_mechanical_step(
        state, driving, support, systems, topologies, common, extensive,
        kinetics, 2e-9, topology_route_enabled=False)
    restored = mechanical_from_checkpoint_arrays(
        mechanical_checkpoint_arrays(first), systems, topologies)
    continuous, continuous_ledger = accepted_v24_mechanical_step(
        first, driving, support, systems, topologies, common, extensive,
        kinetics, 2e-9, topology_route_enabled=False)
    restarted, restarted_ledger = accepted_v24_mechanical_step(
        restored, driving, support, systems, topologies, common, extensive,
        kinetics, 2e-9, topology_route_enabled=False)
    for group in ("common", "density", "reservoir_alignment"):
        left = getattr(continuous, group)
        right = getattr(restarted, group)
        for name in left.__dict__:
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    assert continuous_ledger["accepted_dt_s"] == restarted_ledger["accepted_dt_s"]


def test_active_set_records_geometric_line_stretching_without_moment_clipping():
    args = build_case(16, periodic_nye_consistent=True)
    _, ledger = accepted_v24_mechanical_step(
        args[0], args[1], args[3], args[4], args[5], args[6], args[7],
        args[8], 2e-9, topology_route_enabled=False)
    transport = ledger["transport_capture"]
    for sign in ("plus", "minus"):
        event = transport["sign"][sign]
        assert np.all(event["mura_line_stretching_m2"] >= 0.0)
        assert abs(event["global_scalar_residual_line_per_thickness"]) < 1e-14
    assert not transport["post_step_projection_used"]


@pytest.mark.parametrize("n", (16, 32, 64, 128))
@pytest.mark.parametrize("extent_sign", (-1.0, 1.0))
def test_locking_unlocking_exchange_conserves_tensorial_nye_at_required_grids(
        n, extent_sign):
    state, _, _, _, systems, topologies, _, _, _, _ = build_case(
        n, periodic_nye_consistent=True)
    before = reservoir_nye_m1(
        state.reservoir_alignment, systems,
        state.common.orientation_rad, topologies)["total"]
    extent = np.full(state.density.mobile_plus_m2.shape,
                     extent_sign*1e10)
    _, alignment, ledger = apply_signed_reservoir_exchange(
        state.density, state.reservoir_alignment, "mobile", "forest",
        extent, -extent, systems, state.common.orientation_rad, topologies)
    after = reservoir_nye_m1(
        alignment, systems, state.common.orientation_rad, topologies)["total"]
    np.testing.assert_allclose(after, before, rtol=0.0, atol=5e-13)
    assert np.max(np.abs(ledger["total_nye_residual_m1"])) < 5e-13
    assert not ledger["post_step_projection_used"]


def test_junction_reorientation_is_an_explicit_persistent_source_not_projection():
    # The topology route can change total Nye only through its declared source.
    from tests.test_full_model_v24_mechanical_wall import mechanical_fixture
    args = mechanical_fixture()
    first, ledger1 = accepted_v24_mechanical_step(
        *args, dt_s=1e-9, topology_route_enabled=True)
    second, ledger2 = accepted_v24_mechanical_step(
        first, *args[1:], dt_s=1e-9, topology_route_enabled=True)
    source1 = ledger1["mura_balance_ledger"]["reaction_source_tensor_m1"]
    assert np.sqrt(np.mean(source1**2)) > 0.0
    assert ledger1["nye_suboperator_audit"]["accepted_step_hard_invariant_passed"]
    assert ledger2["nye_suboperator_audit"]["accepted_step_hard_invariant_passed"]
    assert ledger2["mura_face_event"]["cumulative_declared_source_rms_m1"] > 0.0
    second.validate(args[3], args[4])
