import numpy as np

from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallParameters,
)
from full_model.production.extensive_wall import ExtensiveWallParameters
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology, rotated_system_fields,
)
from full_model.production.v23_coupled_wall import initialize_coupled_state
from full_model.production.v24_mechanical_wall import (
    V24MechanicalWallState, V24TopologyKinetics,
    accepted_v24_mechanical_step, mechanical_checkpoint_arrays,
    mechanical_from_checkpoint_arrays,
)
from full_model.production.wall_topology_supply import (
    aligned_state_from_directions,
)


def mechanical_fixture(n=12):
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=1e-9),)
    old = initialize_coupled_state(
        (n, n), systems, topologies, wall_tangle_plus_m2=0.0,
        wall_tangle_minus_m2=0.0, junction_m2=0.0)
    _, slip, normal = rotated_system_fields(
        systems, np.zeros((n, n)))
    line = np.cross(normal, slip)
    alignment = aligned_state_from_directions(old.density, line)
    state = V24MechanicalWallState(old.common, old.density, alignment)
    dx = 2e-7
    common = CommonWallParameters(
        spacing_m=dx, wall_order_enabled=False, transport_scheme="upwind")
    extensive = ExtensiveWallParameters(
        spacing_m=dx, nye_match_coefficient_J_m=0.0,
        disordered_excess_J_m=3e-10, ordered_excess_J_m=1e-10,
        ordering_barrier_eV=.2)
    topology = V24TopologyKinetics(
        ActivatedProcess("junction", 1e10), .2*EV_J, 1e9)
    fixed = np.zeros((n, n, 2, 2)); fixed[n//3:2*n//3, n//3:2*n//3, 0, 1] = .02
    fixed[..., 1, 0] = fixed[..., 0, 1]
    mean = np.array([[0.0, .012], [.012, 0.0]])
    driving = CommonWallDriving(mean_strain=mean, fixed_eigenstrain=fixed)
    support = np.zeros((n, n), bool)
    support[n//3:2*n//3, n//3:2*n//3] = True
    return state, driving, support, systems, topologies, common, extensive, topology


def test_full_elastic_step_uses_no_target_and_accepts_only_ledgered_density_routes():
    args = mechanical_fixture()
    state, ledger = accepted_v24_mechanical_step(
        *args, dt_s=1e-9, topology_route_enabled=False)
    state.validate(args[3], args[4])
    assert not ledger["orientation_target_used"]
    assert not ledger["legacy_common_density_rates_accepted"]
    assert ledger["accepted_dt_s"] > 0.0
    assert np.max(np.abs(ledger["raw_stress_Pa"])) > 0.0
    assert np.min(state.common.temperature_K) > 0.0


def test_topology_on_branch_has_explicit_R_topology_or_zero_physical_supply():
    args = mechanical_fixture()
    state, ledger = accepted_v24_mechanical_step(
        *args, dt_s=1e-9, topology_route_enabled=True)
    if ledger["junction_topology"] is not None:
        source = ledger["junction_topology"]["R_topology_m1_s"]
        assert np.all(np.isfinite(source))
    detailed_balance = ledger["line_reorientation_topology"]["thermodynamics"]
    expected = np.broadcast_to(
        detailed_balance["expected_ratio"],
        detailed_balance["detailed_balance_ratio"].shape)
    np.testing.assert_allclose(
        detailed_balance["detailed_balance_ratio"], expected, rtol=2e-14)


def test_complete_checkpoint_restart_matches_continuous_accepted_map_bitwise():
    args = mechanical_fixture()
    first, _ = accepted_v24_mechanical_step(
        *args, dt_s=1e-9, topology_route_enabled=True)
    restored = mechanical_from_checkpoint_arrays(
        mechanical_checkpoint_arrays(first), args[3], args[4])
    continuous, _ = accepted_v24_mechanical_step(
        first, *args[1:], dt_s=1e-9, topology_route_enabled=True)
    restarted, _ = accepted_v24_mechanical_step(
        restored, *args[1:], dt_s=1e-9, topology_route_enabled=True)
    for group in ("common", "density", "reservoir_alignment"):
        left = getattr(continuous, group); right = getattr(restarted, group)
        for name in left.__dict__:
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
