import json

import numpy as np
import pytest

from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.coupled_front_production import (
    accept_coupled_front_candidate, initialize_coupled_front_runtime,
    runtime_arrays, runtime_from_checkpoint, runtime_metadata_json)
from full_model.production.front_topology import (
    FrontComponent, cut_cell_receiver_fraction,
    diagnostic_ray_crossing_count, initialize_front_topology,
    match_front_topology, snapshot_from_dict, snapshot_to_dict)
from full_model.production.moving_front import (
    DefectState, initialize_declared_boundary_front)


def _circle(n, centre, radius, width=1.0):
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    dx = (x-centre[0]+n/2) % n-n/2
    dy = (y-centre[1]+n/2) % n-n/2
    return np.tanh((radius-np.hypot(dx, dy))/width)


def _two_disks(n, separation, radius=8.0):
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    left = radius-np.hypot(x-(n/2-separation/2), y-n/2)
    right = radius-np.hypot(x-(n/2+separation/2), y-n/2)
    return np.tanh(np.maximum(left, right))


@pytest.mark.parametrize("n", [32, 64])
def test_cut_cell_area_is_exactly_profile_width_invariant(n):
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    level = x-(0.43*n+0.17+0.8*np.sin(2*np.pi*y/n))
    areas = [float(np.sum(cut_cell_receiver_fraction(
        np.tanh(level/width), periodic=False)))
             for width in (0.65, 1.3, 3.7)]
    assert np.ptp(areas) < 2e-12*n*n


def test_ray_tangency_count_change_does_not_change_component_identity():
    before = _circle(64, (31.5, 31.5), 7.5)
    after = _circle(64, (31.5, 31.5), 7.9)
    old_count = diagnostic_ray_crossing_count(before, normal_axis=0)
    new_count = diagnostic_ray_crossing_count(after, normal_axis=0)
    assert old_count != new_count
    previous = initialize_front_topology(before, ray_crossing_count=old_count)
    match = match_front_topology(
        previous, before, after, ray_crossing_count=new_count)
    assert match.event is None
    assert len(match.snapshot.components) == 1
    assert (match.snapshot.components[0].component_id
            == previous.components[0].component_id)
    assert match.signed_receiver_area_cells2 > 0.0
    assert sum(component.signed_receiver_swept_area_cells2
               for component in match.snapshot.components) == pytest.approx(
                   match.signed_receiver_area_cells2, abs=2e-12)


def test_periodic_seam_crossing_loop_and_translation_keep_identity():
    before = _circle(64, (1.2, 30.0), 8.0)
    after = _circle(64, (63.7, 30.0), 8.0)
    previous = initialize_front_topology(before)
    assert len(previous.components) == 1
    assert previous.components[0].winding == (0, 0)
    match = match_front_topology(previous, before, after)
    assert match.event is None
    assert match.maximum_component_distance_cells < 2.0


def test_periodic_winding_pair_has_declared_winding_and_persistent_ids():
    n = 64
    x = np.arange(n, dtype=float)[:, None]
    before = np.broadcast_to(np.tanh(np.sin(2*np.pi*(x-.2)/n)*4), (n, n))
    after = np.broadcast_to(np.tanh(np.sin(2*np.pi*(x-.7)/n)*4), (n, n))
    previous = initialize_front_topology(before)
    assert len(previous.components) == 2
    assert all(component.winding == (0, 1) for component in previous.components)
    match = match_front_topology(previous, before, after)
    assert match.event is None
    assert ({item.component_id for item in match.snapshot.components}
            == {item.component_id for item in previous.components})


@pytest.mark.parametrize("reverse,classification", [
    (False, "FRONT_COMPONENT_SPLIT"),
    (True, "FRONT_COMPONENT_MERGE"),
])
def test_split_and_merge_are_named_nonbijective_events(reverse, classification):
    one = _two_disks(64, 6.0)
    two = _two_disks(64, 20.0)
    before, after = (two, one) if reverse else (one, two)
    match = match_front_topology(
        initialize_front_topology(before, periodic=False), before, after,
        periodic=False)
    assert match.event == classification
    assert match.event_record["old_component_count"] != match.event_record[
        "new_component_count"]


def test_disappearance_distinguishes_consumption_and_winding_pair_annihilation():
    loop = _circle(64, (32, 32), 8)
    uniform = -np.ones_like(loop)
    consumed = match_front_topology(
        initialize_front_topology(loop), loop, uniform)
    assert consumed.event == "GRAIN_CONSUMED"

    x = np.arange(64, dtype=float)[:, None]
    stripe = np.broadcast_to(np.tanh(np.sin(2*np.pi*x/64)*4), (64, 64))
    annihilated = match_front_topology(
        initialize_front_topology(stripe), stripe, uniform)
    assert annihilated.event == "INTERFACE_PAIR_ANNIHILATED"


def test_active_window_exit_is_a_named_terminal():
    n = 64
    x = np.arange(n, dtype=float)[:, None]
    before = np.broadcast_to(np.tanh((x-31.5)/1.2), (n, n))
    after = np.broadcast_to(np.tanh((x-55.0)/1.2), (n, n))
    active = np.zeros((n, n), dtype=bool)
    active[12:52, 12:52] = True
    previous = initialize_front_topology(
        before, active_mask=active, periodic=False)
    assert previous.components
    assert all(component.endpoint_count for component in previous.components)
    match = match_front_topology(
        previous, before, after, active_mask=active, periodic=False)
    assert match.event == "PAIR_LEFT_ACTIVE_WINDOW"


def test_label_exchange_reverses_swept_area_and_preserves_geometry():
    before = _circle(64, (32, 32), 8.0)
    after = _circle(64, (32, 32), 8.4)
    forward = match_front_topology(
        initialize_front_topology(before), before, after)
    reverse = match_front_topology(
        initialize_front_topology(-before), -before, -after)
    assert forward.event is reverse.event is None
    assert forward.signed_receiver_area_cells2 == pytest.approx(
        -reverse.signed_receiver_area_cells2, abs=2e-12)
    assert forward.snapshot.components[0].interface_length_cells == pytest.approx(
        reverse.snapshot.components[0].interface_length_cells)


def test_snapshot_restart_and_contour_point_shuffle_are_invariant():
    phi = _circle(48, (2.0, 24.0), 7.0)
    snapshot = initialize_front_topology(phi)
    restored = snapshot_from_dict(json.loads(json.dumps(snapshot_to_dict(snapshot))))
    assert restored == snapshot
    component = restored.components[0]
    shuffled = FrontComponent(
        component.component_id, component.winding, component.centroid_grid,
        component.interface_length_cells, component.enclosed_area_cells2,
        component.endpoint_count, component.closed,
        component.active_window_relationship,
        component.representative_tangent, component.receiver_normal,
        component.signed_receiver_swept_area_cells2,
        tuple(reversed(component.points_grid)))
    shuffled_snapshot = type(restored)(
        restored.shape, (shuffled,), restored.next_component_id,
        restored.ray_crossing_count)
    matched = match_front_topology(shuffled_snapshot, phi, phi)
    assert matched.event is None
    assert matched.snapshot.components[0].component_id == component.component_id


def test_finite_geometry_frame_is_oriented_toward_receiver():
    phi = _circle(48, (24, 24), 8)
    component = initialize_front_topology(phi).components[0]
    assert np.linalg.norm(component.representative_tangent) == pytest.approx(1.0)
    assert np.linalg.norm(component.receiver_normal) == pytest.approx(1.0)
    assert abs(np.dot(component.representative_tangent,
                      component.receiver_normal)) < 1e-12


def _adapter_fixture(phi, rho_a=1.2e14, rho_b=.8e14):
    b = .5*(1.0+phi)
    eta = np.stack((1.0-b, b), axis=2)
    h = eta**2*(3.0-2.0*eta)
    chi = h[:, :, 1]/np.sum(h, axis=2)
    n = phi.shape[0]

    def defect(rho):
        shape = (n, n, 4)
        return DefectState(
            np.full(shape, .15*rho), np.full(shape, .10*rho),
            np.full(shape, .075*rho), np.full((n, n), .10*rho))

    state = initialize_declared_boundary_front(
        defect(rho_a), defect(rho_b), chi, 0, 1)
    runtime = initialize_coupled_front_runtime(
        state, phi, normal_axis=0, periodic=True)
    return eta, state, runtime


def _adapter_step(state, runtime, before, trial, pressure=2e7):
    return accept_coupled_front_candidate(
        state, runtime, before, trial, spacing_m=2e-9,
        represented_thickness_m=5e-10, dt_s=1e-8,
        temperature_K=1100.0, line_energy_J_m=1e-9,
        process=ActivatedProcess("front", 1e9, 0.0, 1e9),
        h0_J=.35*EV_J, critical_pressure_Pa=1e9, exp_a=2.0,
        exp_n=1.5, exp_floor=.1,
        driving_pressure_a_to_b_Pa=pressure,
        applied_pressure_a_to_b_Pa=pressure, periodic=True,
        transmission_fraction=.5, boundary_storage_fraction=.1,
        neutral_sink_fraction=.05)


def test_production_accepts_tangency_ray_count_change_without_identity_loss():
    before_phi = _circle(64, (31.5, 31.5), 7.5)
    after_phi = _circle(64, (31.5, 31.5), 7.9)
    before, state, runtime = _adapter_fixture(before_phi)
    after = _adapter_fixture(after_phi)[0]
    new_state, new_runtime, accepted, decision = _adapter_step(
        state, runtime, before, after)
    assert decision.ray_crossing_count_before != decision.ray_crossing_count_after
    assert decision.classification != "PAIR_IDENTITY_LOST"
    assert decision.accepted
    assert new_runtime.ledger.accepted == 1
    assert new_state.ledger.swept_volume_m3 > 0.0
    assert not np.array_equal(accepted, before)


def test_production_split_fails_closed_before_any_physical_transaction():
    before_phi = _two_disks(64, 6.0)
    after_phi = _two_disks(64, 20.0)
    before, state, runtime = _adapter_fixture(before_phi)
    after = _adapter_fixture(after_phi)[0]
    new_state, new_runtime, accepted, decision = _adapter_step(
        state, runtime, before, after)
    assert decision.classification == "FRONT_COMPONENT_SPLIT"
    assert decision.topology_event["old_component_count"] == 1
    assert decision.topology_event["new_component_count"] == 2
    assert new_state.ledger == state.ledger
    assert new_runtime.ledger.accepted == runtime.ledger.accepted
    assert new_runtime.topology_event_count == 1
    assert np.array_equal(accepted, before)


def test_checkpoint_roundtrip_persists_topology_and_old_v1_migrates():
    phi = _circle(48, (2.0, 24.0), 7.0)
    _, state, runtime = _adapter_fixture(phi)
    restored = runtime_from_checkpoint(
        runtime_metadata_json(runtime), runtime_arrays(runtime), periodic=True)
    assert snapshot_to_dict(restored.topology) == snapshot_to_dict(runtime.topology)
    assert runtime_metadata_json(restored) == runtime_metadata_json(runtime)

    old_metadata = json.loads(runtime_metadata_json(runtime))
    old_metadata.pop("topology")
    old_metadata.pop("periodic")
    old_metadata.pop("topology_event_count")
    migrated = runtime_from_checkpoint(
        json.dumps(old_metadata), runtime_arrays(runtime), periodic=True)
    assert len(migrated.topology.components) == 1
    assert migrated.topology_event_count == 0
