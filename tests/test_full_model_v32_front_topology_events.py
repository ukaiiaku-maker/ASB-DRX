import json

import numpy as np
import pytest

from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.coupled_front_production import (
    accept_coupled_front_candidate, initialize_coupled_front_runtime)
from full_model.production.front_topology import (
    cut_cell_receiver_fraction, extract_front_components,
    initialize_front_topology, match_front_topology,
    snapshot_from_dict, snapshot_to_dict)
from full_model.production.moving_front import (
    DefectState, initialize_declared_boundary_front)


def _front_with_donor_satellite(radius):
    n = 64
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    main = x-16.0
    if radius is None:
        return np.tanh(main)
    satellite = np.hypot(x-32.0, y-32.0)-float(radius)
    return np.tanh(np.minimum(main, satellite))


def _two_receiver_islands(separation):
    n = 64
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    r0 = np.hypot(x-(32.0-separation/2.0), y-32.0)
    r1 = np.hypot(x-(32.0+separation/2.0), y-32.0)
    return np.tanh(8.0-np.minimum(r0, r1))


def test_subcell_satellite_is_diagnosed_but_does_not_split_material_identity():
    before = _front_with_donor_satellite(None)
    after = _front_with_donor_satellite(.25)
    unfiltered = extract_front_components(
        after, periodic=False, minimum_resolved_loop_area_cells2=0.0,
        minimum_resolved_loop_length_cells=0.0)
    assert len(unfiltered.components) == 2
    loop = [item for item in unfiltered.components if item.closed][0]
    assert loop.enclosed_area_cells2 == pytest.approx(.15088834764821968)
    assert loop.interface_length_cells == pytest.approx(1.4724736459167262)

    previous = initialize_front_topology(before, periodic=False)
    match = match_front_topology(previous, before, after, periodic=False)
    assert match.event is None
    assert len(match.snapshot.components) == 1
    assert match.snapshot.components[0].component_id == previous.components[0].component_id
    assert match.snapshot.filtered_subcell_component_count == 1
    assert match.snapshot.filtered_subcell_area_cells2 == pytest.approx(
        loop.enclosed_area_cells2)
    expected = np.sum(
        cut_cell_receiver_fraction(after, periodic=False)
        -cut_cell_receiver_fraction(before, periodic=False),
        dtype=np.longdouble)
    assert match.signed_receiver_area_cells2 == pytest.approx(float(expected))
    assert match.signed_receiver_area_cells2 < 0.0


def test_resolved_satellite_remains_a_named_unauthorized_island_terminal():
    before = _front_with_donor_satellite(None)
    after = _front_with_donor_satellite(.75)
    match = match_front_topology(
        initialize_front_topology(before, periodic=False), before, after,
        periodic=False)
    assert match.event == "UNAUTHORIZED_PHASE_ISLAND"
    assert match.event_record["old_component_count"] == 1
    assert match.event_record["new_component_count"] == 2
    assert match.snapshot.filtered_subcell_component_count == 0


def test_subcell_filter_is_label_symmetric_and_restart_exact():
    before = _front_with_donor_satellite(None)
    after = _front_with_donor_satellite(.25)
    forward = match_front_topology(
        initialize_front_topology(before, periodic=False), before, after,
        periodic=False)
    reverse = match_front_topology(
        initialize_front_topology(-before, periodic=False), -before, -after,
        periodic=False)
    assert forward.event is reverse.event is None
    assert forward.snapshot.filtered_subcell_component_count == 1
    assert reverse.snapshot.filtered_subcell_component_count == 1
    assert forward.signed_receiver_area_cells2 == pytest.approx(
        -reverse.signed_receiver_area_cells2)
    restored = snapshot_from_dict(json.loads(json.dumps(
        snapshot_to_dict(forward.snapshot))))
    assert restored == forward.snapshot


@pytest.mark.parametrize("reverse,classification", [
    (False, "SUPPORTED_FRONT_COMPONENT_SPLIT"),
    (True, "SUPPORTED_FRONT_COMPONENT_MERGE"),
])
def test_explicit_reconnection_support_preserves_one_identity_and_cut_cell_balance(
        reverse, classification):
    one = _two_receiver_islands(6.0)
    two = _two_receiver_islands(20.0)
    before, after = (two, one) if reverse else (one, two)
    previous = initialize_front_topology(before, periodic=False)
    match = match_front_topology(
        previous, before, after, periodic=False,
        support_component_reconnection=True)
    assert match.event is None
    assert match.event_record["classification"] == classification
    assert ({item.component_id for item in previous.components}
            & {item.component_id for item in match.snapshot.components})
    expected = np.sum(
        cut_cell_receiver_fraction(after, periodic=False)
        -cut_cell_receiver_fraction(before, periodic=False),
        dtype=np.longdouble)
    assert match.signed_receiver_area_cells2 == pytest.approx(float(expected))
    assert sum(item.signed_receiver_swept_area_cells2
               for item in match.snapshot.components) == pytest.approx(
                   float(expected), abs=2e-12)


def _adapter_fixture(phi, periodic=False):
    b = .5*(1.0+phi)
    eta = np.stack((1.0-b, b), axis=2)
    h = eta**2*(3.0-2.0*eta)
    chi = h[:, :, 1]/np.sum(h, axis=2)
    shape = phi.shape+(4,)

    def defect(rho):
        return DefectState(
            np.full(shape, .15*rho), np.full(shape, .10*rho),
            np.full(shape, .075*rho), np.full(phi.shape, .10*rho))

    state = initialize_declared_boundary_front(
        defect(1.2e14), defect(.8e14), chi, 0, 1)
    runtime = initialize_coupled_front_runtime(
        state, phi, normal_axis=0, periodic=periodic)
    return eta, state, runtime


def _large_translation_step(state, runtime, before, trial, backtracking):
    return accept_coupled_front_candidate(
        state, runtime, before, trial, spacing_m=2e-9,
        represented_thickness_m=5e-10, dt_s=1e-8,
        temperature_K=1100.0, line_energy_J_m=1e-9,
        process=ActivatedProcess("front", 1e9, 0.0, 1e9),
        h0_J=.35*EV_J, critical_pressure_Pa=1e9, exp_a=2.0,
        exp_n=1.5, exp_floor=.1,
        driving_pressure_a_to_b_Pa=-2e7,
        applied_pressure_a_to_b_Pa=-2e7, periodic=False,
        transmission_fraction=.5, boundary_storage_fraction=.1,
        neutral_sink_fraction=.05,
        support_component_reconnection=True,
        topology_backtracking_enabled=backtracking)


def test_topology_limited_backtracking_has_exact_off_control_and_closes_transaction():
    n = 64
    x = np.broadcast_to(np.arange(n, dtype=float)[:, None], (n, n))
    phi0 = np.tanh(x-16.0)
    phi1 = np.tanh(x-26.0)
    before, state, runtime = _adapter_fixture(phi0)
    trial = _adapter_fixture(phi1)[0]
    off_state, off_runtime, off_eta, off = _large_translation_step(
        state, runtime, before, trial, False)
    assert off.classification == "PAIR_IDENTITY_LOST"
    assert off.topology_backtrack_fraction == 1.0
    assert off_state.ledger == state.ledger
    assert np.array_equal(off_eta, before)

    new_state, new_runtime, accepted_eta, decision = _large_translation_step(
        state, runtime, before, trial, True)
    assert decision.accepted
    assert 0.0 < decision.topology_backtrack_fraction < 1.0
    assert decision.classification == "ACCEPTED_ATOMIC_COUPLED_FRONT"
    assert new_runtime.ledger.accepted == 1
    assert new_runtime.ledger.maximum_abs_line_closure_m < 1e-15
    assert new_state.ledger.signed_burgers_change_m2 < 1e-20
    assert not np.array_equal(accepted_eta, before)
