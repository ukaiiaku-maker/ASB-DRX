import json

import numpy as np
import pytest

from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.coupled_front_production import (
    accept_coupled_front_candidate, initialize_coupled_front_runtime,
    existing_pair_geometric_envelope, runtime_arrays,
    runtime_from_checkpoint, runtime_metadata_json)
from full_model.production.moving_front import (
    DefectState, initialize_declared_boundary_front, reconstruct_mixture,
    state_arrays, state_from_checkpoint, state_metadata_json,
    total_line_density)


def _phase(n, location=None, width=1.7):
    location = 0.45*n+0.25 if location is None else location
    x = np.arange(n, dtype=float)[:, None]
    phi = np.broadcast_to(np.tanh((x-location)/width), (n, n)).copy()
    b = 0.5*(1.0+phi)
    return np.stack((1.0-b, b), axis=2)


def _defect(n, rho):
    shape = (n, n, 4)
    return DefectState(
        np.full(shape, .15*rho), np.full(shape, .10*rho),
        np.full(shape, .075*rho), np.full((n, n), .10*rho))


def _fixture(n=32, rho_a=1e14, rho_b=1e14):
    eta = _phase(n)
    h = eta**2*(3.0-2.0*eta)
    chi = h[:, :, 1]/np.sum(h, axis=2)
    state = initialize_declared_boundary_front(
        _defect(n, rho_a), _defect(n, rho_b), chi, 0, 1)
    runtime = initialize_coupled_front_runtime(
        state, eta[:, :, 1]-eta[:, :, 0], normal_axis=0)
    return eta, state, runtime


def test_existing_pair_envelope_is_conservative_and_directional():
    eta = _phase(32)
    expanded = existing_pair_geometric_envelope(
        eta, parent_label=0, child_label=1, normal_axis=0,
        fraction=.125, direction=1)
    contracted = existing_pair_geometric_envelope(
        eta, parent_label=0, child_label=1, normal_axis=0,
        fraction=.125, direction=-1)
    np.testing.assert_allclose(np.sum(expanded, axis=2), 1.0)
    np.testing.assert_allclose(np.sum(contracted, axis=2), 1.0)
    assert np.sum(expanded[:, :, 1]) > np.sum(eta[:, :, 1])
    assert np.sum(contracted[:, :, 1]) < np.sum(eta[:, :, 1])
    assert np.all(expanded >= 0.0) and np.all(expanded <= 1.0)
    assert np.all(contracted >= 0.0) and np.all(contracted <= 1.0)


def test_existing_pair_envelope_does_not_consume_a_third_grain():
    eta = np.zeros((16, 16, 3))
    eta[:6, :, 0] = 1.0
    eta[6:11, :, 1] = 1.0
    eta[11:, :, 2] = 1.0
    proposed = existing_pair_geometric_envelope(
        eta, parent_label=0, child_label=1, normal_axis=0,
        fraction=.5, direction=1)
    np.testing.assert_array_equal(proposed[:, :, 2], eta[:, :, 2])
    np.testing.assert_allclose(np.sum(proposed, axis=2), 1.0)
    assert np.sum(proposed[:, :, 1]) > np.sum(eta[:, :, 1])


def _step(state, runtime, before, trial, pressure, mobility=True, capacity=None):
    return accept_coupled_front_candidate(
        state, runtime, before, trial, spacing_m=2e-9,
        represented_thickness_m=5e-10, dt_s=1e-8, temperature_K=1100.0,
        line_energy_J_m=1e-9,
        process=ActivatedProcess("front", 1e9, 0.0, 1e9),
        h0_J=.35*EV_J, critical_pressure_Pa=1e9, exp_a=2.0,
        exp_n=1.5, exp_floor=.1, driving_pressure_a_to_b_Pa=pressure,
        applied_pressure_a_to_b_Pa=pressure,
        mobility_enabled=mobility, periodic=False,
        transmission_fraction=.5, boundary_storage_fraction=.1,
        neutral_sink_fraction=.05,
        boundary_capacity_density_m2=capacity)


@pytest.mark.parametrize("n", [32, 64])
def test_production_equal_state_and_mobility_off_are_exactly_stationary(n):
    eta, state, runtime = _fixture(n)
    proposal = _phase(n, location=.45*n+.15)
    s1, r1, accepted, decision = _step(
        state, runtime, eta, proposal, pressure=0.0)
    assert not decision.accepted
    assert decision.net_velocity_a_to_b_m_s == 0.0
    assert np.array_equal(accepted, eta)
    assert s1.ledger == state.ledger
    s2, r2, accepted2, decision2 = _step(
        state, runtime, eta, proposal, pressure=1e7, mobility=False)
    assert not decision2.accepted
    assert np.array_equal(accepted2, eta)
    assert r1.ledger.rejected_direction == r2.ledger.rejected_direction == 1


@pytest.mark.parametrize("n", [32, 64])
@pytest.mark.parametrize("width", [.7, 1.2, 2.8, 5.0])
def test_width_only_relaxation_has_no_sweep_or_irreversible_ledger(n, width):
    eta, state, runtime = _fixture(n)
    trial = _phase(n, width=width)
    new, run, accepted, decision = _step(
        state, runtime, eta, trial, pressure=1e7)
    assert not decision.accepted
    assert decision.classification == "STATIONARY_GEOMETRY"
    assert decision.accepted_signed_volume_m3 == 0.0
    assert np.array_equal(new.chi, state.chi)
    assert run.ledger.a_to_b_swept_volume_m3 == 0.0
    assert np.array_equal(accepted, trial)


@pytest.mark.parametrize("n", [32, 64])
@pytest.mark.parametrize("shift", [-1.0, -.5, -.1, -.01, .01, .1, .5, 1.0])
def test_both_geometric_directions_require_matching_thermodynamic_sign(n, shift):
    eta, state, runtime = _fixture(n, 1.2e14, .8e14)
    trial = _phase(n, location=.45*n+.25+shift)
    # A shift to negative x grows B, hence positive A->B pressure.
    pressure = -np.sign(shift)*2e7
    new, run, accepted, decision = _step(
        state, runtime, eta, trial, pressure=pressure)
    assert decision.accepted
    assert np.sign(decision.accepted_signed_volume_m3) == -np.sign(shift)
    assert decision.maximum_abs_line_closure_m < 1e-20
    assert decision.maximum_abs_signed_closure_m2 < 1.0
    assert decision.absolute_swept_volume_m3 == (
        decision.positive_swept_volume_m3+decision.negative_swept_volume_m3)
    assert decision.maximum_abs_phase_change > 0.0
    assert decision.rms_phase_change > 0.0
    assert decision.interface_area_m2 > 0.0
    assert decision.cell_volume_m3 == 2e-9*2e-9*5e-10
    assert decision.kinetic_event_volume_m3 == decision.cell_volume_m3
    assert decision.kinetic_event_length_m == 2e-9
    np.testing.assert_allclose(
        decision.expected_normal_velocity_m_s,
        decision.net_velocity_a_to_b_m_s, rtol=2e-15)
    np.testing.assert_allclose(
        decision.expected_signed_swept_volume_m3,
        decision.expected_signed_event_count
        *decision.kinetic_event_volume_m3, rtol=2e-15)
    assert decision.component_motion
    assert all(np.isfinite(item["normal_displacement_m"])
               for item in decision.component_motion)
    assert run.ledger.accepted == 1
    assert run.ledger.heat_J >= 0.0
    assert np.all(total_line_density(reconstruct_mixture(new)) >= 0.0)


def test_advance_retreat_readvance_is_bidirectional_and_closed():
    eta0, state, runtime = _fixture(32, 1.3e14, .7e14)
    eta1 = _phase(32, location=.45*32+.25-.4)
    state, runtime, eta1a, d1 = _step(state, runtime, eta0, eta1, 2e7)
    eta2 = _phase(32, location=.45*32+.25+.2)
    state, runtime, eta2a, d2 = _step(state, runtime, eta1a, eta2, -2e7)
    eta3 = _phase(32, location=.45*32+.25-.3)
    state, runtime, _, d3 = _step(state, runtime, eta2a, eta3, 2e7)
    assert all(d.accepted for d in (d1, d2, d3))
    assert runtime.ledger.a_to_b_swept_volume_m3 > 0.0
    assert runtime.ledger.b_to_a_swept_volume_m3 > 0.0
    assert runtime.ledger.revisit_volume_m3 > 0.0
    assert runtime.ledger.maximum_abs_line_closure_m < 1e-20


def test_finite_boundary_capacity_limits_phase_and_transaction_atomically():
    eta, state, runtime = _fixture(32, 1.3e14, .7e14)
    trial = _phase(32, location=.45*32+.25-.5)
    unlimited = _step(state, runtime, eta, trial, 2e7)[3]
    new, run, accepted, limited = _step(
        state, runtime, eta, trial, 2e7, capacity=1e8)
    assert limited.accepted
    assert 0.0 <= limited.accepted_signed_volume_m3 < unlimited.accepted_signed_volume_m3
    assert np.max(np.abs(accepted-eta)) < np.max(np.abs(trial-eta))
    assert np.max(new.boundary_line_density_m2) <= 1e8*(1+1e-12)
    assert run.ledger.maximum_abs_line_closure_m < 1e-20


def test_geometry_probe_applies_no_work_rate_or_direction_selection():
    eta, state, runtime = _fixture(32, 1.3e14, .7e14)
    trial = _phase(32, location=.45*32+.25-.5)

    def probe(pressure):
        return accept_coupled_front_candidate(
            state, runtime, eta, trial, spacing_m=2e-9,
            represented_thickness_m=5e-10, dt_s=1e-8,
            temperature_K=1100.0, line_energy_J_m=1e-9,
            process=ActivatedProcess("probe", 1e9, .3, 1e9),
            h0_J=.35*EV_J, critical_pressure_Pa=1e9, exp_a=2.0,
            exp_n=1.5, exp_floor=.1,
            driving_pressure_a_to_b_Pa=pressure,
            applied_pressure_a_to_b_Pa=pressure,
            mobility_enabled=True, periodic=False,
            transmission_fraction=.5, boundary_storage_fraction=.1,
            neutral_sink_fraction=.05, proposal_probe_only=True)

    positive = probe(1e9)
    negative = probe(-3e9)
    for left, right in zip(state_arrays(positive[0]).values(),
                           state_arrays(negative[0]).values()):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(positive[2], negative[2])
    for result in (positive, negative):
        decision = result[3]
        assert decision.accepted
        assert decision.classification == (
            "GEOMETRY_PROBE_ONLY_NOT_PHYSICAL_ACCEPTANCE")
        assert decision.proposal_probe_only
        assert decision.proposal_probe_external_work_J == 0.0
        assert not decision.proposal_probe_selected_direction
        assert decision.rate_a_to_b_s == 0.0
        assert decision.rate_b_to_a_s == 0.0
        assert decision.net_velocity_a_to_b_m_s == 0.0
        assert decision.gross_channel_activity_s == 0.0
        assert decision.channel_a_to_b["availability_factor"] == 0.0
        assert decision.channel_b_to_a["availability_factor"] == 0.0


def test_runtime_restart_roundtrip_is_bitwise_and_schema_rejects_partial_state():
    eta, state, runtime = _fixture(32, 1.2e14, .8e14)
    state, runtime, accepted, _ = _step(
        state, runtime, eta, _phase(32, location=.45*32), 2e7)
    restored = runtime_from_checkpoint(
        runtime_metadata_json(runtime), runtime_arrays(runtime))
    assert runtime_metadata_json(restored) == runtime_metadata_json(runtime)
    for key, value in runtime_arrays(runtime).items():
        assert np.array_equal(runtime_arrays(restored)[key], value)
    with pytest.raises(ValueError, match="partial"):
        runtime_from_checkpoint(runtime_metadata_json(runtime), {})
    metadata = json.loads(runtime_metadata_json(runtime))
    assert metadata["schema"].endswith("/v1")


def test_legacy_checkpoint_migration_creates_zero_transaction_history():
    eta, state, _ = _fixture(32)
    migrated = runtime_from_checkpoint(
        None, {}, fallback_state=state,
        fallback_phi=eta[:, :, 1]-eta[:, :, 0], normal_axis=0)
    assert migrated.ledger.attempts == 0
    assert np.array_equal(migrated.maximum_b_fraction, state.chi)
    assert np.array_equal(migrated.minimum_b_fraction, state.chi)


def test_continuous_and_segmented_atomic_history_are_bitwise_identical():
    eta0, state0, runtime0 = _fixture(32, 1.3e14, .7e14)
    proposals = [
        (_phase(32, location=.45*32+.25-.3), 2e7),
        (_phase(32, location=.45*32+.25+.1), -2e7),
        (_phase(32, location=.45*32+.25-.2), 2e7),
    ]

    def advance(state, runtime, eta, sequence):
        for proposal, pressure in sequence:
            state, runtime, eta, _ = _step(
                state, runtime, eta, proposal, pressure)
        return state, runtime, eta

    continuous = advance(state0, runtime0, eta0, proposals)
    state1, runtime1, eta1 = advance(state0, runtime0, eta0, proposals[:1])
    state1 = state_from_checkpoint(
        state_metadata_json(state1), state_arrays(state1))
    runtime1 = runtime_from_checkpoint(
        runtime_metadata_json(runtime1), runtime_arrays(runtime1))
    segmented = advance(state1, runtime1, eta1.copy(), proposals[1:])
    for a, b in zip(state_arrays(continuous[0]).values(),
                    state_arrays(segmented[0]).values()):
        assert np.array_equal(a, b)
    assert runtime_metadata_json(continuous[1]) == runtime_metadata_json(segmented[1])
    for key, value in runtime_arrays(continuous[1]).items():
        assert np.array_equal(value, runtime_arrays(segmented[1])[key])
    assert np.array_equal(continuous[2], segmented[2])
