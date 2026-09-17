import numpy as np

from full_model.production.coupled_front_event import (
    front_barrier_activation_volume_m3,
)


def test_front_activation_volume_is_barrier_derivative_not_event_volume():
    h0 = 1.7e-19; pressure = 2.3e8; critical = 1.1e9
    a = 1.4; n = 1.5; floor = 0.08
    analytic = front_barrier_activation_volume_m3(
        pressure, h0, critical, a, n, floor)
    dp = pressure*1e-5
    from full_model.production.arrhenius_kinetics import exp_floor_enthalpy_j
    numerical = -(exp_floor_enthalpy_j(
        pressure+dp, h0, critical, a, n, floor)-exp_floor_enthalpy_j(
        pressure-dp, h0, critical, a, n, floor))/(2.0*dp)
    assert np.isclose(analytic, numerical, rtol=2e-10)
    assert analytic != 1.0e-29
import pytest

from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.coupled_front_event import (
    FrontEnergyTerms, propose_bidirectional_front_event)
from full_model.production.moving_front import DefectState


def _state(rho):
    shape = (3, 4, 4)
    return DefectState(
        np.full(shape, 0.2*rho), np.full(shape, 0.1*rho),
        np.full(shape, 0.15*rho), np.full(shape[:2], 0.1*rho))


def _event(a, b, eab=0.0, eba=0.0, mobility=True):
    return propose_bidirectional_front_event(
        a, b, event_volume_m3=2e-27, event_length_m=2.8e-10,
        line_energy_J_m=1.1e-9, temperature_K=900.0,
        process=ActivatedProcess("front", 2e10, 0.2, 1e9),
        h0_J=0.3*EV_J, critical_pressure_Pa=1e9, exp_a=2.0,
        exp_n=1.5, exp_floor=0.1,
        energy_a_to_b=FrontEnergyTerms(phase_J=eab),
        energy_b_to_a=FrontEnergyTerms(phase_J=eba),
        mobility_enabled=mobility, transmission_fraction=0.2,
        boundary_storage_fraction=0.1, neutral_sink_fraction=0.05)


def test_equal_state_has_exactly_zero_net_rate_without_projection():
    event = _event(_state(1e14), _state(1e14))
    assert event.net_velocity_a_to_b_m_s == 0.0
    assert event.rate_a_to_b_s == event.rate_b_to_a_s


@pytest.mark.parametrize("bias", [1e-24, 1e-23, 1e-22, 1e-21])
def test_near_equal_energy_response_is_odd_before_and_after_processing(bias):
    plus = _event(_state(1.2e14), _state(0.8e14), -bias, bias)
    minus = _event(_state(0.8e14), _state(1.2e14), bias, -bias)
    assert plus.net_velocity_a_to_b_m_s == pytest.approx(-minus.net_velocity_a_to_b_m_s)


def test_label_material_exchange_covariance_and_favorable_sign_reversal():
    a, b = _state(1.4e14), _state(0.7e14)
    ab = _event(a, b, -2e-21, 2e-21)
    ba = _event(b, a, 2e-21, -2e-21)
    assert ab.net_velocity_a_to_b_m_s == pytest.approx(-ba.net_velocity_a_to_b_m_s)
    assert ab.a_to_b.donor_line_m == pytest.approx(ba.b_to_a.donor_line_m)
    assert ab.net_velocity_a_to_b_m_s > 0.0


def test_mobility_off_is_stationary():
    event = _event(_state(1.4e14), _state(0.7e14), -1e-21, 1e-21, mobility=False)
    assert event.rate_a_to_b_s == event.rate_b_to_a_s == 0.0
    assert event.net_velocity_a_to_b_m_s == 0.0


def test_each_direction_closes_line_burgers_defect_energy_and_heat():
    event = _event(_state(1.4e14), _state(0.7e14))
    for trial in (event.a_to_b, event.b_to_a):
        assert trial.line_closure_m == pytest.approx(0.0, abs=1e-27)
        assert trial.signed_burgers_closure_m2 == pytest.approx(0.0, abs=1e-27)
        assert trial.energy_closure_J == pytest.approx(0.0, abs=1e-31)


def test_rate_ratio_satisfies_detailed_balance():
    event = _event(_state(1.4e14), _state(0.7e14), -1e-22, 1e-22)
    assert event.detailed_balance_log_residual == pytest.approx(0.0, abs=2e-13)


def test_true_reverse_pair_uses_one_delta_f_not_difference_of_opposites():
    thermal = 1.380649e-23*900.0
    delta = .25*thermal
    # Unit transmission removes irreversible processing, so the explicitly
    # overridden pair is a true microscopic reverse independent of density.
    event = propose_bidirectional_front_event(
        _state(1e14), _state(1e14), event_volume_m3=2e-27,
        event_length_m=2.8e-10, line_energy_J_m=1.1e-9,
        temperature_K=900.0,
        process=ActivatedProcess("front", 2e10, 0.2, 1e9),
        h0_J=.3*EV_J, critical_pressure_Pa=1e9, exp_a=2.0,
        exp_n=1.5, exp_floor=.1, transmission_fraction=1.0,
        kinetic_free_energy_a_to_b_J=delta,
        kinetic_free_energy_b_to_a_J=-delta)
    assert event.microscopic_reverse_pair
    assert event.actual_reverse_edge
    assert event.detailed_balance_applicable
    assert np.log(event.rate_a_to_b_s/event.rate_b_to_a_s) == pytest.approx(-.25)
    assert np.log(event.rate_a_to_b_s/event.rate_b_to_a_s) != pytest.approx(-.5)


def test_distinct_downhill_outgoing_channels_do_not_cancel_at_unit_acceptance():
    thermal = 1.380649e-23*900.0
    event = propose_bidirectional_front_event(
        _state(1.4e14), _state(.7e14), event_volume_m3=2e-27,
        event_length_m=2.8e-10, line_energy_J_m=1.1e-9,
        temperature_K=900.0,
        process=ActivatedProcess("front", 2e10, 0.2, 1e9),
        h0_J=.3*EV_J, critical_pressure_Pa=1e9, exp_a=2.0,
        exp_n=1.5, exp_floor=.1, transmission_fraction=.5,
        neutral_sink_fraction=.05,
        kinetic_free_energy_a_to_b_J=-2.0*thermal,
        kinetic_free_energy_b_to_a_J=-.5*thermal,
        actual_reverse_edge=False)
    assert event.a_to_b_channel.acceptance_probability == 1.0
    assert event.b_to_a_channel.acceptance_probability == 1.0
    assert event.a_to_b_channel.transition_state_rate_s > (
        event.b_to_a_channel.transition_state_rate_s)
    assert event.rate_a_to_b_s > event.rate_b_to_a_s
    assert event.net_velocity_a_to_b_m_s > 0.0
    assert not event.actual_reverse_edge
    assert not event.detailed_balance_applicable
    assert event.a_to_b_channel.gross_activity_s == event.rate_a_to_b_s
    assert event.b_to_a_channel.gross_activity_s == event.rate_b_to_a_s


def test_channel_availability_and_mobility_are_exposed_separately():
    event = propose_bidirectional_front_event(
        _state(1e14), _state(1e14), event_volume_m3=2e-27,
        event_length_m=2.8e-10, line_energy_J_m=1.1e-9,
        temperature_K=900.0,
        process=ActivatedProcess("front", 2e10, .2, 1e9),
        h0_J=.3*EV_J, critical_pressure_Pa=1e9, exp_a=2.0,
        exp_n=1.5, exp_floor=.1, transmission_fraction=1.0,
        availability_a_to_b=.25, availability_b_to_a=.75)
    assert event.a_to_b_channel.availability_factor == .25
    assert event.b_to_a_channel.availability_factor == .75
    assert event.actual_reverse_edge
    assert not event.detailed_balance_applicable
    assert event.rate_a_to_b_s/event.rate_b_to_a_s == pytest.approx(1.0/3.0)
    assert event.a_to_b_channel.activation_entropy_over_kB == .2
    assert event.a_to_b_channel.identifiable_prefactor_s == pytest.approx(
        2e10*np.exp(.2))
