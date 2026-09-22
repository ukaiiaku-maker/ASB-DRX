import numpy as np

from full_model.production.v24_mechanical_wall import (
    accepted_geometry_plaquette_transaction, geometry_event_clock,
)
from tests.test_full_model_v43_lattice_geometry import (
    kinetics, prepared_loop, verification_event,
)


def _accepted_clock():
    state, args, _ = prepared_loop()
    event = verification_event(extent=0.25)
    result, ledger = accepted_geometry_plaquette_transaction(
        state, event, args[3], args[4], args[1], args[5], args[6],
        kinetics(), 1e-13)
    assert result is not state and ledger["accepted"]
    return ledger


def test_physical_event_clock_velocity_is_grid_independent():
    coarse = geometry_event_clock(2.5e8, 2.48e-10, 2e-7, 1e-10, 1.0)
    fine = geometry_event_clock(2.5e8, 2.48e-10, 1e-7, 1e-10, 1.0)
    np.testing.assert_allclose(
        coarse["physical_velocity_m_s"],
        fine["physical_velocity_m_s"], rtol=2e-15)
    np.testing.assert_allclose(
        coarse["physical_velocity_m_s"], 2.5e8*2.48e-10, rtol=2e-15)
    # At fixed physical time and speed, displacement is fixed; numerical
    # fractional extent therefore doubles when cell spacing is halved.
    np.testing.assert_allclose(
        fine["maximum_extent"], 2.0*coarse["maximum_extent"], rtol=2e-15)


def test_event_ledger_closes_count_site_jump_area_and_time():
    ledger = _accepted_clock()
    np.testing.assert_allclose(
        ledger["physical_event_count"],
        ledger["active_site_measure"]
        *ledger["microscopic_jumps_per_active_site"], rtol=2e-15)
    np.testing.assert_allclose(
        ledger["physical_displacement_m"],
        ledger["observed_accepted_velocity_m_s"]
        *ledger["consumed_duration_s"], rtol=2e-15)
    np.testing.assert_allclose(
        ledger["proposed_swept_area_m2"],
        ledger["physical_displacement_m"]*float(ledger[
            "physical_plaquette_area_m2"])**0.5, rtol=2e-15)
