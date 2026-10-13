import numpy as np
import pytest

from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters,
    apply_sign_balanced_pair_source,
    boundary_tj_site_field,
    hall_petch_nucleation_hazard_Hz,
    pair_source_rate_m2_s,
    random_bulk_site_field,
    randomized_matched_site_field,
)
from full_model.production.common_tensorial_wall import CommonWallState


def _state(n=12):
    shape = (n, n, 4)
    z = np.zeros(shape)
    return CommonWallState(
        z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy(),
        np.zeros((n, n, 0)), np.zeros((n, n)), np.zeros((n, n)),
        z.copy(), np.zeros((n, n, 3, 3)), np.zeros((n, n, 4, 3)),
        np.zeros((n, n, 4, 3, 3)), np.zeros((n, n)),
        np.full((n, n), 900.0))


def _four_owner_supports(n=12):
    supports = np.zeros((4, n, n))
    supports[0, :n//2, :n//2] = 1.0
    supports[1, n//2:, :n//2] = 1.0
    supports[2, :n//2, n//2:] = 1.0
    supports[3, n//2:, n//2:] = 1.0
    # Diffuse exactly one grid line while preserving the partition of unity.
    supports[:, n//2-1:n//2+1] = 0.5*supports[:, n//2-1:n//2+1]
    deficit = 1.0-np.sum(supports, axis=0)
    supports[0] += deficit
    return supports


def test_boundary_and_random_site_fields_have_matched_measure():
    boundary = boundary_tj_site_field(_four_owner_supports(), tj_site_weight=2.0)
    random = random_bulk_site_field(boundary.shape, seed=31)
    assert np.mean(boundary) == pytest.approx(1.0)
    assert np.mean(random) == pytest.approx(1.0)
    assert not np.allclose(boundary, random)
    np.testing.assert_array_equal(random, random_bulk_site_field(boundary.shape, 31))


def test_matched_random_control_preserves_exact_site_distribution():
    boundary = boundary_tj_site_field(_four_owner_supports(), tj_site_weight=2.0)
    random = randomized_matched_site_field(boundary, seed=19)
    np.testing.assert_array_equal(np.sort(random.ravel()),
                                  np.sort(boundary.ravel()))
    assert not np.array_equal(random, boundary)


def test_hall_petch_hazard_has_temperature_rate_and_stress_dependence():
    p = ArrheniusSourceParameters()
    stress = np.array((0.0, 1e8, 5e8))
    rate = hall_petch_nucleation_hazard_Hz(stress, 900.0, p)
    assert np.all(np.diff(rate) > 0.0)
    assert hall_petch_nucleation_hazard_Hz(0.0, 1100.0, p) > rate[0]


def test_pair_source_and_neutral_sink_close_line_and_burgers_balances():
    p = ArrheniusSourceParameters()
    state = _state()
    sites = random_bulk_site_field(state.orientation_rad.shape, 4)
    rate = pair_source_rate_m2_s(sites, 4e8, 900.0, p)
    sourced, source_ledger = apply_sign_balanced_pair_source(
        state, rate, 1e-5, source_mode="random_bulk")
    final, sink_ledger = apply_sign_balanced_pair_source(
        sourced, np.zeros_like(rate), 1e-5, sink_field=sites,
        neutral_sink_rate_s=2e5, source_mode="random_bulk")
    assert source_ledger.signed_content_change_m2 == 0.0
    assert source_ledger.line_balance_residual_m2 == pytest.approx(0.0, abs=1e-6)
    assert sink_ledger.signed_content_change_m2 == 0.0
    assert sink_ledger.line_balance_residual_m2 == pytest.approx(0.0, abs=1e-5)
    assert np.all(final.mobile_plus_m2 >= 0.0)
    assert np.all(final.mobile_minus_m2 >= 0.0)


def test_location_changes_but_integrated_source_is_matched():
    p = ArrheniusSourceParameters()
    boundary = boundary_tj_site_field(_four_owner_supports())
    random = random_bulk_site_field(boundary.shape, 9)
    br = pair_source_rate_m2_s(boundary, 3e8, 900.0, p)
    rr = pair_source_rate_m2_s(random, 3e8, 900.0, p)
    assert np.sum(br) == pytest.approx(np.sum(rr), rel=2e-15)
    assert np.max(np.abs(br-rr)) > 0.0
