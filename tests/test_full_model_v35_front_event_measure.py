import numpy as np

from full_model.production.front_event_measure import (
    combine_front_event_measures, physical_front_event_measure)
from full_model.analysis.run_v35_front_kinetic_screen import run_screen


def _measure(length=8e-6, thickness=4.96e-10):
    return physical_front_event_measure(
        interface_length_m=length, represented_thickness_m=thickness,
        burgers_m=2.48e-10, rate_a_to_b_per_site_s=3.0e6,
        rate_b_to_a_per_site_s=1.0e6, dt_s=1.0e-7)


def test_b3_b_measure_closes_event_volume_and_velocity_exactly():
    result = _measure()
    assert result.event_volume_m3 == result.burgers_m**3
    assert result.event_length_m == result.burgers_m
    assert result.site_area_m2 == result.burgers_m**2
    assert result.site_count_per_interface_area_m2 == 1.0/result.burgers_m**2
    assert result.expected_signed_swept_volume_m3 == (
        result.expected_signed_event_count*result.event_volume_m3)
    np.testing.assert_allclose(
        result.expected_normal_velocity_m_s,
        result.burgers_m*(result.rate_a_to_b_per_site_s
                          -result.rate_b_to_a_per_site_s), rtol=2e-16)


def test_patch_partition_is_exactly_additive():
    whole = _measure(length=9e-6)
    pieces = [_measure(length=value) for value in (1e-6, 3e-6, 5e-6)]
    combined = combine_front_event_measures(pieces)
    for name in ("physical_site_count", "expected_events_a_to_b",
                 "expected_events_b_to_a", "expected_signed_swept_volume_m3",
                 "expected_absolute_swept_volume_m3"):
        np.testing.assert_allclose(
            getattr(combined, name), getattr(whole, name), rtol=2e-16)


def test_grid_sampling_does_not_change_physical_count():
    length = 10e-6
    samples = []
    for grid in (64, 128, 192):
        dx = length/grid
        segment_lengths = np.full(grid, dx)
        samples.append(combine_front_event_measures(
            [_measure(length=value) for value in segment_lengths]))
    for result in samples[1:]:
        np.testing.assert_allclose(
            result.physical_site_count, samples[0].physical_site_count,
            rtol=3e-15)
        np.testing.assert_allclose(
            result.expected_normal_velocity_m_s,
            samples[0].expected_normal_velocity_m_s, rtol=2e-16)


def test_thickness_extensivity_has_invariant_areal_measure_and_velocity():
    thin = _measure(thickness=2.48e-10)
    thick = _measure(thickness=9.92e-10)
    assert thick.physical_site_count == 4.0*thin.physical_site_count
    assert thick.expected_signed_event_count == (
        4.0*thin.expected_signed_event_count)
    assert thick.expected_signed_swept_volume_m3 == (
        4.0*thin.expected_signed_swept_volume_m3)
    assert thick.site_count_per_interface_area_m2 == (
        thin.site_count_per_interface_area_m2)
    assert thick.event_count_per_interface_area == (
        thin.event_count_per_interface_area)
    assert thick.expected_normal_velocity_m_s == (
        thin.expected_normal_velocity_m_s)


def test_bounded_zero_applied_pressure_screen_has_symmetry_and_balance():
    result = run_screen()
    assert result["case_count"] == 15
    assert result["classification"] == (
        "BOUNDED_HYPOTHESIS_SCREEN_NOT_CALIBRATION")
    records = {(item["temperature_K"], item["stored_energy_pressure_Pa"]): item
               for item in result["records"]}
    for temperature in (900.0, 1100.0, 1300.0):
        neutral = records[(temperature, 0.0)]
        assert neutral["applied_pressure_Pa"] == 0.0
        assert neutral["rate_a_to_b_per_site_s"] == (
            neutral["rate_b_to_a_per_site_s"])
        assert neutral["net_velocity_a_to_b_m_s"] == 0.0
        for pressure in (200e6, 600e6):
            positive = records[(temperature, pressure)]
            negative = records[(temperature, -pressure)]
            assert positive["rate_a_to_b_per_site_s"] == (
                negative["rate_b_to_a_per_site_s"])
            assert positive["rate_b_to_a_per_site_s"] == (
                negative["rate_a_to_b_per_site_s"])
            assert positive["net_velocity_a_to_b_m_s"] == (
                -negative["net_velocity_a_to_b_m_s"])
            assert abs(positive["detailed_balance_log_residual"]) < 2e-15
            assert positive["microscopic_reverse_pair"]
    controls = result["invariance_controls"]
    assert abs(controls["patch_site_count_relative_residual"]) < 3e-16
    assert abs(controls["patch_swept_volume_relative_residual"]) < 3e-16
    assert controls["thickness_site_count_ratio"] == 4.0
    assert controls["thickness_velocity_relative_residual"] == 0.0
