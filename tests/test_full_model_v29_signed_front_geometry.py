import numpy as np
import pytest

from full_model.production.signed_front_geometry import measure_signed_front_motion


def _single_front(n=48, location=20.25, width=1.7):
    x = np.arange(n, dtype=float)[:, None]
    return np.broadcast_to(np.tanh((x-location)/width), (n, 9)).copy()


@pytest.mark.parametrize("shift", [-1.0, -0.5, -0.1, -0.01, 0.01, 0.1, 0.5, 1.0])
def test_manufactured_subcell_translation_has_declared_sign_and_magnitude(shift):
    ref = _single_front()
    cur = _single_front(location=20.25+shift)
    result = measure_signed_front_motion(
        ref, cur, normal_axis=0, spacing_m=2e-9, periodic=False)
    # phi>0 lies to the right, so a positive contour translation consumes it.
    expected = -shift * 9 * (2e-9)**2
    assert result.signed_receiver_area_m2 == pytest.approx(expected, abs=2e-3*abs(expected))
    assert result.crossing_count == 9


@pytest.mark.parametrize("width", [0.45, 0.8, 1.2, 2.5, 4.0, 7.0])
def test_profile_width_sweep_is_zero_motion(width):
    ref = _single_front(width=1.7)
    cur = _single_front(width=width)
    result = measure_signed_front_motion(
        ref, cur, normal_axis=0, spacing_m=1.0, periodic=False)
    assert result.signed_receiver_area_m2 == pytest.approx(0.0, abs=1e-14)


def test_closed_translation_cycle_closes_exactly_within_interpolation_error():
    a = _single_front(location=19.6)
    b = _single_front(location=20.1)
    ab = measure_signed_front_motion(a, b, normal_axis=0, spacing_m=1.0, periodic=False)
    ba = measure_signed_front_motion(b, a, normal_axis=0, spacing_m=1.0, periodic=False)
    assert ab.signed_receiver_area_m2 + ba.signed_receiver_area_m2 == pytest.approx(0.0, abs=1e-14)


def test_periodic_active_window_ignores_wrap_edge_and_measures_selected_front():
    ref = _single_front(location=20.25)
    cur = _single_front(location=20.75)
    mask = np.ones(ref.shape, dtype=bool)
    mask[[0, -1], :] = False
    result = measure_signed_front_motion(
        ref, cur, normal_axis=0, spacing_m=1.0, active_mask=mask, periodic=True)
    assert result.signed_receiver_area_m2 == pytest.approx(-4.5, abs=8e-3)


def test_label_swap_reverses_signed_material_assignment():
    ref = _single_front(location=20.25)
    cur = _single_front(location=20.75)
    direct = measure_signed_front_motion(ref, cur, normal_axis=0, spacing_m=1.0, periodic=False)
    swapped = measure_signed_front_motion(-ref, -cur, normal_axis=0, spacing_m=1.0, periodic=False)
    assert swapped.signed_receiver_area_m2 == pytest.approx(-direct.signed_receiver_area_m2)


def test_coordinate_reflection_preserves_material_gain():
    ref = _single_front(location=20.25)
    cur = _single_front(location=20.75)
    direct = measure_signed_front_motion(ref, cur, normal_axis=0, spacing_m=1.0, periodic=False)
    reflected = measure_signed_front_motion(ref[::-1], cur[::-1], normal_axis=0,
                                            spacing_m=1.0, periodic=False)
    assert reflected.signed_receiver_area_m2 == pytest.approx(direct.signed_receiver_area_m2)


def test_two_interfaces_are_matched_by_orientation_without_cancellation_error():
    n = 64
    x = np.arange(n, dtype=float)[:, None]
    def band(left, right):
        return np.broadcast_to(np.tanh((x-left)/1.4)*np.tanh((right-x)/1.4), (n, 7)).copy()
    ref = band(15.2, 42.6)
    # Receiver band expands by 0.3 on both sides: total area gain = 0.6*7.
    cur = band(14.9, 42.9)
    result = measure_signed_front_motion(ref, cur, normal_axis=0, spacing_m=1.0,
                                         periodic=False)
    assert result.crossing_count == 14
    assert result.signed_receiver_area_m2 == pytest.approx(4.2, abs=1.5e-2)
