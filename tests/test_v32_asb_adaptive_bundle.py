import pytest

from full_model.hpc3.run_v32_asb_adaptive_case import (
    COORDINATES,
    case_definition,
)


def test_initial_screen_is_paired_and_one_factor_at_a_time():
    cases = [case_definition(index) for index in range(2*len(COORDINATES))]
    assert len({case["case_name"] for case in cases}) == 8
    for coordinate, (_, temperature, rate) in enumerate(COORDINATES):
        pair = cases[2*coordinate:2*coordinate+2]
        assert {case["thermal_control"] for case in pair} == {
            "adiabatic", "isothermal"}
        assert {case["temperature_K"] for case in pair} == {temperature}
        assert {case["strain_rate_s-1"] for case in pair} == {rate}
        assert {case["seed"] for case in pair} == {43}
        assert (temperature == 1100.0) ^ (rate == 3.0e4)


def test_case_ids_outside_preregistered_screen_are_rejected():
    with pytest.raises(ValueError):
        case_definition(-1)
    with pytest.raises(ValueError):
        case_definition(2*len(COORDINATES))
