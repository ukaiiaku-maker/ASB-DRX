import json
from pathlib import Path

import pytest

from full_model.analysis.run_v34_finite_coupled_response import resolved_bicrystal
from full_model.analysis.run_v37_priority_a_controls import (
    compare_complete_recurrent_state, equalize_complete_recurrent_state,
    exchange_initial_materials, run_interval, scalar_equal_tensor_contrast,
)
from full_model.analysis.run_v37_front_parameter_screen import evaluate


def _context(fraction=.35):
    return resolved_bicrystal(
        grid=16, length_m=3.2e-6, interface_width_m=4.0e-7,
        child_line_fraction=fraction, temperature_K=1100.0)


def test_scalar_equality_is_not_complete_recurrent_state_equality():
    context = _context(1.0)
    initial = context["state"]
    assert not compare_complete_recurrent_state(initial)[
        "exact_complete_recurrent_state_equal"]
    equal = equalize_complete_recurrent_state(context, initial)
    assert compare_complete_recurrent_state(equal)[
        "exact_complete_recurrent_state_equal"]
    tensor = scalar_equal_tensor_contrast(context, equal)
    comparison = compare_complete_recurrent_state(tensor)
    assert not comparison["exact_complete_recurrent_state_equal"]
    assert comparison["parent_child"]["maximum_relative_difference"] == (
        pytest.approx(.002))


def test_complete_material_exchange_reverses_current_source_response():
    context = _context(.35)
    initial = context["state"]
    _, original = run_interval(
        context, initial, direction=1, dt_s=1e-9,
        proposal_fraction=.0625)
    exchanged = exchange_initial_materials(context, initial)
    _, reverse = run_interval(
        context, exchanged, direction=-1, dt_s=1e-9,
        proposal_fraction=.0625)
    assert original["raw_constitutive_net_velocity_m_s"] == pytest.approx(
        -reverse["raw_constitutive_net_velocity_m_s"], rel=2e-14)
    assert original["accepted_signed_volume_m3"] == pytest.approx(
        -reverse["accepted_signed_volume_m3"], rel=1e-7)
    assert original["processed_line_m"] > 0.0
    assert reverse["processed_line_m"] > 0.0
    for result in (original, reverse):
        assert result["gross_directional_activity_s"] > 0.0
        assert result["complete_energy_decision"]["external_work_J"] == 0.0
        assert "before_energy_components" in result[
            "complete_candidate_endpoints"]["a_to_b_endpoint"]


def test_proposal_reversal_is_not_a_material_exchange():
    context = _context(.35)
    initial = context["state"]
    _, forward = run_interval(
        context, initial, direction=1, dt_s=1e-9,
        proposal_fraction=.0625)
    _, proposal_only = run_interval(
        context, initial, direction=-1, dt_s=1e-9,
        proposal_fraction=.0625)
    assert proposal_only["raw_constitutive_net_velocity_m_s"] < 0.0
    assert abs(proposal_only["raw_constitutive_net_velocity_m_s"]) != (
        pytest.approx(abs(forward["raw_constitutive_net_velocity_m_s"])))


def test_registered_screen_is_bounded_and_preserves_event_geometry():
    path = (Path(__file__).parents[1]/"full_model"/"verification"
            /"v37_front_parameter_screen_registry.json")
    registry = json.loads(path.read_text())
    result = evaluate(registry, -1.1e-23, -4.1e-24)
    assert result["case_count"] == 17
    assert registry["registered_before_response_evaluation"]
    assert not any("cell" in key for row in result["records"]
                   for key in row["parameters"])
    for row in result["records"]:
        assert row["event_volume_m3"] == pytest.approx(
            row["site_area_m2"]*row["jump_length_m"])
        assert row["rate_screen_only_not_trajectory"]
    baseline = next(row for row in result["records"]
                    if row["name"] == "baseline")
    low = next(row for row in result["records"]
               if row["name"] == "prefactor_low")
    high = next(row for row in result["records"]
                if row["name"] == "prefactor_high")
    assert low["net_velocity_m_s"] == pytest.approx(
        .1*baseline["net_velocity_m_s"])
    assert high["net_velocity_m_s"] == pytest.approx(
        10.0*baseline["net_velocity_m_s"])
