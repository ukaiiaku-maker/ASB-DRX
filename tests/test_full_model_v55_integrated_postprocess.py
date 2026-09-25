import numpy as np
import pytest

from full_model.analysis.postprocess_v55_integrated_campaign import (
    comparison_contract,
    concentration,
    outcome_neutral_decision,
)


def test_concentration_has_exact_uniform_and_single_cell_limits():
    uniform = concentration(np.ones((8, 8)))
    localized_field = np.zeros((8, 8)); localized_field[2, 3] = 4.0
    localized = concentration(localized_field)
    assert uniform["participation_fraction"] == 1.0
    assert localized["participation_fraction"] == 1.0/64.0
    assert localized["maximum_to_mean"] == 64.0


def test_concentration_ignores_negative_signed_power_for_support_metrics():
    field = np.array([[3.0, -10.0], [0.0, 1.0]])
    result = concentration(field)
    np.testing.assert_allclose(result["participation_fraction"], .4)
    assert 0.0 < result["top_five_percent_fraction"] <= 1.0
    assert result["positive_sum"] == 4.0
    assert result["negative_sum"] == -10.0
    assert result["signed_sum"] == -6.0


def valid_evidence():
    return {
        "execution_complete": True,
        "all_supporting_cases_hard_valid": True,
        "physical_comparability": True,
        "nontrivial_transformation": True,
        "selected_observable_refinement": True,
        "restart_qualified": True,
        "temporal_refinement_qualified": True,
        "coupled_trajectory_valid": True,
        "causal_controls_valid": True,
    }


@pytest.mark.parametrize("failed_key", [
    "all_supporting_cases_hard_valid",  # invalid fine grid/control
    "physical_comparability",          # changed clock/load/initial state
    "nontrivial_transformation",       # zero transformation
    "selected_observable_refinement",
    "restart_qualified",
    "temporal_refinement_qualified",
    "coupled_trajectory_valid",
    "causal_controls_valid",
])
def test_positive_classification_cannot_survive_failed_condition(failed_key):
    evidence = valid_evidence(); evidence[failed_key] = False
    result = outcome_neutral_decision(evidence)
    assert not result["scientific_gate_passed"]
    assert result["classification"] != (
        "VALID_GENERIC_EXISTING_BOUNDARY_DRX_WITH_"
        "COUPLED_THERMOMECHANICAL_RESPONSE")
    assert failed_key in result["failed_conditions"]


def test_missing_record_is_incomplete_not_conservation_failure():
    evidence = valid_evidence(); del evidence["causal_controls_valid"]
    result = outcome_neutral_decision(evidence)
    assert not result["scientific_gate_passed"]
    assert result["classification"] == "INCOMPLETE_EVIDENCE"
    assert result["missing_conditions"] == ["causal_controls_valid"]


def test_comparison_contract_rejects_changed_clock_and_initial_state():
    base = {
        "physical_time_s": 1e-6, "applied_strain": .03,
        "physical_protocol": {"Nx": 64, "L_phys": 1e-5},
        "initial_child_material_fraction": .5,
    }
    assert comparison_contract(base, dict(base))["comparable"]
    changed_clock = dict(base, physical_time_s=1.1e-6)
    assert not comparison_contract(base, changed_clock)["comparable"]
    changed_origin = dict(base, initial_child_material_fraction=.4)
    assert not comparison_contract(base, changed_origin)["comparable"]
