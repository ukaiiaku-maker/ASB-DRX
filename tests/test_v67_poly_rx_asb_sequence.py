import json

import pytest

from full_model.analysis.run_v67_poly_rx_asb_sequence import (
    build_plan, validate_result,
)


def test_plan_alternates_controls_and_uses_exact_branch_parent():
    plan = build_plan("case", 400, 500, 100)
    assert [(row["output_name"], row["parent_name"]) for row in plan] == [
        ("case_ad_400", "case_ad_300"),
        ("case_bath_400", "case_bath_300"),
        ("case_ad_500", "case_ad_400"),
        ("case_bath_500", "case_bath_400"),
    ]


def test_plan_rejects_nonintegral_horizon():
    with pytest.raises(ValueError):
        build_plan("case", 400, 550, 100)


def _result(control):
    return {
        "source_tree_clean": True,
        "configuration": {
            "thermal_control": (
                "adiabatic" if control == "ad" else "strong_bath"),
            "thermal_bath_rate_s": 0.0 if control == "ad" else 5e9,
        },
        "decision": {
            "all_hard_absorption_balances_passed": True,
            "mura_hard_balance_passed": True,
            "first_law_hard_balance_passed": True,
            "grain_ids_unchanged": True,
            "front_transmission_fraction": .25,
        },
    }


def test_result_validation_is_fail_closed_for_bath_and_balances(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_result("bath")))
    assert validate_result(path, "bath") == (True, [])
    payload = _result("bath")
    payload["configuration"]["thermal_bath_rate_s"] = 0.0
    payload["decision"]["mura_hard_balance_passed"] = False
    path.write_text(json.dumps(payload))
    valid, failures = validate_result(path, "bath")
    assert not valid
    assert "mura_hard_balance_passed" in failures
    assert "strong_bath_rate_not_positive" in failures
