import json

import pytest

from full_model.hpc3.run_v37_conduction_case import (
    load_cases,
    validate_source_identity,
)


def test_registered_v37_cases_are_positive_conduction_and_unique():
    cases = load_cases(
        __import__("pathlib").Path("full_model/hpc3/v37_conduction_cases.json"))
    assert len(cases) == 6
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["conductivity_W_m_K"] == 0.15 for case in cases)
    assert {case["strain_rate_s"] for case in cases} == {1e4, 3e4, 1e5}


def test_case_table_rejects_zero_conductivity(tmp_path):
    path = tmp_path/"cases.json"
    path.write_text(json.dumps([{
        "id": "bad", "T0_K": 900.0, "strain_rate_s": 3e4,
        "particle_radius_um": 0.75, "conductivity_W_m_K": 0.0}]))
    with pytest.raises(ValueError, match="positive conductivity"):
        load_cases(path)


def test_exact_frozen_source_identity_does_not_require_descendant_diff():
    from pathlib import Path
    from unittest.mock import patch
    head = "a"*40
    with patch(
            "full_model.hpc3.run_v37_conduction_case.subprocess.check_output",
            side_effect=[head+"\n", ""]):
        actual, exact = validate_source_identity(Path("."), head)
    assert actual == head and exact
