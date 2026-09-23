import json
from pathlib import Path

import pytest

from full_model.analysis.run_v53_mechanism_priority import (
    elapsed_campaign_seconds, parse_utc, validate_case_table,
)


def test_v53_asb_pair_is_matched_and_high_rate():
    cases = validate_case_table(Path("full_model/hpc3/v53_asb_mechanism_pair.json"))
    assert cases[0]["causal_temperature_ablation"] == "none"
    assert cases[1]["causal_temperature_ablation"] == "freeze_flow"
    assert cases[0]["strain_rate_s"] == cases[1]["strain_rate_s"] == 30000.0


def test_v53_campaign_clock_is_timezone_aware_and_nonnegative():
    stamp = "2026-09-23T20:17:08.312432+00:00"
    assert parse_utc(stamp).utcoffset().total_seconds() == 0.0
    assert elapsed_campaign_seconds(stamp) >= 0.0


def test_v53_asb_pair_rejects_unmatched_parameters(tmp_path):
    source = Path("full_model/hpc3/v53_asb_mechanism_pair.json")
    cases = json.loads(source.read_text())
    cases[1]["conductivity_W_m_K"] = 0.2
    path = tmp_path/"bad.json"
    path.write_text(json.dumps(cases))
    with pytest.raises(ValueError, match="does not share"):
        validate_case_table(path)
