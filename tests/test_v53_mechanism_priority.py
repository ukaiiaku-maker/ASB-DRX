import json
from pathlib import Path

import pytest

from full_model.analysis.run_v53_mechanism_priority import (
    elapsed_campaign_seconds, parse_utc, recovered_v53_handoff,
    validate_case_table,
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


def test_recovery_accepts_only_complete_hashed_n192_and_matched_comparison(
        tmp_path):
    mechanism = tmp_path/"v53-bulk"/"mechanism-priority"
    loading = mechanism.parent/"n192"/"loading"
    verification = mechanism.parent/"verification"
    loading.mkdir(parents=True); verification.mkdir()
    checkpoint = loading/"checkpoint_000052.npz"
    checkpoint.write_bytes(b"attributable-checkpoint")
    import hashlib
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = loading/"run_manifest.json"
    manifest.write_text(json.dumps({
        "status": "COMPLETE", "completed_intervals": 52,
        "latest_checkpoint": str(checkpoint),
        "latest_checkpoint_sha256": checksum,
    }))
    comparison = verification/"v53_spatial_comparison_052.json"
    comparison.write_text(json.dumps({
        "comparison_interval": 52,
        "comparison_preconditions": {"all_preconditions_passed": True},
    }))
    handoff = {"state": "FAILED_CONTROLLER",
               "failure": "predecessor ended without valid terminal"}
    assert recovered_v53_handoff(handoff, mechanism) is not None
    comparison.write_text(json.dumps({"comparison_interval": 52}))
    assert recovered_v53_handoff(handoff, mechanism) is None
