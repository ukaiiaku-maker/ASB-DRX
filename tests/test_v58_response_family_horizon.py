import json

from full_model.analysis.classify_v58_response_family import load


def _write_case(root, steps):
    root.mkdir()
    (root / "result.json").write_text(json.dumps({"source_commit": "abc"}))
    (root / "classification.json").write_text(json.dumps({
        "source_commit": "abc",
    }))
    (root / "history.json").write_text(json.dumps([
        {"step": step, "time_s": time_s} for step, time_s in steps
    ]))


def test_refinement_horizon_matches_physical_time_not_step(tmp_path):
    coarse = tmp_path / "coarse"
    refined = tmp_path / "refined"
    _write_case(coarse, [(63, 3.15e-6)])
    _write_case(refined, [(126, 3.15e-6)])

    assert load(coarse, common_time_s=3.15e-6)["row"]["step"] == 63
    assert load(refined, common_time_s=3.15e-6)["row"]["step"] == 126
