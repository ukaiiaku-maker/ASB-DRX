import json

from full_model.analysis.manage_v37_hpc import atomic_json, read_record


def test_atomic_manager_state_and_missing_record(tmp_path):
    state = tmp_path/"state.json"
    atomic_json(state, {"terminal": False, "iteration": 1})
    assert json.loads(state.read_text())["iteration"] == 1
    assert not state.with_suffix(".json.tmp").exists()
    assert read_record(tmp_path, "missing") == {}
