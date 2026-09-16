import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from full_model.hpc3.run_v30_front_case import (
    build_parameters, checkpoint_records, segment_for_step)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/"full_model"/"hpc3"/"v30_front_case_manifest.json"


def test_manifest_preregisters_restart_policy_operator_and_dependency_order():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["checkpoint_policy"]["wallclock_seconds"] <= 900
    base = manifest["base_parameters"]
    assert base["sibm_front_operator"] == "coupled_bidirectional_v30"
    assert base["sibm_legacy_afterburner_reproduction"] is False
    assert base["sibm_equal_state_projection_enabled"] is False
    ids = [case["id"] for case in manifest["cases"]]
    assert len(ids) == len(set(ids)) == 33
    assert all(case["dependency_state"] == "READY"
               for case in manifest["cases"][:24])
    assert all(case["dependency_state"] == "BLOCKED_BY_DEPENDENCY"
               for case in manifest["cases"][24:])


def test_delta_config_is_odd_and_restart_is_exactly_declared(tmp_path):
    manifest = json.loads(MANIFEST.read_text())
    plus = manifest["cases"][12]
    minus = manifest["cases"][13]
    restart = tmp_path/"state.npz"
    restart.touch()
    p = build_parameters(manifest, plus, plus["segments"][0], 17, restart)
    m = build_parameters(manifest, minus, minus["segments"][0], 17, restart)
    base = 2.5e17
    assert np.isclose(p["sibm_clean_parent_density_m2"]-base, -(m[
        "sibm_clean_parent_density_m2"]-base), rtol=0.0, atol=64.0)
    assert np.isclose(p["sibm_clean_child_density_m2"]-base, -(m[
        "sibm_clean_child_density_m2"]-base), rtol=0.0, atol=64.0)
    assert p["restart_file"] == str(restart)
    assert p["restart_reset_clock"] is False
    assert p["nSteps"] == 17


def test_segment_resume_selects_only_unfinished_physical_progress():
    case = {"segments": [
        {"steps": 600, "applied_pressure_Pa": 5e6},
        {"steps": 600, "applied_pressure_Pa": -5e6},
        {"steps": 600, "applied_pressure_Pa": 5e6}]}
    assert segment_for_step(case, 0)[:3] == (0, 0, 600)
    assert segment_for_step(case, 599)[:3] == (0, 0, 600)
    assert segment_for_step(case, 600)[:3] == (1, 600, 1200)
    assert segment_for_step(case, 1799)[:3] == (2, 1200, 1800)
    assert segment_for_step(case, 1800) is None


def test_checkpoint_discovery_ignores_partial_files_and_selects_step(tmp_path):
    run = tmp_path/"segments"/"segment-00"/"attempt-000"
    run.mkdir(parents=True)
    np.savez(run/"a.npz", step=np.array(3))
    np.savez(run/"b.npz", step=np.array(9),
             coupled_front_metadata_json=np.array("{}"))
    (run/"partial.npz").write_bytes(b"not an npz")
    records = checkpoint_records(tmp_path)
    assert [record[0] for record in records] == [9]
    assert records[-1][3] is True


def test_empty_postprocess_preserves_incomplete_decision(tmp_path):
    output = tmp_path/"decision.json"
    index = tmp_path/"index.csv"
    subprocess.run([
        sys.executable,
        str(ROOT/"full_model"/"analysis"/"postprocess_v30_front_hpc.py"),
        "--output-root", str(tmp_path/"missing"),
        "--output", str(output), "--index", str(index)], check=True)
    decision = json.loads(output.read_text())
    assert decision["classification"] == "V30_FRONT_A1_INCOMPLETE"
    assert decision["a2_authorized"] is False
    assert len(decision["records"]) == 33
    assert index.exists()


def test_scheduler_arrays_cover_each_case_once_and_a2_requires_promotion():
    anchor = (ROOT/"full_model"/"hpc3"/
              "submit_v30_front_a1_anchor.sbatch").read_text()
    delta = (ROOT/"full_model"/"hpc3"/
             "submit_v30_front_a1_delta.sbatch").read_text()
    a2 = (ROOT/"full_model"/"hpc3"/"submit_v30_front_a2.sbatch").read_text()
    assert "--array=0-11%4" in anchor
    assert "--array=12-23%4" in delta
    assert "--array=24-32%3" in a2
    assert "V30_A2_AUTHORIZED" in a2
    assert "V30_EXPECTED_SOURCE_SHA" in anchor+delta+a2
    assert "V30_EXPECTED_REMOTE_SHA" in anchor+delta+a2
