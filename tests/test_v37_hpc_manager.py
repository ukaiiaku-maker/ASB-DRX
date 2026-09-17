import json
import tarfile

from full_model.analysis.manage_v37_hpc import (
    ACTIVE_MARKERS, atomic_json, parse_submission, read_record,
    extract_result_archives, run_postprocessor,
)


def test_atomic_manager_state_and_missing_record(tmp_path):
    state = tmp_path/"state.json"
    atomic_json(state, {"terminal": False, "iteration": 1})
    assert json.loads(state.read_text())["iteration"] == 1
    assert not state.with_suffix(".json.tmp").exists()
    assert read_record(tmp_path, "missing") == {}


def test_scheduler_active_markers_cover_array_queue_states():
    assert any(marker in "1_0|RUNNING|node|" for marker in ACTIVE_MARKERS)
    assert any(marker in "1_[2-5]|PENDING|reason|" for marker in ACTIVE_MARKERS)
    assert not any(marker in "1_0|COMPLETED|node|" for marker in ACTIVE_MARKERS)


def test_parse_followup_submission_identity():
    run_id, job_id = parse_submission(
        "Run ID: 20260917T000000Z-abcd123-123456\nJob ID: 98765\n")
    assert run_id == "20260917T000000Z-abcd123-123456"
    assert job_id == "98765"
    run_id, job_id = parse_submission(
        "Submitted run 20260917T000001Z-beef456-654321 as Slurm job 12345\n")
    assert run_id == "20260917T000001Z-beef456-654321"
    assert job_id == "12345"


def test_postprocessor_records_artifact(tmp_path):
    output = tmp_path/"artifact.txt"
    result = run_postprocessor([
        "python", "-c", f"from pathlib import Path; Path({str(output)!r}).write_text('ok')"
    ], output, tmp_path)
    assert result["returncode"] == 0
    assert result["artifact_exists"]


def test_extract_scoped_result_archives(tmp_path):
    fetched = tmp_path/"fetched"
    results = fetched/"results"
    results.mkdir(parents=True)
    source = tmp_path/"source.txt"
    source.write_text("checkpoint")
    with tarfile.open(results/"results-2.tar.gz", "w:gz") as stream:
        stream.add(source, arcname="output/case/checkpoint.txt")
    staging = tmp_path/"staging"
    archives = extract_result_archives(fetched, staging)
    assert len(archives) == 1
    assert (staging/"2/output/case/checkpoint.txt").read_text() == "checkpoint"
