import hashlib
import json

import pytest

from full_model.analysis.run_v52_completion_manager import (
    LogicalLock, command_owns_output, inspect_manifest,
)


def identity(pid, command, cwd="/work", start="Mon Jan 1 00:00:00 2024"):
    return {"pid": pid, "ppid": 1, "start_identity": start,
            "command": command, "cwd": cwd}


def test_recognizes_direct_and_module_invocation_output_owner(tmp_path):
    output = tmp_path/"run"
    assert command_owns_output(
        f"python full_model/analysis/run_v49_physical_continuation.py "
        f"--output-dir {output}", "/work", output)
    assert command_owns_output(
        f"python -m full_model.analysis.run_v49_physical_continuation "
        f"--output-dir={output}", "/work", output)
    assert not command_owns_output(
        "python run_v49_physical_continuation.py --output-dir elsewhere",
        "/work", output)


def write_manifest(output, status, completed):
    output.mkdir()
    checkpoint = output/"checkpoint.npz"; checkpoint.write_bytes(b"state")
    manifest = {"status": status, "completed_intervals": completed,
                "records": [{}]*completed, "latest_checkpoint": str(checkpoint),
                "latest_checkpoint_sha256": hashlib.sha256(b"state").hexdigest()}
    (output/"run_manifest.json").write_text(json.dumps(manifest))


def test_stale_running_manifest_is_restartable_and_live_owner_is_healthy(tmp_path):
    output = tmp_path/"run"; write_manifest(output, "RUNNING", 3)
    assert inspect_manifest(output, 8, identities=[])["classification"] == (
        "RESTARTABLE_INFRASTRUCTURE_INTERRUPTION")
    owner = identity(12, f"python -m full_model.analysis."
                     f"run_v49_physical_continuation --output-dir {output}")
    assert inspect_manifest(output, 8, identities=[owner])["classification"] == (
        "HEALTHY_RUNNING")


def test_completed_stage_resumes_and_early_terminal_is_valid(tmp_path):
    complete = tmp_path/"complete"; write_manifest(complete, "COMPLETE", 8)
    assert inspect_manifest(complete, 8, identities=[])["classification"] == (
        "VALID_COMPLETED")
    terminal = tmp_path/"terminal"
    write_manifest(terminal, "PHYSICAL_TERMINAL", 5)
    assert inspect_manifest(terminal, 8, identities=[])["classification"] == (
        "VALID_PHYSICAL_TERMINAL")


def test_corrupt_manifest_is_not_restartable(tmp_path):
    output = tmp_path/"bad"; output.mkdir()
    (output/"run_manifest.json").write_text("not-json")
    assert inspect_manifest(output, 8, identities=[])["classification"] == (
        "UNKNOWN_OR_CORRUPT")


def test_two_managers_cannot_acquire_same_logical_lock(tmp_path, monkeypatch):
    import os
    current = identity(os.getpid(), "python manager.py")
    monkeypatch.setattr(
        "full_model.analysis.run_v52_completion_manager.process_identity",
        lambda pid: current if int(pid) == os.getpid() else None)
    lock = LogicalLock(tmp_path/"logical.lock"); lock.acquire()
    with pytest.raises(RuntimeError, match="live manager"):
        LogicalLock(tmp_path/"logical.lock").acquire()
    lock.release()
