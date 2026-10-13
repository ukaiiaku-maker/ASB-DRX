import json
from pathlib import Path
import zipfile

import pytest

import full_model.analysis.build_v65_delivery as delivery
from full_model.analysis.build_v65_delivery import (
    build, require_complete_finalizer, select_checkpoints, verify,
)


def checkpoint(tmp_path: Path, index: int) -> Path:
    path = tmp_path / f"checkpoint_adaptive_{index:06d}.npz"
    path.touch()
    return path


def test_delivery_checkpoint_selection_keeps_endpoints_and_milestones(tmp_path):
    paths = [checkpoint(tmp_path, index) for index in (10, 20, 30, 40, 50)]
    times = {10: 1.01e-6, 20: 1.05e-6, 30: 1.091e-6,
             40: 1.112e-6, 50: 1.25e-6}
    selected = select_checkpoints(paths, times)
    assert [int(path.stem.rsplit("_", 1)[-1]) for path in selected] == [
        10, 20, 30, 50,
    ]


def test_delivery_requires_complete_verified_finalizer(tmp_path):
    (tmp_path / "finalizer_status.json").write_text(json.dumps({
        "status": "POSTPROCESS_OR_TEST_FAILURE",
        "all_required_artifacts_present": False,
    }))
    with pytest.raises(RuntimeError, match="not complete"):
        require_complete_finalizer(tmp_path)


def test_delivery_accepts_complete_verified_finalizer(tmp_path):
    payload = {"status": "COMPLETE", "all_required_artifacts_present": True}
    (tmp_path / "finalizer_status.json").write_text(json.dumps(payload))
    assert require_complete_finalizer(tmp_path) == payload


def test_delivery_builds_and_reverifies_atomic_archive(tmp_path, monkeypatch):
    campaign = tmp_path/"campaign"
    worktree = tmp_path/"worktree"
    campaign.mkdir(); worktree.mkdir()
    (campaign/"finalizer_status.json").write_text(json.dumps({
        "status": "COMPLETE",
        "all_required_artifacts_present": True,
        "completed_utc": "2026-10-14T00:00:00Z",
    }))
    payload = campaign/"evidence.json"
    payload.write_text('{"result":"controlled"}\n')
    monkeypatch.setattr(delivery, "required_files",
                        lambda *_: [(payload, "analysis/evidence.json")])
    monkeypatch.setattr(delivery, "git_archive",
                        lambda _, commit: f"source:{commit}".encode())
    output = campaign/"definitive.zip"
    result = build(campaign, worktree, output, "production", "evidence")
    assert result["payload_count"] == 3
    assert result["archive_sha256"]
    assert output.with_name("definitive.zip.sha256").is_file()
    verify(output)
    with zipfile.ZipFile(output) as archive:
        inventory = json.loads(archive.read("INVENTORY.json"))
        assert inventory["production_source_commit"] == "production"
        assert inventory["evidence_source_commit"] == "evidence"
        assert {item["path"] for item in inventory["payloads"]} == {
            "analysis/evidence.json",
            "source/production_source.tar",
            "source/evidence_source.tar",
        }
    with pytest.raises(FileExistsError):
        build(campaign, worktree, output, "production", "evidence")
