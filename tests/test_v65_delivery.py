import json
from pathlib import Path

import pytest

from full_model.analysis.build_v65_delivery import (
    require_complete_finalizer, select_checkpoints,
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
