import json
from pathlib import Path

import numpy as np

from full_model.hpc3.run_v31_front_continuation import (
    _checkpoint, _selected_case, _sha256)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/"full_model"/"hpc3"/"v31_front_continuation_manifest.json"


def test_manifest_selects_only_failed_discriminating_cases_and_frozen_inputs():
    manifest = json.loads(MANIFEST.read_text())
    archive = Path(manifest["frozen_archive_root"])
    assert manifest["checkpoint_policy"][
        "maximum_checkpoint_spacing_seconds"] <= 900
    assert len(manifest["cases"]) == 10
    identifiers = [case["id"] for case in manifest["cases"]]
    assert not any(name.endswith("_equal") or "mobility_off" in name
                   for name in identifiers)
    assert {case["grid"] for case in manifest["cases"]} == {64, 128}
    assert {case["comparison"] for case in manifest["cases"]} == {
        "favorable", "reversed", "label_swap_favorable",
        "near_equal_plus", "near_equal_minus"}
    for case in manifest["cases"]:
        checkpoint = archive/case["checkpoint"]
        parameters = archive/case["parameters"]
        assert _sha256(checkpoint) == case["checkpoint_sha256"]
        assert _sha256(parameters) == case["parameters_sha256"]
        status = json.loads((archive/case["id"]/"case_status.json").read_text())
        assert status["state"] == "FAILED_SCIENTIFIC"
        assert _checkpoint(checkpoint) < case["target_step"]


def test_case_selection_is_exact_and_checkpoint_validation_rejects_partial(tmp_path):
    manifest = json.loads(MANIFEST.read_text())
    selected = _selected_case(manifest, "a1_n64_favorable")
    assert selected["comparison"] == "favorable"
    bad = tmp_path/"bad.npz"
    np.savez_compressed(bad, step=np.array(3))
    try:
        _checkpoint(bad)
    except ValueError as error:
        assert "lacks coupled-front state" in str(error)
    else:
        raise AssertionError("partial checkpoint was accepted")
