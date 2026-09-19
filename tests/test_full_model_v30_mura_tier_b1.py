import json
import hashlib
from pathlib import Path
import subprocess
import sys
import shutil


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT/"full_model/analysis/run_v30_mura_tier_b1_case.py"
POST = ROOT/"full_model/analysis/postprocess_v30_mura_tier_b1.py"


def _run(case, target, max_steps, mode):
    command = [
        sys.executable, str(RUNNER), "--case-dir", str(case),
        "--grid", "16", "--condition", "broadband_noise",
        "--initial-strain", ".01", "--target-strain", str(target),
        "--max-steps", str(max_steps), "--history-interval", "1",
        "--progress-checkpoint-strain", ".000005",
        "--trial-dt-s", "1e-9", "--max-wall-s", "30",
        "--start-mode", mode,
    ]
    if mode == "restart":
        checkpoint = sorted(case.glob("checkpoint_step_*.npz"))[-1]
        renamed = case/"ordinary_explicit_seed.npz"
        shutil.copy2(checkpoint, renamed)
        checkpoint = renamed
        status = json.loads((case/"status.json").read_text())
        command.extend([
            "--restart-file", str(checkpoint),
            "--expected-restart-sha256",
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "--expected-restart-source-sha", "UNRECORDED",
            "--expected-restart-step", str(status["step"]),
            "--expected-restart-time-s", str(status["physical_time_s"]),
            "--expected-restart-strain", str(status["applied_strain"]),
        ])
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)


def test_tier_b1_case_checkpoint_resume_and_partial_postprocess(tmp_path):
    cases = tmp_path/"cases"
    case = cases/"broadband_noise_16"
    _run(case, .01001, 2, "initialize")
    first = json.loads((case/"status.json").read_text())
    assert first["step"] == 1
    assert first["status"] == "COMPLETED"
    _run(case, .01002, 4, "restart")
    second = json.loads((case/"status.json").read_text())
    assert second["step"] == 2
    assert second["resumed_from"] is not None
    assert len(list(case.glob("checkpoint_step_*.npz"))) == 2
    output = tmp_path/"decision.json"
    index = tmp_path/"index.csv"
    subprocess.run([
        sys.executable, str(POST), "--cases", str(cases),
        "--output", str(output), "--index-csv", str(index),
    ], cwd=ROOT, check=True, capture_output=True, text=True)
    decision = json.loads(output.read_text())
    assert decision["classification"] == (
        "TIER_B1_PARTIAL_VALID_RESTARTABLE_EVIDENCE")
    assert decision["cases"][0]["valid_so_far"]
    assert not decision["scientific_gate_passed"]
    assert index.exists()
