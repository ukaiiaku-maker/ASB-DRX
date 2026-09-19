#!/usr/bin/env python3
"""Exercise the strict V43 restart contract from an archived source tree."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, cwd, env, check=True):
    return subprocess.run(command, cwd=cwd, env=env, check=check,
                          capture_output=True, text=True)


def base_command(python, runner, case, target, source):
    return [python, str(runner), "--case-dir", str(case), "--grid", "16",
            "--condition", "broadband_noise", "--initial-strain", ".01",
            "--target-strain", str(target), "--max-steps", "4",
            "--history-interval", "1", "--progress-checkpoint-strain", ".000005",
            "--trial-dt-s", "1e-9", "--max-wall-s", "30"]


def main():
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign = (Path("/Users/sdillon/HPC3/local-results/asb-drx-v43-restart")
                /source[:7]/("attempt-"+token))
    campaign.mkdir(parents=True, exist_ok=True)
    archive = campaign/"source.tar"
    with archive.open("wb") as stream:
        subprocess.run(["git", "archive", "HEAD"], cwd=ROOT, check=True,
                       stdout=stream)
    staged = campaign/"staged-source"
    staged.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(staged, filter="data")
    runner = staged/"full_model/analysis/run_v30_mura_tier_b1_case.py"
    env = dict(os.environ, PYTHONPATH=str(staged), V37_SOURCE_SHA=source)
    python = os.environ.get("PYTHON", "python")
    case = campaign/"valid-renamed-seed"
    init = base_command(python, runner, case, .01001, source)+[
        "--start-mode", "initialize"]
    run(init, staged, env)
    status0 = json.loads((case/"status.json").read_text())
    seed = sorted(case.glob("checkpoint_step_*.npz"))[-1]
    renamed = case/"seed-with-ordinary-name.npz"
    shutil.copy2(seed, renamed)
    seed_sha = digest(renamed)
    explicit = [
        "--start-mode", "restart", "--restart-file", str(renamed),
        "--expected-restart-sha256", seed_sha,
        "--expected-restart-source-sha", source,
        "--expected-restart-step", str(status0["step"]),
        "--expected-restart-time-s", str(status0["physical_time_s"]),
        "--expected-restart-strain", str(status0["applied_strain"]),
    ]
    run(base_command(python, runner, case, .01002, source)+explicit,
        staged, env)
    status1 = json.loads((case/"status.json").read_text())

    failures = {}
    variants = {
        "wrong_checksum": [
            *explicit[:],
        ],
        "wrong_source_lineage": [
            *explicit[:],
        ],
        "missing_path": [
            *explicit[:],
        ],
    }
    variants["wrong_checksum"][variants["wrong_checksum"].index(seed_sha)] = "0"*64
    variants["wrong_source_lineage"][
        variants["wrong_source_lineage"].index(source)] = "not-the-seed-source"
    variants["missing_path"][
        variants["missing_path"].index(str(renamed))] = str(case/"absent-seed.npz")
    for name, options in variants.items():
        bad_case = campaign/("reject-"+name)
        result = run(base_command(python, runner, bad_case, .01002, source)+options,
                     staged, env, check=False)
        failures[name] = {
            "returncode": result.returncode,
            "rejected": result.returncode != 0,
            "error_tail": result.stderr.strip().splitlines()[-1],
            "checkpoint_created": any(bad_case.glob("checkpoint_step_*.npz")),
            "status_created": (bad_case/"status.json").exists(),
        }
    payload = {
        "schema": "asb-drx/v43/restart-contract/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source, "source_archive": str(archive),
        "source_archive_sha256": digest(archive),
        "staged_archive_execution": True,
        "ordinary_filename_loaded_explicitly": (
            status1["resumed_from"] == str(renamed.resolve())),
        "restart_validation": status1["restart_validation"],
        "initial_step": status0["step"], "continued_step": status1["step"],
        "continued_physical_time_s": status1["physical_time_s"],
        "failure_challenges": failures,
        "failed_restart_advanced_zero_steps": all(
            row["rejected"] and not row["checkpoint_created"]
            and not row["status_created"] for row in failures.values()),
        "contract_passed": bool(
            status1["resumed_from"] == str(renamed.resolve())
            and all(row["rejected"] and not row["checkpoint_created"]
                    and not row["status_created"] for row in failures.values())),
    }
    output = ROOT/"full_model/verification/v43_restart_contract.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"contract_passed": payload["contract_passed"],
                      "source_sha": source}, sort_keys=True))


if __name__ == "__main__":
    main()
