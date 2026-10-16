#!/usr/bin/env python3
"""Run the predeclared matched V67 polycrystal horizon sequentially.

This is a local orchestration layer only.  It calls the production overlap
runner one case at a time, retains every atomic checkpoint, and advances each
thermal branch only from a result that passes the declared conservation and
identity audits.  A failure in one branch is recorded without deleting or
overwriting the other branch's evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNNER = "full_model.analysis.run_v67_drx_asb_overlap"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _case_name(prefix: str, control: str, total: int) -> str:
    return f"{prefix}_{control}_{total}"


def build_plan(prefix: str, first_total: int, final_total: int,
               chunk: int) -> list[dict]:
    if chunk <= 0 or first_total <= chunk or final_total < first_total:
        raise ValueError("invalid positive cumulative chunk horizon")
    if (final_total - first_total) % chunk:
        raise ValueError("final horizon must be an integer number of chunks")
    plan = []
    for total in range(first_total, final_total + 1, chunk):
        for control in ("ad", "bath"):
            plan.append({
                "control": control,
                "total_intervals": total,
                "parent_name": _case_name(prefix, control, total - chunk),
                "output_name": _case_name(prefix, control, total),
            })
    return plan


def validate_result(path: Path, control: str) -> tuple[bool, list[str]]:
    failures = []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return False, [f"unreadable_result:{error}"]
    decision = payload.get("decision", {})
    configuration = payload.get("configuration", {})
    required = (
        "all_hard_absorption_balances_passed",
        "mura_hard_balance_passed",
        "first_law_hard_balance_passed",
        "grain_ids_unchanged",
    )
    failures.extend(key for key in required if not decision.get(key, False))
    if not payload.get("source_tree_clean", False):
        failures.append("source_tree_dirty")
    expected = "adiabatic" if control == "ad" else "strong_bath"
    if configuration.get("thermal_control") != expected:
        failures.append("thermal_control_mismatch")
    bath_rate = float(configuration.get("thermal_bath_rate_s", 0.0))
    if control == "ad" and bath_rate != 0.0:
        failures.append("adiabatic_bath_rate_nonzero")
    if control == "bath" and bath_rate <= 0.0:
        failures.append("strong_bath_rate_not_positive")
    if decision.get("front_transmission_fraction") != 0.25:
        failures.append("front_transmission_changed")
    return not failures, failures


def _command(parent: Path, output: Path, chunk: int) -> list[str]:
    return [
        sys.executable, "-m", RUNNER, str(parent), str(output),
        "--intervals", str(chunk), "--production-rx",
        "--front-transmission-fraction", "0.25",
        "--override-dt", "2e-9", "--override-shear-rate", "8e4",
        "--thermal-control", "inherited",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification-root", type=Path, default=(
        ROOT / "full_model" / "verification"))
    parser.add_argument("--prefix", default="v67_poly_rx_asb_seed19p375")
    parser.add_argument("--first-total", type=int, default=400)
    parser.add_argument("--final-total", type=int, default=800)
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--state", type=Path, default=(
        ROOT / "full_model" / "verification"
        / "v67_poly_rx_asb_long_horizon_sequence.json"))
    args = parser.parse_args()
    plan = build_plan(
        args.prefix, args.first_total, args.final_total, args.chunk)
    state = {
        "schema": "v67-poly-rx-asb-sequence-v1",
        "created_utc": _utc(),
        "updated_utc": _utc(),
        "one_heavy_solver_at_a_time": True,
        "execution_location": "local_only",
        "plan": plan,
        "cases": {},
    }
    if args.state.exists():
        old = json.loads(args.state.read_text())
        state["created_utc"] = old.get("created_utc", state["created_utc"])
        state["cases"] = old.get("cases", {})
    args.state.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.state, state)

    for item in plan:
        name = item["output_name"]
        parent_root = args.verification_root / item["parent_name"]
        parent = parent_root / "checkpoint_latest.npz"
        output = args.verification_root / name
        result = output / "result.json"
        if result.exists():
            valid, failures = validate_result(result, item["control"])
            state["cases"][name] = {
                "status": "PRESERVED_VALID" if valid else "INVALID_EXISTING",
                "failures": failures, "updated_utc": _utc(),
            }
            state["updated_utc"] = _utc()
            _atomic_json(args.state, state)
            if valid:
                continue
            # Never overwrite invalid evidence. Recovery is an explicit
            # diagnosis task, while the independent matched branch may proceed.
            continue
        if (output / "checkpoint_latest.npz").exists():
            state["cases"][name] = {
                "status": "PARTIAL_REQUIRES_RECOVERY",
                "checkpoint": str(output / "checkpoint_latest.npz"),
                "updated_utc": _utc(),
            }
            state["updated_utc"] = _utc()
            _atomic_json(args.state, state)
            continue
        if not parent.exists():
            state["cases"][name] = {
                "status": "BLOCKED_MISSING_PARENT",
                "parent": str(parent), "updated_utc": _utc(),
            }
            state["updated_utc"] = _utc()
            _atomic_json(args.state, state)
            continue
        parent_result = parent_root / "result.json"
        parent_valid, parent_failures = validate_result(
            parent_result, item["control"])
        if not parent_valid:
            state["cases"][name] = {
                "status": "BLOCKED_INVALID_PARENT",
                "parent": str(parent), "failures": parent_failures,
                "updated_utc": _utc(),
            }
            state["updated_utc"] = _utc()
            _atomic_json(args.state, state)
            continue
        output.mkdir(parents=True, exist_ok=True)
        command = _command(parent, output, args.chunk)
        state["cases"][name] = {
            "status": "RUNNING", "command": command,
            "started_utc": _utc(), "parent": str(parent),
        }
        state["updated_utc"] = _utc()
        _atomic_json(args.state, state)
        with (output / "sequence_stdout.log").open("a") as stream:
            completed = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                text=True, check=False)
        valid, failures = validate_result(result, item["control"])
        state["cases"][name].update({
            "status": "PASSED" if completed.returncode == 0 and valid
                      else "FAILED",
            "returncode": completed.returncode,
            "failures": failures,
            "finished_utc": _utc(),
        })
        state["updated_utc"] = _utc()
        _atomic_json(args.state, state)


if __name__ == "__main__":
    main()
