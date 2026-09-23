#!/usr/bin/env python3
"""Run one budget-qualified current-source ASB causal pair after V53 closure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from full_model.analysis.run_v52_completion_manager import (
    LogicalLock, atomic_json, digest, process_identity, same_identity, utc,
)


SCHEMA = "asb-drx/v53/mechanism-priority/v1"
COMPLETE = "COMPLETE_AT_ACHIEVED_SCOPE"


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("campaign timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def elapsed_campaign_seconds(created_utc: str) -> float:
    return max((datetime.now(timezone.utc)-parse_utc(created_utc)).total_seconds(), 0.0)


def validate_case_table(path: Path) -> list[dict[str, object]]:
    cases = json.loads(path.read_text())
    if not isinstance(cases, list) or len(cases) != 2:
        raise ValueError("ASB mechanism test requires exactly two cases")
    common = ("T0_K", "strain_rate_s", "particle_radius_um",
              "conductivity_W_m_K")
    if any(any(key not in case for key in common) for case in cases):
        raise ValueError("ASB case is missing a physical parameter")
    if any(cases[0][key] != cases[1][key] for key in common):
        raise ValueError("ASB causal pair does not share physical parameters")
    if cases[0].get("causal_temperature_ablation", "none") != "none":
        raise ValueError("first ASB case must use full thermal feedback")
    if cases[1].get("causal_temperature_ablation") != "freeze_flow":
        raise ValueError("second ASB case must be the flow-temperature control")
    if float(cases[0]["strain_rate_s"]) < 1.0e4:
        raise ValueError("ASB mechanism test is not in the declared high-rate regime")
    if float(cases[0]["conductivity_W_m_K"]) <= 0.0:
        raise ValueError("ASB mechanism test requires positive conductivity")
    return cases


def persist(path: Path, state: dict, status: str | None = None) -> None:
    if status is not None:
        state["state"] = status
    state["updated_utc"] = utc()
    atomic_json(path, state)


def run_logged(command: list[str], cwd: Path, log: Path) -> tuple[int, float]:
    started = time.perf_counter()
    with log.open("a") as stream:
        stream.write("COMMAND "+json.dumps(command)+"\n")
        stream.flush()
        result = subprocess.run(
            command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
            text=True, env={**os.environ, "PYTHONPATH": "src:."})
    return result.returncode, time.perf_counter()-started


def checkpoint_for(case_dir: Path, step: int) -> Path:
    return case_dir/f"drx_v25_restart_{step:06d}.npz"


def finish_with_regression(state: dict, state_path: Path, root: Path,
                           log: Path) -> None:
    persist(state_path, state, "RUNNING_CANONICAL_REGRESSION")
    code, wall = run_logged(
        [sys.executable, "-m", "pytest", "-q", "tests"], root, log)
    state["canonical_regression"] = {
        "returncode": code, "wall_seconds": wall,
        "scope": "tests at the immutable mechanism-manager source",
    }
    if code:
        raise RuntimeError("canonical regression failed")
    state["completed_utc"] = utc()
    persist(state_path, state, COMPLETE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-state", type=Path, required=True)
    parser.add_argument("--campaign-origin-state", type=Path, required=True)
    parser.add_argument("--case-table", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manager-state", type=Path, required=True)
    parser.add_argument("--budget-s", type=float, default=12*3600.0)
    parser.add_argument("--full-pair-estimate-s", type=float, default=2.25*3600.0)
    parser.add_argument("--closure-reserve-s", type=float, default=900.0)
    parser.add_argument("--target-step", type=int, default=2500)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--poll-s", type=float, default=30.0)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if dirty:
        raise RuntimeError("mechanism manager requires an immutable clean source")
    cases = validate_case_table(args.case_table.resolve())
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = args.manager_state.resolve()
    log = output/"v53_mechanism_priority.log"
    lock = LogicalLock(output/".v53-mechanism-priority.lock")
    lock.acquire()
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("schema") != SCHEMA or state.get("source_sha") != source:
            raise RuntimeError("mechanism journal source or schema mismatch")
        state["resume_count"] = int(state.get("resume_count", 0))+1
    else:
        state = {
            "schema": SCHEMA, "created_utc": utc(), "source_sha": source,
            "source_worktree": str(root), "execution_location": "local",
            "hpc3_jobs_submitted": 0, "resume_count": 0, "stages": {},
            "case_table": str(args.case_table.resolve()),
            "case_table_sha256": digest(args.case_table),
        }
    state["manager"] = process_identity(os.getpid())
    persist(state_path, state, "STARTING")
    try:
        while True:
            handoff = json.loads(args.handoff_state.read_text())
            identity = handoff.get("manager", {})
            live = process_identity(identity.get("pid", -1))
            if same_identity(identity, live):
                state["handoff"] = {"live": True, "state": handoff.get("state"),
                                    "manager": identity}
                persist(state_path, state, "WAITING_FOR_V53_HANDOFF")
                time.sleep(args.poll_s)
                continue
            state["handoff"] = {"live": False, "state": handoff.get("state"),
                                "manager": identity}
            if handoff.get("state") != COMPLETE:
                raise RuntimeError(
                    f"V53 handoff ended without valid terminal: {handoff.get('state')}")
            break

        origin = json.loads(args.campaign_origin_state.read_text())
        origin_utc = origin["created_utc"]
        state["campaign_origin_utc"] = origin_utc
        state["campaign_budget_s"] = args.budget_s

        # Two tiny current-source executions verify configuration, checkpoint
        # publication, and accepted-trajectory field availability before any
        # physical-horizon claim is attempted.
        run_root = output/"asb-pair"
        driver = root/"full_model/hpc3/run_v37_conduction_case.py"
        for index, case in enumerate(cases):
            name = f"PREFLIGHT_{index}_{case['id']}"
            record = run_root/str(case["id"])/"v37_conduction_run_record.json"
            if not (record.exists() and json.loads(record.read_text()).get("latest_step", -1) >= 2):
                command = [
                    sys.executable, str(driver), "--case-id", str(index),
                    "--case-table", str(args.case_table.resolve()),
                    "--source-root", str(root), "--expected-source-sha", source,
                    "--run-root", str(run_root), "--grid", str(args.grid),
                    "--target-step", str(args.target_step), "--preflight",
                ]
                persist(state_path, state, "RUNNING_"+name)
                code, wall = run_logged(command, root, log)
                if code:
                    raise RuntimeError(f"{name} failed with return code {code}")
                state["stages"][name] = {"returncode": code, "wall_seconds": wall}
                persist(state_path, state)
            payload = json.loads(record.read_text())
            latest = Path(payload["latest_checkpoint"])
            with np.load(latest, allow_pickle=True) as raw:
                required = {"asb_last_plastic_power_W_m3",
                            "asb_last_heat_production_W_m3", "T"}
                missing = sorted(required-set(raw.files))
            if missing:
                raise RuntimeError(f"{name} lacks accepted-trajectory fields: {missing}")
            state["stages"].setdefault(name, {}).update({
                "classification": "VALID_CURRENT_SOURCE_PREFLIGHT",
                "run_record": str(record), "run_record_sha256": digest(record),
                "latest_checkpoint": str(latest),
                "latest_checkpoint_sha256": digest(latest),
            })
            persist(state_path, state)

        elapsed = elapsed_campaign_seconds(origin_utc)
        remaining = args.budget_s-elapsed
        required = args.full_pair_estimate_s+args.closure_reserve_s
        state["budget_decision"] = {
            "campaign_elapsed_s": elapsed, "remaining_s": remaining,
            "full_pair_estimate_s": args.full_pair_estimate_s,
            "closure_reserve_s": args.closure_reserve_s,
            "full_pair_authorized": remaining >= required,
        }
        persist(state_path, state)
        if remaining < required:
            state["runnable_next_work"] = {
                "classification": "CURRENT_SOURCE_PREFLIGHT_COMPLETE_FULL_PAIR_BUDGET_DEFERRED",
                "command_template": (
                    "python full_model/analysis/run_v53_mechanism_priority.py "
                    "--handoff-state <terminal-handoff.json> "
                    "--campaign-origin-state <bulk-manager.json> "
                    "--case-table full_model/hpc3/v53_asb_mechanism_pair.json "
                    "--output-root <output> --manager-state <journal>"),
                "physical_asb_result_claimed": False,
            }
            finish_with_regression(state, state_path, root, log)
            return

        for index, case in enumerate(cases):
            name = f"PHYSICAL_HORIZON_{index}_{case['id']}"
            case_dir = run_root/str(case["id"])
            target = checkpoint_for(case_dir, args.target_step)
            if not target.exists():
                command = [
                    sys.executable, str(driver), "--case-id", str(index),
                    "--case-table", str(args.case_table.resolve()),
                    "--source-root", str(root), "--expected-source-sha", source,
                    "--run-root", str(run_root), "--grid", str(args.grid),
                    "--target-step", str(args.target_step),
                ]
                persist(state_path, state, "RUNNING_"+name)
                code, wall = run_logged(command, root, log)
                state["stages"][name] = {"returncode": code, "wall_seconds": wall}
                if code:
                    raise RuntimeError(f"{name} failed with return code {code}")
            record = case_dir/"v37_conduction_run_record.json"
            payload = json.loads(record.read_text())
            if not payload.get("terminal") or payload.get("latest_step") != args.target_step:
                raise RuntimeError(f"{name} did not reach its requested horizon")
            state["stages"].setdefault(name, {}).update({
                "classification": "VALID_COMPUTATIONAL_COMPLETION",
                "run_record": str(record), "run_record_sha256": digest(record),
                "checkpoint": str(target), "checkpoint_sha256": digest(target),
            })
            persist(state_path, state)

        result = output/"v53_asb_mechanism_decision.json"
        figure = output/"v53_asb_mechanism_fields.png"
        baseline = checkpoint_for(run_root/str(cases[0]["id"]), args.target_step)
        control_dir = run_root/str(cases[1]["id"])
        control = checkpoint_for(control_dir, args.target_step)
        command = [
            sys.executable,
            str(root/"full_model/analysis/postprocess_v41_thermal_response.py"),
            "--baseline", str(baseline), "--control", str(control),
            "--control-ledger", str(control_dir/"v31_common_mura_asb_ledger.json"),
            "--output", str(result), "--figure", str(figure),
        ]
        persist(state_path, state, "RUNNING_ASB_CLASSIFICATION")
        code, wall = run_logged(command, root, log)
        if code or not result.exists():
            raise RuntimeError("ASB classification failed")
        decision = json.loads(result.read_text())
        state["stages"]["ASB_CLASSIFICATION"] = {
            "classification": decision["classification"],
            "strict_asb_claimed": bool(decision["strict_asb_claimed"]),
            "wall_seconds": wall, "result": str(result),
            "result_sha256": digest(result), "figure": str(figure),
            "figure_sha256": digest(figure),
        }
        finish_with_regression(state, state_path, root, log)
    except Exception as error:
        state["failure"] = f"{type(error).__name__}: {error}"
        state["failed_utc"] = utc()
        persist(state_path, state, "FAILED_CONTROLLER")
        raise
    finally:
        lock.release()


if __name__ == "__main__":
    main()
