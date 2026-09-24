#!/usr/bin/env python3
"""Single-owner, block-interleaved V54 ASB causal anchor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SCHEMA = "asb-drx/v54/asb-anchor-manager/v1"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)


def run(command: list[str], cwd: Path, log: Path) -> tuple[int, float]:
    started = time.perf_counter()
    with log.open("a") as stream:
        stream.write("COMMAND "+json.dumps(command)+"\n")
        stream.flush()
        result = subprocess.run(command, cwd=cwd, stdout=stream,
                                stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONPATH": "src:."})
    return result.returncode, time.perf_counter()-started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--case-table", type=Path, required=True)
    parser.add_argument("--prefix-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--targets", type=int, nargs="+",
                        default=[300, 700, 1200, 1800, 2500])
    args = parser.parse_args()
    root = args.source_root.resolve(); output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.case_table.read_text())
    allocation = json.loads(args.allocation.read_text())
    deadline = datetime.fromisoformat(
        allocation["deadline_utc"].replace("Z", "+00:00"))
    if len(cases) != 2 or [c.get("causal_temperature_ablation", "none")
                           for c in cases] != ["none", "freeze_flow"]:
        raise ValueError("V54 requires the exact none/freeze_flow causal pair")
    state_path = output/"v54_asb_anchor_manager.json"
    log = output/"v54_asb_anchor_manager.log"
    state = ({"schema": SCHEMA, "allocation_id": allocation["allocation_id"],
              "source_sha": args.source_sha, "created_utc": utc(),
              "owner": {"pid": os.getpid()}, "targets": args.targets,
              "prefix_checkpoint": str(args.prefix_checkpoint.resolve()),
              "prefix_sha256": digest(args.prefix_checkpoint), "blocks": []}
             if not state_path.exists() else json.loads(state_path.read_text()))
    if state.get("source_sha") != args.source_sha:
        raise RuntimeError("manager source changed across resume")
    if state.get("allocation_id") != allocation["allocation_id"]:
        state.setdefault("continuation_allocations", []).append({
            "allocation_id": allocation["allocation_id"],
            "started_utc": allocation.get("started_utc"),
            "deadline_utc": allocation["deadline_utc"],
            "resumed_utc": utc(),
        })
        state["active_allocation_id"] = allocation["allocation_id"]
    state["targets"] = args.targets
    state["state"] = "RUNNING"; state["updated_utc"] = utc()
    write_json(state_path, state)
    driver = root/"full_model/hpc3/run_v37_conduction_case.py"
    try:
        for target in args.targets:
            for index, case in enumerate(cases):
                if datetime.now(timezone.utc) >= deadline:
                    state["state"] = "DEADLINE_REACHED_WITH_ACCEPTED_PREFIX"
                    write_json(state_path, state); return
                record_path = (output/"asb-pair"/case["id"]/
                               "v37_conduction_run_record.json")
                if record_path.exists():
                    record = json.loads(record_path.read_text())
                    if int(record.get("latest_step", -1)) >= target:
                        continue
                    if "VALIDITY_BOUNDARY" in record.get("terminal_reason", ""):
                        state.setdefault("validity_boundaries", {})[case["id"]] = record
                        continue
                command = [sys.executable, str(driver), "--case-id", str(index),
                           "--case-table", str(args.case_table.resolve()),
                           "--source-root", str(root), "--expected-source-sha",
                           args.source_sha, "--run-root", str(output/"asb-pair"),
                           "--grid", str(args.grid), "--target-step", str(target),
                           "--initial-checkpoint", str(args.prefix_checkpoint.resolve())]
                state["active"] = {"case": case["id"], "target": target,
                                   "started_utc": utc()}
                write_json(state_path, state)
                code, wall = run(command, root, log)
                if code:
                    raise RuntimeError(f"{case['id']} target {target} failed: {code}")
                record = json.loads(record_path.read_text())
                state["blocks"].append({"case": case["id"], "target": target,
                                        "wall_seconds": wall,
                                        "latest_step": record["latest_step"],
                                        "terminal_reason": record["terminal_reason"],
                                        "record_sha256": digest(record_path),
                                        "completed_utc": utc()})
                state.pop("active", None); write_json(state_path, state)
        result = output/"v54_asb_mechanism_decision.json"
        command = [sys.executable,
                   str(root/"full_model/analysis/postprocess_v53_asb_mechanism.py"),
                   "--baseline-dir", str(output/"asb-pair"/cases[0]["id"]),
                   "--control-dir", str(output/"asb-pair"/cases[1]["id"]),
                   "--output", str(result)]
        code, wall = run(command, root, log)
        if code:
            raise RuntimeError(f"postprocessing failed: {code}")
        decision = json.loads(result.read_text())
        state["decision"] = {"path": str(result), "sha256": digest(result),
                             "classification": decision["classification"],
                             "wall_seconds": wall}
        state["state"] = "COMPLETE"; state["completed_utc"] = utc()
        write_json(state_path, state)
    except Exception as error:
        state["state"] = "FAILED_CONTROLLER"
        state["failure"] = f"{type(error).__name__}: {error}"
        state["failed_utc"] = utc(); write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
