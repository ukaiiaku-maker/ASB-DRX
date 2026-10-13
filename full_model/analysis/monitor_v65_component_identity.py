#!/usr/bin/env python3
"""Autonomously audit plastic-power component identity in a live V65 run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from full_model.analysis.analyze_v65_power_component_transition import analyze


def checkpoint_interval(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def pending_pairs(
    checkpoints: list[Path], output_dir: Path, start_after_interval: int
) -> list[tuple[Path, Path, Path]]:
    ordered = sorted(checkpoints, key=checkpoint_interval)
    result = []
    for left, right in zip(ordered, ordered[1:]):
        left_interval = checkpoint_interval(left)
        right_interval = checkpoint_interval(right)
        if left_interval < start_after_interval:
            continue
        output = output_dir / (
            f"power_component_transition_{left_interval}_{right_interval}.json"
        )
        if not output.is_file():
            result.append((left, right, output))
    return result


def process_available(
    run: Path, output_dir: Path, start_after_interval: int
) -> list[dict]:
    completed = []
    pairs = pending_pairs(
        list(run.glob("checkpoint_adaptive_*.npz")),
        output_dir,
        start_after_interval,
    )
    for left, right, output in pairs:
        payload = analyze([left, right])
        atomic_json(output, payload)
        completed.append({
            "from_interval": checkpoint_interval(left),
            "to_interval": checkpoint_interval(right),
            "output": str(output.resolve()),
            "classification": payload["decision"]["classification"],
        })
    return completed


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--solver-pid", required=True, type=int)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--start-after-interval", type=int, default=0)
    args = parser.parse_args()
    status_path = args.output_dir / "component_identity_monitor_status.json"
    total = []
    while True:
        total.extend(process_available(
            args.run, args.output_dir, args.start_after_interval
        ))
        alive = process_alive(args.solver_pid)
        atomic_json(status_path, {
            "schema": "asb-drx-v65-component-identity-monitor-v1",
            "status": "RUNNING" if alive else "COMPLETE",
            "solver_pid": args.solver_pid,
            "start_after_interval": args.start_after_interval,
            "completed_pair_count": len(total),
            "latest_completed_pair": total[-1] if total else None,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        if not alive:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
