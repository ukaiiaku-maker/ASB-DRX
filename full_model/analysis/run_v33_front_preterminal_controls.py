#!/usr/bin/env python3
"""Run a bounded, restart-based control set around the n128 front terminal."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT/"full_model"/"production"


CASES = {
    "baseline": {"nSteps": 1},
    "half_dt_matched": {"nSteps": 2, "dt_base": 5e-8},
    "quarter_dt_matched": {"nSteps": 4, "dt_base": 2.5e-8},
    "half_capillarity": {"nSteps": 1, "kappa_eta": 2.5e-7},
    "double_capillarity": {"nSteps": 1, "kappa_eta": 1.0e-6},
    "half_bulk_barrier": {"nSteps": 1, "W_eta": 2.5e6},
    "double_bulk_barrier": {"nSteps": 1, "W_eta": 1.0e7},
    "stored_energy_off": {"nSteps": 1,
                          "stored_energy_coupling_mode": "disabled"},
    "phase_frozen": {"nSteps": 1, "freeze_kwc_eta": True},
}


def _step(path):
    with np.load(path, allow_pickle=True) as data:
        return int(data["step"])


def _run(case_id, overrides, *, source, restart, root):
    out = root/case_id
    out.mkdir(parents=True, exist_ok=True)
    parameters = json.loads(source.read_text())
    parameters.update(
        restart_file=str(restart), restart_reset_clock=False,
        restart_prefix="v33_front_restart", restart_interval=1,
        restart_wallclock_interval_s=840.0, save_interval=100000,
        plot_interval=100000, save_main_panels=False,
        save_signed_panels=False, write_field_npz=False,
        moving_front_support_component_reconnection=True,
        moving_front_topology_backtracking_enabled=True,
        diagnostic_write_front_terminal_trial=True,
        **overrides)
    (out/"parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True)+"\n")
    environment = dict(
        os.environ, DRX_OUTDIR=str(out), MPLBACKEND="Agg",
        OMP_NUM_THREADS="1",
        DRX_PARAMS=json.dumps(parameters, separators=(",", ":")))
    with (out/"stdout.log").open("w") as stdout, \
            (out/"stderr.log").open("w") as stderr:
        completed = subprocess.run(
            [sys.executable, "drx_full_v34_recovery.py"], cwd=PRODUCTION,
            env=environment, stdout=stdout, stderr=stderr, check=False)
    checkpoints = sorted(out.glob("v33_front_restart_*.npz"), key=_step)
    terminals = sorted(out.glob("sibm_terminal_event.json"))
    record = {
        "case": case_id, "overrides": overrides,
        "returncode": completed.returncode,
        "terminal": bool(terminals),
        "terminal_classification": (
            json.loads(terminals[0].read_text())["classification"]
            if terminals else None),
        "final_step": _step(checkpoints[-1]) if checkpoints else None,
        "checkpoint": str(checkpoints[-1]) if checkpoints else None,
    }
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(
            _run, case_id, overrides, source=args.parameters,
            restart=args.restart, root=args.root): case_id
            for case_id, overrides in CASES.items()}
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: item["case"])
    result = {
        "schema": "asb-drx/v33-front-preterminal-controls/v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"),
        "source_checkpoint": str(args.restart),
        "source_step": _step(args.restart),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(records, indent=2))
    return int(any(item["returncode"] for item in records))


if __name__ == "__main__":
    raise SystemExit(main())
