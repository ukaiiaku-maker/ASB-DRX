#!/usr/bin/env python3
"""Replay a saved front checkpoint with the V34 phase proposal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT/"full_model"/"production"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_step(path):
    with np.load(path, allow_pickle=True) as data:
        return int(data["step"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--grid", type=int)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument(
        "--topology-active-set", choices=("on", "off"), default="on")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    parameters = json.loads(args.parameters.read_text())
    parameters.update(
        restart_file=str(args.restart), restart_reset_clock=False,
        restart_prefix="v34_phase_restart", restart_interval=1,
        restart_wallclock_interval_s=840.0, nSteps=args.steps,
        save_interval=100000, plot_interval=100000,
        save_main_panels=False, save_signed_panels=False,
        write_field_npz=False, diagnostic_write_front_terminal_trial=True,
        diagnostic_write_phase_proposal=True,
        ac_topology_active_set_enabled=(args.topology_active_set == "on"),
        ac_phase_proposal_mode=(
            "legacy_sequential_euler" if args.legacy
            else "adaptive_simplex_imex"))
    if args.grid is not None:
        parameters.update(Nx=args.grid, Ny=args.grid)
    (args.output/"parameters.json").write_text(
        json.dumps(parameters, indent=2, sort_keys=True)+"\n")
    environment = dict(
        os.environ, DRX_OUTDIR=str(args.output), MPLBACKEND="Agg",
        OMP_NUM_THREADS="1",
        DRX_PARAMS=json.dumps(parameters, separators=(",", ":")))
    with (args.output/"stdout.log").open("w") as stdout, \
            (args.output/"stderr.log").open("w") as stderr:
        completed = subprocess.run(
            [sys.executable, "drx_full_v34_recovery.py"], cwd=PRODUCTION,
            env=environment, stdout=stdout, stderr=stderr, check=False)
    terminal_path = args.output/"sibm_terminal_event.json"
    terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else None
    diagnostics_path = args.output/"ac_phase_proposal_diagnostics.jsonl"
    diagnostics = []
    if diagnostics_path.exists():
        diagnostics = [json.loads(line) for line in diagnostics_path.read_text().splitlines()]
    result = {
        "schema": "asb-drx/v34-phase-proposal-replay/v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "case": args.case,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_parameters": str(args.parameters),
        "source_restart": str(args.restart),
        "source_restart_sha256": _sha256(args.restart),
        "source_step": _checkpoint_step(args.restart),
        "requested_steps": args.steps,
        "returncode": completed.returncode,
        "terminal_classification": terminal["classification"] if terminal else None,
        "phase_proposal": diagnostics,
    }
    (args.output/"v34_phase_replay_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
