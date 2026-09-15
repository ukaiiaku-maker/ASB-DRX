#!/usr/bin/env python3
"""Replay a frozen V19 branch with regular exact spatial checkpoints.

The source parameters are read from the final V19 checkpoint rather than
reconstructed by hand.  Output-only controls are changed, and the resulting
parameter differences are recorded beside the replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


OUTPUT_OVERRIDES = {
    "diag_interval": 100,
    # Restart emission is nested inside the driver's save cadence.  Field and
    # plot writes remain disabled below, so this changes checkpoint cadence only.
    "save_interval": 100,
    "plot_interval": 100_000_000,
    "restart_interval": 100,
    "restart_wallclock_interval_s": 1.0e30,
    "write_field_npz": False,
    "save_main_panels": False,
    "save_signed_panels": False,
    "diag_csv_name": "diagnostics.csv",
    "v20_trajectory_diagnostics": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    driver = args.driver.resolve()
    with np.load(checkpoint, allow_pickle=True) as data:
        parameters = json.loads(str(data["P_json"].item()))
    original = dict(parameters)
    parameters.update(OUTPUT_OVERRIDES)
    parameters["restart_file"] = None
    parameters["restart_reset_clock"] = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["DRX_PARAMS"] = json.dumps(parameters, separators=(",", ":"))
    environment["DRX_OUTDIR"] = str(args.output_dir.resolve())
    completed = subprocess.run(
        [sys.executable, str(driver)], cwd=driver.parent, env=environment,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.output_dir / "replay.log").write_text(completed.stdout)
    changed = {
        key: {"source": original.get(key), "replay": parameters.get(key)}
        for key in parameters if original.get(key) != parameters.get(key)
    }
    record = {
        "schema": "asb-drx/v20-v19-trajectory-replay/v1",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "source_driver": str(driver),
        "source_driver_sha256": sha256(driver),
        "output_only_parameter_changes": changed,
        "return_code": completed.returncode,
    }
    (args.output_dir / "replay_provenance.json").write_text(
        json.dumps(record, indent=2) + "\n")
    if completed.returncode:
        print(completed.stdout[-4000:], file=sys.stderr)
        raise SystemExit(completed.returncode)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
