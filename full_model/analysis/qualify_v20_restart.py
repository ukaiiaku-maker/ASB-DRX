#!/usr/bin/env python3
"""Continuous-versus-segmented restart qualification for V20 state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


STATE_KEYS = (
    "rho", "rp", "rm", "rho_forest", "rho_wall", "rho_wall_plus",
    "rho_wall_minus", "q_wall_v19", "T", "psi_plastic", "E_tot",
    "v20_slip", "v20_beta_p", "v20_alignment_m2", "v20_family_nye_m1",
)


def run(driver, output, parameters):
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["DRX_PARAMS"] = json.dumps(parameters, separators=(",", ":"))
    environment["DRX_OUTDIR"] = str(output)
    completed = subprocess.run(
        [sys.executable, driver.name], cwd=driver.parent, env=environment,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output/"run.log").write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(completed.stdout[-5000:])
    return sorted(output.glob("drx_v25_restart_*.npz"))[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.source_checkpoint, allow_pickle=True) as z:
        base = json.loads(str(z["P_json"].item()))
    base.update(
        v20_tensorial_nye_enabled=True, v20_trajectory_diagnostics=True,
        diag_interval=1000, save_interval=1000, plot_interval=1000,
        restart_interval=1000, restart_wallclock_interval_s=1e30,
        write_field_npz=False, save_main_panels=False, save_signed_panels=False,
        restart_file=None, restart_reset_clock=True)
    root = Path(tempfile.mkdtemp(prefix="v20-restart-"))
    continuous_parameters = dict(base, nSteps=20, restart_interval=19)
    continuous = run(args.driver.resolve(), root/"continuous", continuous_parameters)
    first_parameters = dict(base, nSteps=8, restart_interval=7)
    first = run(args.driver.resolve(), root/"first", first_parameters)
    second_parameters = dict(base, nSteps=12, restart_interval=11,
                             restart_file=str(first), restart_reset_clock=False)
    segmented = run(args.driver.resolve(), root/"segmented", second_parameters)
    checks = {}
    maxima = {}
    with np.load(continuous, allow_pickle=True) as left, np.load(segmented, allow_pickle=True) as right:
        for key in STATE_KEYS:
            checks[key] = bool(np.array_equal(left[key], right[key]))
            maxima[key] = float(np.max(np.abs(left[key]-right[key])))
        grain_count_equal = int(left["Ng"]) == int(right["Ng"]) == 1
        step_equal = int(left["step"]) == int(right["step"]) == 19
        time_equal = float(left["sim_time"]) == float(right["sim_time"])
    result = {
        "schema": "asb-drx/v20-tensorial-restart-qualification/v1",
        "continuous_checkpoint": str(continuous),
        "segmented_checkpoint": str(segmented),
        "array_bitwise_identity": checks,
        "maximum_absolute_difference": maxima,
        "grain_count_remained_one": grain_count_equal,
        "step_identity": step_equal,
        "time_identity": time_equal,
        "fixture_passed": bool(all(checks.values()) and grain_count_equal
                               and step_equal and time_equal),
        "scientific_gate_passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"fixture_passed": result["fixture_passed"],
                      "temporary_runs": str(root)}, indent=2))
    if not result["fixture_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
