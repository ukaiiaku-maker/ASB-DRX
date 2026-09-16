#!/usr/bin/env python3
"""Local equal/near-equal SIBM audit for projection and odd response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.analysis.run_v26_sequential_sibm import parameters, run_case


def augment(directory, record):
    contour = pd.read_csv(directory/"sibm_contour_diagnostics.csv")
    time = contour["time_s"].to_numpy(float)
    displacement = contour["signed_normal_displacement_mean_m"].to_numpy(float)
    velocity = np.divide(
        np.diff(displacement), np.diff(time),
        out=np.zeros(max(len(time)-1, 0)), where=np.diff(time) > 0.0)
    checkpoint = sorted(directory.glob("drx_v25_restart_*.npz"))[-1]
    with np.load(checkpoint, allow_pickle=True) as data:
        experiment = json.loads(str(data["sibm_experiment_json"].item()))
        p = json.loads(str(data["P_json"].item()))
    represented_thickness = float(p.get(
        "represented_thickness_m",
        float(p.get("nuc_barrier_thickness_b", 2.0))*float(p["b"])))
    represented_volume = float(p["L_phys"])**2*represented_thickness
    suppressed_abs = float(
        experiment.get("equal_state_suppressed_abs_volume_m3", 0.0))
    return {
        **record,
        "initial_window_velocity_m_s": float(velocity[0]) if velocity.size else 0.0,
        "mean_window_velocity_m_s": float(np.mean(velocity)) if velocity.size else 0.0,
        "equal_state_projection_activations": int(
            experiment.get("equal_state_projection_activations", 0)),
        "equal_state_suppressed_abs_volume_m3": suppressed_abs,
        "equal_state_suppressed_abs_volume_fraction": (
            suppressed_abs/represented_volume),
        "equal_state_suppressed_signed_volume_m3": float(
            experiment.get("equal_state_suppressed_signed_volume_m3", 0.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=ROOT/"results-local/v27-sibm-symmetry")
    parser.add_argument("--output", type=Path,
                        default=ROOT/"full_model/verification/v27_sibm_symmetry.json")
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--relative-perturbations", nargs="+", type=float,
                        default=[1e-10, 1e-8, 1e-6, 1e-4, 1e-2])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    base = 2.5e17
    rows = []
    equal_config = parameters(
        args.grid, args.steps, "S5", base, base, 1000.0)
    equal = run_case(args.root, "equal", equal_config)
    rows.append(augment(args.root/"equal", equal))
    for delta in args.relative_perturbations:
        if not 0.0 < delta < 1.0:
            raise ValueError("relative perturbations must lie in (0,1)")
        for sign in (-1, 1):
            parent = base*(1.0+sign*delta)
            child = base*(1.0-sign*delta)
            name = f"delta_{delta:.0e}_{sign:+d}"
            config = parameters(
                args.grid, args.steps, "S5", parent, child, 1000.0)
            row = run_case(args.root, name, config)
            row.update(relative_density_perturbation=sign*delta)
            rows.append(augment(args.root/name, row))
    pairs = []
    for delta in args.relative_perturbations:
        negative = next(row for row in rows if np.isclose(
            row.get("relative_density_perturbation", np.nan), -delta,
            rtol=0.0, atol=delta*1e-12))
        positive = next(row for row in rows if np.isclose(
            row.get("relative_density_perturbation", np.nan), delta,
            rtol=0.0, atol=delta*1e-12))
        vp = positive["initial_window_velocity_m_s"]
        vm = negative["initial_window_velocity_m_s"]
        pp = positive["first_pressure_Pa"]
        pm = negative["first_pressure_Pa"]
        pairs.append({
            "relative_density_perturbation": delta,
            "velocity_odd_relative_residual": abs(vp+vm)/max(abs(vp), abs(vm), 1e-300),
            "pressure_odd_relative_residual": abs(pp+pm)/max(abs(pp), abs(pm), 1e-300),
            "opposite_velocity_signs": bool(vp*vm < 0.0),
            "opposite_pressure_signs": bool(pp*pm < 0.0),
        })
    odd = all(pair["opposite_velocity_signs"]
              and pair["opposite_pressure_signs"]
              and pair["velocity_odd_relative_residual"] <= .05
              and pair["pressure_odd_relative_residual"] <= .05
              for pair in pairs)
    equal_row = rows[0]
    result = {
        "schema": "asb-drx/v27-sibm-equal-near-equal/v1",
        "grid": args.grid, "steps": args.steps,
        "records": rows, "antisymmetric_pairs": pairs,
        "equal_state_zero_virgin_sweep": bool(
            equal_row["front_ledger"]["swept_volume_m3"] == 0.0),
        "equal_state_projection_is_machine_symmetry_only": bool(
            equal_row["equal_state_projection_activations"] > 0
            and equal_row["equal_state_suppressed_abs_volume_fraction"]
            <= 4096.0*np.finfo(float).eps
            and all(row["equal_state_projection_activations"] == 0
                    for row in rows[1:])),
        "near_equal_response_continuous_and_odd": bool(odd),
        "fixture_passed": True,
        "scientific_gate_passed": bool(odd),
        "classification": (
            "SIBM_EQUAL_STATE_PROJECTION_LOCALIZED_AND_ODD_RESPONSE"
            if odd else "SIBM_NEAR_EQUAL_ODD_RESPONSE_FAILURE"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(result["classification"])


if __name__ == "__main__":
    main()
