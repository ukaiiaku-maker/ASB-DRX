#!/usr/bin/env python3
"""Decompose dual-Nye mismatch from a periodically admissible wall state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.production.nye_consistency import periodic_nye_decomposition
from full_model.production.v24_mechanical_wall import accepted_v24_mechanical_step
from full_model.production.wall_topology_supply import reservoir_nye_m1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (state, driving, _, support, systems, topologies,
     common, extensive, kinetics, dx) = build_case(
        args.grid, periodic_nye_consistent=True)
    records = []
    for step in range(args.steps):
        state, ledger = accepted_v24_mechanical_step(
            state, driving, support, systems, topologies, common, extensive,
            kinetics, 2e-9, topology_route_enabled=False)
        if step < 10 or step in (args.steps//4, args.steps//2, args.steps-1):
            reservoir = reservoir_nye_m1(
                state.reservoir_alignment, systems,
                state.common.orientation_rad, topologies)["total"]
            curl_beta = np.sum(state.common.family_nye_m1, axis=2)
            audit = periodic_nye_decomposition(reservoir, curl_beta, dx)
            records.append({
                "step": step+1,
                **{key: value.tolist() if isinstance(value, np.ndarray) else value
                   for key, value in audit.items()},
            })
    maximum = max(row["total_relative_rms"] for row in records)
    result = {
        "schema": "asb-drx/v28-dual-nye-decomposition/v1",
        "grid": args.grid, "steps": args.steps, "spacing_m": dx,
        "periodic_initial_signed_population_balanced": True,
        "records": records,
        "maximum_dual_nye_relative_rms": maximum,
        "classification": ("FB_TARGET_REACHABLE_WITH_CONSISTENT_CURRENT_STATE"
                           if maximum <= .05 else
                           "DUAL_NYE_STATE_STILL_INCONSISTENT"),
        "fixture_passed": True,
        "scientific_gate_passed": bool(maximum <= .05),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(result["classification"])


if __name__ == "__main__":
    main()
