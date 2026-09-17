#!/usr/bin/env python3
"""Equivalence and timing audit for the V38 bounded-moment acceleration."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from full_model.production.common_front_state import (
    _minimum_change_bounded_moments,
    _reference_minimum_change_bounded_moments,
)


def benchmark_case(n: int, families: int, seed: int):
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    phase = 0.5*(1.0+np.tanh((x-n/2.0)/2.0))
    weights = [np.broadcast_to(1.0-phase, (n, n)).copy(),
               np.broadcast_to(phase, (n, n)).copy(),
               np.zeros((n, n))]
    bounds = [1.0+rng.random((n, n, families)) for _ in range(3)]
    baselines = []
    for bound in bounds:
        vector = rng.normal(size=(n, n, families, 3))
        vector /= np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True),
                             1e-300)
        baselines.append(0.4*bound[..., None]*vector)
    target = sum(weight[..., None, None]*moment
                 for weight, moment in zip(weights, baselines))
    # Only the diffuse interface needs the difficult full-polarization solve.
    stripe = (phase > 1e-5)&(phase < 1.0-1e-5)
    direction = np.array([0.0, 1.0, 0.0])
    full = sum(weight[..., None]*bound
               for weight, bound in zip(weights, bounds))[..., None]*direction
    target[stripe] = full[stripe]
    return baselines, bounds, weights, target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--families", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = benchmark_case(args.grid, args.families, 38)
    timings = {"reference_s": [], "compact_s": []}
    reference = compact = None
    for name, function in (
            ("reference_s", _reference_minimum_change_bounded_moments),
            ("compact_s", _minimum_change_bounded_moments)):
        for _ in range(args.repeats):
            start = time.perf_counter(); value = function(*inputs)
            timings[name].append(time.perf_counter()-start)
        if name == "reference_s": reference = value
        else: compact = value
    maximum_difference = max(float(np.max(np.abs(a-b)))
                             for a, b in zip(reference, compact))
    reference_median = float(np.median(timings["reference_s"]))
    compact_median = float(np.median(timings["compact_s"]))
    result = {
        "schema": "asb-drx/v38/bounded-moment-projection-benchmark/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "grid": args.grid, "families": args.families,
        "diffuse_interface_points": int(np.count_nonzero(
            (inputs[2][1][0] > 1e-5)&(inputs[2][1][0] < 1.0-1e-5))),
        "timings": timings,
        "reference_median_s": reference_median,
        "compact_median_s": compact_median,
        "speedup": reference_median/compact_median,
        "maximum_abs_state_difference": maximum_difference,
        "map_equivalent": maximum_difference <= 2e-12,
        "constraint_role": (
            "minimum-change owner-history projection onto exact common first moment "
            "and per-owner |kappa|<=rho balls; target is never reconstructed from Nye"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
