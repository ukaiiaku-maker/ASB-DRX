#!/usr/bin/env python3
"""Assemble an exact ASB checkpoint continuation without hiding its seam."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def step(path):
    return int(re.search(r"(\d+)$", path.stem).group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--restart", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.restart, allow_pickle=True) as data:
        restart_step = int(data["step"])
        restart_time = float(data["sim_time"])
    name = "drx_v25_restart_asb_diagnostics.csv"
    prefix = pd.read_csv(args.prefix/name)
    continuation = pd.read_csv(args.continuation/name)
    retained_prefix = prefix.loc[prefix["t_us"]*1e-6 <= restart_time*(1+1e-12)]
    combined = pd.concat((retained_prefix, continuation), ignore_index=True)
    combined = combined.sort_values("t_us").drop_duplicates("t_us", keep="last")
    combined.to_csv(args.output/name, index=False)
    linked = {}
    for directory in (args.prefix, args.continuation):
        for source in sorted(directory.glob("drx_v25_restart_*.npz"), key=step):
            target = args.output/source.name
            if target.exists() or target.is_symlink():
                continue
            target.symlink_to(source.resolve())
            linked[source.name] = {
                "source": str(source.resolve()), "sha256": sha256(source)}
    for filename in ("stdout.log", "stderr.log", "drx_v25_summary.png",
                     "drx_v25_diagnostic_audit.png", "drx_v25_potential.png"):
        source = args.continuation/filename
        target = args.output/filename
        if source.exists() and not target.exists():
            target.symlink_to(source.resolve())
    record = {
        "schema": "asb-drx/v27-exact-continuation-assembly/v1",
        "prefix_directory": str(args.prefix.resolve()),
        "continuation_directory": str(args.continuation.resolve()),
        "restart_checkpoint": str(args.restart.resolve()),
        "restart_checkpoint_sha256": sha256(args.restart),
        "restart_step": restart_step, "restart_time_s": restart_time,
        "prefix_rows_total": len(prefix),
        "prefix_rows_retained_through_checkpoint": len(retained_prefix),
        "continuation_rows": len(continuation), "assembled_rows": len(combined),
        "assembled_time_strictly_increasing": bool(np.all(
            np.diff(combined["t_us"].to_numpy(float)) > 0.0)),
        "linked_checkpoints": linked,
    }
    (args.output/"continuation_assembly.json").write_text(
        json.dumps(record, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key: record[key] for key in (
        "restart_step", "assembled_rows", "assembled_time_strictly_increasing")},
        indent=2))


if __name__ == "__main__":
    main()
