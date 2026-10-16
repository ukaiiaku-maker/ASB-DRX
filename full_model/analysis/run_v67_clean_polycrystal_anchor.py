#!/usr/bin/env python3
"""Build a zero-load polycrystal anchor from the current clean source.

The anchor uses the established V66 three-owner preparation but publishes it
before any production loading interval.  This avoids inheriting a historical
mechanical ledger or inserting an otherwise unnecessary first strain step.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _save_checkpoint
from full_model.analysis.run_v66_coupled_rx_asb import (
    _configuration, _physics, _prepared_state,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--length", type=float, default=4.8e-6)
    parser.add_argument("--interface-width", type=float, default=6e-7)
    parser.add_argument("--temperature", type=float, default=1000.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Supply the complete argument surface used by the inherited constructor.
    args.shear_rate = 0.0
    args.dt = 2.0e-9
    args.steps = 0
    args.thermal_control = "adiabatic"
    args.bath_rate = 0.0
    args.maximum_mechanical_subdivisions = 14
    args.source_mode = "disabled"
    args.source_grain_size = 2.4e-6
    args.sink_mode = "disabled"
    args.sink_activation_enthalpy = 1.4
    args.sink_stress_assisted = False
    args.interface_sink_mode = "disabled"
    args.structural_absorption_mode = "disabled"
    args.plastic_history_mode = "material_continuity"

    physics = _physics(args)
    state, runtime, preparation = _prepared_state(args, physics)
    context = physics[-1]
    initial_volume = (np.sum(state.supports, axis=(1, 2))
                      *context.spacing_m**2
                      *context.represented_thickness_m)
    configuration = _configuration(args)
    configuration.update(
        schema="v67-clean-polycrystal-anchor-v1",
        initial_state="fresh V66 three-owner preparation at zero load",
        target_steps=0)
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=ROOT, text=True).strip())
    provenance = {
        "campaign": "V67",
        "source_commit": commit,
        "source_dirty": dirty,
        "scientific_scope": "fresh zero-load polycrystal anchor",
    }
    checkpoint = args.out/"checkpoint_latest.npz"
    temporary = args.out/"checkpoint_latest.tmp.npz"
    _save_checkpoint(
        temporary, state, runtime, 0, 0.0, initial_volume,
        configuration, provenance)
    os.replace(temporary, checkpoint)
    payload = {
        "schema": "v67-clean-polycrystal-anchor-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "preparation": preparation,
        "provenance": provenance,
        "result": {
            "grain_ids": list(state.grain_ids),
            "owner_core_cells": [
                int(np.count_nonzero(np.asarray(item) > .75))
                for item in state.supports],
            "owner_orientation_span_deg": [
                float(np.degrees(np.ptp(
                    owner.orientation_rad[np.asarray(support) > .75])))
                for support, owner in zip(state.supports, state.owners)],
            "applied_shear_strain": 0.0,
            "physical_time_s": runtime.ledger.physical_time_s,
            "grain_count": len(state.grain_ids),
        },
        "decision": {
            "zero_load_anchor": True,
            "source_tree_clean": not dirty,
            "grain_ids_unchanged_by_publication": True,
            "production_loading_intervals": 0,
        },
        "artifacts": {"checkpoint_latest.npz": _sha(checkpoint)},
    }
    (args.out/"result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], sort_keys=True))
    print(json.dumps(payload["result"], sort_keys=True))


if __name__ == "__main__":
    main()
