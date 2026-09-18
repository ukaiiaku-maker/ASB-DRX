#!/usr/bin/env python3
"""Classify a checksum-verified partial rescue of the frozen V37 Mura run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v30_mura_tier_b1_case import create_case, load_checkpoint
from full_model.production.density_state_map import derived_density_fields
from full_model.production.wall_topology_supply import reservoir_nye_m1


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument(
        "--scheduler-terminal", action="store_true",
        help="classify a completed scheduler run rather than a live rescue")
    args = parser.parse_args()
    config = json.loads((args.case_dir/"case_config.json").read_text())
    checkpoints = sorted(args.case_dir.glob("checkpoint_step_*.npz"))
    checkpoint = checkpoints[-1]
    (_, _, _, systems, topologies, _, _, _, spacing) = create_case(
        config["grid"], config["condition"], config["seed"], config["length_m"])
    state, metadata = load_checkpoint(checkpoint, systems, topologies)
    checkpoint_strain = float(metadata["applied_strain"])
    history_all = [json.loads(line) for line in (
        args.case_dir/"history.jsonl").read_text().splitlines() if line.strip()]
    history = [row for row in history_all
               if float(row["applied_strain"]) <= checkpoint_strain+1e-12]
    if not history:
        raise ValueError("rescue has no history at or before its checkpoint")
    hard = all(row["accepted_step_hard_invariant_passed"]
               and not row["post_step_projection_used"]
               and row["minimum_heat_increment_J_m3"] >= 0.0
               for row in history)
    tail = history[-3:]
    legacy_persistent = (len(tail) == 3 and all(
        row["orientation_span_deg"] >= 1.0
        and row["ordered_fraction_wall_local"] > 0.0 for row in tail))
    fields = derived_density_fields(state.density, topologies)
    nye = reservoir_nye_m1(
        state.reservoir_alignment, systems, state.common.orientation_rad,
        topologies)["total"]
    area = spacing**2
    result = {
        "schema": "asb-drx/v39/mura-valid-partial/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": ("VALID_FINAL_RETAINED_CHECKPOINT_WITH_UNCHECKPOINTED_TAIL"
                   if args.scheduler_terminal else "VALID_PARTIAL_ACTIVE_RUN"),
        "scheduler_terminal": bool(args.scheduler_terminal),
        "archive": str(args.archive.resolve()), "archive_sha256": sha256(args.archive),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_source_sha": metadata.get("source_sha"),
        "checkpoint_step": int(metadata["step"]),
        "checkpoint_strain": checkpoint_strain,
        "history_records_at_or_before_checkpoint": len(history),
        "history_ahead_of_checkpoint_excluded": len(history_all)-len(history),
        "hard_invariants_passed": hard,
        "legacy_source_persistent_last_three_records": legacy_persistent,
        "ordered_line_m_per_m_thickness": float(np.sum(
            fields["rho_wall_ordered_m2"], dtype=np.longdouble)*area),
        "tangle_line_m_per_m_thickness": float(np.sum(
            fields["rho_wall_tangle_m2"], dtype=np.longdouble)*area),
        "nye_rms_m1": float(np.sqrt(np.mean(nye*nye))),
        "orientation_span_deg": float(np.rad2deg(
            np.max(state.common.orientation_rad)-np.min(
                state.common.orientation_rad))),
        "classification": (
            "LEGACY_SOURCE_PERSISTENT_ORIENTATION_GRADIENT_WITH_"
            "ORDERED_RESERVOIR_UNQUALIFIED" if hard and legacy_persistent else
            "LEGACY_SOURCE_VALID_PARTIAL_WITHOUT_PERSISTENT_WALL"),
        "lagb_precursor_claimed_current_source": False,
        "phase_support_present": False, "drx_claimed": False,
        "claim_boundary": (
            "frozen V37 kinematics/history; ordered inventory predates V38/V39 "
            "complete-time correction and must be compared at matched state"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    strain = [row["applied_strain"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    axes[0].plot(strain, [row["ordered_fraction_wall_local"] for row in history])
    axes[1].plot(strain, [row["ordered_polarization_wall_local_mean"] for row in history])
    axes[2].plot(strain, [row["orientation_span_deg"] for row in history])
    for axis, label in zip(axes, ("ordered fraction", "polarization", "orientation span (deg)")):
        axis.set(xlabel="applied strain", ylabel=label); axis.grid(alpha=.25)
    figure.tight_layout(); args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=180); plt.close(figure)


if __name__ == "__main__":
    main()
