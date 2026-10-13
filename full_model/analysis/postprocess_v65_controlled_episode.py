#!/usr/bin/env python3
"""Outcome-neutral DRX/ASB reduction for V64/V65 adaptive trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from full_model.analysis.postprocess_v59_physical_asb import (
    PhysicalASBCriteria, _checkpoint_map, _history_map,
    _sampling_segment_metadata, checkpoint_snapshot,
    classify_physical_episode,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = _checkpoint_map(args.run)
    histories = _history_map(args.run)
    segments = _sampling_segment_metadata(args.run)
    rows = []
    components = {}
    for step, checkpoint in sorted(checkpoints.items()):
        pre_front = histories.get(step)
        row, component = checkpoint_snapshot(checkpoint, pre_front)
        segment = next(
            (str(directory.resolve()) for directory in args.run
             if checkpoint.parent.resolve() == directory.resolve()),
            str(checkpoint.parent.resolve()))
        metadata = segments.get(segment, {})
        row.update({
            "sampling_segment_id": segment,
            "declared_sampling_cadence_s": metadata.get(
                "declared_sampling_cadence_s"),
            "source_seam_verified": False,
        })
        rows.append(row); components[step] = component
    for index, row in enumerate(rows):
        if index == 0:
            continue
        previous = rows[index-1]
        row["source_seam_verified"] = bool(
            row["sampling_segment_id"] == previous["sampling_segment_id"]
            or row.get("parent_checkpoint_sha256")
            == previous.get("checkpoint_sha256"))
    episode = classify_physical_episode(
        rows, components, PhysicalASBCriteria(),
        matched_control_available=False, refinement_passed=False,
        qualifying_operator_id="local_realized_event")
    scalar_history = [row for step, row in sorted(histories.items())]
    peak = (None if not scalar_history else
            max(scalar_history, key=lambda row: abs(row["shear_stress_Pa"])))
    final = None if not scalar_history else scalar_history[-1]
    scalar_summary = {
        "record_count": len(scalar_history),
        "peak_shear_stress_Pa": (
            None if peak is None else peak["shear_stress_Pa"]),
        "peak_physical_time_s": (
            None if peak is None else peak["physical_time_s"]),
        "final_shear_stress_Pa": (
            None if final is None else final["shear_stress_Pa"]),
        "final_temperature_mean_K": (
            None if final is None else final["temperature_mean_K"]),
        "final_temperature_contrast_K": (
            None if final is None else final["temperature_contrast_K"]),
        "post_peak_softening_fraction": (
            None if peak is None else
            (abs(peak["shear_stress_Pa"])-abs(final["shear_stress_Pa"]))
            /max(abs(peak["shear_stress_Pa"]), 1e-300)),
        "semantics": (
            "dense accepted-time scalar history; spatial conjunction and "
            "persistence are evaluated only at raw checkpoint times"),
    }
    payload = {
        "schema": "asb-drx-v65-controlled-episode-v1",
        "runs": [str(path.resolve()) for path in args.run],
        "checkpoint_count": len(rows),
        "scalar_history": scalar_summary,
        "physical_episode": episode,
        "rows": rows,
        "strict_asb": bool(episode["inherited_strict_asb"]),
        "strict_asb_missing_requirements": episode[
            "missing_or_failed_requirements"],
        "claim_scope": (
            "prepared four-grain existing-boundary trajectory; no spontaneous "
            "grain-birth claim and no all-history matched-temperature control"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output),
                      "checkpoint_count": len(rows),
                      "strict_asb": payload["strict_asb"]}))


if __name__ == "__main__":
    main()
