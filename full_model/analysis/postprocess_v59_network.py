#!/usr/bin/env python3
"""Outcome-neutral audit of a resolved V59 prepared-grain network trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v58_three_grain_production import (
    _load_checkpoint, network_interfaces,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def network_topology_metrics(state) -> dict:
    labels = np.argmax(state.supports, axis=0)
    interfaces = network_interfaces(state)
    return {
        "declared_grain_count": len(state.grain_ids),
        "represented_dominant_grain_ids": [
            state.grain_ids[index] for index in sorted(np.unique(labels).tolist())],
        "dominant_adjacency_count": len(interfaces),
        "dominant_adjacencies": [[item.grain_a_id, item.grain_b_id]
                                  for item in interfaces],
        "maximum_support_by_grain": [float(np.max(value))
                                     for value in state.supports],
        "resolved_core_count_support_gt_0p8": int(sum(
            np.max(value) > .8 for value in state.supports)),
        "partition_maximum_absolute_error": float(np.max(np.abs(
            np.sum(state.supports, axis=0)-1.0))),
    }


def analyze(root: Path) -> dict:
    result_path = root/"result.json"
    if not result_path.is_file():
        raise ValueError("network run is not complete")
    result = json.loads(result_path.read_text())
    checkpoint = Path(result["checkpoint"])
    state, runtime, step, gamma, initial_volume, configuration, provenance = (
        _load_checkpoint(checkpoint))
    history = json.loads((root/"history.json").read_text())
    spacing = float(configuration["length_m"])/int(configuration["n"])
    thickness = float(result["represented_thickness_m"])
    volume = np.sum(state.supports, axis=(1, 2))*spacing**2*thickness
    force_rows = [row for row in history if row.get("front")]
    force_consistent = bool(force_rows and all(row["front"].get(
        "selected_rate_conjugate_to_recorded_force", False) for row in force_rows))
    factors = [row["front"].get("joint_pressure_factor") for row in force_rows]
    factors = [float(value) for value in factors if value is not None]
    topology = network_topology_metrics(state)
    genuine = bool(
        len(state.grain_ids) >= 4
        and topology["resolved_core_count_support_gt_0p8"] >= 4
        and topology["dominant_adjacency_count"] >= 4
        and runtime.ledger.physical_time_s > 0.0
        and runtime.ledger.accepted_events > 0
        and runtime.ledger.rejected_events == 0
        and force_consistent
        and topology["partition_maximum_absolute_error"] <= 1e-12)
    return {
        "schema": "asb-drx-v59-prepared-network-audit-v1",
        "classification": ("VALID_EVOLVED_PREPARED_MULTIGRAIN_NETWORK"
                           if genuine else "INCOMPLETE_OR_INVALID_NETWORK"),
        "genuine_network_trajectory": genuine,
        "run_directory": str(root.resolve()),
        "source_commit": result["source_commit"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest(checkpoint),
        "configuration": configuration,
        "provenance": provenance,
        "step": step, "physical_time_s": runtime.ledger.physical_time_s,
        "applied_engineering_shear": gamma,
        "initial_grain_volume_m3": initial_volume.tolist(),
        "final_grain_volume_m3": volume.tolist(),
        "net_grain_volume_change_m3": (volume-initial_volume).tolist(),
        "fresh_sweep_volume_m3": state.ledger.fresh_sweep_fraction*spacing**2*thickness,
        "revisit_sweep_volume_m3": state.ledger.revisit_sweep_fraction*spacing**2*thickness,
        "runtime_ledger": runtime.ledger.__dict__,
        "topology": topology,
        "force_rate": {
            "all_saved_front_records_conjugate": force_consistent,
            "joint_pressure_factor_range": ([min(factors), max(factors)]
                                             if factors else None),
        },
        "restart_exercised": bool((provenance or {}).get("parent_checkpoint_sha256")),
        "spontaneous_grain_birth": False,
        "claim_limit": (
            "Prepared-grain network evolution only; no intragranular birth or "
            "general polycrystal calibration claim."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    output = args.output or args.run/"network_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: result[key] for key in (
        "classification", "physical_time_s", "topology", "force_rate")},
        indent=2))


if __name__ == "__main__":
    main()
