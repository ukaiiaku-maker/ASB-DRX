#!/usr/bin/env python3
"""Decision-grade diagnostics for V40 current-source one-grain paths."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.postprocess_v22_long_wall import axial_frank_bilby_audit
from full_model.analysis.postprocess_v37_mura_organization import (
    checkpoint_scoped_history, last_checkpoint, summarize,
)
from full_model.analysis.run_v30_mura_tier_b1_case import create_case, load_checkpoint
from full_model.analysis.run_v37_conduction_localization import field_metrics
from full_model.production.density_state_map import derived_density_fields
from full_model.production.wall_topology_supply import reservoir_nye_m1


FRANK_BILBY_RELATIVE_TOLERANCE = 0.20


def qualified_lagb_candidate(row: dict) -> bool:
    """Require persistence, an angle-bearing section, and independent closure."""
    frank = row["independent_frank_bilby"]
    return bool(
        row["persistent_last_three_records"]
        and frank["candidate_wall_present"]
        and frank["relative_residual"] <= FRANK_BILBY_RELATIVE_TOLERANCE)


def diagnose(directory: Path) -> dict:
    summary = summarize(directory)
    config = json.loads((directory/"case_config.json").read_text())
    (_, _, _, systems, topologies, _, _, _, spacing) = create_case(
        config["grid"], config["condition"], config["seed"],
        config["length_m"])
    checkpoint = last_checkpoint(directory)
    state, metadata = load_checkpoint(checkpoint, systems, topologies)
    nye = reservoir_nye_m1(
        state.reservoir_alignment, systems, state.common.orientation_rad,
        topologies)["total"]
    nye_norm = np.linalg.norm(nye, axis=(-2, -1))
    fields = derived_density_fields(state.density, topologies)
    orientation_deg = np.rad2deg(
        state.common.orientation_rad-np.mean(state.common.orientation_rad))
    history = [json.loads(line) for line in (
        directory/"history.jsonl").read_text().splitlines() if line.strip()]
    history = checkpoint_scoped_history(history, metadata)
    persistent = bool(len(history) >= 3 and all(
        row["orientation_span_deg"] >= 1.0
        and row["ordered_fraction_wall_local"] > 0.0
        for row in history[-3:]))
    frank = axial_frank_bilby_audit(
        state.common.orientation_rad, nye, spacing)
    wall_metrics = field_metrics(fields["rho_wall_m2"], spacing)
    return {
        "summary": summary,
        "checkpoint_metadata": metadata,
        "orientation": {
            "span_deg": float(np.ptp(orientation_deg)),
            "p95_minus_p05_deg": float(
                np.percentile(orientation_deg, 95)
                -np.percentile(orientation_deg, 5)),
            "gradient_rms_deg_m": float(np.sqrt(np.mean(sum(
                value*value for value in np.gradient(
                    orientation_deg, spacing))))),
        },
        "nye": {
            "rms_m1": float(np.sqrt(np.mean(nye_norm*nye_norm))),
            "maximum_m1": float(np.max(nye_norm)),
        },
        "independent_frank_bilby": frank,
        "frank_bilby_relative_tolerance": FRANK_BILBY_RELATIVE_TOLERANCE,
        "frank_bilby_closed": bool(
            frank["relative_residual"] <= FRANK_BILBY_RELATIVE_TOLERANCE),
        "wall_geometry": wall_metrics,
        "persistent_last_three_records": persistent,
        "donor_capacity": summary["integrated_inventory_m_per_m_thickness"],
        "reaction_exposure": metadata.get("cumulative_ledger", {}),
        "unloading_release_test": (
            "TRIGGERED_PENDING" if persistent and frank["candidate_wall_present"]
            else "NOT_TRIGGERED_NO_QUALIFIED_BOUNDARY"),
        "fields": {
            "orientation_deg": orientation_deg,
            "nye_norm_m1": nye_norm,
            "wall_total_m2": fields["rho_wall_m2"],
            "wall_ordered_m2": fields["rho_wall_ordered_m2"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    diagnosed = [diagnose(path) for path in args.case_dir]
    serial = []
    for row in diagnosed:
        serial.append({key: value for key, value in row.items()
                       if key != "fields"})
    hard = all(row["summary"]["hard_invariants_passed"] for row in diagnosed)
    candidates = [row for row in diagnosed if qualified_lagb_candidate(row)]
    result = {
        "schema": "asb-drx/v40/current-one-grain-decision/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "fixture_passed": hard,
        "scientific_gate_passed": bool(candidates),
        "classification": (
            "PERSISTENT_LAGB_CANDIDATE_REQUIRES_RELEASE" if candidates else
            "NO_CURRENT_SOURCE_PHYSICAL_LAGB_PRECURSOR_AT_EXPOSURE"),
        "cases": serial,
        "phase_or_grain_allocation_present": False,
        "drx_claimed": False,
        "claim_boundary": (
            "One-grain signed-dislocation organization only; a global "
            "orientation span or ordered fraction alone is not a LAGB."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    figure, axes = plt.subplots(
        len(diagnosed), 4, figsize=(14, 3.4*len(diagnosed)), squeeze=False,
        constrained_layout=True)
    for row_index, row in enumerate(diagnosed):
        condition = row["summary"]["condition"]
        topology = ("on" if row["summary"]["topology_route_enabled"]
                    else "off")
        panels = (
            (row["fields"]["wall_total_m2"], "wall total", "viridis"),
            (row["fields"]["wall_ordered_m2"], "wall ordered", "viridis"),
            (row["fields"]["nye_norm_m1"], "Nye norm", "magma"),
            (row["fields"]["orientation_deg"], "orientation (deg)", "coolwarm"),
        )
        for axis, (field, title, cmap) in zip(axes[row_index], panels):
            image = axis.imshow(field.T, origin="lower", cmap=cmap)
            axis.set_title(f"{condition}, topology {topology}: {title}",
                           fontsize=9)
            axis.set_xticks([]); axis.set_yticks([])
            figure.colorbar(image, ax=axis, shrink=.72)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=180); plt.close(figure)


if __name__ == "__main__":
    main()
