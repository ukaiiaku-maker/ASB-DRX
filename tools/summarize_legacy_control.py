#!/usr/bin/env python3
"""Extract raw legacy-control observables without assigning physical regimes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def numeric(rows: list[dict[str, str]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def extrema(rows: list[dict[str, str]], key: str) -> dict[str, float | None]:
    values = numeric(rows, key)
    return {
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--control", required=True, choices=("v32", "v33", "v34"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.diagnostics.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("empty diagnostics")

    keys = [
        "step", "eps_pct", "sigma_MPa", "T_mean", "T_max", "rho_mean", "rho_std",
        "n_grains", "topo_components", "grain_hazard_births", "grain_topology_births",
        "nuc_candidate_active", "nuc_candidate_new", "nuc_candidate_promotable",
        "nuc_candidate_age_max", "asb_T_std", "asb_T_range", "asb_rho_hot_over_cold",
        "asb_corr_T_logrho", "asb_qdot_top5_frac", "asb_gdot_top5_frac",
        "asb_band_anisotropy_T", "asb_band_angle_deg", "asb_band_alignment_to_slip_deg",
    ]
    report = {
        "schema": "asb-drx-legacy-control-summary/v1",
        "control": args.control,
        "diagnostics_path": str(args.diagnostics.resolve()),
        "diagnostics_sha256": digest(args.diagnostics),
        "row_count": len(rows),
        "raw_observables": {key: extrema(rows, key) for key in keys},
        "interpretation": {
            "allocated_label_count_is_physical_grain_count": False,
            "hazard_birth_is_physical_grain": False,
            "candidate_is_physical_grain": False,
            "physical_grain_count": None,
            "physical_grain_status": "not_evaluable_from_legacy_diagnostics_missing_finite_support_purity_persistence_growth_and_provenance",
            "asb_status": "not_evaluable_without_sustained_localization_and_mesh_converged_band_width",
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
