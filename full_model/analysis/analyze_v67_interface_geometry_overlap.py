#!/usr/bin/env python3
"""Audit whether retained interface reservoirs detach from moving GB support."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.tensorial_nye import spectral_derivatives


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _audit(path, spacing):
    state, _, step, gamma, _, _, provenance = _load_checkpoint(path)
    owner = {grain_id: index for index, grain_id in enumerate(state.grain_ids)}
    rows = []
    for component in state.interfaces:
        left = owner[component.grain_a_id]
        right = owner[component.grain_b_id]
        gx, gy = spectral_derivatives(
            state.supports[right]-state.supports[left], spacing)
        band = np.hypot(gx, gy)
        normalized = band/max(float(np.max(band, initial=0.0)), 1e-300)
        fields = {
            "line": np.sum(
                component.boundary_plus_m2+component.boundary_minus_m2,
                axis=(-2, -1)),
            "burgers_norm": np.linalg.norm(
                component.boundary_burgers_m1, axis=(-2, -1)),
        }
        metrics = {}
        for name, value in fields.items():
            total = float(np.sum(value, dtype=np.longdouble))
            metrics[name] = {
                "fraction_within_normalized_band_gt_0p1": float(np.sum(
                    value[normalized > .1], dtype=np.longdouble)
                    /max(total, 1e-300)),
                "content_weighted_normalized_band": float(np.sum(
                    value*normalized, dtype=np.longdouble)
                    /max(total, 1e-300)),
                "pearson_correlation_with_band": float(np.corrcoef(
                    value.ravel(), normalized.ravel())[0, 1]),
            }
        rows.append({"interface": component.component_id, **metrics})
    return {
        "checkpoint": str(path.resolve()), "checkpoint_sha256": _sha(path),
        "step": int(step), "gamma": float(gamma), "provenance": provenance,
        "interfaces": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spacing-m", type=float, required=True)
    args = parser.parse_args()
    before = _audit(args.before, args.spacing_m)
    after = _audit(args.after, args.spacing_m)
    old = {row["interface"]: row for row in before["interfaces"]}
    new = {row["interface"]: row for row in after["interfaces"]}
    changes = {}
    for edge in old:
        changes[edge] = {}
        for field in ("line", "burgers_norm"):
            a = old[edge][field]["content_weighted_normalized_band"]
            b = new[edge][field]["content_weighted_normalized_band"]
            changes[edge][field] = {
                "absolute_change": b-a,
                "relative_change": (b-a)/max(abs(a), 1e-300),
            }
    minimum_inside = min(
        row[field]["fraction_within_normalized_band_gt_0p1"]
        for row in after["interfaces"] for field in ("line", "burgers_norm"))
    maximum_relative_drop = max(
        max(-entry[field]["relative_change"], 0.0)
        for entry in changes.values() for field in ("line", "burgers_norm"))
    gross_detachment = bool(minimum_inside < .95 or maximum_relative_drop > .05)
    payload = {
        "schema": "v67-interface-geometry-overlap-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "before": before, "after": after, "changes": changes,
        "decision": {
            "minimum_after_fraction_within_boundary_band": minimum_inside,
            "maximum_relative_weighted_overlap_drop": maximum_relative_drop,
            "gross_reservoir_detachment_detected": gross_detachment,
            "classification": (
                "GROSS_INTERFACE_RESERVOIR_DETACHMENT" if gross_detachment
                else "NO_GROSS_INTERFACE_RESERVOIR_DETACHMENT"),
        },
        "claim_boundary": (
            "This overlap audit can detect a gross Eulerian trail but does not "
            "prove a local conservative comoving flux or Mura/disconnection "
            "conjugacy."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
