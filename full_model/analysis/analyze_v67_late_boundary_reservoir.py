#!/usr/bin/env python3
"""Diagnose the explicit interface reservoir at a late RX reversal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.compare_v67_late_source_geography import (
    _component_bias,
)
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.tensorial_nye import spectral_derivatives


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spacing-m", type=float, required=True)
    args = parser.parse_args()
    result_path = args.case/"result.json"
    checkpoint_path = args.case/"checkpoint_latest.npz"
    result = json.loads(result_path.read_text())
    state, runtime, step, gamma, initial, configuration, provenance = (
        _load_checkpoint(checkpoint_path))
    owner_index = {grain_id: index for index, grain_id in enumerate(
        state.grain_ids)}
    rows = []
    normal_sq = tangent_sq = 0.0
    total_line = required_line = removable_line = 0.0
    burgers_m = float(result["configuration"].get(
        "burgers_m", 2.86e-10))
    for component in state.interfaces:
        left = owner_index[component.grain_a_id]
        right = owner_index[component.grain_b_id]
        eta = np.asarray(state.supports[right])-np.asarray(state.supports[left])
        gx, gy = spectral_derivatives(eta, args.spacing_m)
        magnitude = np.sqrt(gx*gx+gy*gy)
        active = magnitude > 1e-12*max(float(np.max(magnitude)), 1.0)
        nx = np.divide(gx, magnitude, out=np.zeros_like(gx), where=active)
        ny = np.divide(gy, magnitude, out=np.zeros_like(gy), where=active)
        normal = np.stack((nx, ny, np.zeros_like(nx)), axis=-1)
        tangent = np.stack((-ny, nx, np.zeros_like(nx)), axis=-1)
        burgers = np.asarray(component.boundary_burgers_m1)
        normal_projection = np.sum(
            burgers*normal[..., None, :], axis=-1)
        tangent_projection = np.sum(
            burgers*tangent[..., None, :], axis=-1)
        n2 = float(np.sum(normal_projection**2, dtype=np.longdouble))
        t2 = float(np.sum(tangent_projection**2, dtype=np.longdouble))
        normal_sq += n2
        tangent_sq += t2
        line_by_row = np.sum(
            component.boundary_plus_m2+component.boundary_minus_m2,
            axis=-1)
        required_by_row = np.linalg.norm(burgers, axis=-1)/burgers_m
        removable_by_row = np.maximum(line_by_row-required_by_row, 0.0)
        line = float(np.sum(line_by_row, dtype=np.longdouble))
        required = float(np.sum(required_by_row, dtype=np.longdouble))
        removable = float(np.sum(removable_by_row, dtype=np.longdouble))
        total_line += line
        required_line += required
        removable_line += removable
        rows.append({
            "interface": component.component_id,
            "normal_burgers_l2_m1": n2**.5,
            "tangential_burgers_l2_m1": t2**.5,
            "normal_fraction_l2": n2**.5/max((n2+t2)**.5, 1e-300),
            "boundary_line_cell_sum_m2": line,
            "burgers_capacity_line_cell_sum_m2": required,
            "vector_neutral_removable_line_cell_sum_m2": removable,
            "vector_neutral_removable_fraction": removable/max(line, 1e-300),
        })
    components = _component_bias(result["after_directional_audit"])
    total = (normal_sq+tangent_sq)**.5
    payload = {
        "schema": "v67-late-boundary-reservoir-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": result["source_commit"],
        "source_tree_clean": result["source_tree_clean"],
        "result_sha256": _sha(result_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "interfaces": rows,
        "directional_component_growth_minus_shrink_Pa": components,
        "decision": {
            "aggregate_normal_burgers_fraction_l2": (
                normal_sq**.5/max(total, 1e-300)),
            "aggregate_tangential_burgers_fraction_l2": (
                tangent_sq**.5/max(total, 1e-300)),
            "defect_storage_favors_growth_Pa": components[
                "defect_storage_J"],
            "boundary_excess_opposes_growth_Pa": components[
                "boundary_excess_J"],
            "net_growth_minus_shrink_Pa": components["helmholtz_J"],
            "boundary_reservoir_limits_growth": bool(
                components["defect_storage_J"] > 0.0
                and components["boundary_excess_J"] < 0.0
                and components["helmholtz_J"] < 0.0),
            "common_axis_absorption_can_address_dominant_content": bool(
                normal_sq >= tangent_sq),
            "vector_neutral_removable_line_fraction": (
                removable_line/max(total_line, 1e-300)),
            "burgers_capacity_line_fraction": (
                required_line/max(total_line, 1e-300)),
        },
        "selected_hypothesis": (
            "First test the existing vector-neutral GB junction reaction, "
            "which removes only scalar line above the exact Burgers-capacity "
            "bound and leaves the tangential Burgers tensor unchanged. If "
            "that cannot restore favorable RX competition, test a "
            "conservative tangential Burgers/disconnection or boundary-"
            "sliding reaction coupled to compatible beta-p/Nye and boundary "
            "geometry; scalar Burgers deletion is not authorized."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
