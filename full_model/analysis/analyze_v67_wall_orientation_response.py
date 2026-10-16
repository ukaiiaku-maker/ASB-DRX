#!/usr/bin/env python3
"""Compare matched-rate ordered-wall orientation continuations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.analysis.run_v67_interface_structural_bridge import (
    _args_from_configuration,
)
from full_model.production.common_subgrain_recognition import (
    recognize_common_orientation_plateau,
)
from full_model.production.orientation_wall_compatibility import (
    evaluate_orientation_wall_compatibility,
)


def _total_density(owner):
    return np.sum(
        owner.mobile_plus_m2+owner.mobile_minus_m2
        +owner.forest_plus_m2+owner.forest_minus_m2
        +owner.wall_plus_m2+owner.wall_minus_m2, axis=2)


def _serial_recognition(value):
    return {key: item for key, item in value.items()
            if key not in {"component_mask", "interior_mask", "shell_mask",
                           "frank_bilby_rays"}}


def _case(directory):
    directory = Path(directory)
    result = json.loads((directory/"result.json").read_text())
    state, runtime, step, load, initial_volume, configuration, provenance = (
        _load_checkpoint(directory/"checkpoint_latest.npz"))
    args = _args_from_configuration(configuration)
    args.shear_rate = float(configuration["shear_rate_s"])
    spacing, _, systems, _, _, _, _ = _physics(args)
    owners = {}
    any_qualified = False
    for grain_id, support, owner in zip(
            state.grain_ids, state.supports, state.owners):
        clearance = max(0, int(np.ceil(2.0e-7/spacing)))
        domain = np.asarray(support) > .75
        if clearance:
            domain = ndimage.binary_erosion(domain, iterations=clearance)
        if np.count_nonzero(domain) < 2:
            owners[str(grain_id)] = {
                "qualified": False, "reason": "unresolved_owner_interior",
                "domain_cells": int(np.sum(domain)),
            }
            continue
        compatibility = evaluate_orientation_wall_compatibility(
            owner, systems, domain, spacing)
        recognition = recognize_common_orientation_plateau(
            owner.orientation_rad, owner.family_nye_m1, owner.wall_order,
            _total_density(owner), spacing, domain_mask=domain)
        threshold_scan = {}
        for threshold_deg in (.05, .10, .20, .50, 1.0, 2.0):
            threshold_scan[f"{threshold_deg:g}"] = _serial_recognition(
                recognize_common_orientation_plateau(
                    owner.orientation_rad, owner.family_nye_m1,
                    owner.wall_order, _total_density(owner), spacing,
                    minimum_misorientation_deg=threshold_deg,
                    domain_mask=domain))
        any_qualified |= bool(recognition.get("qualified", False))
        owners[str(grain_id)] = {
            "domain_cells": int(np.sum(domain)),
            "orientation_span_deg": float(np.degrees(np.ptp(
                np.asarray(owner.orientation_rad)[domain]))),
            "wall_order_mean": float(np.mean(
                np.asarray(owner.wall_order)[domain])),
            "compatibility_before_relative_mismatch": (
                compatibility.before_relative_mismatch),
            "compatibility_reconstructed_relative_mismatch": (
                compatibility.reconstructed_relative_mismatch),
            "compatible_reconstructed_span_deg": float(np.degrees(
                compatibility.reconstructed_orientation_span_rad)),
            "recognition": _serial_recognition(recognition),
            "misorientation_threshold_scan_deg": threshold_scan,
        }
    return {
        "source_commit": result["source_commit"],
        "source_tree_clean": result["source_tree_clean"],
        "configuration": result["configuration"],
        "owner_area_m2": result["owner_area_m2"],
        "ledger": result["ledger"],
        "owners": owners,
        "any_strict_subgrain_qualified": any_qualified,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("high_rate", type=Path)
    parser.add_argument("low_rate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    high = _case(args.high_rate)
    low = _case(args.low_rate)
    payload = {
        "schema": "v67-wall-orientation-matched-rate-response-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": {"high_rate": high, "low_rate": low},
        "decision": {
            "same_wall_match_parameterization": bool(
                high["configuration"]["wall_nye_match_coefficient_J_m"]
                ==low["configuration"]["wall_nye_match_coefficient_J_m"]),
            "high_rate_strict_subgrain": high["any_strict_subgrain_qualified"],
            "low_rate_strict_subgrain": low["any_strict_subgrain_qualified"],
            "lower_rate_selectively_qualifies_subgrain": bool(
                low["any_strict_subgrain_qualified"]
                and not high["any_strict_subgrain_qualified"]),
        },
        "claim_boundary": (
            "Common-origin, matched-added-strain rate comparison. Strict "
            "subgrain qualification requires owner-domain enclosure, resolved "
            "misorientation, ordered shell, full-vector Frank--Bilby closure, "
            "and lower-density interior; smooth rotation is not DRX."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
