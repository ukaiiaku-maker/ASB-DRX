#!/usr/bin/env python3
"""Classify restartable continuation toward one-percent existing-boundary RX."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checkpoint_row(path, grain_id, spacing_m, thickness_m):
    state, runtime, step, gamma, initial, configuration, provenance = (
        _load_checkpoint(path))
    index = state.grain_ids.index(grain_id)
    area = float(np.sum(state.supports[index], dtype=np.longdouble))*spacing_m**2
    initial_area = float(initial[index])/thickness_m
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": _sha(path),
        "step": int(step),
        "gamma": float(gamma),
        "area_m2": area,
        "initial_area_m2": initial_area,
        "growth_percent": 100.0*(area/initial_area-1.0),
    }


def _directional_bias(audit, grain_id):
    edges = {}
    for row in audit["directional_prices"]:
        if not (row["target_owner_grows"] or row["target_owner_shrinks"]):
            continue
        edge = edges.setdefault(row["interface"], {})
        pressure = -float(row["structural_directional_energy_density_Pa"])
        edge["growth_pressure_Pa" if row["target_owner_grows"]
             else "shrink_pressure_Pa"] = pressure
    for edge in edges.values():
        edge["growth_minus_shrink_Pa"] = (
            edge.get("growth_pressure_Pa", 0.0)
            -edge.get("shrink_pressure_Pa", 0.0))
    return {
        "by_interface": edges,
        "net_growth_minus_shrink_Pa": sum(
            edge["growth_minus_shrink_Pa"] for edge in edges.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("bundles", type=Path, nargs="+")
    parser.add_argument("--grain-id", type=int, default=20)
    parser.add_argument("--spacing-m", type=float, required=True)
    parser.add_argument("--represented-thickness-m", type=float, required=True)
    args = parser.parse_args()
    endpoints = [_checkpoint_row(
        args.parent_checkpoint, args.grain_id, args.spacing_m,
        args.represented_thickness_m)]
    bundles = []
    for directory in args.bundles:
        result_path = directory/"result.json"
        checkpoint_path = directory/"checkpoint_latest.npz"
        result = json.loads(result_path.read_text())
        endpoint = _checkpoint_row(
            checkpoint_path, args.grain_id, args.spacing_m,
            args.represented_thickness_m)
        endpoints.append(endpoint)
        decision = result["decision"]
        bundles.append({
            "directory": str(directory.resolve()),
            "result_sha256": _sha(result_path),
            "source_commit": result["source_commit"],
            "source_tree_clean": result["source_tree_clean"],
            "intervals": int(result["configuration"]["intervals"]),
            "first_law_hard_balance_passed": decision[
                "first_law_hard_balance_passed"],
            "mura_hard_balance_passed": decision["mura_hard_balance_passed"],
            "grain_ids_unchanged": decision["grain_ids_unchanged"],
            "area_change_m2": result["owner_area_m2"][str(args.grain_id)][
                "change"],
            "endpoint": endpoint,
            "directional_bias": _directional_bias(
                result["after_directional_audit"], args.grain_id),
        })
    increments = [right["growth_percent"]-left["growth_percent"]
                  for left, right in zip(endpoints[:-1], endpoints[1:])]
    interval_counts = [bundle["intervals"] for bundle in bundles]
    rates = [increment/count for increment, count in zip(
        increments, interval_counts)]
    remaining = 1.0-endpoints[-1]["growth_percent"]
    latest_rate = rates[-1] if rates else 0.0
    projected = (math.ceil(remaining/latest_rate)
                 if remaining > 0.0 and latest_rate > 0.0 else None)
    hard = all(
        bundle["source_tree_clean"]
        and bundle["first_law_hard_balance_passed"]
        and bundle["mura_hard_balance_passed"]
        and bundle["grain_ids_unchanged"] for bundle in bundles)
    monotone = all(value >= 0.0 for value in increments)
    directional = bundles[-1]["directional_bias"][
        "net_growth_minus_shrink_Pa"] if bundles else 0.0
    reached = endpoints[-1]["growth_percent"] >= 1.0
    payload = {
        "schema": "v67-existing-boundary-one-percent-continuation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grain_id": args.grain_id,
        "endpoints": endpoints,
        "bundles": bundles,
        "decision": {
            "all_hard_audits_passed": hard,
            "growth_monotone": monotone,
            "growth_increment_percent_by_bundle": increments,
            "growth_rate_percent_per_interval_by_bundle": rates,
            "latest_growth_percent": endpoints[-1]["growth_percent"],
            "one_percent_reached": reached,
            "latest_net_directional_growth_bias_Pa": directional,
            "physical_reversal_observed": bool(
                not monotone or directional < 0.0),
            "projected_additional_intervals_at_latest_rate": projected,
            "classification": (
                "QUALIFIED_ONE_PERCENT_EXISTING_BOUNDARY_RX" if hard and reached
                else ("PHYSICAL_RX_REVERSAL" if hard and (
                    not monotone or directional < 0.0)
                      else "VALID_CONTINUING_EXISTING_BOUNDARY_RX")),
        },
        "claim_boundary": (
            "Existing-boundary RX/SIBM only. Reaching one percent does not "
            "demonstrate spontaneous intragranular grain birth or DRX inside "
            "a qualified shear band."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
