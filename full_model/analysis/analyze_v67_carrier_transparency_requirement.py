#!/usr/bin/env python3
"""Infer the carrier-transparency lower bound from two accepted V67 states."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.analysis.run_v66_coupled_rx_asb import _physics


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_transparency(rows: dict[str, dict]) -> str:
    if not rows or not all(row["hard_audits_passed"] for row in rows.values()):
        return "TRANSPARENCY_BOUND_NOT_AUDIT_QUALIFIED"
    required = max(row["minimum_transparency_for_no_net_mobile_loss"]
                   for row in rows.values())
    if required > 1.0:
        return "TRANSPARENCY_ALONE_CANNOT_CLOSE_MOBILE_BALANCE"
    if required >= .90:
        return "NEAR_COMPLETE_TRANSPARENCY_REQUIRED_NOT_PRODUCTION_IDENTIFIED"
    return "PARTIAL_TRANSPARENCY_COULD_CLOSE_MOBILE_BALANCE"


def _load(root: Path) -> dict:
    checkpoint = root/"checkpoint_latest.npz"
    result = json.loads((root/"result.json").read_text())
    state, runtime, step, load, _, config, provenance = _load_checkpoint(
        checkpoint)
    args = SimpleNamespace(
        n=config["n"], length=config["length_m"],
        temperature=config["temperature_K"],
        shear_rate=config["shear_rate_s"], dt=config["dt_s"],
        thermal_control=config["thermal_control"],
        bath_rate=config["thermal_bath_rate_s"])
    thickness = _physics(args)[1]
    spacing = config["length_m"]/config["n"]
    mobile_cell_sum = float(sum(np.sum(
        np.asarray(support)[..., None]
        *(np.asarray(owner.mobile_plus_m2)+np.asarray(owner.mobile_minus_m2)),
        dtype=np.longdouble)
        for support, owner in zip(state.supports, state.owners)))
    decision = result["decision"]
    return {
        "directory": str(root.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "parent_provenance": provenance,
        "step": step,
        "load": load,
        "thermal_control": config["thermal_control"],
        "cell_volume_m3": spacing*spacing*thickness,
        "mobile_line_length_m": mobile_cell_sum*spacing*spacing*thickness,
        "cumulative_lattice_absorbed_line_length_m": (
            runtime.ledger.lattice_absorbed_line_length_m),
        "cumulative_source_created_line_cell_sum_m2": (
            runtime.ledger.source_created_line_cell_sum_m2),
        "hard_audits_passed": bool(
            result["source_tree_clean"]
            and decision["all_hard_absorption_balances_passed"]
            and decision["mura_hard_balance_passed"]
            and decision["first_law_hard_balance_passed"]
            and decision["grain_ids_unchanged"]),
    }


def interval_balance(start: dict, end: dict) -> dict:
    if end["step"] <= start["step"] or end["load"] <= start["load"]:
        raise ValueError("end state must follow start state")
    if end["thermal_control"] != start["thermal_control"]:
        raise ValueError("thermal control changed inside transparency audit")
    if end["cell_volume_m3"] != start["cell_volume_m3"]:
        raise ValueError("represented cell volume changed")
    mobile_loss = start["mobile_line_length_m"]-end["mobile_line_length_m"]
    absorbed = (end["cumulative_lattice_absorbed_line_length_m"]
                -start["cumulative_lattice_absorbed_line_length_m"])
    source = ((end["cumulative_source_created_line_cell_sum_m2"]
               -start["cumulative_source_created_line_cell_sum_m2"])
              *start["cell_volume_m3"])
    required = max(mobile_loss, 0.0)/max(absorbed, np.finfo(float).tiny)
    return {
        "start": start,
        "end": end,
        "hard_audits_passed": (
            start["hard_audits_passed"] and end["hard_audits_passed"]),
        "mobile_line_loss_m": mobile_loss,
        "lattice_absorbed_line_m": absorbed,
        "source_created_line_m": source,
        "mobile_loss_over_absorbed_line": required,
        "source_created_over_absorbed_line": (
            source/max(absorbed, np.finfo(float).tiny)),
        "minimum_transparency_for_no_net_mobile_loss": required,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adiabatic", nargs=2, type=Path, required=True,
                        metavar=("START", "END"))
    parser.add_argument("--bath", nargs=2, type=Path, required=True,
                        metavar=("START", "END"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = {
        "adiabatic": interval_balance(*map(_load, args.adiabatic)),
        "strong_bath": interval_balance(*map(_load, args.bath)),
    }
    classification = classify_transparency(rows)
    payload = {
        "schema": "v67-carrier-transparency-requirement-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "intervals": rows,
        "decision": {
            "classification": classification,
            "production_transmission_kernel_identified": False,
            "coefficient_fitted_or_promoted": False,
        },
        "claim_boundary": (
            "This is a balance-derived lower bound, not a fitted boundary "
            "transmission coefficient. Existing DD evidence does not identify "
            "a signed transfer kernel or memory law, so near-complete "
            "transparency is not promoted to production."),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
