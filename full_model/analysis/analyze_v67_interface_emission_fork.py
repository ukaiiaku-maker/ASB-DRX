#!/usr/bin/env python3
"""Compare the exact-parent GB baseline with interface carrier emission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from full_model.analysis.analyze_v67_mobile_supply_collapse import load_case


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(parent: dict, baseline: dict, emission: dict) -> dict:
    same_parent = bool(
        baseline["parent_checkpoint_sha256"]
        == emission["parent_checkpoint_sha256"]
        == parent["checkpoint_sha256"])
    hard_valid = bool(
        same_parent and parent["hard_audits_passed"]
        and baseline["hard_audits_passed"]
        and emission["hard_audits_passed"])
    bflow = baseline["flow_exposure_fraction"]
    eflow = emission["flow_exposure_fraction"]
    bmobile = baseline["extensive_cell_inventory_m2"]["mobile"]
    emobile = emission["extensive_cell_inventory_m2"]["mobile"]
    ratios = {
        "emission_to_baseline_flow_exposure": eflow/max(bflow, 1e-300),
        "emission_to_baseline_mobile_inventory": emobile/max(bmobile, 1e-300),
        "emitted_pair_line_to_baseline_mobile": (
            2.0*emission["emitted_pair_increment_cell_sum_m2"]
            /max(bmobile, 1e-300)),
    }
    if not hard_valid:
        outcome = "INTERFACE_EMISSION_FORK_INVALID"
    elif eflow >= .01 and bflow < .01:
        outcome = "INTERFACE_EMISSION_RESTORES_PHYSICAL_FLOW"
    elif (ratios["emission_to_baseline_flow_exposure"] >= 2.0
          and ratios["emission_to_baseline_mobile_inventory"] >= 1.25):
        outcome = "INTERFACE_EMISSION_IMPROVES_SUPPLY_WITHOUT_FLOW_RECOVERY"
    else:
        outcome = "INTERFACE_EMISSION_INSUFFICIENT_TO_PREVENT_FLOW_ARREST"
    return {"same_parent_checkpoint": same_parent, "hard_valid": hard_valid,
            "ratios": ratios, "classification": outcome}


def _child(root: Path, parent_emitted_pairs: float) -> dict:
    result = json.loads((root/"result.json").read_text())
    case = load_case(root)
    emitted = result["ledger"].get(
        "interface_mobile_emitted_pair_cell_sum_m2", 0.0)
    case.update(
        parent_checkpoint_sha256=result["parent_checkpoint_sha256"],
        interface_mobile_emission_mode=result["configuration"].get(
            "interface_mobile_emission_mode", "disabled"),
        emitted_pair_increment_cell_sum_m2=max(
            emitted-parent_emitted_pairs, 0.0))
    return case


def analyze(parent_root: Path, baseline_root: Path, emission_root: Path) -> dict:
    parent_checkpoint = parent_root/"checkpoint_latest.npz"
    parent_result = json.loads((parent_root/"result.json").read_text())
    parent = load_case(parent_root)
    parent.update(checkpoint_sha256=_sha(parent_checkpoint))
    parent_emitted = parent_result["ledger"].get(
        "interface_mobile_emitted_pair_cell_sum_m2", 0.0)
    baseline = _child(baseline_root, parent_emitted)
    emission = _child(emission_root, parent_emitted)
    return {
        "schema": "v67-interface-emission-fork-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "Hall--Petch-activated conservative emission from the explicit "
            "neutral GB reservoir prevents mobile-carrier starvation."),
        "cases": {"parent": parent, "baseline": baseline,
                  "interface_emission": emission},
        "decision": classify(parent, baseline, emission),
        "claim_boundary": (
            "This exact-parent fork adds only the ledgered reservoir-transfer "
            "channel; it does not retune Peierls, Taylor, Hall--Petch, front, "
            "source, or thermal parameters."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parent", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("emission", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    payload = analyze(args.parent, args.baseline, args.emission)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
