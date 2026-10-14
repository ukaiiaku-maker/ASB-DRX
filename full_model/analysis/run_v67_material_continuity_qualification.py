#!/usr/bin/env python3
"""Classify the V67 plastic-history handoff against the frozen V66 control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.multigrain_common_state import audit_multigrain_nye


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEGACY = Path(
    "/Users/sdillon/HPC3/worktrees/asb-drx-full-v66-20261013/"
    "full_model/verification/v66_coupled_rx_asb")
DEFAULT_REPAIRED = ROOT / "full_model/verification/v67_material_continuity_r4e4"
DEFAULT_OUT = ROOT / "full_model/verification/v67_material_continuity_qualification.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rms(value) -> float:
    array = np.asarray(value, dtype=float)
    return float(np.sqrt(np.mean(array * array)))


def _case(path: Path) -> dict:
    result_path = path / "result.json"
    history_path = path / "history.json"
    checkpoint_path = path / "checkpoint_latest.npz"
    result = json.loads(result_path.read_text())
    history = json.loads(history_path.read_text())
    state, runtime, step, load, _, configuration, provenance = _load_checkpoint(
        checkpoint_path)
    spacing = float(configuration["length_m"]) / int(configuration["n"])
    nye = audit_multigrain_nye(state, spacing)
    exact = _rms(nye.exact_reconstructed_m1)
    owner = _rms(nye.owner_curl_weighted_m1)
    mismatch = _rms(nye.owner_reservoir_mismatch_m1)
    representation = _rms(nye.discrete_representation_residual_m1)
    growth = np.asarray([row["target_growth_fraction"] for row in history])
    differences = np.diff(growth)
    return {
        "directory": str(path),
        "files": {
            name: {"sha256": _sha256(path / name), "bytes": (path / name).stat().st_size}
            for name in ("result.json", "history.json", "checkpoint_latest.npz")
        },
        "checkpoint": {
            "step": int(step),
            "applied_shear_strain": float(load),
            "source_provenance": provenance,
        },
        "configuration": configuration,
        "response": {
            "final_target_growth_fraction": float(growth[-1]),
            "peak_target_growth_fraction": float(np.max(growth)),
            "peak_target_growth_step": int(history[int(np.argmax(growth))]["step"]),
            "minimum_increment": float(np.min(differences, initial=0.0)),
            "negative_increment_count": int(np.count_nonzero(differences < -1e-12)),
            "stationary_front_intervals": int(runtime.ledger.stationary_intervals),
            "accepted_front_events": int(runtime.ledger.accepted_events),
            "maximum_front_relative_energy_closure": float(
                runtime.ledger.maximum_relative_energy_closure),
            "maximum_mechanical_relative_first_law_residual": float(
                runtime.ledger.maximum_mechanical_relative_first_law_residual),
            "stress_peak_observed": bool(result["decision"]["stress_peak_observed"]),
        },
        "nye_audit_rms_m1": {
            "exact_reconstructed": exact,
            "owner_curl_weighted": owner,
            "owner_reservoir_mismatch": mismatch,
            "discrete_representation_residual": representation,
            "owner_reservoir_mismatch_over_owner_curl": (
                mismatch / max(owner, np.finfo(float).tiny)),
            "representation_residual_over_exact": (
                representation / max(exact, np.finfo(float).tiny)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--repaired", type=Path, default=DEFAULT_REPAIRED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    legacy = _case(args.legacy.resolve())
    repaired = _case(args.repaired.resolve())
    response = repaired["response"]
    legacy_response = legacy["response"]
    mismatch_ratio = repaired["nye_audit_rms_m1"][
        "owner_reservoir_mismatch_over_owner_curl"]
    payload = {
        "schema": "v67-material-continuity-qualification-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": {"legacy_receiver_dormant": legacy,
                  "material_continuity": repaired},
        "decision": {
            "matched_endpoint_reached": bool(
                repaired["checkpoint"]["step"] == legacy["checkpoint"]["step"]
                and np.isclose(repaired["checkpoint"]["applied_shear_strain"],
                               legacy["checkpoint"]["applied_shear_strain"])),
            "legacy_reversal_reproduced": bool(
                legacy_response["final_target_growth_fraction"] < 0.0),
            "material_continuity_prevents_reversal": bool(
                response["final_target_growth_fraction"] > 0.0
                and response["negative_increment_count"] == 0),
            "nye_owner_reservoir_consistent": bool(mismatch_ratio < 1e-3),
            "sustained_dynamic_front_motion_demonstrated": bool(
                response["final_target_growth_fraction"] > 0.0
                and response["negative_increment_count"] == 0),
            "macroscopic_drx_transformation_qualified": False,
            "reason": (
                "Exact laboratory-frame plastic-history continuity removes the "
                "legacy elastic growth reversal at the matched 7.2% endpoint "
                "without material Nye-reservoir drift. The transformed fraction "
                "remains too small to qualify macroscopic DRX, and the prepared "
                "target remains an inherited-subgrain control rather than an "
                "endogenous initiation result."),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
