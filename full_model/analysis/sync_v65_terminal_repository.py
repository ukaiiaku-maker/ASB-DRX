#!/usr/bin/env python3
"""Synchronize terminal V65 production evidence into repository records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


TERMINAL_HEADING = "## Terminal production state"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def synchronize(
    manifest: dict, result: dict, latest: dict, identity: dict
) -> tuple[dict, str]:
    if result.get("execution_status") != "TARGET_REACHED":
        raise ValueError("production result has not reached its target")
    checkpoint = Path(result["checkpoint"])
    if not checkpoint.is_file() or digest(checkpoint) != result.get(
        "checkpoint_sha256"
    ):
        raise ValueError("terminal checkpoint bytes do not match result")
    if latest.get("checkpoint_sha256") != result["checkpoint_sha256"]:
        raise ValueError("live classifier has not consumed terminal checkpoint")
    if identity.get("status") != "COMPLETE":
        raise ValueError("component identity monitor is not complete")
    target = float(result["target_time_s"])
    final = float(result["final_time_s"])
    if abs(final-target) > 1e-18:
        raise ValueError("terminal physical time does not match target")

    updated = json.loads(json.dumps(manifest))
    production = updated["active_production"]
    production["latest_durable_checkpoint"] = {
        "adaptive_interval": int(latest["adaptive_interval"]),
        "physical_time_s": float(latest["physical_time_s"]),
        "sha256": latest["checkpoint_sha256"],
    }
    production["terminal_checkpoint"] = str(checkpoint.resolve())
    production["terminal_checkpoint_sha256"] = result["checkpoint_sha256"]
    production["final_applied_shear_strain"] = float(result["final_load"])
    production["status"] = "TARGET_REACHED"
    updated["live_checkpoint_monitor"]["latest_classified_interval"] = int(
        latest["adaptive_interval"]
    )
    updated["live_checkpoint_monitor"]["status"] = "COMPLETE"
    identity_record = updated["component_identity_monitor"]
    identity_record["verified_completed_pair_count"] = int(
        identity["completed_pair_count"]
    )
    identity_record["latest_verified_pair"] = identity.get(
        "latest_completed_pair"
    )
    identity_record["status"] = "COMPLETE"
    updated["execution_status"] = "TARGET_REACHED"
    updated["scientific_status"] = "SEE_FINAL_DECISION_RECORD"
    updated["autonomous_finalizer"]["status"] = (
        "POSTPROCESSING_OR_COMPLETE_SEE_FINALIZER_STATUS"
    )
    updated["autonomous_closure"]["status"] = (
        "WAITING_OR_COMPLETE_SEE_CLOSURE_STATUS"
    )
    updated["terminal_repository_sync"] = {
        "status": "SYNCHRONIZED_SEE_REPOSITORY_SYNC_STATUS",
        "terminal_checkpoint_sha256": result["checkpoint_sha256"],
        "terminal_interval": int(latest["adaptive_interval"]),
        "trajectory_modified": False,
    }

    localization = latest["localization"]
    grain = latest["grain_20"]
    paragraph = f"""{TERMINAL_HEADING}

Production reached its exact target at {final:.12e} s and engineering shear
{float(result['final_load']):.12g}. The terminal checkpoint is adaptive interval
{int(latest['adaptive_interval'])}, SHA-256
`{result['checkpoint_sha256']}`. Grain 20 retains
{100.0*float(grain['volume_growth_fraction']):.6g}% net growth and
{float(grain['equivalent_contour_displacement_cells']):.6g} cell displacement.
The accepted-state stress is {float(localization['stress_Pa'])/1e9:.6g} GPa,
peak-minus-mean temperature is
{float(localization['temperature_peak_minus_mean_K']):.6g} K, power
participation is {float(localization['power_participation']):.6g}, and
heat/power overlap is {float(localization['heat_power_overlap']):.6g}. The live
strict-ASB snapshot is `{str(bool(localization['strict_snapshot'])).lower()}`;
the final cumulative classifier and tests remain post-processing work.
"""
    return updated, paragraph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--execution-record", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--latest-classification", required=True, type=Path)
    parser.add_argument("--identity-status", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = json.loads(args.result.read_text())
    latest = json.loads(args.latest_classification.read_text())
    identity = json.loads(args.identity_status.read_text())
    updated, terminal = synchronize(manifest, result, latest, identity)
    document = args.execution_record.read_text()
    if TERMINAL_HEADING in document:
        document = document.split(TERMINAL_HEADING, 1)[0].rstrip()+"\n\n"
    else:
        document = document.rstrip()+"\n\n"
    atomic_text(args.manifest, json.dumps(updated, indent=2)+"\n")
    atomic_text(args.execution_record, document+terminal)
    print(json.dumps({
        "terminal_interval": latest["adaptive_interval"],
        "terminal_checkpoint_sha256": result["checkpoint_sha256"],
    }))


if __name__ == "__main__":
    main()
