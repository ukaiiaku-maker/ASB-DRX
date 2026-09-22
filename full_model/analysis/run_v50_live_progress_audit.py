#!/usr/bin/env python3
"""Read-only audit of the retained immutable V49 physical continuation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_record(pid):
    try:
        fields = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "pid=,lstart=,etime=,%cpu=,%mem=,command="],
            text=True).strip()
    except subprocess.CalledProcessError:
        return {"pid": int(pid), "live": False}
    try:
        cwd_lines = subprocess.check_output(
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            text=True, stderr=subprocess.DEVNULL).splitlines()
        cwd = next((line[1:] for line in cwd_lines if line.startswith("n")), None)
    except (subprocess.CalledProcessError, FileNotFoundError):
        cwd = None
    return {"pid": int(pid), "live": True, "ps_identity": fields,
            "working_directory": cwd}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--target-intervals", type=int, default=52)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = manifest["records"]
    valid = 0; shortening = []
    for record in records:
        clean = True
        for segment in record["segments"]:
            requested = float(segment["requested_duration_s"])
            accepted = float(segment["accepted_duration_s"])
            tolerance = 64*2.220446049250313e-16*max(requested, 1e-300)
            if accepted < requested-tolerance:
                clean = False
                shortening.append({
                    "interval": record["interval"],
                    "requested_duration_s": requested,
                    "accepted_duration_s": accepted,
                })
        if not clean:
            break
        valid += 1
    recent_costs = [sum(float(s["wall_seconds"]) for s in r["segments"])
                    for r in records[max(0, valid-4):valid]]
    cost = statistics.median(recent_costs) if recent_costs else None
    remaining = max(args.target_intervals-valid, 0)
    endpoint = records[valid-1] if valid else None
    process = process_record(args.pid)
    expected_root = str(args.manifest.parent.resolve())
    command_output_match = expected_root in process.get("ps_identity", "")
    if process.get("working_directory"):
        try:
            relative_root = str(Path(expected_root).relative_to(
                Path(process["working_directory"])))
            command_output_match = command_output_match or (
                f"--output-dir {relative_root}" in process.get("ps_identity", ""))
        except ValueError:
            pass
    identity_matches = bool(
        process["live"]
        and command_output_match
        and manifest["source_sha"] == "37065c25e02507da1bc25491574c3ef6fbb01966")
    latest = Path(manifest["latest_checkpoint"])
    payload = {
        "schema": "asb-drx/v50/retained-v49-live-audit/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "audit_is_read_only": True,
        "retained_source_sha": manifest["source_sha"],
        "retained_manifest": str(args.manifest.resolve()),
        "retained_manifest_sha256": digest(args.manifest),
        "process": {**process, "identity_matches_retained_run": identity_matches},
        "run_status": manifest["status"],
        "published_intervals": len(records),
        "valid_full_duration_prefix_intervals": valid,
        "first_shortened_interval": shortening[0] if shortening else None,
        "conditional_v49_shortening_defect_exposed": bool(shortening),
        "latest_valid_checkpoint": str(latest.resolve()) if valid == len(records) else None,
        "latest_checkpoint_sha256_verified": bool(
            latest.exists() and digest(latest) == manifest["latest_checkpoint_sha256"]),
        "target_intervals": args.target_intervals,
        "remaining_intervals": remaining,
        "physical_time_s": None if endpoint is None else endpoint["physical_time_end_s"],
        "additional_engineering_shear": None if endpoint is None else (
            2*float(manifest["strain_rate_s"])*endpoint["load_elapsed_time_s"]),
        "target_physical_time_s": args.target_intervals*float(manifest["macro_dt_s"]),
        "target_additional_engineering_shear": (
            2*float(manifest["strain_rate_s"])*args.target_intervals
            *float(manifest["macro_dt_s"])),
        "recent_complete_interval_wall_seconds": recent_costs,
        "median_recent_interval_wall_seconds": cost,
        "estimated_remaining_wall_seconds": None if cost is None else remaining*cost,
        "endpoint_history": [{
            "interval": r["interval"],
            "physical_time_s": r["physical_time_end_s"],
            "engineering_total_shear": r["endpoint_observables"][
                "engineering_total_shear_gamma"],
            "engineering_plastic_shear": r["endpoint_observables"][
                "engineering_plastic_shear_gamma_p"],
            "mean_shear_stress_Pa": r["endpoint_observables"][
                "mean_shear_stress_sigma_12_Pa"],
            "temperature_mean_K": r["endpoint_observables"][
                "temperature_mean_K"],
            "temperature_peak_K": r["endpoint_observables"][
                "temperature_peak_K"],
            "cumulative_first_law_residual_J": r[
                "cumulative_first_law_residual_J"],
            "ordering_linear_iterations": sum(
                int(s.get("ordering_linear_iterations") or 0)
                for s in r["segments"]),
            "wall_seconds": sum(float(s["wall_seconds"]) for s in r["segments"]),
        } for r in records[:valid]],
        "next_automatic_action": (
            "retain worker and re-audit each published macro; postprocess at completion"
            if process["live"] and identity_matches and not shortening
            else "preserve valid prefix and diagnose before any continuation"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    temporary.replace(args.output)
    print(json.dumps({
        "valid_prefix": valid, "published": len(records),
        "remaining": remaining, "process_identity_matches": identity_matches,
        "output_sha256": digest(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
