#!/usr/bin/env python3
"""Build and verify the single compact V65 evidence delivery."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import zipfile


MILESTONES_S = (1.05e-6, 1.10e-6, 1.25e-6)


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def checkpoint_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def telemetry_times(path: Path) -> dict[int, float]:
    answer: dict[int, float] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            answer[int(row["adaptive_interval"])] = float(
                row["physical_time_s"])
    return answer


def select_checkpoints(
    checkpoints: list[Path], times: dict[int, float],
    milestones_s: tuple[float, ...] = MILESTONES_S,
) -> list[Path]:
    """Select endpoints and checkpoint states nearest requested milestones."""
    ordered = sorted(checkpoints, key=checkpoint_index)
    if not ordered:
        raise ValueError("no production checkpoints found")
    selected = {ordered[0], ordered[-1]}
    timed = [path for path in ordered if checkpoint_index(path) in times]
    for milestone in milestones_s:
        if timed:
            selected.add(min(
                timed,
                key=lambda path: abs(
                    times[checkpoint_index(path)] - milestone),
            ))
    return sorted(selected, key=checkpoint_index)


def require_complete_finalizer(campaign: Path) -> dict:
    path = campaign / "finalizer_status.json"
    if not path.is_file():
        raise RuntimeError("V65 finalizer status is absent")
    payload = json.loads(path.read_text())
    if payload.get("status") != "COMPLETE":
        raise RuntimeError(
            f"V65 finalizer is not complete: {payload.get('status')}")
    if not payload.get("all_required_artifacts_present", False):
        raise RuntimeError("V65 finalizer did not verify every required artifact")
    return payload


def stable_tree_files(campaign: Path, root: Path) -> list[Path]:
    """Return closed evidence files, excluding the builder's live stdout."""
    if not root.is_dir():
        return []
    live_builder_log = campaign / "logs/build_v65_delivery.log"
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path != live_builder_log)


def required_files(campaign: Path, worktree: Path) -> list[tuple[Path, str]]:
    production = campaign / "runs/production_1p035_to1p25us_6debbe0"
    candidate = campaign / "runs/candidate_to1p035us_0637685"
    baseline = campaign / "runs/baseline_to1p035us_f82d3a6"
    reconstructed = campaign / "reconstructed_v64"
    fixed = [
        campaign / "finalizer_status.json",
        campaign / "repository_sync_status.json",
        campaign / "analysis/work_precision_1p030_to1p035us.json",
        campaign / "analysis/interval454_cost_error_trace.json",
        campaign / "analysis/grain20_selection_energy_affinity.json",
        campaign / "analysis/power_component_transition_2900_3028.json",
        campaign / "analysis/production_to1p25us_summary.json",
        campaign / "analysis/merged_controlled_history.json",
        campaign / "analysis/final_asb_through1p25us.json",
        campaign / "analysis/final_grain20_selection.json",
        campaign / "analysis/v65_final_decision.json",
        campaign / "analysis/v65_final_decision.md",
        reconstructed / "checkpoint_adaptive_002585.npz",
        reconstructed / "trajectory_checkpoints/n48_analytic_origin_bab4c95.npz",
        reconstructed / "trajectory_checkpoints/checkpoint_adaptive_000454.npz",
        reconstructed / "trajectory_checkpoints/checkpoint_adaptive_000886.npz",
        reconstructed / "histories/terminal_through2184.json",
        reconstructed / "histories/terminal_extension.json",
        reconstructed / "histories/final_allocation.json",
        reconstructed / "result.json",
        reconstructed / "v64_campaign_manifest.json",
        reconstructed / "v64_numerical_source_b579a16.tar.gz",
        baseline / "history.json",
        baseline / "result.json",
        baseline / "checkpoint_adaptive_002653.npz",
        candidate / "accepted_telemetry.jsonl",
        candidate / "history.json",
        candidate / "result.json",
        candidate / "checkpoint_adaptive_002644.npz",
        production / "accepted_telemetry.jsonl",
        production / "history.json",
        production / "result.json",
        worktree / "full_model/verification/v65_campaign_manifest.json",
        worktree / "full_model/docs/v65_execution_record.md",
        worktree / "full_model/docs/v65_restart_commands.md",
    ]
    missing = [str(path) for path in fixed if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required V65 payloads:\n" +
                                "\n".join(missing))
    pairs: list[tuple[Path, str]] = []
    for path in fixed:
        if path.is_relative_to(campaign):
            arcname = str(path.relative_to(campaign))
        else:
            arcname = "repository/" + str(path.relative_to(worktree))
        pairs.append((path, arcname))

    for root in (campaign / "analysis/live_monitor", campaign / "logs",
                 campaign / "visuals"):
        for path in stable_tree_files(campaign, root):
            pairs.append((path, str(path.relative_to(campaign))))

    times = telemetry_times(production / "accepted_telemetry.jsonl")
    checkpoints = select_checkpoints(
        list(production.glob("checkpoint_adaptive_*.npz")), times)
    for path in checkpoints:
        pairs.append((path, str(path.relative_to(campaign))))
    unique: dict[str, Path] = {}
    for path, arcname in pairs:
        if arcname in unique and unique[arcname] != path:
            raise RuntimeError(f"conflicting archive member {arcname}")
        unique[arcname] = path
    return [(path, arcname) for arcname, path in sorted(unique.items())]


def git_archive(worktree: Path, commit: str) -> bytes:
    return subprocess.check_output(
        ["git", "archive", "--format=tar", commit], cwd=worktree)


def build(
    campaign: Path, worktree: Path, output: Path,
    production_commit: str, evidence_commit: str,
) -> dict:
    finalizer = require_complete_finalizer(campaign)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite definitive archive {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    entries: list[dict] = []
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path, arcname in required_files(campaign, worktree):
            payload_hash = digest_file(path)
            archive.write(path, arcname)
            entries.append({
                "path": arcname,
                "bytes": path.stat().st_size,
                "sha256": payload_hash,
            })
        for commit, name in (
            (production_commit, "source/production_source.tar"),
            (evidence_commit, "source/evidence_source.tar"),
        ):
            payload = git_archive(worktree, commit)
            archive.writestr(name, payload)
            entries.append({
                "path": name, "bytes": len(payload),
                "sha256": digest_bytes(payload), "git_commit": commit,
            })
        inventory = {
            "schema": "asb-drx-v65-definitive-delivery-inventory-v1",
            "production_source_commit": production_commit,
            "evidence_source_commit": evidence_commit,
            "finalizer_completed_utc": finalizer.get("completed_utc"),
            "payload_count": len(entries),
            "payloads": entries,
        }
        inventory_bytes = (json.dumps(
            inventory, indent=2, sort_keys=True) + "\n").encode()
        sums = "".join(
            f"{item['sha256']}  {item['path']}\n" for item in entries
        ).encode()
        archive.writestr("INVENTORY.json", inventory_bytes)
        archive.writestr("SHA256SUMS", sums)
    os.replace(temporary, output)
    verify(output)
    archive_hash = digest_file(output)
    sidecar = output.with_name(output.name + ".sha256")
    sidecar.write_text(f"{archive_hash}  {output.name}\n")
    return {
        "archive": str(output.resolve()),
        "archive_sha256": archive_hash,
        "archive_bytes": output.stat().st_size,
        "payload_count": len(entries),
        "sidecar": str(sidecar.resolve()),
    }


def verify(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        inventory = json.loads(archive.read("INVENTORY.json"))
        if inventory["payload_count"] != len(inventory["payloads"]):
            raise RuntimeError("inventory payload count mismatch")
        for item in inventory["payloads"]:
            payload = archive.read(item["path"])
            if len(payload) != item["bytes"]:
                raise RuntimeError(f"size mismatch for {item['path']}")
            if digest_bytes(payload) != item["sha256"]:
                raise RuntimeError(f"checksum mismatch for {item['path']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--production-commit", required=True)
    parser.add_argument("--evidence-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(build(
        args.campaign.resolve(), args.worktree.resolve(), args.output.resolve(),
        args.production_commit, args.evidence_commit,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
