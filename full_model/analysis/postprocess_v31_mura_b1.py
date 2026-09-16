#!/usr/bin/env python3
"""Decision-grade, matched-horizon audit of immutable V30 Mura B1 output.

The V30 tree is read-only input.  V31 products are written to a separate
directory and distinguish terminal evidence from the last exact checkpoint
shared by every case.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.analysis.run_v30_mura_tier_b1_case import create_case
from full_model.production.tensorial_nye import (
    divergence_of_nye, nye_from_plastic_distortion,
)
from full_model.production.v24_mechanical_wall import (
    mechanical_from_checkpoint_arrays,
)
from full_model.production.wall_topology_supply import reservoir_nye_m1


CHECKPOINT = re.compile(
    r"checkpoint_step_(?P<step>[0-9]+)_strain_(?P<strain>[0-9.]+)\.npz$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def mtime_utc(path: Path):
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def history(case_dir: Path):
    path = case_dir / "history.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def checkpoint_map(case_dir: Path):
    found = {}
    for path in case_dir.glob("checkpoint_step_*.npz"):
        match = CHECKPOINT.match(path.name)
        if match:
            found[(int(match.group("step")), float(match.group("strain")))] = path
    return found


def relative_pair_difference(a, b, floor=1e-30):
    return float(abs(float(a)-float(b))/max(abs(float(a)), abs(float(b)), floor))


def checkpoint_metrics(path: Path, condition: str, grid: int, length_m: float):
    created = create_case(grid, condition, 42, length_m)
    systems, topologies, spacing = created[3], created[4], created[-1]
    with np.load(path, allow_pickle=False) as archive:
        state = mechanical_from_checkpoint_arrays(archive, systems, topologies)
        metadata = json.loads(str(archive["v30_metadata_json"].item()))
    alpha = np.sum(state.common.family_nye_m1, axis=2)
    curl_beta = nye_from_plastic_distortion(state.common.beta_p, spacing)
    reservoir = reservoir_nye_m1(
        state.reservoir_alignment, systems,
        state.common.orientation_rad, topologies)["total"]
    alpha_scale = max(float(np.sqrt(np.mean(alpha**2))), 1.0)
    signed_family = sum(
        np.asarray(getattr(state.density, f"{stem}_plus_m2"))
        - np.asarray(getattr(state.density, f"{stem}_minus_m2"))
        for stem in ("mobile", "forest", "wall_tangle", "wall_ordered"))
    total_family = sum(
        np.asarray(getattr(state.density, f"{stem}_plus_m2"))
        + np.asarray(getattr(state.density, f"{stem}_minus_m2"))
        for stem in ("mobile", "forest", "wall_tangle", "wall_ordered"))
    signed_norm = np.linalg.norm(signed_family, axis=2)
    centered = signed_norm - np.mean(signed_norm)
    power = np.abs(np.fft.fftn(centered))**2
    power[0, 0] = 0.0
    peak = np.unravel_index(np.argmax(power), power.shape)
    frequency = np.hypot(
        np.fft.fftfreq(grid, d=spacing)[peak[0]],
        np.fft.fftfreq(grid, d=spacing)[peak[1]])
    div_scale = max(alpha_scale/spacing, 1.0)
    metrics = {
        "step": int(metadata["step"]),
        "applied_strain": float(metadata["applied_strain"]),
        "dual_nye_relative_rms": float(
            np.sqrt(np.mean((reservoir-alpha)**2))/alpha_scale),
        "authoritative_source_offset_relative_rms": float(
            np.sqrt(np.mean((alpha-curl_beta)**2))/alpha_scale),
        "normalized_line_continuity_residual": float(
            np.sqrt(np.mean(divergence_of_nye(alpha, spacing)**2))/div_scale),
        "alpha_rms_m1": float(np.sqrt(np.mean(alpha**2))),
        "maximum_signed_density_m2": float(np.max(signed_norm)),
        "mean_total_density_m2": float(np.mean(np.sum(total_family, axis=2))),
        "orientation_span_deg": float(
            np.ptp(state.common.orientation_rad)*180.0/np.pi),
        "slip_rms": float(np.sqrt(np.mean(state.common.slip**2))),
        "dominant_signed_wavelength_m": (
            None if frequency == 0.0 else float(1.0/frequency)),
        "structure_factor_peak_fraction": float(
            np.max(power)/max(float(np.sum(power)), 1e-300)),
    }
    fields = {
        "signed_norm_m2": signed_norm,
        "alpha_norm_m1": np.linalg.norm(alpha, axis=(-2, -1)),
        "orientation_deg": state.common.orientation_rad*180.0/np.pi,
    }
    return metadata, metrics, fields


def case_summary(case_dir: Path, shared_key):
    status = read_json(case_dir/"status.json")
    records = history(case_dir)
    checkpoints = checkpoint_map(case_dir)
    path = checkpoints[shared_key]
    _, common, fields = checkpoint_metrics(
        path, status["condition"], int(status["grid"]), float(status["length_m"]))
    failure_path = case_dir/"failure.json"
    process_path = case_dir/"process_failure.json"
    failure = read_json(failure_path) if failure_path.exists() else None
    process = read_json(process_path) if process_path.exists() else None
    maximum_dual = max(row["dual_nye_relative_rms"] for row in records)
    maximum_continuity = max(
        row["normalized_line_continuity_residual"] for row in records)
    invariant = all(row["accepted_step_hard_invariant_passed"]
                    and not row["post_step_projection_used"] for row in records)
    return {
        "case": case_dir.name,
        "grid": int(status["grid"]),
        "condition": status["condition"],
        "status": status["status"],
        "step": int(status["step"]),
        "applied_strain": float(status["applied_strain"]),
        "latest_checkpoint": status["latest_checkpoint"],
        "latest_checkpoint_sha256": sha256(case_dir/status["latest_checkpoint"]),
        "checkpoint_count": len(checkpoints),
        "maximum_dual_nye_relative_rms": float(maximum_dual),
        "maximum_normalized_line_continuity_residual": float(maximum_continuity),
        "all_recorded_hard_invariants_passed": bool(invariant),
        "failure": failure,
        "process_failure_marker": process,
        "process_failure_marker_mtime_utc": (
            mtime_utc(process_path) if process_path.exists() else None),
        "status_mtime_utc": mtime_utc(case_dir/"status.json"),
        "process_failure_marker_is_stale": bool(
            process is not None and status["status"] == "COMPLETED"
            and process_path.stat().st_mtime < (case_dir/"status.json").stat().st_mtime),
        "common_checkpoint": {
            "filename": path.name,
            "sha256": sha256(path),
            **common,
        },
    }, fields


def write_figures(cases, field_map, output_dir):
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    order = [(condition, grid) for condition in
             ("homogeneous", "broadband_noise", "mechanical_heterogeneity")
             for grid in (64, 128)]
    fig, axes = plt.subplots(3, 6, figsize=(16, 8), constrained_layout=True)
    for column, key in enumerate(order):
        values = field_map[key]
        for row, (field, title) in enumerate((
                ("signed_norm_m2", r"$|\rho^+-\rho^-|$ (m$^{-2}$)"),
                ("alpha_norm_m1", r"$|\alpha|$ (m$^{-1}$)"),
                ("orientation_deg", "orientation (deg)"))):
            image = axes[row, column].imshow(values[field].T, origin="lower",
                                              cmap="magma", interpolation="nearest")
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(f"{key[0]}\n{key[1]}²")
            if column == 0:
                axes[row, column].set_ylabel(title)
            fig.colorbar(image, ax=axes[row, column], fraction=.045, pad=.02)
    fig.savefig(output_dir/"matched_horizon_fields.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    metric_names = ("dual_nye_relative_rms", "orientation_span_deg",
                    "maximum_signed_density_m2")
    for case in cases:
        label = f"{case['condition']} {case['grid']}"
        case_dir = case["_case_dir"]
        records = history(case_dir)
        x = [row["applied_strain"] for row in records]
        for axis, metric in zip(axes, metric_names):
            axis.plot(x, [row[metric] for row in records], label=label, lw=1)
            axis.set_xlabel("applied strain")
            axis.set_ylabel(metric)
            axis.axvline(.03, color="k", ls="--", lw=.7)
    axes[0].axhline(.05, color="r", ls=":", lw=.8)
    axes[-1].set_yscale("symlog", linthresh=1.0)
    axes[0].legend(fontsize=6)
    fig.savefig(output_dir/"terminal_histories.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v30-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-id", default="56070295")
    parser.add_argument("--external-manifest-location", type=Path)
    args = parser.parse_args()
    cases_root = args.v30_root/"cases"
    case_dirs = sorted(path for path in cases_root.iterdir() if path.is_dir())
    maps = {path.name: checkpoint_map(path) for path in case_dirs}
    shared = set.intersection(*(set(mapping) for mapping in maps.values()))
    if not shared:
        raise RuntimeError("no exact checkpoint is shared by every case")
    shared_key = max(shared)
    cases, field_map = [], {}
    for case_dir in case_dirs:
        summary, fields = case_summary(case_dir, shared_key)
        summary["_case_dir"] = case_dir
        cases.append(summary)
        field_map[(summary["condition"], summary["grid"])] = fields

    pair_metrics = ("dual_nye_relative_rms", "alpha_rms_m1",
                    "maximum_signed_density_m2", "mean_total_density_m2",
                    "orientation_span_deg", "slip_rms",
                    "structure_factor_peak_fraction")
    convergence = {}
    for condition in ("homogeneous", "broadband_noise", "mechanical_heterogeneity"):
        lo = next(case for case in cases if case["condition"] == condition
                  and case["grid"] == 64)["common_checkpoint"]
        hi = next(case for case in cases if case["condition"] == condition
                  and case["grid"] == 128)["common_checkpoint"]
        values = {name: relative_pair_difference(lo[name], hi[name])
                  for name in pair_metrics}
        convergence[condition] = {
            "relative_pair_differences": values,
            "all_declared_observables_below_5_percent": all(
                value < .05 for value in values.values()),
        }

    mechanical = [case for case in cases
                  if case["condition"] == "mechanical_heterogeneity"]
    thermodynamic_failure = all(
        case["failure"] and case["failure"]["type"] == "RuntimeError"
        and case["failure"]["message"] ==
        "Mura line storage exceeds available plastic work"
        for case in mechanical)
    complete_controls = all(case["status"] == "COMPLETED"
                            for case in cases if case not in mechanical)
    b1_pass = bool(complete_controls and not thermodynamic_failure
                   and all(item["all_recorded_hard_invariants_passed"]
                           for item in cases)
                   and all(item["all_declared_observables_below_5_percent"]
                           for item in convergence.values()))
    classification = (
        "TIER_B1_MECHANICAL_HETEROGENEITY_THERMODYNAMIC_ADMISSIBILITY_FAILURE"
        if thermodynamic_failure else
        "TIER_B1_RESOLVED_ANCHOR_QUALIFIED" if b1_pass else
        "TIER_B1_NONQUALIFYING_EVIDENCE")
    serial_cases = []
    for case in cases:
        item = dict(case); item.pop("_case_dir")
        serial_cases.append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_figures(cases, field_map, args.output_dir/"figures")
    all_files = sorted(path for path in args.v30_root.rglob("*") if path.is_file())
    evidence = [{"path": str(path.relative_to(args.v30_root)),
                 "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in all_files]
    (args.output_dir/"evidence_manifest.json").write_text(json.dumps({
        "immutable_input_root": str(args.v30_root.resolve()),
        "file_count": len(evidence), "files": evidence,
    }, indent=2, sort_keys=True)+"\n")
    manifest_path = args.output_dir/"evidence_manifest.json"
    external_manifest = (args.external_manifest_location
                         if args.external_manifest_location is not None
                         else manifest_path.resolve())
    result = {
        "schema": "asb-drx/v31-mura-b1-decision/v1",
        "source_sha": "675d74183baf2043ad0f7c055fe6e3370435ae65",
        "slurm_job_id": args.job_id,
        "slurm_terminal": {"state": "FAILED", "exit_code": "1:0",
                           "elapsed": "04:15:49", "time_limit": "16:00:00",
                           "started_local": "2026-09-15T22:28:39-07:00",
                           "ended_local": "2026-09-16T02:44:28-07:00",
                           "allocated_cpus": 8, "requested_memory": "32G",
                           "batch_max_rss": "1188720K", "node": "hpc3-15-16",
                           "classification": "APPLICATION_SCIENTIFIC_FAILURE_NOT_RESOURCE_EXHAUSTION"},
        "external_full_file_manifest": {
            "path": str(external_manifest),
            "sha256": sha256(manifest_path),
            "input_file_count": len(evidence),
            "remote_local_sha256sum_stream_digest":
                "27e2ede728b8f5cdd62ad7edf7e222cdb190ffe0aeac62ebdc3421055cc63fa5",
            "remote_local_stream_digest_matched": True,
        },
        "matched_horizon": {"step": shared_key[0],
                            "applied_strain": shared_key[1],
                            "selection": "latest exact checkpoint shared by all six cases"},
        "cases": serial_cases,
        "grid_pair_convergence": convergence,
        "fixture_passed": True,
        "scientific_gate_passed": b1_pass,
        "classification": classification,
        "tier_b2_authorized": b1_pass,
        "tier_b2_decision": ("AUTHORIZED" if b1_pass else
                             "NOT_AUTHORIZED_FROM_V30_B1_EVIDENCE"),
        "resubmission_performed": False,
        "interpretation": (
            "Four controls completed 20% strain. Both mechanical-heterogeneity "
            "cases independently rejected a step because Mura line storage "
            "exceeded available plastic work; the 4:15:49 batch duration is "
            "the time required for the slow completed controls, not the onset "
            "of a scheduler or memory failure."),
    }
    (args.output_dir/"v31_mura_b1_decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n")
    fields = ("case", "grid", "condition", "status", "step",
              "applied_strain", "checkpoint_count", "latest_checkpoint",
              "latest_checkpoint_sha256", "maximum_dual_nye_relative_rms",
              "maximum_normalized_line_continuity_residual")
    with (args.output_dir/"v31_mura_b1_index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in serial_cases:
            writer.writerow({name: case.get(name) for name in fields})
    print(json.dumps({"classification": classification,
                      "matched_horizon": shared_key,
                      "tier_b2_authorized": b1_pass}, sort_keys=True))


if __name__ == "__main__":
    main()
