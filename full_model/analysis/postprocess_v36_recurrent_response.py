#!/usr/bin/env python3
"""Assemble V36 recurrent-response evidence and compact physical plots."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCHEMA = "asb-drx/v36/recurrent-physical-response-decision/v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discover(roots):
    cases = {}
    for root in roots:
        for path in Path(root).glob("*/result.json"):
            data = json.loads(path.read_text())
            cases[path.parent.name] = {
                "path": str(path.resolve()), "sha256": sha256(path),
                "data": data,
            }
    return cases


def relative_difference(a, b):
    return abs(float(a)-float(b))/max(abs(float(a)), abs(float(b)), 1e-300)


def summarize_case(item):
    data = item["data"]
    records = data["records"]
    first = records[0] if records else None
    last = records[-1] if records else None
    return {
        "path": item["path"], "sha256": item["sha256"],
        "source_commit": data["source_commit"],
        "configuration": data["configuration"],
        "completed_intervals": data["completed_intervals"],
        "physical_time_s": data["physical_time_s"],
        "accepted_front_intervals": data["accepted_front_intervals"],
        "cumulative_signed_sweep_m3": data["cumulative_signed_sweep_m3"],
        "gross_expected_event_count": data["gross_expected_event_count"],
        "net_expected_event_count": data["net_expected_event_count"],
        "first_net_velocity_m_s": (
            None if first is None else first["net_velocity_m_s"]),
        "last_net_velocity_m_s": (
            None if last is None else last["net_velocity_m_s"]),
        "last_front_published": (
            None if last is None else last["front_published"]),
        "runner_classification": data["classification"],
    }


def latest_checkpoint(case_path):
    files = sorted(Path(case_path).parent.glob("checkpoint_*.npz"))
    return files[-1] if files else None


def plot_fields(checkpoint, output):
    with np.load(checkpoint, allow_pickle=False) as data:
        eta = np.asarray(data["eta"])[..., 1]
        prefixes = ("mobile", "forest", "wall")
        total = sum(np.asarray(data[f"mechanical__v24_common__{p}_{s}_m2"])
                    for p in prefixes for s in ("plus", "minus"))
        total = np.sum(total, axis=2)+np.sum(np.asarray(
            data["mechanical__v24_common__junction_m2"]), axis=2)
        nye_tensor = np.asarray(
            data["mechanical__v24_common__family_nye_m1"])
        nye = np.sqrt(np.sum(nye_tensor*nye_tensor, axis=(2, 3, 4)))
        slip = np.linalg.norm(np.asarray(
            data["mechanical__v24_common__slip"]), axis=2)
        orientation = np.asarray(
            data["mechanical__v24_common__orientation_rad"])*180.0/np.pi
        temperature = np.asarray(
            data["mechanical__v24_common__temperature_K"])
    fields = (
        (eta, "child phase", "viridis"),
        (total, r"total line density [m$^{-2}$]", "magma"),
        (nye, r"Nye norm [m$^{-1}$]", "magma"),
        (slip, "slip norm", "cividis"),
        (orientation, "orientation [degree]", "coolwarm"),
        (temperature, "temperature [K]", "inferno"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, (field, title, cmap) in zip(axes.flat, fields):
        image = axis.imshow(field.T, origin="lower", cmap=cmap)
        axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=.78)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_histories(cases, output):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                             constrained_layout=True)
    for name, item in sorted(cases.items()):
        data = item["data"]
        if not data["records"]:
            continue
        t = np.asarray([r["physical_time_end_s"] for r in data["records"]])
        velocity = np.asarray([r["net_velocity_m_s"] or 0.0
                               for r in data["records"]])
        sweep = np.cumsum([r["signed_sweep_m3"]
                           for r in data["records"]])
        if data["configuration"]["grid"] >= 64 or "dt" in name:
            axes[0].plot(t, velocity, marker=".", label=name)
            axes[1].plot(t, sweep, marker=".", label=name)
    axes[0].set_ylabel(r"net velocity [m s$^{-1}$]")
    axes[1].set_ylabel(r"cumulative sweep [m$^3$]")
    axes[1].set_xlabel("physical time [s]")
    axes[0].legend(fontsize=7, ncol=2)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_decision(cases):
    summaries = {name: summarize_case(item) for name, item in cases.items()}
    checks = {}
    if {"hold_dt1e5_20", "hold_dt5e6_40"} <= cases.keys():
        a = cases["hold_dt1e5_20"]["data"]["cumulative_signed_sweep_m3"]
        b = cases["hold_dt5e6_40"]["data"]["cumulative_signed_sweep_m3"]
        checks["n16_timestep_relative_difference_at_0p2ms"] = relative_difference(a, b)
    if {"hold_n128_pf00625_1", "hold_n192_pf00625_1"} <= cases.keys():
        a = cases["hold_n128_pf00625_1"]["data"]["records"][0]["net_velocity_m_s"]
        b = cases["hold_n192_pf00625_1"]["data"]["records"][0]["net_velocity_m_s"]
        checks["n128_n192_initial_velocity_relative_difference"] = relative_difference(a, b)
    proposal_names = ("hold_pf00625", "hold_pf0125", "hold_pf025")
    if set(proposal_names) <= cases.keys():
        values = [cases[name]["data"]["cumulative_signed_sweep_m3"]
                  for name in proposal_names]
        checks["n16_proposal_sweep_relative_span"] = (
            (max(values)-min(values))/max(abs(x) for x in values))
    initial_grid_pass = checks.get(
        "n128_n192_initial_velocity_relative_difference", np.inf) < .05
    timestep_pass = checks.get(
        "n16_timestep_relative_difference_at_0p2ms", np.inf) < .05
    n128_long = summaries.get("hold_n128_pf00625_10")
    n192_long = summaries.get("hold_n192_pf00625_10")
    if n128_long and n192_long:
        trajectory_grid_error = relative_difference(
            n128_long["cumulative_signed_sweep_m3"],
            n192_long["cumulative_signed_sweep_m3"])
        checks["n128_n192_trajectory_sweep_relative_difference"] = trajectory_grid_error
    else:
        trajectory_grid_error = None
    if trajectory_grid_error is not None and trajectory_grid_error < .05:
        classification = "RESOLVED_ZERO_WORK_RECURRENT_RESPONSE_QUALIFIED"
        scientific_gate = True
    elif initial_grid_pass and timestep_pass:
        classification = "INITIAL_RATE_QUALIFIED_LONG_HORIZON_RESOLUTION_PENDING"
        scientific_gate = False
    else:
        classification = "RESPONSE_OBSERVED_RESOLUTION_NOT_QUALIFIED"
        scientific_gate = False
    return {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "fixture_passed": bool(initial_grid_pass and timestep_pass),
        "scientific_gate_passed": scientific_gate,
        "applied_front_work_Pa": 0.0,
        "frozen_closure_interpretation": (
            "ZERO_DRIFT_UNDER_SATURATED_DIRECTIONAL_CLOSURE_"
            "PHYSICAL_ARREST_UNRESOLVED"),
        "selected_closure": "CHANNEL_RESOLVED_NONREVERSE_EXP_FLOOR",
        "grid_policy": {
            "n16_n32_n64": "FIXTURE_OR_CONVERGENCE_EVIDENCE_ONLY",
            "minimum_promoted_grid": 128 if initial_grid_pass else None,
            "provisional_relative_threshold": .05,
        },
        "convergence": checks,
        "cases": summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--field-case", default="hold_n128_pf00625_10")
    args = parser.parse_args()
    cases = discover(args.root)
    decision = build_decision(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, indent=2, sort_keys=True)+"\n")
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_histories(cases, args.figures_dir/"v36_recurrent_histories.png")
    selected = cases.get(args.field_case)
    if selected:
        checkpoint = latest_checkpoint(selected["path"])
        if checkpoint:
            plot_fields(checkpoint, args.figures_dir/"v36_recurrent_fields.png")
    print(decision["classification"])


if __name__ == "__main__":
    main()
