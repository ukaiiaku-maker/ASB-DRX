#!/usr/bin/env python3
"""Classify the matched fresh-source V49 n128 loading/hold partial."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt


ROOT = Path("full_model/production/results-local/v49-physical/n128")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifests = {name: json.loads((ROOT/name/"run_manifest.json").read_text())
                 for name in ("loading", "hold")}
    if manifests["loading"]["source_checkpoint_sha256"] != manifests[
            "hold"]["source_checkpoint_sha256"]:
        raise ValueError("physical pair does not share one initial state")
    endpoints = {name: value["records"][-1]["endpoint_observables"]
                 for name, value in manifests.items()}
    difference = {key: endpoints["loading"][key]-endpoints["hold"][key]
                  for key in endpoints["loading"]}
    achieved_extra_engineering_shear = difference[
        "engineering_total_shear_gamma"]
    target = .005
    measured_wall = sum(value["wall_seconds"] for value in manifests.values())
    loading_interval_wall = manifests["loading"]["wall_seconds"]
    remaining_intervals = max(target/achieved_extra_engineering_shear-1.0, 0.0)
    payload = {
        "schema": "asb-drx/v49/fresh-n128-physical-pair/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "calculation_source_sha": manifests["loading"]["source_sha"],
        "grid": 128, "domain_m": 3.2e-6,
        "shared_initial_checkpoint_sha256": manifests[
            "loading"]["source_checkpoint_sha256"],
        "loading_checkpoint_sha256": manifests[
            "loading"]["latest_checkpoint_sha256"],
        "hold_checkpoint_sha256": manifests["hold"]["latest_checkpoint_sha256"],
        "manifest_sha256": {name: sha(ROOT/name/"run_manifest.json")
                            for name in manifests},
        "physical_time_s": manifests["loading"]["physical_time_s"],
        "engineering_strain_rate_s-1": 200.0,
        "endpoints": endpoints,
        "loading_minus_hold": difference,
        "balances": {name: {
            key: value["records"][-1][key] for key in (
                "cumulative_external_work_J",
                "cumulative_internal_energy_change_J",
                "cumulative_first_law_residual_J")}
            for name, value in manifests.items()},
        "ordering_linear_iterations": {name: value["records"][-1][
            "segments"][0]["ordering_linear_iterations"]
            for name, value in manifests.items()},
        "measured_pair_wall_seconds": measured_wall,
        "planning_target_additional_engineering_shear": target,
        "achieved_additional_engineering_shear": (
            achieved_extra_engineering_shear),
        "target_fraction_complete": achieved_extra_engineering_shear/target,
        "projected_remaining_loading_wall_seconds_at_first_interval_cost": (
            remaining_intervals*loading_interval_wall),
        "durable_partial": True,
        "physical_history": (
            "fresh current-source initial state; no inherited V46 trajectory"),
        "front_enabled": False,
        "classification": "VALID_RESOLVED_FRESH_SOURCE_PHYSICAL_PARTIAL",
        "drx_claimed": False, "lagb_claimed": False, "asb_claimed": False,
    }
    output = Path("full_model/verification/v49_physical_pair_n128.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    labels = ("stress (MPa)", "plastic shear (1e-6)", "mean dT (mK)")
    values = (difference["mean_shear_stress_sigma_12_Pa"]*1e-6,
              difference["engineering_plastic_shear_gamma_p"]*1e6,
              difference["temperature_mean_K"]*1e3)
    fig, ax = plt.subplots(figsize=(7, 3.8)); ax.bar(labels, values)
    ax.axhline(0, color="k", lw=.8)
    ax.set_title("V49 n128 loading minus hold, 0.488 μs")
    fig.tight_layout()
    figure = Path("full_model/verification/v49_physical_pair_n128.png")
    fig.savefig(figure, dpi=180); plt.close(fig)
    print(json.dumps({"classification": payload["classification"],
                      "sha256": sha(output), "figure_sha256": sha(figure)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
