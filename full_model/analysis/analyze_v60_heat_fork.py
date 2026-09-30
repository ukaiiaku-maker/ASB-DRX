#!/usr/bin/env python3
"""Audit a same-parent local versus uniform front-heat fork."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.multigrain_common_state import reconstruct_multigrain_common


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoints(directory: Path) -> dict[int, Path]:
    return {int(path.stem.rsplit("_", 1)[-1]): path
            for path in directory.glob("checkpoint_*.npz")}


def analyze(local_directory: Path, uniform_directory: Path) -> dict:
    local_result = json.loads((local_directory/"result.json").read_text())
    uniform_result = json.loads((uniform_directory/"result.json").read_text())
    local_config = dict(local_result["configuration"])
    uniform_config = dict(uniform_result["configuration"])
    local_mode = local_config.pop("front_heat_deposition")
    uniform_mode = uniform_config.pop("front_heat_deposition")
    local_checkpoints = _checkpoints(local_directory)
    uniform_checkpoints = _checkpoints(uniform_directory)
    common = sorted(set(local_checkpoints) & set(uniform_checkpoints))
    rows = []
    for step in common:
        local_path = local_checkpoints[step]
        uniform_path = uniform_checkpoints[step]
        (local_state, local_runtime, local_step, local_gamma, _,
         local_checkpoint_config, local_provenance) = _load_checkpoint(local_path)
        (uniform_state, uniform_runtime, uniform_step, uniform_gamma, _,
         uniform_checkpoint_config, uniform_provenance) = _load_checkpoint(uniform_path)
        spacing = float(local_checkpoint_config["length_m"])/int(
            local_checkpoint_config["n"])
        local_common, _ = reconstruct_multigrain_common(local_state, spacing)
        uniform_common, _ = reconstruct_multigrain_common(uniform_state, spacing)
        temperature_difference = (
            np.asarray(local_common.temperature_K)
            -np.asarray(uniform_common.temperature_K))
        rows.append({
            "step": step,
            "physical_time_s": local_runtime.ledger.physical_time_s,
            "loading_coordinate": local_gamma,
            "local_checkpoint": str(local_path),
            "uniform_checkpoint": str(uniform_path),
            "local_checkpoint_sha256": _digest(local_path),
            "uniform_checkpoint_sha256": _digest(uniform_path),
            "temperature_local_minus_uniform_K": {
                "mean": float(np.mean(temperature_difference)),
                "rms": float(np.sqrt(np.mean(temperature_difference**2))),
                "minimum": float(np.min(temperature_difference)),
                "maximum": float(np.max(temperature_difference)),
                "peak_to_peak": float(np.ptp(temperature_difference)),
            },
            "support_maximum_absolute_difference": float(np.max(np.abs(
                local_state.supports-uniform_state.supports))),
            "generated_heat_difference_J": float(
                local_runtime.ledger.generated_heat_J
                -uniform_runtime.ledger.generated_heat_J),
            "mean_temperature_local_K": float(np.mean(local_common.temperature_K)),
            "mean_temperature_uniform_K": float(np.mean(uniform_common.temperature_K)),
            "temperature_contrast_local_K": float(np.ptp(local_common.temperature_K)),
            "temperature_contrast_uniform_K": float(np.ptp(uniform_common.temperature_K)),
            "source_and_parent_checks": {
                "same_step": local_step == uniform_step == step,
                "same_physical_time": local_runtime.ledger.physical_time_s
                    == uniform_runtime.ledger.physical_time_s,
                "same_loading_coordinate": local_gamma == uniform_gamma,
                "same_source_commit": local_provenance.get("source_commit")
                    == uniform_provenance.get("source_commit"),
                "same_parent_checkpoint": local_provenance.get(
                    "parent_checkpoint_sha256") == uniform_provenance.get(
                    "parent_checkpoint_sha256"),
            },
        })
    checks = {
        "local_mode_is_production": local_mode == "local_realized_event",
        "uniform_mode_is_labeled_ablation": uniform_mode == "uniform_ablation",
        "otherwise_identical_configuration": local_config == uniform_config,
        "same_executing_source": local_result["source_commit"]
            == uniform_result["source_commit"],
        "common_checkpoint_pair_exists": bool(rows),
        "every_pair_has_common_source_and_parent": all(
            all(row["source_and_parent_checks"].values()) for row in rows),
        "spatial_deposition_has_nonuniform_effect": any(
            row["temperature_local_minus_uniform_K"]["peak_to_peak"] > 0.0
            for row in rows),
        "no_rejected_front_event": (
            local_result["runtime"]["rejected_events"] == 0
            and uniform_result["runtime"]["rejected_events"] == 0),
    }
    return {
        "schema": "asb-drx-v60-same-parent-front-heat-fork-v1",
        "classification": (
            "LOCAL_FRONT_HEAT_OPERATOR_ACTIVATED_SHORT_HORIZON"
            if all(checks.values()) else "INVALID_OR_INCONCLUSIVE_FORK"),
        "scientific_scope": (
            "operator/provenance discrimination only; this 50 ns fork is not "
            "a strict-ASB, persistence, or grid-convergence qualification"),
        "checks": checks,
        "local_result_sha256": _digest(local_directory/"result.json"),
        "uniform_result_sha256": _digest(uniform_directory/"result.json"),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--uniform", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.local, args.uniform)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["checks"], sort_keys=True))


if __name__ == "__main__":
    main()
