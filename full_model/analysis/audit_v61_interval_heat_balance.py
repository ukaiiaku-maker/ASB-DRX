"""Verify a fixed-Eulerian finite-interval thermal balance from checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.postprocess_v59_physical_asb import (
    checkpoint_snapshot, source_association,
)
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.multigrain_common_state import (
    reconstruct_multigrain_common,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _history_row(checkpoint: Path, step: int) -> dict | None:
    path = checkpoint.parent/"history.json"
    if not path.is_file():
        return None
    return next((row for row in json.loads(path.read_text())
                 if int(row["step"]) == step), None)


def _signed_integral(field, mask, cell_volume):
    value = np.asarray(field, dtype=float)
    region = np.asarray(mask, dtype=bool)
    return float(np.sum(value[region], dtype=np.longdouble)*cell_volume)


def audit(before_checkpoint: Path, after_checkpoint: Path,
          arrays_output: Path | None = None) -> dict:
    (before_state, before_runtime, before_step, _before_gamma,
     _before_volume, before_config, _before_provenance) = _load_checkpoint(
         before_checkpoint)
    (after_state, after_runtime, after_step, _after_gamma,
     _after_volume, after_config, after_provenance) = _load_checkpoint(
         after_checkpoint)
    if after_step != before_step+1:
        raise ValueError("thermal balance requires consecutive macro checkpoints")
    if before_config != after_config:
        raise ValueError("thermal balance checkpoints have different configuration")
    dt = (after_runtime.ledger.physical_time_s
          -before_runtime.ledger.physical_time_s)
    if not np.isclose(dt, float(after_config["dt_s"]), rtol=0.0, atol=1e-18):
        raise ValueError("checkpoint time gap is not one configured interval")
    spacing = float(after_config["length_m"])/int(after_config["n"])
    thickness = 2.0*2.48e-10
    cell_volume = spacing**2*thickness
    heat_capacity = 3.8e6
    before_common, _ = reconstruct_multigrain_common(before_state, spacing)
    after_common, _ = reconstruct_multigrain_common(after_state, spacing)
    direct_storage = heat_capacity*(
        np.asarray(after_common.temperature_K)
        -np.asarray(before_common.temperature_K))
    history_row = _history_row(after_checkpoint, after_step)
    snapshot, candidate = checkpoint_snapshot(after_checkpoint, history_row)
    with np.load(after_checkpoint, allow_pickle=False) as data:
        required = (
            "diagnostic_front_heat_source_J_by_cell",
            "diagnostic_mechanical_heat_J_m3_cells",
            "diagnostic_thermal_conduction_J_m3_cells",
            "diagnostic_thermal_bath_exchange_J_m3_cells",
        )
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"checkpoint lacks interval ledgers: {missing}")
        front = np.asarray(data[required[0]], dtype=float)/cell_volume
        mechanical = np.asarray(data[required[1]], dtype=float)
        conduction = np.asarray(data[required[2]], dtype=float)
        bath = np.asarray(data[required[3]], dtype=float)
    reconstructed = mechanical+conduction+bath+front
    residual = direct_storage-reconstructed
    fields = {
        "direct_thermal_storage": direct_storage,
        "mechanical_reaction_heat": mechanical,
        "front_heat": front,
        "thermal_conduction": conduction,
        "thermal_bath_exchange": bath,
        "independent_source_sum": reconstructed,
        "balance_residual": residual,
    }
    whole = np.ones(candidate.shape, dtype=bool)
    budgets = {}
    for name, field in fields.items():
        budgets[name] = {
            "whole_domain_J": _signed_integral(field, whole, cell_volume),
            "candidate_fixed_eulerian_J": _signed_integral(
                field, candidate, cell_volume),
            "source_association": source_association(field, candidate),
        }
    residual_scale = max(
        abs(budgets["direct_thermal_storage"]["whole_domain_J"]),
        abs(budgets["independent_source_sum"]["whole_domain_J"]), 1e-300)
    local_scale = max(
        abs(budgets["direct_thermal_storage"]["candidate_fixed_eulerian_J"]),
        abs(budgets["independent_source_sum"]["candidate_fixed_eulerian_J"]),
        1e-300)
    front_recorded = (None if not history_row or not history_row.get("front")
                      else history_row["front"].get("generated_heat_J"))
    front_integral = budgets["front_heat"]["whole_domain_J"]
    result = {
        "schema": "asb-drx-v61-fixed-eulerian-interval-heat-balance-v1",
        "before_checkpoint": str(before_checkpoint.resolve()),
        "after_checkpoint": str(after_checkpoint.resolve()),
        "before_checkpoint_sha256": _sha(before_checkpoint),
        "after_checkpoint_sha256": _sha(after_checkpoint),
        "source_commit": (after_provenance or {}).get("source_commit"),
        "before_step": before_step, "after_step": after_step,
        "interval_s": dt,
        "cell_volume_m3": cell_volume,
        "candidate_rule": (
            "largest periodic accepted-state plastic-power component above "
            "mean plus one standard deviation; fixed Eulerian cells over the "
            "entire interval, so no boundary-motion transport term exists"),
        "candidate_cell_count": int(np.count_nonzero(candidate)),
        "candidate_area_fraction": float(np.mean(candidate)),
        "budgets": budgets,
        "whole_domain_relative_balance_residual": abs(
            budgets["balance_residual"]["whole_domain_J"])/residual_scale,
        "candidate_relative_balance_residual": abs(
            budgets["balance_residual"]["candidate_fixed_eulerian_J"])
            /local_scale,
        "maximum_cell_relative_balance_residual": float(
            np.max(np.abs(residual))/max(
                float(np.max(np.abs(direct_storage))),
                float(np.max(np.abs(reconstructed))), 1e-300)),
        "front_source_pre_normalization": {
            "saved_cell_integral_J": front_integral,
            "history_independent_generated_heat_J": front_recorded,
            "relative_mismatch": (None if front_recorded is None else
                abs(front_integral-float(front_recorded))/max(
                    abs(float(front_recorded)), 1e-300)),
        },
        "snapshot_checkpoint_sha256": snapshot["checkpoint_sha256"],
    }
    if arrays_output is not None:
        arrays_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays_output, candidate_mask=candidate,
            temperature_before_K=before_common.temperature_K,
            temperature_after_K=after_common.temperature_K,
            **{f"{name}_J_m3": value for name, value in fields.items()})
        result["arrays"] = str(arrays_output.resolve())
        result["arrays_sha256"] = _sha(arrays_output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arrays-output", type=Path)
    args = parser.parse_args()
    result = audit(args.before, args.after, args.arrays_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: result[key] for key in (
        "before_step", "after_step", "whole_domain_relative_balance_residual",
        "candidate_relative_balance_residual",
        "maximum_cell_relative_balance_residual",
        "front_source_pre_normalization")}, indent=2))


if __name__ == "__main__":
    main()
