#!/usr/bin/env python3
"""Quantify the global-temperature approximation on resolved front segments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.postprocess_v53_asb_mechanism import periodic_components
from full_model.analysis.run_v58_three_grain_production import (
    _load_checkpoint, network_interfaces,
)
from full_model.production.arrhenius_kinetics import (
    ActivatedProcess, activated_rate_s, exp_floor_enthalpy_j,
)
from full_model.production.multigrain_common_state import reconstruct_multigrain_common
from full_model.production.multigrain_production import _geometric_sweep_weight


def weighted_temperature(temperature: np.ndarray, weight: np.ndarray) -> dict:
    total = float(np.sum(weight, dtype=np.longdouble))
    if total <= 0.0:
        raise ValueError("interface weight must have positive measure")
    mean = float(np.sum(temperature*weight, dtype=np.longdouble)/total)
    variance = float(np.sum(weight*(temperature-mean)**2,
                            dtype=np.longdouble)/total)
    active = weight > .05*float(np.max(weight))
    components = periodic_components(active)
    segments = []
    for component in components:
        local_weight = np.where(component, weight, 0.0)
        local_total = float(np.sum(local_weight, dtype=np.longdouble))
        if local_total <= 0.0:
            continue
        segments.append({
            "weight_fraction": local_total/total,
            "temperature_K": float(np.sum(
                temperature*local_weight, dtype=np.longdouble)/local_total),
            "cells": int(np.count_nonzero(component)),
        })
    return {
        "weighted_mean_K": mean,
        "weighted_standard_deviation_K": variance**.5,
        "active_minimum_K": float(np.min(temperature[active])),
        "active_maximum_K": float(np.max(temperature[active])),
        "segment_count": len(segments),
        "segments": sorted(segments, key=lambda item: -item["weight_fraction"]),
    }


def analyze(run: Path) -> dict:
    result = json.loads((run/"result.json").read_text())
    checkpoint = Path(result["checkpoint"])
    state, _, step, _, _, configuration, provenance = _load_checkpoint(checkpoint)
    history = json.loads((run/"history.json").read_text())
    row = next(item for item in reversed(history) if int(item["step"]) == step)
    spacing = float(configuration["length_m"])/int(configuration["n"])
    common, _ = reconstruct_multigrain_common(state, spacing)
    temperature = np.asarray(common.temperature_K, dtype=float)
    global_temperature = float(np.mean(temperature))
    process = ActivatedProcess(
        "V59 locality audit", float(configuration["front_attempt_frequency_s"]),
        0.0, float(configuration["front_attempt_frequency_s"]))
    interfaces = network_interfaces(state)
    index = {grain_id: position for position, grain_id in enumerate(state.grain_ids)}
    records = []
    for interface in interfaces:
        key = interface.component_id
        if not row.get("front") or key not in row["front"]["directions"]:
            continue
        direction = row["front"]["directions"][key]
        donor, receiver = ((interface.grain_a_id, interface.grain_b_id)
                           if direction == "a_to_b" else
                           (interface.grain_b_id, interface.grain_a_id))
        weight = _geometric_sweep_weight(
            state, index[donor], index[receiver], spacing)
        thermal = weighted_temperature(temperature, weight)
        pressure = float(row["front"]["pressures_Pa"][key])
        enthalpy = exp_floor_enthalpy_j(
            pressure, .35*1.602176634e-19, 1e9, 2.0, 1.5, .10)
        global_rate = activated_rate_s(process, enthalpy, global_temperature)
        local_rate = activated_rate_s(
            process, enthalpy, thermal["weighted_mean_K"])
        segment_rates = [activated_rate_s(
            process, enthalpy, item["temperature_K"])
            for item in thermal["segments"]]
        records.append({
            "interface_id": key, "direction": direction,
            "pressure_Pa": pressure,
            "selected_global_velocity_m_s": float(
                row["front"]["selected_velocities_m_s"][key]),
            "temperature": thermal,
            "global_temperature_K": global_temperature,
            "local_minus_global_temperature_K": (
                thermal["weighted_mean_K"]-global_temperature),
            "transition_rate_local_to_global_ratio": local_rate/global_rate,
            "segment_transition_rate_ratio_range": (
                [min(segment_rates)/global_rate, max(segment_rates)/global_rate]
                if segment_rates else None),
        })
    maximum_temperature_offset = max(
        (abs(item["local_minus_global_temperature_K"]) for item in records),
        default=0.0)
    maximum_rate_ratio_error = max(
        (abs(item["transition_rate_local_to_global_ratio"]-1.0)
         for item in records), default=0.0)
    return {
        "schema": "asb-drx-v59-front-locality-audit-v1",
        "run_directory": str(run.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_commit": (provenance or {}).get("source_commit"),
        "step": step, "global_temperature_K": global_temperature,
        "interfaces": records,
        "maximum_interface_temperature_offset_K": maximum_temperature_offset,
        "maximum_local_transition_rate_relative_change": maximum_rate_ratio_error,
        "scope": (
            "Diagnostic comparison only. Production still assigns one scalar "
            "speed per named pair; segment-specific force and capacity are not "
            "integrated by this audit."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    output = args.output or args.run/"front_locality_audit.json"
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: result[key] for key in (
        "maximum_interface_temperature_offset_K",
        "maximum_local_transition_rate_relative_change", "scope")}, indent=2))


if __name__ == "__main__":
    main()
