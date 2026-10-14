#!/usr/bin/env python3
"""Work-ledgered unloading followed by fixed-zero-strain annealing.

This continuation consumes the exact terminal checkpoint of the V66 coupled
loading calculation.  "Static" means zero imposed rate at zero macroscopic
shear strain; residual internal stress is retained and is not relabelled as a
stress-free condition.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from full_model.analysis.run_v58_three_grain_production import (
    _load_checkpoint, _save_checkpoint,
)
from full_model.analysis.run_v66_coupled_rx_asb import (
    _atomic_json, _physics, _record, _source_provenance,
)
from full_model.production.coupled_multigrain_interval import (
    advance_complete_multigrain_interval, make_endpoint,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/"full_model"/"verification"/"v66_coupled_rx_asb"/"checkpoint_latest.npz"
DEFAULT_OUT = ROOT/"full_model"/"verification"/"v66_unload_static"


def _configuration(args):
    return {
        "schema": "v66-unload-static-configuration-v1",
        "source_checkpoint": str(args.source.relative_to(ROOT)),
        "n": args.n, "length_m": args.length,
        "temperature_parameter_K": args.temperature,
        "unload_rate_s": -abs(args.unload_rate),
        "unload_dt_s": args.unload_dt,
        "unload_steps": args.unload_steps,
        "hold_rate_s": 0.0,
        "hold_dt_s": args.hold_dt,
        "hold_steps": args.hold_steps,
        "hold_constraint": "fixed zero macroscopic shear strain; residual stress retained",
    }


def _summary(history, endpoint, loaded_volume, configuration):
    unload_steps = configuration["unload_steps"]
    unload = history[unload_steps-1]
    final = history[-1]
    hold = history[unload_steps:]
    hold_target = np.asarray([row["target_growth_fraction"] for row in hold])
    unloading_work = sum(
        row["mechanical_external_work_J"] for row in history[:unload_steps])
    hold_work = sum(row["mechanical_external_work_J"] for row in hold)
    directions = [row["front_directions"].get("edge-10-20") for row in hold]
    return {
        "schema": "v66-unload-static-result-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration,
        "result": {
            "zero_strain_reached": bool(abs(unload["applied_shear_strain"]) < 1e-12),
            "unloading_external_work_J": unloading_work,
            "hold_external_work_J": hold_work,
            "unload_terminal_shear_stress_Pa": unload["mean_shear_stress_Pa"],
            "final_residual_shear_stress_Pa": final["mean_shear_stress_Pa"],
            "unload_target_change_fraction": unload["target_growth_fraction"],
            "hold_target_change_fraction": final["target_growth_fraction"],
            "hold_target_change_from_unloaded_fraction": float(
                final["target_growth_fraction"]-unload["target_growth_fraction"]),
            "hold_target_monotone": bool(
                len(hold_target) < 2 or np.all(np.diff(hold_target) >= -1e-12)),
            "hold_edge_10_20_direction_counts": {
                name: directions.count(name) for name in sorted(set(directions))},
            "final_temperature_mean_K": final["temperature_mean_K"],
            "final_temperature_contrast_K": final["temperature_contrast_K"],
            "maximum_mechanical_relative_first_law_residual": (
                endpoint.runtime.ledger.maximum_mechanical_relative_first_law_residual),
            "maximum_front_relative_energy_closure": (
                endpoint.runtime.ledger.maximum_relative_energy_closure),
            "accepted_front_events_total": endpoint.runtime.ledger.accepted_events,
            "rejected_front_events_total": endpoint.runtime.ledger.rejected_events,
        },
        "decision": {
            "retained_loaded_state_used": True,
            "static_front_active": bool(any(name is not None for name in directions)),
            "static_target_reactivated": bool(
                final["target_growth_fraction"] > unload["target_growth_fraction"]),
            "stress_free_claim": False,
            "reason": (
                "The protocol removes imposed strain through the complete mechanical "
                "ledger and then holds zero strain. Any residual internal stress remains "
                "a physical driver, so this is not called a zero-stress anneal."),
        },
        "loaded_target_volume_m3": float(loaded_volume[1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--length", type=float, default=4.8e-6)
    parser.add_argument("--interface-width", type=float, default=6e-7)
    parser.add_argument("--temperature", type=float, default=1000.0)
    parser.add_argument("--unload-rate", type=float, default=4e4)
    parser.add_argument("--unload-dt", type=float, default=2e-9)
    parser.add_argument("--unload-steps", type=int, default=900)
    parser.add_argument("--hold-dt", type=float, default=2e-8)
    parser.add_argument("--hold-steps", type=int, default=300)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    # Reuse the identical V66 coupled physical parameter builder; these aliases
    # are only its input names, not an additional physical law.
    args.shear_rate = abs(args.unload_rate)
    args.dt = args.unload_dt
    configuration = _configuration(args)
    provenance = dict(
        _source_provenance(),
        scientific_scope="retained-state unload and fixed-strain anneal")
    _, _, _, _, _, _, base_context = _physics(args)
    unload_context = replace(
        base_context, shear_rate_s=-abs(args.unload_rate),
        maximum_front_substep_s=args.unload_dt)
    hold_context = replace(
        base_context, shear_rate_s=0.0,
        maximum_front_substep_s=args.hold_dt)
    total_steps = args.unload_steps+args.hold_steps
    history_path = args.out/"history.json"
    if args.resume is None:
        state, runtime, _, load, _, _, _ = _load_checkpoint(args.source)
        if not np.isclose(load, args.unload_steps*args.unload_rate*args.unload_dt,
                          rtol=0.0, atol=1e-12):
            raise ValueError("unload discretization does not terminate at zero strain")
        endpoint = make_endpoint(state, runtime, load, unload_context)
        loaded_volume = (np.sum(state.supports, axis=(1, 2))
                         *base_context.spacing_m**2
                         *base_context.represented_thickness_m)
        history = []
        start_step = 0
        _atomic_json(args.out/"loaded_baseline.json", {
            "source": str(args.source.relative_to(ROOT)),
            "load": load, "loaded_volume_m3": loaded_volume.tolist(),
            "source_physical_time_s": runtime.ledger.physical_time_s,
        })
    else:
        state, runtime, start_step, load, loaded_volume, saved, saved_provenance = (
            _load_checkpoint(args.resume))
        if saved != configuration or saved_provenance != provenance:
            raise ValueError("resume configuration/provenance mismatch")
        context = unload_context if start_step < args.unload_steps else hold_context
        endpoint = make_endpoint(state, runtime, load, context)
        history = json.loads(history_path.read_text())
        if len(history) != start_step:
            raise ValueError("history/checkpoint mismatch")
    elapsed_physical = sum(
        args.unload_dt if index < args.unload_steps else args.hold_dt
        for index in range(start_step))
    for global_step in range(start_step+1, total_steps+1):
        unloading = global_step <= args.unload_steps
        context = unload_context if unloading else hold_context
        duration = args.unload_dt if unloading else args.hold_dt
        if global_step == args.unload_steps+1:
            endpoint = make_endpoint(
                endpoint.state, endpoint.runtime, endpoint.load, hold_context,
                numerical_hints=endpoint.numerical_hints)
        started = time.perf_counter()
        interval = advance_complete_multigrain_interval(endpoint, duration, context)
        endpoint = interval.endpoint
        elapsed_physical += duration
        row = _record(
            global_step, endpoint, interval, loaded_volume, context,
            time.perf_counter()-started)
        row["protocol_time_s"] = elapsed_physical
        row["phase"] = "unload" if unloading else "fixed_zero_strain_hold"
        history.append(row)
        if (global_step == 1 or global_step % args.checkpoint_every == 0
                or global_step == total_steps):
            temporary = args.out/"checkpoint_latest.tmp.npz"
            checkpoint = args.out/"checkpoint_latest.npz"
            _save_checkpoint(
                temporary, endpoint.state, endpoint.runtime, global_step,
                endpoint.load, loaded_volume, configuration, provenance)
            os.replace(temporary, checkpoint)
            _atomic_json(history_path, history)
            if global_step >= args.unload_steps+1:
                payload = _summary(
                    history, endpoint, loaded_volume, configuration)
                payload["decision"]["run_complete"] = global_step == total_steps
                _atomic_json(args.out/"result.json", payload)
            print(json.dumps({
                "global_step": global_step, "target": total_steps,
                "phase": row["phase"], "strain": endpoint.load,
                "stress_MPa": row["mean_shear_stress_Pa"]/1e6,
                "temperature_contrast_K": row["temperature_contrast_K"],
                "target_change_fraction": row["target_growth_fraction"],
            }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
