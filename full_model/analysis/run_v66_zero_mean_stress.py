#!/usr/bin/env python3
"""Feedback-controlled release toward zero mean shear stress.

The production map is strain controlled.  This driver therefore changes the
applied strain through ordinary complete, work-ledgered intervals.  A local
elastic secant predicts the strain needed for zero mean stress and a bounded
increment approaches it.  It is a near-zero-mean-stress control, not an
instantaneous stress-boundary-condition solver.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

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
SOURCE = ROOT/"full_model"/"verification"/"v66_unload_static"/"checkpoint_latest.npz"
DEFAULT_OUT = ROOT/"full_model"/"verification"/"v66_zero_mean_stress"


def _mean_shear(endpoint):
    return float(np.mean(
        np.asarray(endpoint.observables["common_stress_tensor_Pa"])[..., 0, 1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--dt", type=float, default=2e-9)
    parser.add_argument("--probe-strain", type=float, default=1e-4)
    parser.add_argument("--maximum-strain-change", type=float, default=5e-4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    physical_args = SimpleNamespace(
        n=16, length=4.8e-6, temperature=1000.0,
        shear_rate=0.0, dt=args.dt)
    _, _, _, _, _, _, base_context = _physics(physical_args)
    state, runtime, _, load, _, _, _ = _load_checkpoint(args.source)
    endpoint = make_endpoint(state, runtime, load, base_context)
    initial_volume = (np.sum(state.supports, axis=(1, 2))
                      *base_context.spacing_m**2
                      *base_context.represented_thickness_m)
    configuration = {
        "schema": "v66-zero-mean-stress-feedback-v1",
        "source_checkpoint": str(args.source.relative_to(ROOT)),
        "steps": args.steps, "dt_s": args.dt,
        "probe_strain": args.probe_strain,
        "maximum_strain_change_per_interval": args.maximum_strain_change,
        "controller": "bounded local elastic secant; complete work-ledgered interval",
    }
    provenance = dict(
        _source_provenance(),
        scientific_scope="near-zero-mean-stress discriminator")
    history = []
    for step in range(1, args.steps+1):
        stress0 = _mean_shear(endpoint)
        probe = make_endpoint(
            endpoint.state, endpoint.runtime, endpoint.load+args.probe_strain,
            base_context, numerical_hints=endpoint.numerical_hints)
        tangent = (_mean_shear(probe)-stress0)/args.probe_strain
        if not np.isfinite(tangent) or tangent <= 0.0:
            raise RuntimeError("mean shear-stress tangent is not positive finite")
        predicted = endpoint.load-stress0/tangent
        increment = float(np.clip(
            predicted-endpoint.load, -args.maximum_strain_change,
            args.maximum_strain_change))
        context = replace(base_context, shear_rate_s=increment/args.dt)
        started = time.perf_counter()
        interval = advance_complete_multigrain_interval(endpoint, args.dt, context)
        endpoint = interval.endpoint
        row = _record(
            step, endpoint, interval, initial_volume, context,
            time.perf_counter()-started)
        row.update({
            "controller_stress_before_Pa": stress0,
            "controller_elastic_tangent_Pa": tangent,
            "controller_predicted_zero_stress_strain": predicted,
            "controller_strain_increment": increment,
        })
        history.append(row)
        if step == 1 or step % args.checkpoint_every == 0 or step == args.steps:
            temporary = args.out/"checkpoint_latest.tmp.npz"
            _save_checkpoint(
                temporary, endpoint.state, endpoint.runtime, step,
                endpoint.load, initial_volume, configuration, provenance)
            os.replace(temporary, args.out/"checkpoint_latest.npz")
            _atomic_json(args.out/"history.json", history)
            stress = np.asarray([item["mean_shear_stress_Pa"] for item in history])
            target = np.asarray([item["target_growth_fraction"] for item in history])
            settled = np.abs(stress) <= 10e6
            first_settled = int(np.argmax(settled)) if np.any(settled) else None
            payload = {
                "schema": "v66-zero-mean-stress-result-v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "configuration": configuration,
                "result": {
                    "steps_completed": step,
                    "final_applied_shear_strain": endpoint.load,
                    "final_mean_shear_stress_Pa": float(stress[-1]),
                    "minimum_absolute_mean_shear_stress_Pa": float(
                        np.min(np.abs(stress))),
                    "first_step_below_10MPa": (
                        None if first_settled is None else first_settled+1),
                    "final_target_change_fraction": float(target[-1]),
                    "target_change_after_first_below_10MPa": (
                        None if first_settled is None else
                        float(target[-1]-target[first_settled])),
                    "cumulative_external_work_J": float(sum(
                        item["mechanical_external_work_J"] for item in history)),
                    "maximum_mechanical_relative_first_law_residual": (
                        endpoint.runtime.ledger.maximum_mechanical_relative_first_law_residual),
                    "maximum_front_relative_energy_closure": (
                        endpoint.runtime.ledger.maximum_relative_energy_closure),
                    "rejected_front_events_total": endpoint.runtime.ledger.rejected_events,
                },
                "decision": {
                    "run_complete": step == args.steps,
                    "near_zero_mean_stress_reached": first_settled is not None,
                    "stress_free_claim": False,
                    "static_rx_at_near_zero_mean_stress_supported": bool(
                        first_settled is not None and target[-1] > target[first_settled]),
                    "reason": (
                        "Mean stress is feedback-controlled by work-ledgered strain "
                        "increments; local residual stresses and finite controller error remain."),
                },
            }
            _atomic_json(args.out/"result.json", payload)
            print(json.dumps({
                "step": step, "target": args.steps,
                "load": endpoint.load, "mean_stress_MPa": stress[-1]/1e6,
                "target_change_fraction": target[-1],
            }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
