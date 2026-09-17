#!/usr/bin/env python3
"""Evaluate the preregistered V37 front-rate hypotheses from actual endpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.coupled_front_event import (
    propose_bidirectional_front_event,
)
from full_model.production.moving_front import DefectState


SCHEMA = "asb-drx/v37/front-parameter-screen/v1"


def evaluate(registry, endpoint_a_to_b_J, endpoint_b_to_a_J):
    fixed = registry["fixed"]
    b = float(fixed["burgers_m"])
    temperature = float(fixed["temperature_K"])
    shape = (2, 2, 4)
    unit = DefectState(
        np.ones(shape), np.ones(shape), np.ones(shape), np.ones(shape[:2]))
    records = []
    for case in registry["cases"]:
        volume = float(case["event_volume_b3"])*b**3
        length = float(case["jump_length_b"])*b
        site_area = volume/length
        process = ActivatedProcess(
            case["name"], float(case["attempt_frequency_s"]),
            float(case["activation_entropy_kB"]),
            negative_barrier_mode=fixed["negative_free_barrier_policy"])
        event = propose_bidirectional_front_event(
            unit, unit, event_volume_m3=volume, event_length_m=length,
            line_energy_J_m=1.0e-9, temperature_K=temperature,
            process=process,
            h0_J=float(case["activation_h0_eV"])*EV_J,
            critical_pressure_Pa=float(fixed["critical_pressure_Pa"]),
            exp_a=float(case["exp_a"]), exp_n=float(case["exp_n"]),
            exp_floor=float(fixed["exp_floor"]),
            transmission_fraction=.5,
            kinetic_free_energy_a_to_b_J=float(endpoint_a_to_b_J),
            kinetic_free_energy_b_to_a_J=float(endpoint_b_to_a_J),
            availability_a_to_b=float(case["availability"]),
            availability_b_to_a=float(case["availability"]),
            actual_reverse_edge=False)
        speed = event.net_velocity_a_to_b_m_s
        records.append({
            "name": case["name"], "parameters": case,
            "event_volume_m3": volume, "jump_length_m": length,
            "site_area_m2": site_area,
            "site_density_m-2": 1.0/site_area,
            "identifiable_prefactor_s": process.identifiable_prefactor_s,
            "channel_a_to_b": asdict(event.a_to_b_channel),
            "channel_b_to_a": asdict(event.b_to_a_channel),
            "gross_activity_s": event.rate_a_to_b_s+event.rate_b_to_a_s,
            "net_velocity_m_s": speed,
            "time_to_quarter_width_s": (
                np.inf if speed <= 0.0 else .25*4.0e-7/speed),
            "rate_screen_only_not_trajectory": True,
        })
    positive = sorted(
        (row for row in records if row["net_velocity_m_s"] > 0.0),
        key=lambda row: row["net_velocity_m_s"])
    baseline = next(row for row in records if row["name"] == "baseline")
    intermediate = positive[len(positive)//2]
    measurable = positive[-1]
    return {
        "schema": SCHEMA,
        "classification": "PREREGISTERED_RATE_SCREEN_NOT_TRAJECTORY_OR_CALIBRATION",
        "endpoint_a_to_b_J": float(endpoint_a_to_b_J),
        "endpoint_b_to_a_J": float(endpoint_b_to_a_J),
        "registry_schema": registry["schema"],
        "case_count": len(records),
        "records": records,
        "provisional_full_solver_candidates": {
            "slow_reference": baseline["name"],
            "intermediate": intermediate["name"],
            "measurable": measurable["name"],
            "selection_requires_full_solver_verification": True
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    controls = json.loads(args.controls.read_text())
    endpoints = controls["material_exchange_control"]["original"][
        "complete_candidate_endpoints"]
    result = evaluate(
        registry, endpoints["a_to_b_event_J"], endpoints["b_to_a_event_J"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result["provisional_full_solver_candidates"], indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
