#!/usr/bin/env python3
"""Same-state V41 audit of complete directional front energetics.

Every row starts from an immutable checkpoint.  Both geometric directions are
constructed before publication, and the retained V40 traffic-difference law
is compared with the complete-dissipation production law.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from dataclasses import replace

from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, resolved_bicrystal, run_i3_cycle,
)
from full_model.analysis.run_v36_recurrent_physical_response import (
    driving_at_time, geometric_envelope,
)
from full_model.analysis.run_v39_common_horizon import load_stage


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact(audit, direction, fraction, law):
    decision = audit["front_decision"]
    energy = audit["complete_energy"]["front_decision"]
    directional = audit["complete_directional_kinetics"]
    endpoints = {
        key: directional[key]
        for key in ("a_to_b_endpoint", "b_to_a_endpoint")
    }
    return {
        "proposal_direction": direction,
        "proposal_fraction": fraction,
        "deterministic_rate_law": law,
        "starting_state_was_mutated": False,
        "trial_constructed": directional is not None,
        "kinetic_rule_admitted_trial": bool(decision["accepted"]),
        "complete_physical_energy_admitted_trial": bool(energy["accepted"]),
        "state_published": bool(audit["candidate_sweep_published"]),
        "kinetic_classification": decision["classification"],
        "energy_classification": energy["classification"],
        "net_velocity_a_to_b_m_s": decision["net_velocity_a_to_b_m_s"],
        "rate_a_to_b_s": decision["rate_a_to_b_s"],
        "rate_b_to_a_s": decision["rate_b_to_a_s"],
        "proposed_signed_volume_m3": decision["proposed_signed_volume_m3"],
        "accepted_signed_volume_m3": decision["accepted_signed_volume_m3"],
        "actual_complete_candidate_delta_helmholtz_J": energy[
            "delta_helmholtz_J"],
        "external_work_J": energy["external_work_J"],
        "generated_heat_J": energy["generated_heat_J"],
        "material_sink_export_J": energy["material_sink_export_J"],
        "a_to_b_event_J": directional["a_to_b_event_J"],
        "b_to_a_event_J": directional["b_to_a_event_J"],
        "directional_endpoints": endpoints,
        "channel_a_to_b": decision["channel_a_to_b"],
        "channel_b_to_a": decision["channel_b_to_a"],
        "event_volume_m3": decision["front_event_volume_m3"],
        "jump_length_m": decision["front_jump_length_m"],
        "physical_site_count": decision["physical_site_count"],
        "line_balance_residual_m": decision["maximum_abs_line_closure_m"],
        "signed_balance_residual_m2": decision[
            "maximum_abs_signed_closure_m2"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="name=path; repeat for each immutable state")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=128)
    args = parser.parse_args()
    context = resolved_bicrystal(
        grid=args.grid, length_m=3.2e-6, interface_width_m=4e-7,
        temperature_K=1100.0, child_line_fraction=.35)
    driving = driving_at_time(args.grid, .01, "hold", 0.0, 0.0)
    controls = I3Controls(
        mura_enabled=False, front_enabled=True, trial_dt_s=1.5625e-5,
        front_dt_s=1.5625e-5, front_exp_n=1.0,
        driving_pressure_a_to_b_Pa=0.0,
        applied_pressure_a_to_b_Pa=0.0)
    states = []
    for item in args.checkpoint:
        name, raw_path = item.split("=", 1)
        path = Path(raw_path).resolve()
        state, metadata = load_stage(path, context)
        rows = []
        for fraction in (.0625, .03125, .015625, .0078125):
            for direction in (1, -1):
                proposal = geometric_envelope(
                    state.eta, fraction=fraction, direction=direction)
                for law in ("legacy_independent_metropolis",
                            "complete_dissipation"):
                    _, audit = run_i3_cycle(
                        context, state, proposal, driving,
                        replace(controls, deterministic_front_rate_law=law))
                    rows.append(compact(audit, direction, fraction, law))
        states.append({
            "name": name,
            "checkpoint": str(path),
            "checkpoint_sha256": digest(path),
            "stage_metadata": {key: metadata.get(key) for key in (
                "source_sha", "stage", "grid", "macro_dt_s",
                "interval_index", "completed_intervals", "physical_time_s")},
            "trials": rows,
        })
    result = {
        "schema": "asb-drx/v41/front-direction-energy-decision/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "grid": args.grid,
        "zero_applied_front_work": True,
        "candidate_status_fields_are_separate": True,
        "states": states,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
