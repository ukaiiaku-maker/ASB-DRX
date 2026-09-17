#!/usr/bin/env python3
"""Reproduce the V36 non-reverse front-channel discriminator.

This is a bounded operator/integration check, not a material calibration.  It
compares the frozen V35 shared-base construction with independently priced
outgoing channels, then records equal-state, exchange, geometry-probe, proposal
amplitude, and publication-ledger controls from one small resolved bicrystal.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, checkpoint_payload, resolved_bicrystal, run_i3_cycle,
)
from full_model.production.arrhenius_kinetics import (
    ActivatedProcess, EV_J, KB_J_K, activated_rate_s, exp_floor_enthalpy_j,
)
from full_model.production.common_tensorial_wall import CommonWallDriving
from full_model.production.coupled_front_event import (
    propose_bidirectional_front_event,
)
from full_model.production.moving_front import DefectState


SCHEMA = "asb-drx/v36-front-channel-kinetics/v1"


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _context():
    context = resolved_bicrystal(
        grid=16, length_m=3.2e-6, interface_width_m=4.0e-7,
        child_line_fraction=.35, temperature_K=1100.0)
    driving = CommonWallDriving(
        mean_strain=np.array([[0.0, .01], [.01, 0.0]]),
        fixed_eigenstrain=np.zeros((16, 16, 2, 2)))
    controls = I3Controls(
        driving_pressure_a_to_b_Pa=0.0,
        applied_pressure_a_to_b_Pa=0.0,
        geometric_probe_pressure_Pa=1.0e8,
        trial_dt_s=1.0e-10, front_dt_s=1.0e-3)
    return context, context["state"], driving, controls


def _trial(context, displacement_cells):
    n = context["state"].eta.shape[0]
    spacing = context["spacing_m"]
    length = n*spacing
    width = context["interface_width_m"]
    x = np.arange(n)*spacing
    displacement = float(displacement_cells)*spacing
    child = .5*(
        np.tanh((x-(.25*length-displacement))/width)
        -np.tanh((x-(.75*length+displacement))/width))
    child = np.broadcast_to(child[:, None], (n, n))
    return np.stack((1.0-child, child), axis=2)


def _payload_digest(state, context):
    payload = checkpoint_payload(state, context)
    digest = hashlib.sha256()
    for name in sorted(payload):
        value = payload[name]
        digest.update(name.encode("utf-8"))
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _frozen_shared_base(delta_ab, delta_ba, *, temperature, event_volume,
                        event_length):
    """Evaluate, but never use, the removed V35 shared-base formula."""
    process = ActivatedProcess("frozen-v35-front", 1.0e8)
    pressure = max(abs(delta_ab), abs(delta_ba))/event_volume
    enthalpy = exp_floor_enthalpy_j(
        pressure, .35*EV_J, 1.0e9, 2.0, 1.5, .10)
    base = activated_rate_s(process, enthalpy, temperature)
    accept_ab = np.exp(-min(max(delta_ab/(KB_J_K*temperature), 0.0), 700.0))
    accept_ba = np.exp(-min(max(delta_ba/(KB_J_K*temperature), 0.0), 700.0))
    rate_ab = base*accept_ab
    rate_ba = base*accept_ba
    return {
        "classification": "FROZEN_V35_COMPARATOR_NOT_PRODUCTION",
        "shared_pressure_Pa": pressure,
        "shared_activation_enthalpy_J": enthalpy,
        "shared_transition_state_rate_s": base,
        "acceptance_a_to_b": float(accept_ab),
        "acceptance_b_to_a": float(accept_ba),
        "rate_a_to_b_s": rate_ab,
        "rate_b_to_a_s": rate_ba,
        "net_velocity_m_s": event_length*(rate_ab-rate_ba),
    }


def _explicit_event(delta_ab, delta_ba, *, event_volume, event_length,
                    temperature):
    shape = (2, 2, 4)
    state = DefectState(
        np.ones(shape), np.ones(shape), np.ones(shape), np.ones(shape[:2]))
    return propose_bidirectional_front_event(
        state, state, event_volume_m3=event_volume,
        event_length_m=event_length, line_energy_J_m=1.0e-9,
        temperature_K=temperature,
        process=ActivatedProcess("v36-control", 1.0e8),
        h0_J=.35*EV_J, critical_pressure_Pa=1.0e9,
        exp_a=2.0, exp_n=1.5, exp_floor=.10,
        transmission_fraction=.5,
        kinetic_free_energy_a_to_b_J=delta_ab,
        kinetic_free_energy_b_to_a_J=delta_ba,
        actual_reverse_edge=False)


def build_report(source_commit="WORKTREE"):
    context, initial, driving, controls = _context()
    amplitudes = (.025, .05, .10)
    amplitude_records = []
    states = {}
    audits = {}
    for amplitude in amplitudes:
        state, audit = run_i3_cycle(
            context, initial, _trial(context, amplitude), driving, controls)
        states[amplitude] = state
        audits[amplitude] = audit
        decision = audit["front_decision"]
        directional = audit["complete_directional_kinetics"]
        amplitude_records.append({
            "proposal_displacement_cells": amplitude,
            "published": audit["candidate_sweep_published"],
            "a_to_b_event_J": directional["a_to_b_event_J"],
            "b_to_a_event_J": directional["b_to_a_event_J"],
            "rate_a_to_b_s": decision["rate_a_to_b_s"],
            "rate_b_to_a_s": decision["rate_b_to_a_s"],
            "net_velocity_m_s": decision["net_velocity_a_to_b_m_s"],
            "accepted_signed_volume_m3": decision[
                "accepted_signed_volume_m3"],
            "gross_channel_activity_s": decision[
                "gross_channel_activity_s"],
        })

    base_audit = audits[.05]
    base_decision = base_audit["front_decision"]
    directional = base_audit["complete_directional_kinetics"]
    burgers = float(context["wall_parameters"].burgers_m)
    event_volume = burgers**3
    temperature = float(base_decision["channel_a_to_b"]["temperature_K"])
    delta_ab = directional["a_to_b_event_J"]
    delta_ba = directional["b_to_a_event_J"]

    equal = _explicit_event(
        0.0, 0.0, event_volume=event_volume, event_length=burgers,
        temperature=temperature)
    exchange = _explicit_event(
        delta_ba, delta_ab, event_volume=event_volume, event_length=burgers,
        temperature=temperature)

    alternate_controls = replace(
        controls, geometric_probe_pressure_Pa=-9.0e8)
    alternate_state, alternate_audit = run_i3_cycle(
        context, initial, _trial(context, .05), driving, alternate_controls)
    base_digest = _payload_digest(states[.05], context)
    alternate_digest = _payload_digest(alternate_state, context)

    energy = base_audit["complete_energy"]["front_decision"]
    return {
        "schema": SCHEMA,
        "source_commit": source_commit,
        "classification": "CHANNEL_RESOLVED_NONREVERSE_OUTGOING_KINETICS",
        "calibration_claimed": False,
        "applied_pressure_Pa": 0.0,
        "actual_reverse_edge": directional["actual_reverse_edge"],
        "reverse_edge_status": directional["reverse_edge_status"],
        "endpoint_channels": {
            "a_to_b": directional["a_to_b_endpoint"],
            "b_to_a": directional["b_to_a_endpoint"],
        },
        "corrected_rate_channels": {
            "a_to_b": base_decision["channel_a_to_b"],
            "b_to_a": base_decision["channel_b_to_a"],
            "gross_channel_activity_s": base_decision[
                "gross_channel_activity_s"],
            "net_velocity_m_s": base_decision[
                "net_velocity_a_to_b_m_s"],
        },
        "frozen_v35_comparator": _frozen_shared_base(
            delta_ab, delta_ba, temperature=temperature,
            event_volume=event_volume, event_length=burgers),
        "equal_control": {
            "rate_a_to_b_s": equal.rate_a_to_b_s,
            "rate_b_to_a_s": equal.rate_b_to_a_s,
            "net_velocity_m_s": equal.net_velocity_a_to_b_m_s,
        },
        "exchange_control": {
            "original_net_velocity_m_s": base_decision[
                "net_velocity_a_to_b_m_s"],
            "exchanged_net_velocity_m_s": exchange.net_velocity_a_to_b_m_s,
            "odd_residual_m_s": (exchange.net_velocity_a_to_b_m_s
                                  +base_decision["net_velocity_a_to_b_m_s"]),
        },
        "proposal_probe_control": {
            "probe_pressures_Pa": [1.0e8, -9.0e8],
            "base_state_sha256": base_digest,
            "alternate_state_sha256": alternate_digest,
            "states_bitwise_identical": base_digest == alternate_digest,
            "physical_decisions_identical": (
                base_audit["front_decision"]
                ==alternate_audit["front_decision"]),
            "probe_external_work_J": 0.0,
            "probe_selected_direction": False,
        },
        "proposal_amplitude_control": amplitude_records,
        "publication_accounting": {
            "candidate_sweep_published": base_audit[
                "candidate_sweep_published"],
            "accepted_signed_volume_m3": base_decision[
                "accepted_signed_volume_m3"],
            "inventory_change": base_audit["actual_inventory_change"],
            "material_sink_change_m": base_audit[
                "actual_material_sink_change_m"],
            "generated_heat_J": energy["generated_heat_J"],
            "thermostat_export_J": energy["thermostat_export_J"],
            "material_sink_export_J": energy["material_sink_export_J"],
            "first_law_residual_J": energy["first_law_residual_J"],
            "external_work_J": energy["external_work_J"],
        },
        "hard_checks": {
            "frozen_comparator_cancels": (
                _frozen_shared_base(
                    delta_ab, delta_ba, temperature=temperature,
                    event_volume=event_volume,
                    event_length=burgers)["net_velocity_m_s"] == 0.0),
            "corrected_nonzero_direction": (
                base_decision["net_velocity_a_to_b_m_s"] > 0.0),
            "equal_control_stationary": equal.net_velocity_a_to_b_m_s == 0.0,
            "exchange_is_odd": abs(
                exchange.net_velocity_a_to_b_m_s
                +base_decision["net_velocity_a_to_b_m_s"]) <= 1e-20,
            "probe_is_invariant": base_digest == alternate_digest,
            "all_amplitudes_publish_same_direction": all(
                row["published"] and row["net_velocity_m_s"] > 0.0
                for row in amplitude_records),
            "zero_external_work": energy["external_work_J"] == 0.0,
            "first_law_closed": abs(energy["first_law_residual_J"]) < 1e-24,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", default="WORKTREE")
    args = parser.parse_args()
    report = build_report(args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        report, indent=2, sort_keys=True, default=_json_default)+"\n")
    if not all(report["hard_checks"].values()):
        raise SystemExit("V36 front-channel discriminator failed")
    print(json.dumps({
        "output": str(args.output),
        "classification": report["classification"],
        "hard_checks": report["hard_checks"],
    }, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
