#!/usr/bin/env python3
"""Restartable V36 recurrent response on the authoritative common state.

The phase field supplied to :func:`run_i3_cycle` is only a resolved geometric
envelope.  Complete-state directional energies and production kinetics choose
whether any prefix is publishable.  The envelope does no physical work and is
therefore reported separately from the accepted physical sweep.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from full_model.analysis.run_v34_finite_coupled_response import (
    I3Controls, checkpoint_payload, resolved_bicrystal, run_i3_cycle,
    state_from_payload,
)
from full_model.production.common_tensorial_wall import CommonWallDriving


SCHEMA = "asb-drx/v36/recurrent-physical-response/v1"


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_commit():
    frozen = os.environ.get("V36_SOURCE_SHA", "").strip()
    if frozen:
        return frozen
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()


def geometric_envelope(eta, fraction=0.125, direction=1):
    """Return a topology-preserving one-cell dilation/erosion envelope.

    This bounded stencil is deliberately not a kinetic equation.  It exposes
    both existing interfaces to the production transaction while the signed
    complete-state rate limits the accepted prefix.  ``fraction`` controls
    proposal resolution only and must pass subdivision independence checks.
    """
    eta = np.asarray(eta, dtype=float)
    if eta.ndim != 3 or eta.shape[2] != 2:
        raise ValueError("V36 recurrent response requires two phase fields")
    if not 0.0 < float(fraction) <= 1.0 or direction not in (-1, 1):
        raise ValueError("invalid geometric envelope control")
    child = eta[:, :, 1]
    if direction > 0:
        bound = np.maximum.reduce((child, np.roll(child, 1, axis=0),
                                   np.roll(child, -1, axis=0)))
    else:
        bound = np.minimum.reduce((child, np.roll(child, 1, axis=0),
                                   np.roll(child, -1, axis=0)))
    proposed_child = np.clip(
        child+float(fraction)*(bound-child), 0.0, 1.0)
    return np.stack((1.0-proposed_child, proposed_child), axis=2)


def driving_at_time(grid, initial_shear, protocol, strain_rate_s, time_s):
    shear = float(initial_shear)
    if protocol == "continued_deformation":
        shear += float(strain_rate_s)*float(time_s)
    fixed = np.zeros((int(grid), int(grid), 2, 2))
    return CommonWallDriving(
        mean_strain=np.array([[0.0, shear], [shear, 0.0]]),
        fixed_eigenstrain=fixed)


def _record(index, time_s, driving, audit):
    front = audit["front_decision"]
    kinetic = audit["complete_directional_kinetics"]
    measure = audit["physical_site_event_measure"]
    signed_sweep = float(audit["sweep"]["net_m3"])
    interface_area = (None if front is None else
                      float(front["interface_area_m2"]))
    record = {
        "interval": int(index),
        "physical_time_end_s": float(time_s),
        "mean_shear_strain": float(driving.mean_strain[0, 1]),
        "front_classification": None if front is None else front["classification"],
        "front_channel_diagnostics": front,
        "front_published": bool(audit["candidate_sweep_published"]),
        "signed_sweep_m3": signed_sweep,
        "absolute_sweep_m3": float(audit["sweep"]["absolute_m3"]),
        "positive_sweep_m3": float(audit["sweep"]["positive_m3"]),
        "negative_sweep_m3": float(audit["sweep"]["negative_m3"]),
        "proposed_signed_volume_m3": (None if front is None else float(
            front["proposed_signed_volume_m3"])),
        "interface_area_m2": interface_area,
        "accepted_contour_displacement_m": (
            0.0 if not interface_area else signed_sweep/interface_area),
        "component_contour_displacements_m": [float(
            row["normal_displacement_m"])
            for row in audit["sweep"]["components"]],
        "child_fraction_change": float(audit["phase"]["child_fraction_change"]),
        "rate_a_to_b_per_site_s": None if front is None else float(
            front["rate_a_to_b_s"]),
        "rate_b_to_a_per_site_s": None if front is None else float(
            front["rate_b_to_a_s"]),
        "net_velocity_m_s": None if front is None else float(
            front["net_velocity_a_to_b_m_s"]),
        "gross_expected_events": None if measure is None else float(
            measure["expected_events_a_to_b"]+measure["expected_events_b_to_a"]),
        "net_expected_events": None if measure is None else float(
            measure["expected_signed_event_count"]),
        "directional_event_energy_J": None if kinetic is None else {
            "a_to_b": float(kinetic["a_to_b_event_J"]),
            "b_to_a": float(kinetic["b_to_a_event_J"]),
        },
        "inventory_change": audit["actual_inventory_change"],
        "boundary_inventory_change_m": float(
            audit["actual_boundary_inventory_change_m"]),
        "material_sink_change_m": float(audit["actual_material_sink_change_m"]),
        "processed_line_change_m": float(
            audit["actual_processed_line_change_m"]),
        "temperature_range_K": audit["temperature_range_K"],
        "mura": audit["mura"],
        "complete_energy_delta_J": float(
            audit["complete_energy"]["delta_helmholtz_J"]),
    }
    # Checkpoint metadata is JSON.  Normalize tuples and NumPy scalar subclasses
    # immediately so continuous and restarted records have identical semantics.
    return json.loads(json.dumps(record, sort_keys=True))


def _write_checkpoint(path, state, context, metadata):
    payload = checkpoint_payload(state, context)
    payload["v36_response_metadata_json"] = np.asarray(json.dumps(
        metadata, sort_keys=True))
    temporary = path.with_suffix(path.suffix+".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _read_checkpoint(path, context):
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]).copy()
                   for name in archive.files}
    metadata = json.loads(str(payload.pop(
        "v36_response_metadata_json").item()))
    if metadata.get("schema") != SCHEMA:
        raise ValueError("unsupported V36 response checkpoint")
    return state_from_payload(payload, context), metadata


def run_response(*, output_dir, protocol, grid=16, intervals=10,
                 dt_s=2.0e-9, initial_shear=0.01, strain_rate_s=1.0e3,
                 temperature_K=1100.0, child_line_fraction=0.35,
                 length_m=3.2e-6, interface_width_m=4.0e-7,
                 proposal_fraction=0.125, proposal_direction=1,
                 front_enabled=True, mura_enabled=True,
                 checkpoint_every=10,
                 front_activation_h0_eV=.35, front_exp_a=2.0,
                 front_exp_n=1.5, front_exp_floor=.10,
                 front_attempt_frequency_s=1.0e8,
                 front_activation_entropy_kB=0.0,
                 front_event_volume_b3=1.0, front_jump_length_b=1.0,
                 front_symmetric_availability=1.0,
                 resume=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = resolved_bicrystal(
        grid=grid, length_m=length_m, interface_width_m=interface_width_m,
        temperature_K=temperature_K,
        child_line_fraction=child_line_fraction)
    source = _source_commit()
    records = []
    start = 0
    physical_time = 0.0
    state = context["state"]
    if resume is not None:
        state, saved = _read_checkpoint(Path(resume), context)
        if saved.get("source_commit") != source:
            raise ValueError("restart source differs from the frozen checkpoint source")
        immutable = saved["configuration"]
        # V36 checkpoints predate the explicit V37 kinetic-family controls.
        # Their implicit values are exactly the defaults below, so schema
        # migration preserves the frozen trajectory rather than invalidating it.
        immutable = dict(immutable)
        immutable.setdefault("front_activation_h0_eV", .35)
        immutable.setdefault("front_exp_a", 2.0)
        immutable.setdefault("front_exp_n", 1.5)
        immutable.setdefault("front_exp_floor", .10)
        immutable.setdefault("front_attempt_frequency_s", 1.0e8)
        immutable.setdefault("front_activation_entropy_kB", 0.0)
        immutable.setdefault("front_event_volume_b3", 1.0)
        immutable.setdefault("front_jump_length_b", 1.0)
        immutable.setdefault("front_symmetric_availability", 1.0)
        requested = {
            "protocol": protocol, "grid": int(grid), "dt_s": float(dt_s),
            "initial_shear": float(initial_shear),
            "strain_rate_s": float(strain_rate_s),
            "temperature_K": float(temperature_K),
            "child_line_fraction": float(child_line_fraction),
            "length_m": float(length_m),
            "interface_width_m": float(interface_width_m),
            "proposal_fraction": float(proposal_fraction),
            "proposal_direction": int(proposal_direction),
            "front_enabled": bool(front_enabled),
            "mura_enabled": bool(mura_enabled),
            "front_activation_h0_eV": float(front_activation_h0_eV),
            "front_exp_a": float(front_exp_a),
            "front_exp_n": float(front_exp_n),
            "front_exp_floor": float(front_exp_floor),
            "front_attempt_frequency_s": float(front_attempt_frequency_s),
            "front_activation_entropy_kB": float(front_activation_entropy_kB),
            "front_event_volume_b3": float(front_event_volume_b3),
            "front_jump_length_b": float(front_jump_length_b),
            "front_symmetric_availability": float(front_symmetric_availability),
        }
        if immutable != requested:
            raise ValueError("restart configuration differs from checkpoint")
        records = list(saved["records"])
        start = int(saved["completed_intervals"])
        physical_time = float(saved["physical_time_s"])
    configuration = {
        "protocol": protocol, "grid": int(grid), "dt_s": float(dt_s),
        "initial_shear": float(initial_shear),
        "strain_rate_s": float(strain_rate_s),
        "temperature_K": float(temperature_K),
        "child_line_fraction": float(child_line_fraction),
        "length_m": float(length_m),
        "interface_width_m": float(interface_width_m),
        "proposal_fraction": float(proposal_fraction),
        "proposal_direction": int(proposal_direction),
        "front_enabled": bool(front_enabled),
        "mura_enabled": bool(mura_enabled),
        "front_activation_h0_eV": float(front_activation_h0_eV),
        "front_exp_a": float(front_exp_a),
        "front_exp_n": float(front_exp_n),
        "front_exp_floor": float(front_exp_floor),
        "front_attempt_frequency_s": float(front_attempt_frequency_s),
        "front_activation_entropy_kB": float(front_activation_entropy_kB),
        "front_event_volume_b3": float(front_event_volume_b3),
        "front_jump_length_b": float(front_jump_length_b),
        "front_symmetric_availability": float(front_symmetric_availability),
    }
    controls = I3Controls(
        mura_enabled=bool(mura_enabled), front_enabled=bool(front_enabled),
        driving_pressure_a_to_b_Pa=0.0,
        applied_pressure_a_to_b_Pa=0.0,
        geometric_probe_pressure_Pa=1.0e8,
        trial_dt_s=dt_s, front_dt_s=dt_s,
        front_activation_h0_eV=front_activation_h0_eV,
        front_exp_a=front_exp_a, front_exp_n=front_exp_n,
        front_exp_floor=front_exp_floor,
        front_attempt_frequency_s=front_attempt_frequency_s,
        front_activation_entropy_kB=front_activation_entropy_kB,
        front_event_volume_b3=front_event_volume_b3,
        front_jump_length_b=front_jump_length_b,
        front_symmetric_availability=front_symmetric_availability)
    wall_start = time.monotonic()
    for index in range(start, int(intervals)):
        driving = driving_at_time(
            grid, initial_shear, protocol, strain_rate_s, physical_time)
        envelope = geometric_envelope(
            state.eta, proposal_fraction, direction=proposal_direction)
        state, audit = run_i3_cycle(
            context, state, envelope, driving, controls)
        if mura_enabled:
            accepted_dt = float(audit["mura"]["accepted_dt_s"])
            if not np.isclose(accepted_dt, dt_s, rtol=0.0,
                              atol=32*np.finfo(float).eps*max(dt_s, 1e-300)):
                raise RuntimeError("Mura and front failed the declared common clock")
        physical_time += float(dt_s)
        records.append(_record(index, physical_time, driving, audit))
        if ((index+1) % int(checkpoint_every) == 0
                or index+1 == int(intervals)):
            metadata = {
                "schema": SCHEMA, "source_commit": source,
                "configuration": configuration,
                "completed_intervals": index+1,
                "physical_time_s": physical_time,
                "records": records,
            }
            _write_checkpoint(
                output_dir/f"checkpoint_{index+1:06d}.npz",
                state, context, metadata)
    accepted = sum(item["front_published"] for item in records)
    cumulative = float(sum(item["signed_sweep_m3"] for item in records))
    gross = float(sum(item["gross_expected_events"] or 0.0 for item in records))
    net = float(sum(item["net_expected_events"] or 0.0 for item in records))
    classification = (
        "STORED_ENERGY_DRIVEN_MIGRATION_WITHOUT_APPLIED_FRONT_WORK"
        if accepted and cumulative != 0.0 else
        "INSUFFICIENT_PHYSICAL_EXPOSURE_OR_KINETIC_ARREST")
    result = {
        "schema": SCHEMA, "created_utc": _utc_now(),
        "source_commit": source, "configuration": configuration,
        "applied_front_work_Pa": 0.0,
        "geometric_probe_is_nonphysical_and_unledgered": True,
        "completed_intervals": len(records),
        "physical_time_s": physical_time,
        "accepted_front_intervals": accepted,
        "cumulative_signed_sweep_m3": cumulative,
        "gross_expected_event_count": gross,
        "net_expected_event_count": net,
        "wall_seconds_this_invocation": time.monotonic()-wall_start,
        "classification": classification,
        "records": records,
    }
    result_path = output_dir/"result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    result["result_sha256"] = hashlib.sha256(
        result_path.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=("hold", "continued_deformation"),
                        required=True)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--intervals", type=int, default=10)
    parser.add_argument("--dt-s", type=float, default=2.0e-9)
    parser.add_argument("--initial-shear", type=float, default=0.01)
    parser.add_argument("--strain-rate-s", type=float, default=1.0e3)
    parser.add_argument("--temperature-K", type=float, default=1100.0)
    parser.add_argument("--child-line-fraction", type=float, default=0.35)
    parser.add_argument("--length-m", type=float, default=3.2e-6)
    parser.add_argument("--interface-width-m", type=float, default=4.0e-7)
    parser.add_argument("--proposal-fraction", type=float, default=0.125)
    parser.add_argument("--proposal-direction", type=int, choices=(-1, 1),
                        default=1)
    parser.add_argument("--disable-front", action="store_false",
                        dest="front_enabled")
    parser.add_argument("--disable-mura", action="store_false",
                        dest="mura_enabled")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--front-activation-h0-eV", type=float, default=.35)
    parser.add_argument("--front-exp-a", type=float, default=2.0)
    parser.add_argument("--front-exp-n", type=float, default=1.5)
    parser.add_argument("--front-exp-floor", type=float, default=.10)
    parser.add_argument("--front-attempt-frequency-s", type=float, default=1.0e8)
    parser.add_argument("--front-activation-entropy-kB", type=float, default=0.0)
    parser.add_argument("--front-event-volume-b3", type=float, default=1.0)
    parser.add_argument("--front-jump-length-b", type=float, default=1.0)
    parser.add_argument("--front-symmetric-availability", type=float, default=1.0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    result = run_response(**vars(args))
    print(json.dumps({key: result[key] for key in (
        "classification", "completed_intervals", "physical_time_s",
        "accepted_front_intervals", "cumulative_signed_sweep_m3",
        "wall_seconds_this_invocation")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
