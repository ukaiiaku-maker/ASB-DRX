#!/usr/bin/env python3
"""Bounded current-source response family for the V43 geometry channel."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v43_geometry_verification import prepare_block
from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.v24_mechanical_wall import (
    V43GeometryKinetics, accepted_v24_mechanical_step,
)


def run_case(name, *, enabled, enthalpy_eV, entropy_over_kB,
             attempt_frequency_s=1e12, release=False):
    state, data, _ = prepare_block(16, [(7, 7)])
    _, loading, unloading, support, systems, topologies, common, extensive, oldk, dx = data
    driving = unloading if release else loading
    kinetics = V43GeometryKinetics(
        ActivatedProcess(
            "represented-plaquette-sweep", attempt_frequency_s,
            entropy_over_kB=entropy_over_kB, negative_barrier_mode="drag"),
        enthalpy_J=enthalpy_eV*EV_J, critical_stress_Pa=1e9,
        exp_a=2.2, exp_n=2.5, exp_floor=.05)
    path = ((8, 7), (8, 8), (7, 8), (6, 8))
    events = []
    for cell in path:
        event = ({"cell": cell, "family": 0, "burgers_sign": 1,
                  "proposed_extent": 1.0} if enabled else None)
        state, ledger = accepted_v24_mechanical_step(
            state, driving, support, systems, topologies, common, extensive,
            oldk, 1e-9, topology_route_enabled=False,
            geometry_event=event,
            geometry_kinetics=kinetics if enabled else None)
        item = ledger["geometry_event_energy_kinematics"]
        if item is not None:
            events.append({
                "cell": list(cell), "accepted": bool(item["accepted"]),
                "classification": item["classification"],
                "extent": float(item.get("accepted_extent", 0.0)),
                "event_rate_s": float(item["event_rate_s"]),
                "complete_energy_change_J_m3_cells": float(
                    item.get("complete_energy_change_J_m3_cells", 0.0)),
                "heat_J_m3_cells": float(np.sum(
                    item["irreversible_heat_increment_J_m3"])),
            })
    alpha = np.sum(state.common.family_nye_m1, axis=2)
    return {
        "name": name, "geometry_enabled": enabled,
        "loading_history": "fixed_total_strain_release" if release
                           else "fixed_total_strain_heterogeneous_load",
        "enthalpy_eV": enthalpy_eV,
        "entropy_over_kB": entropy_over_kB,
        "attempt_frequency_s": attempt_frequency_s,
        "accepted_nonzero_event_count": sum(
            row["accepted"] and row["extent"] != 0.0 for row in events),
        "events": events,
        "persistent_event_count": int(state.geometry.accepted_event_count),
        "line_length_m": float(np.sum(
            np.abs(state.geometry.edge_x_quanta)
            +np.abs(state.geometry.edge_y_quanta))*dx),
        "beta_p_rms": float(np.sqrt(np.mean(state.common.beta_p**2))),
        "nye_rms_m1": float(np.sqrt(np.mean(alpha**2))),
        "orientation_span_deg": float(np.ptp(state.common.orientation_rad)*180/np.pi),
        "maximum_temperature_K": float(np.max(state.common.temperature_K)),
        "grain_count": 1,
    }


def main():
    cases = [
        run_case("disabled_control", enabled=False, enthalpy_eV=.02,
                 entropy_over_kB=-.2),
        run_case("low_barrier_load", enabled=True, enthalpy_eV=.02,
                 entropy_over_kB=-.2),
        run_case("low_barrier_release", enabled=True, enthalpy_eV=.02,
                 entropy_over_kB=-.2, release=True),
        run_case("mid_barrier_load", enabled=True, enthalpy_eV=.2,
                 entropy_over_kB=-.2, attempt_frequency_s=1e9),
        run_case("high_barrier_load", enabled=True, enthalpy_eV=.5,
                 entropy_over_kB=-.2, attempt_frequency_s=1e8),
        run_case("signed_entropy_hypothesis", enabled=True, enthalpy_eV=.2,
                 entropy_over_kB=.5, attempt_frequency_s=1e9),
    ]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = {
        "schema": "asb-drx/v43/physical-response/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source,
        "parameter_scope": (
            "generic response hypotheses, not material calibration; EXP-floor "
            "enthalpy 0.02-0.5 eV and signed entropy -0.2 to +0.5 kB"),
        "shared_parameters": {
            "attempt_frequency_range_s": [1e8, 1e12],
            "critical_stress_Pa": 1e9,
            "exp_a": 2.2, "exp_n": 2.5, "exp_floor": .05,
            "external_front_pressure_Pa": 0.0,
            "geometry_external_work_J": 0.0,
        },
        "cases": cases,
        "qualified_nonzero_physical_path": any(
            row["accepted_nonzero_event_count"] >= 1 for row in cases
            if row["geometry_enabled"]),
        "disabled_control_changed_geometry": cases[0]["persistent_event_count"] != 1,
        "grain_label_allocation_possible": False,
        "drx_claimed": False, "strict_asb_claimed": False,
        "classification": (
            "BOUNDED_GEOMETRY_RESPONSE_FAMILY_COMPLETED;"
            "FIRST_ADVANCE_ACCEPTED_SUBSEQUENT_PATH_ENERGY_PINNED"),
    }
    out = Path("full_model/verification/v43_physical_response.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    names = [row["name"] for row in cases]; x = np.arange(len(names))
    axes[0].bar(x, [row["accepted_nonzero_event_count"] for row in cases])
    axes[0].set_ylabel("accepted events")
    axes[1].bar(x, [row["nye_rms_m1"] for row in cases])
    axes[1].set_ylabel("Nye RMS [m$^{-1}$]")
    axes[2].bar(x, [row["line_length_m"] for row in cases])
    axes[2].set_ylabel("represented line [m]")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7)
    fig.savefig("full_model/verification/v43_physical_response.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"source_sha": source,
                      "qualified_nonzero_physical_path": result[
                          "qualified_nonzero_physical_path"]}, sort_keys=True))


if __name__ == "__main__":
    main()
