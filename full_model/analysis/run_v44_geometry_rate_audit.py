#!/usr/bin/env python3
"""Independent geometry/continuum reconstruction and timed V44 response."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v43_geometry_verification import prepare_block
from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.lattice_line_geometry import (
    geometry_link_nye_mimetic, geometry_link_nye_spectral_transfer,
    geometry_plastic_distortion,
    geometry_surface_nye_mimetic,
)
from full_model.production.tensorial_nye import nye_from_plastic_distortion
from full_model.production.v24_mechanical_wall import (
    V43GeometryKinetics, accepted_geometry_plaquette_transaction,
)


ROOT = Path("full_model/verification")


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value)**2)))


def fixed_square(n):
    # Same 0.8 um square in a 3.2 um periodic section.
    width = n//4; start = (n-width)//2
    cells = [(i, j) for i in range(start, start+width)
             for j in range(start, start+width)]
    return prepare_block(n, cells)[0]


def geometry_connection():
    rows = []
    for n in (16, 32, 64):
        state = fixed_square(n)
        link = geometry_link_nye_mimetic(state.geometry)
        transferred = geometry_link_nye_spectral_transfer(state.geometry)
        surface = geometry_surface_nye_mimetic(state.geometry)
        family_beta = geometry_plastic_distortion(state.geometry)
        spectral = np.stack([
            nye_from_plastic_distortion(family_beta[..., a, :, :],
                                        float(state.geometry.spacing_m))
            for a in range(link.shape[2])], axis=2)
        line_length = float(np.sum(
            np.abs(state.geometry.edge_x_quanta)
            +np.abs(state.geometry.edge_y_quanta))*state.geometry.spacing_m)
        scale = max(rms(link), 1e-300)
        rows.append({
            "grid": n, "spacing_m": float(state.geometry.spacing_m),
            "line_length_m": line_length,
            "link_nye_rms_m1": rms(link),
            "link_surface_commuting_residual_rms_m1": rms(link-surface),
            "link_surface_commuting_relative_rms": rms(link-surface)/scale,
            "spectral_transfer_residual_rms_m1": rms(link-spectral),
            "spectral_transfer_relative_rms": rms(link-spectral)/scale,
            "explicit_link_to_spectral_residual_rms_m1": rms(
                transferred-spectral),
            "explicit_link_to_spectral_relative_rms": rms(
                transferred-spectral)/max(rms(spectral), 1e-300),
        })
    return rows


def timed_response():
    state, data, _ = prepare_block(16, [(7, 7)])
    _, loading, _, _, systems, topologies, common, extensive, _, _ = data
    kinetics = V43GeometryKinetics(
        ActivatedProcess("v44-represented-sweep", 1e12,
                         entropy_over_kB=-.2, negative_barrier_mode="drag"),
        enthalpy_J=.02*EV_J, critical_stress_Pa=1e9,
        material_exchange_model="equilibrated_point_defect_reservoir")
    requested_horizon = 1e-9
    elapsed = 0.0; attempted = 0; accepted = 0; rows = []
    path = ((8, 7), (8, 8), (7, 8), (6, 8))
    while elapsed < requested_horizon-1e-18 and attempted < 32:
        cell = path[attempted % len(path)]
        remaining = requested_horizon-elapsed
        event = {"cell": cell, "family": 0, "burgers_sign": 1,
                 "proposed_extent": 1.0, "search_partial_extent": True,
                 "partial_extent_levels": 18}
        candidate, ledger = accepted_geometry_plaquette_transaction(
            state, event, systems, topologies, loading, common, extensive,
            kinetics, remaining)
        consumed = float(ledger.get("accepted_time_s", 0.0))
        rows.append({
            "attempt": attempted, "cell": list(cell),
            "state_event_count_before": int(state.geometry.accepted_event_count),
            "accepted": bool(ledger["accepted"]),
            "classification": ledger["classification"],
            "extent": float(ledger.get("accepted_extent", 0.0)),
            "event_rate_s": float(ledger["event_rate_s"]),
            "requested_remaining_time_s": remaining,
            "accepted_time_s": consumed,
            "rate_exposure": float(ledger.get("accepted_rate_exposure", 0.0)),
            "physical_plaquette_area_m2": float(
                ledger.get("physical_plaquette_area_m2", 0.0)),
            "swept_area_m2": float(ledger.get("swept_area_m2", 0.0)),
            "same_state_partial_search": ledger.get(
                "same_state_partial_extent_search"),
            "mechanism": ledger.get("mechanism"),
            "material_exchange_model": ledger.get("material_exchange_model"),
        })
        attempted += 1
        if not ledger["accepted"]:
            # No represented nearby direction in the four-cell path advanced.
            # Move to the next path direction without advancing physical time.
            if attempted % len(path) == 0 and not any(
                    row["accepted"] for row in rows[-len(path):]):
                break
            continue
        state = candidate; accepted += 1; elapsed += consumed
        if consumed <= 0.0:
            raise RuntimeError("accepted geometry event made no clock progress")
    return {
        "requested_time_s": requested_horizon,
        "accepted_time_s": elapsed,
        "unintegrated_time_s": requested_horizon-elapsed,
        "attempted_updates": attempted, "accepted_updates": accepted,
        "final_event_count": int(state.geometry.accepted_event_count),
        "final_line_length_m": float(np.sum(
            np.abs(state.geometry.edge_x_quanta)
            +np.abs(state.geometry.edge_y_quanta))*state.geometry.spacing_m),
        "events": rows,
        "clock_classification": (
            "HORIZON_COMPLETE" if elapsed >= requested_horizon-1e-18
            else "CONSTRAINED_ARREST_NO_REPRESENTED_PATH"),
    }


def main():
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    payload = {
        "schema": "asb-drx/v44/geometry-rate-audit/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha": source,
        "independent_geometry_connection": geometry_connection(),
        "timed_response": timed_response(),
        "grain_label_allocation_possible": False,
        "drx_claimed": False, "strict_asb_claimed": False,
    }
    out = ROOT/"v44_geometry_rate_audit.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "clock_classification": payload["timed_response"]["clock_classification"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
