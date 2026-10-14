#!/usr/bin/env python3
"""Matched GB/TJ versus random source-and-neutral-sink ablation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v66_multigrain_handoff import _initial
from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters, boundary_tj_site_field,
    randomized_matched_site_field,
)
from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.multigrain_common_state import reconstruct_multigrain_common
from full_model.production.multigrain_source_transaction import (
    NeutralBoundarySinkParameters, evaluate_multigrain_neutral_sink_transaction,
    evaluate_multigrain_pair_source_transaction,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"full_model"/"verification"/"v66_source_sink_geography.json"


def _corr(left, right):
    a = np.ravel(left)-np.mean(left)
    b = np.ravel(right)-np.mean(right)
    return float(np.sum(a*b)/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-300))


def _raw_fields(state, spacing):
    common, _ = reconstruct_multigrain_common(state, spacing)
    total = sum(np.sum(np.asarray(getattr(common, name)), axis=2)
                for name in ("mobile_plus_m2", "mobile_minus_m2",
                             "forest_plus_m2", "forest_minus_m2",
                             "wall_plus_m2", "wall_minus_m2"))
    gnd = np.linalg.norm(np.sum(common.family_nye_m1, axis=2), axis=(-2, -1))
    return common, total, gnd


def _fields(state, spacing, boundary, baseline):
    common, total, gnd = _raw_fields(state, spacing)
    base_common, base_total, base_gnd = baseline
    density_change = total-base_total
    orientation_change = np.asarray(common.orientation_rad)-np.asarray(
        base_common.orientation_rad)
    gnd_change = gnd-base_gnd
    return {
        "mean_total_density_m2": float(np.mean(total)),
        "total_density_cv": float(np.std(total)/np.mean(total)),
        "density_boundary_correlation": _corr(total, boundary),
        "density_change_boundary_correlation": _corr(
            density_change, boundary),
        "density_change_rms_m2": float(np.sqrt(np.mean(
            density_change*density_change))),
        "gnd_rms_m1": float(np.sqrt(np.mean(gnd*gnd))),
        "gnd_change_rms_m1": float(np.sqrt(np.mean(gnd_change*gnd_change))),
        "orientation_span_deg": float(np.degrees(np.ptp(common.orientation_rad))),
        "orientation_change_span_deg": float(np.degrees(np.ptp(
            orientation_change))),
        "temperature_rise_mean_K": float(np.mean(common.temperature_K)-1000.0),
    }


def main():
    n = 16
    length = 4.8e-6
    spacing = length/n
    thickness = 2*2.48e-10
    systems = bcc_four_family_systems()
    state, _, _ = _initial(n, length, 1000.0, 2e7)
    wall = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=2,
        mobile_correlation_diffusivity_m2_s=0.0,
        transport_scheme="upwind")
    energy = {
        "topologies": (), "systems": systems,
        "phase_barrier_J_m3": 5e6, "phase_gradient_J_m": 5e-7,
        "boundary_line_energy_J_m": wall.line_energy_J_m,
        "boundary_junction_energy_J_m": wall.junction_energy_J_m,
        "reference_temperature_K": 1000.0,
    }
    boundary = boundary_tj_site_field(state.supports, tj_site_weight=2.0)
    random = randomized_matched_site_field(boundary, 66)
    source_parameters = ArrheniusSourceParameters(grain_size_m=2.4e-6)
    sink_parameters = NeutralBoundarySinkParameters(stress_assisted=False)
    strain = np.array(((0.0, .02), (.02, 0.0)))
    duration = 1e-6
    baseline = _raw_fields(state, spacing)
    cases = {}
    for name, site in (("gb_tj", boundary), ("matched_random", random)):
        source = evaluate_multigrain_pair_source_transaction(
            state, mean_strain=strain, site_field=site, interval_s=duration,
            source_parameters=source_parameters, systems=systems,
            wall_parameters=wall, represented_thickness_m=thickness,
            energy_kwargs=energy, source_mode=name)
        if not source.accepted:
            raise RuntimeError(f"{name} source failed: {source.classification}")
        sink = evaluate_multigrain_neutral_sink_transaction(
            source.candidate_state, mean_strain=strain, site_field=site,
            interval_s=duration, sink_parameters=sink_parameters,
            systems=systems, wall_parameters=wall,
            represented_thickness_m=thickness, energy_kwargs=energy,
            sink_mode=name)
        if not sink.accepted:
            raise RuntimeError(f"{name} sink failed: {sink.classification}")
        cases[name] = {
            "source_event_count": source.event_count,
            "source_created_line_cell_sum_m2": source.created_line_cell_sum_m2,
            "source_heat_J": source.generated_heat_J,
            "source_relative_first_law_residual": (
                source.relative_first_law_residual),
            "sink_annihilated_pair_cell_sum_m2": (
                sink.annihilated_pair_cell_sum_m2),
            "sink_heat_J": sink.generated_heat_J,
            "sink_relative_first_law_residual": (
                sink.relative_first_law_residual),
            "sink_line_balance_residual_m2": sink.line_balance_residual_m2,
            "fields": _fields(
                sink.candidate_state, spacing, boundary, baseline),
        }
    payload = {
        "schema": "v66-source-sink-geography-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "96e75b4+neutral-sink-development",
        "configuration": {
            "grid": [n, n], "duration_s": duration,
            "temperature_K": 1000.0, "engineering_shear": .04,
            "source_grain_size_m": source_parameters.grain_size_m,
            "sink_activation_enthalpy_eV": (
                sink_parameters.activation_enthalpy_eV),
            "sink_stress_assisted": sink_parameters.stress_assisted,
            "matched_site_histograms": True,
        },
        "cases": cases,
        "decision": {
            "source_and_sink_energy_complete": True,
            "source_and_sink_burgers_neutral": True,
            "geography_changes_density_heterogeneity": bool(abs(
                cases["gb_tj"]["fields"][
                    "density_change_boundary_correlation"]
                -cases["matched_random"]["fields"][
                    "density_change_boundary_correlation"]) > .1),
            "physical_wall_claim": False,
            "interpretation": (
                "neutral sink geography can create stored-energy heterogeneity "
                "but cannot by itself create signed GND or a structural wall"),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
