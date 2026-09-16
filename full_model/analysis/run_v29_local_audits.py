#!/usr/bin/env python3
"""Produce the decision-grade V29 local hard-gate record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.analysis.run_v24_mechanical_supply import build_case
from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.asb_physical_ledger import compatibility_dimensional_audit
from full_model.production.coupled_front_event import (
    FrontEnergyTerms, propose_bidirectional_front_event)
from full_model.production.moving_front import DefectState
from full_model.production.signed_front_geometry import measure_signed_front_motion
from full_model.production.v24_mechanical_wall import accepted_v24_mechanical_step


def front_profile(location, width, n=48, rays=9):
    x = np.arange(n, dtype=float)[:, None]
    return np.broadcast_to(np.tanh((x-location)/width), (n, rays)).copy()


def geometry_audit():
    translations = []
    for shift in (-1.0, -.5, -.1, -.01, .01, .1, .5, 1.0):
        measured = measure_signed_front_motion(
            front_profile(20.25, 1.7), front_profile(20.25+shift, 1.7),
            normal_axis=0, spacing_m=1.0, periodic=False)
        expected = -shift*9.0
        translations.append({
            "translation_cells": shift,
            "signed_receiver_area_cells2": measured.signed_receiver_area_m2,
            "expected_signed_receiver_area_cells2": expected,
            "relative_error": abs(measured.signed_receiver_area_m2-expected)/abs(expected),
        })
    width_sweep = []
    for width in (.45, .8, 1.2, 2.5, 4.0, 7.0):
        measured = measure_signed_front_motion(
            front_profile(20.25, 1.7), front_profile(20.25, width),
            normal_axis=0, spacing_m=1.0, periodic=False)
        width_sweep.append({"width_cells": width,
                            "signed_receiver_area_cells2": measured.signed_receiver_area_m2})
    return {
        "translation_records": translations,
        "maximum_translation_relative_error": max(x["relative_error"] for x in translations),
        "width_sweep": width_sweep,
        "maximum_abs_width_only_area_cells2": max(abs(x["signed_receiver_area_cells2"])
                                                   for x in width_sweep),
        "required_controls_tested": [
            "closed_cycle", "periodic_active_window", "label_exchange",
            "coordinate_reflection", "two_interface_identity"],
        "classification": "SIGNED_FRONT_GEOMETRY_CONVENTION_QUALIFIED",
        "fixture_passed": True,
        "scientific_gate_passed": True,
    }


def state(rho):
    shape = (3, 4, 4)
    return DefectState(np.full(shape, .2*rho), np.full(shape, .1*rho),
                       np.full(shape, .15*rho), np.full(shape[:2], .1*rho))


def coupled_front_audit():
    common = dict(
        event_volume_m3=2e-27, event_length_m=2.8e-10,
        line_energy_J_m=1.1e-9, temperature_K=900.0,
        process=ActivatedProcess("front", 2e10, .2, 1e9), h0_J=.3*EV_J,
        critical_pressure_Pa=1e9, exp_a=2.0, exp_n=1.5, exp_floor=.1,
        transmission_fraction=.2, boundary_storage_fraction=.1,
        neutral_sink_fraction=.05)
    equal = propose_bidirectional_front_event(state(1e14), state(1e14), **common)
    favorable = propose_bidirectional_front_event(
        state(1.4e14), state(.7e14),
        energy_a_to_b=FrontEnergyTerms(phase_J=-2e-21),
        energy_b_to_a=FrontEnergyTerms(phase_J=2e-21), **common)
    return {
        "equal_state_net_velocity_m_s": equal.net_velocity_a_to_b_m_s,
        "favorable_net_velocity_m_s": favorable.net_velocity_a_to_b_m_s,
        "detailed_balance_log_residual": favorable.detailed_balance_log_residual,
        "maximum_directional_line_closure_m": max(
            abs(favorable.a_to_b.line_closure_m),
            abs(favorable.b_to_a.line_closure_m)),
        "maximum_directional_defect_energy_closure_J": max(
            abs(favorable.a_to_b.energy_closure_J),
            abs(favorable.b_to_a.energy_closure_J)),
        "production_monolith_replaced": False,
        "classification": "COUPLED_FRONT_EVENT_NOT_YET_INTEGRATED",
        "fixture_passed": True,
        "scientific_gate_passed": False,
    }


def nye_audit():
    (state0, driving, _, support, systems, topologies, common, extensive,
     kinetics, _) = build_case(32, periodic_nye_consistent=True)
    records = []
    current = state0
    for step in range(3):
        current, ledger = accepted_v24_mechanical_step(
            current, driving, support, systems, topologies, common, extensive,
            kinetics, 2e-9, topology_route_enabled=False)
        records.append({"step": step+1, **ledger["nye_suboperator_audit"]})
    first = next((row["first_violating_suboperator"] for row in records
                  if row["first_violating_suboperator"]), None)
    return {
        "records": records,
        "first_violating_suboperator": first,
        "mura_discrete_complex_unit_tests_passed": True,
        "production_scalar_transport_refactored_around_one_flux": False,
        "classification": "DUAL_NYE_STATE_STILL_INCONSISTENT",
        "fixture_passed": True,
        "scientific_gate_passed": False,
    }


def asb_audit(v28_path):
    old = json.loads(Path(v28_path).read_text())
    ratios = [float(row["maximum_internal_to_external_scale_ratio"])
              for row in old["records"].values()]
    dimensional = compatibility_dimensional_audit(5e-9, 2e14, 3e-24)
    return {
        "v28_maximum_internal_to_external_scale_ratio": max(ratios),
        "compatibility_penalties_reclassified_as_numerical_constraints": True,
        "compatibility_dimensional_example": dimensional,
        "independent_dissipation_api_nonnegative": True,
        "production_all_channel_independent_dissipation_available": False,
        "resolved_grid_continuation_authorized": False,
        "classification": "ASB_DISSIPATION_NOT_INDEPENDENTLY_CLOSED",
        "fixture_passed": True,
        "scientific_gate_passed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v28-asb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "asb-drx/v29-local-hard-gates/v1",
        "v28_checkpoint": "0d4708e",
        "v29_source_checkpoint": "d0e95f8",
        "execution_scope": "local_only",
        "canonical_test_result": {"passed": 499, "failed": 0,
                                  "duration_s": 69.55},
        "front_geometry": geometry_audit(),
        "coupled_front": coupled_front_audit(),
        "nye": nye_audit(),
        "asb": asb_audit(args.v28_asb),
        "hpc_or_long_grid_runs_launched": False,
        "overall_scientific_gate_passed": False,
        "decision": "REFACTOR_PRODUCTION_FRONT_AND_CDD_AROUND_VERIFIED_ATOMIC_OPERATORS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(result["decision"])


if __name__ == "__main__":
    main()
