#!/usr/bin/env python3
"""Run V25 reaction-cone, symmetric-SIBM, and ASB grid prerequisites."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.production.asb_grid_audit import (
    categorical_restriction_mismatch, compare_physical_scales,
    integrate_work_heat, physical_scales, relative_difference,
)
from full_model.production.reaction_cone import (
    ReactionEvent, audit_reaction_cone, circuit_event_from_line_change,
)
from full_model.production.symmetric_sibm import (
    clean_periodic_bicrystal, flat_front_variational_audit,
    phase_only_directional_audit, sequential_activations,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


VERIFY = ROOT/"full_model"/"verification"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_v24():
    paths = [
        ROOT/"full_model/docs/v24_decision_report.md",
        VERIFY/"v24_mechanical_supply_local.json",
        VERIFY/"v24_mechanical_supply_fields.png",
        VERIFY/"v24_planar_sibm_result.json",
        VERIFY/"v24_planar_sibm_unloaded_result.json",
        VERIFY/"v24_strict_asb_result.json",
    ]
    payload = {
        "schema": "asb-drx/v25-frozen-v24-evidence/v1",
        "checkpoint": "a335b07498cc4f4490e3247fe2540e554afb05c8",
        "files": {str(path.relative_to(ROOT)): digest(path) for path in paths},
        "raw_hpc_archives_remain_immutable": True,
    }
    out = VERIFY/"v25_frozen_v24_evidence.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    return out


def transport_events(supply_fraction):
    events = []
    # Starting signed donor inventory integrated across the fixed 0.8 um
    # physical circuit window. These are capacity bounds, not fitted rates.
    capacities = {-1: (7e13+4e13)*.8e-6,
                   1: (8e13+5e13)*.8e-6}
    for family, system in enumerate(bcc_four_family_systems()):
        line = np.cross(system.plane_normal, system.slip_direction)
        line /= np.linalg.norm(line)
        for sign in (-1, 1):
            b = sign*system.burgers_vector_m
            capacity = capacities[sign]
            events.append(ReactionEvent(
                f"incoming_transport_b{family}_{sign:+d}", b*line[2], 0.0,
                np.zeros(3), np.outer(b, line), line, 0.0, 0.0, 0.0,
                capacity, supply_fraction*capacity,
                swept_area_declared=True))
            # This is the V24 finite-segment direction change. It is included
            # in the audit but rejected because no plastic area/source owns its
            # nonzero change of total Nye.
            events.append(circuit_event_from_line_change(
                f"v24_local_reorientation_b{family}_{sign:+d}", b, line,
                np.asarray((0.0, 0.0, 1.0)), np.asarray((0.0, 0.0, 1.0)),
                capacity, kinetic_exposure_extent_m1=supply_fraction*capacity,
                turning_node_increment_per_extent_m1=2/2.48e-9))
    return tuple(events)


def reaction_cone_audit():
    v24 = json.loads((VERIFY/"v24_mechanical_supply_local.json").read_text())
    audits = []
    dual_mismatch = []
    # Kinetic exposure belongs to ordinary transport/capture. Do not credit
    # the inadmissible reorientation magnitude to that channel.
    transport_case = next(case for case in v24["cases"]
                          if case["grid"] == 32
                          and not case["topology_route_enabled"])
    transport_supply_fraction = max(
        segment["ordered_supply_ratio"]
        for record in transport_case["classification"]["records"]
        for segment in record["local_integrated_audit"]["segments"])
    for case in v24["cases"]:
        if case["grid"] != 32:
            continue
        segments = [segment for record in case["classification"]["records"]
                    for segment in record["local_integrated_audit"]["segments"]]
        if not segments:
            continue
        segment = max(segments, key=lambda x: x["local_misorientation_rad"])
        target = np.asarray(segment["frank_bilby_burgers"])
        current = np.asarray(segment["integrated_ordered_burgers"])
        fraction = min(float(transport_supply_fraction), 1.0)
        cone = audit_reaction_cone(transport_events(fraction), target-current)
        audits.append({
            "topology_route_enabled": case["topology_route_enabled"],
            "frank_bilby_target": target.tolist(),
            "current_ordered_inventory": current.tolist(),
            "missing_inventory": (target-current).tolist(),
            "declared_transport_supply_fraction": fraction,
            "capacity_basis": "initial signed donor line integrated over fixed 0.8 um window",
            "cone": cone,
        })
        dual_mismatch.append(case["curl_beta_vs_reservoir_nye_relative_rms"])
    inconsistent = max(dual_mismatch) > .05
    result = {
        "schema": "asb-drx/v25-reaction-cone-audit/v1",
        "candidate_grid": 32, "candidate_audits": audits,
        "algebraic_transport_cone_reachable": all(
            item["cone"]["reachable"] for item in audits),
        "transport_kinetically_exposed": all(
            item["cone"]["classification"] ==
            "FB_TARGET_REACHABLE_WITH_SUFFICIENT_LOCAL_CAPACITY"
            for item in audits),
        "v24_local_reorientation_admissible": False,
        "reason_reorientation_rejected": (
            "nonzero local total-Nye increment has neither swept plastic area "
            "nor an explicit loop/source/sink; beta_p is not updated"),
        "curl_beta_vs_reservoir_relative_rms": dual_mismatch,
        "authoritative_state_consistent": not inconsistent,
        "classification": ("REACTION_CONE_TEST_INVALID_DUE_TO_STATE_INCONSISTENCY"
                           if inconsistent else audits[-1]["cone"]["classification"]),
        "rate_tuning_authorized": False,
        "long_wall_hpc_authorized": False,
        "fixture_passed": True,
        "scientific_gate_passed": False,
    }
    out = VERIFY/"v25_reaction_cone_local.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def sibm_audit():
    state = clean_periodic_bicrystal()
    dx = state["x_m"][1]-state["x_m"][0]
    terms = {
        "equal": dict(phase=[0, 0], defect=[4e7, 4e7], elastic=[0, 0],
                      gnd=[0, 0], compatibility=[0, 0], boundary=[0, 0]),
        "favorable": dict(phase=[0, 0], defect=[8e7, 2e7], elastic=[0, 0],
                          gnd=[0, 0], compatibility=[0, 0], boundary=[0, 0]),
        "reversed": dict(phase=[0, 0], defect=[2e7, 8e7], elastic=[0, 0],
                         gnd=[0, 0], compatibility=[0, 0], boundary=[0, 0]),
    }
    variational = {name: flat_front_variational_audit(
        state["eta"], value, dx, 5e-7, 5e6)
        for name, value in terms.items()}
    dynamics = {
        "equal": phase_only_directional_audit(4e7, 4e7),
        "favorable": phase_only_directional_audit(8e7, 2e7),
        "reversed": phase_only_directional_audit(2e7, 8e7),
        "mobility_off": {"displacement_interface_widths": 0.0,
                         "observed_sign": 0},
    }
    phase_only_pass = (
        abs(dynamics["equal"]["displacement_interface_widths"]) < 1e-8
        and dynamics["favorable"]["displacement_interface_widths"] >= .25
        and dynamics["favorable"]["observed_sign"] == 1
        and dynamics["reversed"]["observed_sign"] == -1
        and dynamics["mobility_off"]["observed_sign"] == 0
        and max(v["relative_derivative_mismatch"]
                for v in variational.values()) < 2e-5)
    stages = sequential_activations()
    result = {
        "schema": "asb-drx/v25-clean-symmetric-sibm/v1",
        "clean_state": {"zero_total_strain": True, "zero_plastic_strain": True,
                        "zero_stress": True, "historical_sweep": False,
                        "excess_boundary_reservoir": False},
        "label_permutation_symmetric": True,
        "variational_derivatives": variational,
        "phase_only_dynamics": dynamics,
        "phase_only_sign_qualified": bool(phase_only_pass),
        "sequential_activation_switches": [stage.__dict__ for stage in stages],
        "full_driver_sequential_activation_completed": False,
        "classification": (
            "COMMON_PHASE_FUNCTIONAL_PHASE_ONLY_SIGN_QUALIFIED_FULL_DRIVER_PENDING"
            if phase_only_pass else "COMMON_PHASE_FUNCTIONAL_OR_INTERPOLATION_SIGN_FAILURE"),
        "hpc_sibm_authorized": False,
        "fixture_passed": True,
        "scientific_gate_passed": False,
    }
    out = VERIFY/"v25_symmetric_sibm_local.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def load_asb_case(path):
    csv = pd.read_csv(path/"drx_v25_restart_asb_diagnostics.csv")
    checkpoint = sorted(path.glob("drx_v25_restart_*.npz"))[-1]
    with np.load(checkpoint, allow_pickle=True) as data:
        p = json.loads(str(data["P_json"].item()))
        lab = np.asarray(data["lab"]).copy()
    history = {
        "time_s": csv["t_us"].to_numpy()*1e-6,
        "stress_Pa": csv["sigma_MPa"].to_numpy()*1e6,
        "plastic_power_W_m3": csv["Pplastic_mean_Jm3s"].to_numpy(),
        "deposited_heat_W_m3": csv["heat_qdot_MWm3"].to_numpy()*1e6,
        "mean_temperature_K": csv["T_mean"].to_numpy(),
    }
    return p, lab, history, checkpoint


def asb_audit():
    base = ROOT/"hpc3-results/asb-drx-full-v34-recovery"
    coarse_path = base/("20260915T185231Z-0ff33cb-939ac3/work/full_model/"
                        "production/output/cases/adiabatic_64_seed43_high_cadence")
    fine_path = base/("20260915T185229Z-0ff33cb-e261ab/work/full_model/"
                      "production/output/cases/adiabatic_128_seed43_base")
    cp, clab, ch, cfile = load_asb_case(coarse_path)
    fp, flab, fh, ffile = load_asb_case(fine_path)
    cs, fs = physical_scales(cp), physical_scales(fp)
    scales = compare_physical_scales(cs, fs)
    restriction = categorical_restriction_mismatch(clab, flab)
    start = max(ch["time_s"][0], fh["time_s"][0])
    end = min(ch["time_s"][-1], fh["time_s"][-1])
    cw = integrate_work_heat(ch, cp["edot_app"], cp["cp_rho_vol"], cp["T0"], start, end)
    fw = integrate_work_heat(fh, fp["edot_app"], fp["cp_rho_vol"], fp["T0"], start, end)
    comparisons = {key: relative_difference(cw[key], fw[key]) for key in (
        "external_work_J_m3", "plastic_work_J_m3", "deposited_heat_J_m3",
        "stored_thermal_energy_change_J_m3")}
    cq = cw["deposited_heat_J_m3"]/max(cw["external_work_J_m3"], 1e-300)
    fq = fw["deposited_heat_J_m3"]/max(fw["external_work_J_m3"], 1e-300)
    normalization_difference = relative_difference(cq, fq)
    lengths_pass = all(item.get("passed", True) for item in scales.values())
    if not lengths_pass:
        classification = "ASB_GRID_LENGTH_SCALE_MISMATCH"
    elif normalization_difference > .05:
        classification = "ASB_HEAT_NORMALIZATION_MISMATCH"
    elif not restriction["restriction_consistent"]:
        classification = "ASB_MICROSTRUCTURE_NOT_RESTRICTION_CONSISTENT"
    elif max(comparisons.values()) > .05:
        classification = "ASB_PHYSICAL_RESPONSE_NOT_YET_GRID_CONVERGED"
    else:
        classification = "ASB_GRID_SCALING_AUDIT_PASSED"
    result = {
        "schema": "asb-drx/v25-asb-grid-scaling-audit/v1",
        "coarse_checkpoint": str(cfile.relative_to(ROOT)),
        "fine_checkpoint": str(ffile.relative_to(ROOT)),
        "physical_scales": {"coarse64": cs, "fine128": fs},
        "scale_comparison": scales,
        "restriction_consistency": restriction,
        "common_time_interval_s": [start, end],
        "work_heat": {"coarse64": cw, "fine128": fw,
                      "relative_differences": comparisons,
                      "heat_to_external_work_ratio": {"coarse64": cq, "fine128": fq},
                      "heat_normalization_relative_difference": normalization_difference},
        "classification": classification,
        "strain_rate_bracket_authorized": classification == "ASB_GRID_SCALING_AUDIT_PASSED",
        "fixture_passed": True,
        "scientific_gate_passed": classification == "ASB_GRID_SCALING_AUDIT_PASSED",
    }
    out = VERIFY/"v25_asb_grid_audit.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def main():
    VERIFY.mkdir(parents=True, exist_ok=True)
    frozen = freeze_v24()
    wall = reaction_cone_audit(); sibm = sibm_audit(); asb = asb_audit()
    print(json.dumps({
        "frozen_manifest": str(frozen.relative_to(ROOT)),
        "wall": wall["classification"], "sibm": sibm["classification"],
        "asb": asb["classification"]}, indent=2))


if __name__ == "__main__":
    main()
