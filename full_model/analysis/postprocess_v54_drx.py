#!/usr/bin/env python3
"""Independent outcome-neutral classification of the V54 boundary cases."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


BURGERS_M = 2.48e-10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(path):
    result=json.loads(path.read_text()); rows=result["records"]; end=rows[-1]
    load=[r.get("loading_energy_audit") for r in rows]
    load=[r for r in load if r is not None]
    load_allowance=sum(float(r["first_law_scale_J"])
                       *float(r["first_law_relative_tolerance"]) for r in load)
    load_cumulative=(None if not load else float(
        load[-1]["cumulative_first_law_residual_J"]))
    front=[r["complete_energy_decision"] for r in rows
           if r.get("complete_energy_decision") is not None]
    front_allowance=sum(float(r["tolerance_J"]) for r in front)
    front_cumulative=sum(float(r["first_law_residual_J"]) for r in front)
    displacement=sum(float(r["accepted_contour_displacement_m"]) for r in rows)
    return {
        "result":str(path.resolve()),"sha256":digest(path),
        "configuration":result["configuration"],
        "boundary_initialization":result["boundary_initialization"],
        "completed_intervals":int(result["completed_intervals"]),
        "physical_time_s":float(result["physical_time_s"]),
        "direct_signed_contour_displacement_m":displacement,
        "displacement_in_burgers_vectors":displacement/BURGERS_M,
        "newly_swept_volume_m3":float(end["cumulative_newly_swept_volume_m3"]),
        "revisit_volume_m3":float(end["cumulative_revisit_volume_m3"]),
        "processed_line_m":float(end["cumulative_processed_line_m"]),
        "boundary_stored_line_m":float(end["cumulative_boundary_stored_line_m"]),
        "child_fraction":float(end["current_child_fraction"]),
        "child_owner_line_density_mean_m2":float(end[
            "child_owner_total_line_density_mean_m2"]),
        "stress_Pa":float(end["mean_shear_stress_Pa"]),
        "plastic_shear":float(end["engineering_plastic_shear"]),
        "hard_loading_ledger_passed":bool(load and all(
            r["first_law_passed"] for r in load)
            and abs(load_cumulative)<=load_allowance),
        "maximum_abs_first_law_residual_J":max(
            (abs(float(r["first_law_residual_J"])) for r in load),default=None),
        "cumulative_loading_first_law_residual_J":load_cumulative,
        "accumulated_loading_first_law_allowance_J":load_allowance,
        "complete_front_first_law_passed":bool(
            not front or abs(front_cumulative)<=front_allowance),
        "cumulative_complete_front_first_law_residual_J":front_cumulative,
        "accumulated_complete_front_first_law_allowance_J":front_allowance,
        "child_owner_rehardening_rate_m2_s":(
            (float(rows[-1]["child_owner_total_line_density_mean_m2"])
             -float(rows[0]["child_owner_total_line_density_mean_m2"]))
            /max(float(rows[-1]["physical_time_end_s"]
                       -rows[0]["physical_time_end_s"]),1e-300)),
        "minimum_topology_backtrack_fraction":min(
            (float(r["front_channel_diagnostics"]["topology_backtrack_fraction"])
             for r in rows if r.get("front_channel_diagnostics") is not None),
            default=None),
        "all_mura_intervals_use_compatible_transport":all(
            r["mura"]["transport_operator"]=="compatible_dealiased" for r in rows),
    }


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    cases={p.parent.name:summarize(p) for p in args.root.glob("*/result.json")}
    required={"misoriented_front","misoriented_front_disabled",
              "same_orientation_density_front","equal_complete_state",
              "contrast_reversal","zero_front_exposure"}
    missing=sorted(required-set(cases))
    anchor=cases.get("misoriented_front")
    motion=False if anchor is None else bool(
        anchor["newly_swept_volume_m3"]>0.0 and
        anchor["direct_signed_contour_displacement_m"]!=0.0)
    substantial=False if anchor is None else bool(
        abs(anchor["direct_signed_contour_displacement_m"])>=BURGERS_M)
    control_contract={
        "front_disabled_has_zero_sweep": (
            cases.get("misoriented_front_disabled",{}).get("newly_swept_volume_m3")==0.0),
        "zero_exposure_has_zero_sweep": (
            cases.get("zero_front_exposure",{}).get("newly_swept_volume_m3")==0.0),
        "equal_complete_state_is_actually_equal": bool(
            cases.get("equal_complete_state",{}).get("configuration",{}).get(
                "misorientation_deg")==0.0
            and cases.get("equal_complete_state",{}).get("configuration",{}).get(
                "parent_line_fraction")==1.0
            and cases.get("equal_complete_state",{}).get("configuration",{}).get(
                "child_line_fraction")==1.0),
    }
    hard_valid=bool(not missing and all(c["hard_loading_ledger_passed"] and
                    c["complete_front_first_law_passed"] and
                    c["all_mura_intervals_use_compatible_transport"]
                    for c in cases.values()) and all(control_contract.values()))
    if missing:
        classification="INCOMPLETE_MISSING_REQUIRED_CONTROLS"
    elif not hard_valid:
        classification="HARD_INVALID_OR_CONTROL_CONTRACT_FAILED"
    elif substantial:
        classification="PREPARED_EXISTING_BOUNDARY_SUBSTANTIAL_GROWTH_DEMONSTRATED"
    elif motion:
        classification="VALID_ATOMIC_MOTION_WITH_NEGLIGIBLE_GEOMETRIC_GROWTH"
    else:
        classification="VALID_ARREST_OR_NO_RESOLVED_GROWTH"
    payload={"schema":"asb-drx/v54/existing-boundary-decision/v2",
             "generated_utc":datetime.now(timezone.utc).isoformat(),
             "classification":classification,"hard_valid":hard_valid,
             "missing_cases":missing,"control_contract":control_contract,
             "mechanism_status":{"implemented":True,"enabled":True,
                                 "exercised":anchor is not None,
                                 "direct_motion_observed":motion,
                                 "substantial_growth_demonstrated":substantial},
             "substantial_motion_definition":{
                 "minimum_direct_displacement_m":BURGERS_M,
                 "basis":"one declared BCC Burgers-vector jump length"},
             "cases":cases,"spontaneous_grain_birth_claimed":False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"classification":classification,"hard_valid":hard_valid,
                      "output":str(args.output),"sha256":digest(args.output)}))


if __name__=="__main__":main()
