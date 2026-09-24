#!/usr/bin/env python3
"""Single-owner V54 misoriented existing-boundary DRX anchor and controls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SCHEMA = "asb-drx/v54/existing-boundary-manager/v1"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    temporary.replace(path)


CASES = (
    {"id": "misoriented_front", "misorientation_deg": 20.0,
     "parent_line_fraction": 1.0, "child_line_fraction": .35,
     "proposal_direction": 1, "front_enabled": True,
     "front_symmetric_availability": 1.0},
    {"id": "misoriented_front_disabled", "misorientation_deg": 20.0,
     "parent_line_fraction": 1.0, "child_line_fraction": .35,
     "proposal_direction": 1, "front_enabled": False,
     "front_symmetric_availability": 1.0},
    {"id": "same_orientation_density_front", "misorientation_deg": 0.0,
     "parent_line_fraction": 1.0, "child_line_fraction": .35,
     "proposal_direction": 1, "front_enabled": True,
     "front_symmetric_availability": 1.0},
    {"id": "equal_complete_state", "misorientation_deg": 20.0,
     "parent_line_fraction": 1.0, "child_line_fraction": 1.0,
     "proposal_direction": 1, "front_enabled": True,
     "front_symmetric_availability": 1.0},
    {"id": "contrast_reversal", "misorientation_deg": 20.0,
     "parent_line_fraction": .35, "child_line_fraction": 1.0,
     "proposal_direction": -1, "front_enabled": True,
     "front_symmetric_availability": 1.0},
    {"id": "zero_front_exposure", "misorientation_deg": 20.0,
     "parent_line_fraction": 1.0, "child_line_fraction": .35,
     "proposal_direction": 1, "front_enabled": True,
     "front_symmetric_availability": 0.0},
)


def latest_checkpoint(directory):
    paths = sorted(directory.glob("checkpoint_*.npz"))
    return paths[-1] if paths else None


def run_case(root, source_sha, output, case, target, log):
    directory = output/case["id"]; directory.mkdir(parents=True, exist_ok=True)
    result_path = directory/"result.json"
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if int(old.get("completed_intervals", -1)) >= target:
            return old, 0.0
    command = [
        sys.executable,
        str(root/"full_model/analysis/run_v36_recurrent_physical_response.py"),
        "--output-dir", str(directory), "--protocol", "continued_deformation",
        "--grid", "64", "--intervals", str(target), "--dt-s", "2e-9",
        "--initial-shear", ".01", "--strain-rate-s", "100",
        "--temperature-K", "1100", "--length-m", "3.2e-6",
        "--interface-width-m", "4e-7", "--proposal-fraction", ".125",
        "--proposal-direction", str(case["proposal_direction"]),
        "--misorientation-deg", str(case["misorientation_deg"]),
        "--parent-line-fraction", str(case["parent_line_fraction"]),
        "--child-line-fraction", str(case["child_line_fraction"]),
        "--mura-transport-operator", "compatible_dealiased",
        "--qualified-midpoint-loading", "--checkpoint-every", "10",
        "--front-symmetric-availability",
        str(case["front_symmetric_availability"]),
    ]
    if not case["front_enabled"]:
        command.append("--disable-front")
    restart = latest_checkpoint(directory)
    if restart is not None:
        command.extend(("--resume", str(restart)))
    started = time.perf_counter()
    with log.open("a") as stream:
        stream.write("COMMAND "+json.dumps(command)+"\n"); stream.flush()
        result = subprocess.run(
            command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": "src:.",
                 "V36_SOURCE_SHA": source_sha})
    wall = time.perf_counter()-started
    if result.returncode:
        raise RuntimeError(f"{case['id']} target {target} failed: {result.returncode}")
    return json.loads(result_path.read_text()), wall


def compact(result):
    row = result["records"][-1]
    first_law = [item.get("loading_energy_audit") for item in result["records"]]
    first_law = [item for item in first_law if item is not None]
    return {
        "completed_intervals": result["completed_intervals"],
        "physical_time_s": result["physical_time_s"],
        "classification": result["classification"],
        "boundary_initialization": result["boundary_initialization"],
        "transport_operator": result["configuration"]["mura_transport_operator"],
        "front_enabled": result["configuration"]["front_enabled"],
        "front_exposure": result["configuration"]["front_symmetric_availability"],
        "cumulative_signed_sweep_m3": result["cumulative_signed_sweep_m3"],
        "newly_swept_volume_m3": row["cumulative_newly_swept_volume_m3"],
        "current_child_fraction": row["current_child_fraction"],
        "accepted_contour_displacement_m": sum(
            item["accepted_contour_displacement_m"] for item in result["records"]),
        "processed_line_m": row["cumulative_processed_line_m"],
        "boundary_stored_line_m": row["cumulative_boundary_stored_line_m"],
        "child_owner_total_line_density_mean_m2": row[
            "child_owner_total_line_density_mean_m2"],
        "mean_shear_stress_Pa": row["mean_shear_stress_Pa"],
        "engineering_plastic_shear": row["engineering_plastic_shear"],
        "all_loading_first_law_checks_passed": bool(
            first_law and all(item["first_law_passed"] for item in first_law)),
        "maximum_abs_loading_first_law_residual_J": max(
            (abs(item["first_law_residual_J"]) for item in first_law), default=None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=[1, 10, 50, 100, 200, 500])
    args = parser.parse_args()
    root=args.source_root.resolve(); output=args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    allocation=json.loads(args.allocation.read_text())
    deadline=datetime.fromisoformat(allocation["deadline_utc"].replace("Z", "+00:00"))
    state_path=output/"v54_drx_anchor_manager.json"; log=output/"v54_drx_anchor.log"
    state={"schema": SCHEMA, "created_utc": utc(), "source_sha": args.source_sha,
           "allocation_id": allocation["allocation_id"], "owner": {"pid": os.getpid()},
           "state": "RUNNING", "blocks": [], "cases": list(CASES)}
    atomic_json(state_path,state)
    try:
        # Exercise the physical anchor first, then all bounded one-step controls.
        ordered=[CASES[0], CASES[1], *CASES[2:]]
        for case in ordered:
            result,wall=run_case(root,args.source_sha,output,case,1,log)
            state["blocks"].append({"case":case["id"],"target":1,
                                    "wall_seconds":wall,"completed_utc":utc()})
            atomic_json(state_path,state)
        # Continue front-on/off at paired physical horizons. Cost and the fixed
        # allocation clock decide whether the 1-us endpoint is affordable.
        per_interval=[]
        for target in args.targets:
            if target <= 1: continue
            prior=max(int(item["target"]) for item in state["blocks"]
                      if item["case"] in (CASES[0]["id"],CASES[1]["id"]))
            remaining_intervals=2*(target-prior)
            estimate=(max(per_interval,default=30.0)*remaining_intervals)
            remaining=(deadline-datetime.now(timezone.utc)).total_seconds()-1800.0
            if estimate > remaining:
                state["budget_stop"]={"next_target":target,"estimate_s":estimate,
                                      "remaining_before_reserve_s":remaining}
                break
            for case in CASES[:2]:
                state["active"]={"case":case["id"],"target":target,"started_utc":utc()}
                atomic_json(state_path,state)
                result,wall=run_case(root,args.source_sha,output,case,target,log)
                increment=max(target-prior,1); per_interval.append(wall/increment)
                state["blocks"].append({"case":case["id"],"target":target,
                                        "wall_seconds":wall,"completed_utc":utc()})
                state.pop("active",None); atomic_json(state_path,state)
        summaries={}
        for case in CASES:
            path=output/case["id"]/"result.json"
            if path.exists(): summaries[case["id"]]=compact(json.loads(path.read_text()))
        anchor=summaries[CASES[0]["id"]]
        demonstrated=bool(anchor["newly_swept_volume_m3"]>0.0 and
                          anchor["accepted_contour_displacement_m"]!=0.0)
        decision={
            "schema":"asb-drx/v54/existing-boundary-decision/v1",
            "generated_utc":utc(),"source_sha":args.source_sha,"cases":summaries,
            "hard_valid":all(item["all_loading_first_law_checks_passed"]
                             for item in summaries.values()),
            "mechanism_status":{"implemented":True,"enabled":True,
                                "exercised":True,"demonstrated":demonstrated},
            "scientific_classification":(
                "PREPARED_EXISTING_BOUNDARY_GROWTH_AND_PROCESSING_DEMONSTRATED"
                if demonstrated else "VALID_EXISTING_BOUNDARY_ARREST_OR_NEGLIGIBLE_GROWTH"),
            "claim_boundary":"prepared existing-grain growth; not spontaneous grain birth",
        }
        decision_path=output/"v54_existing_boundary_decision.json"
        atomic_json(decision_path,decision)
        state["decision"]={"path":str(decision_path),"sha256":digest(decision_path),
                           "classification":decision["scientific_classification"]}
        state["state"]="COMPLETE";state["completed_utc"]=utc();atomic_json(state_path,state)
    except Exception as error:
        state["state"]="FAILED_CONTROLLER";state["failure"]=f"{type(error).__name__}: {error}"
        state["failed_utc"]=utc();atomic_json(state_path,state);raise


if __name__ == "__main__":
    main()
