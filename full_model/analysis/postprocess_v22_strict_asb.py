#!/usr/bin/env python3
"""Strict matched-control ASB classification from authoritative checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re

import numpy as np

from full_model.production.asb_classifier import (
    ASBCriteria, classify, matched_history, refinement_passes)


def checkpoints(path: Path):
    return sorted(path.glob("drx_v25_restart_*.npz"))


def matched_pair(adiabatic: Path, control: Path):
    a_paths, c_paths = checkpoints(adiabatic), checkpoints(control)
    by_step = {int(re.search(r"(\d+)$", p.stem).group(1)): p for p in c_paths}
    rate=[]; hot=[]; cold=[]; stress=[]; time=[]; used=[]
    for path in a_paths:
        step=int(re.search(r"(\d+)$", path.stem).group(1))
        if step not in by_step: continue
        with np.load(path,allow_pickle=True) as a, np.load(by_step[step],allow_pickle=True) as c:
            rate.append(np.asarray(a["asb_last_gdot_abs"]))
            hot.append(np.asarray(a["T"])); cold.append(np.asarray(c["T"]))
            stress.append(float(a["sigma_bar"])); time.append(float(a["sim_time"]))
            used.append(step)
    if len(used)<2: raise ValueError("fewer than two matched ASB checkpoints")
    with np.load(a_paths[-1],allow_pickle=True) as z:
        p=json.loads(str(z["P_json"].item())); dx=float(p["L_phys"])/int(p["Nx"])
        interface=float(np.sqrt(float(p["kappa_eta"])/float(p["W_eta"])))
    history=matched_history(np.asarray(rate),np.asarray(hot),np.asarray(cold),
                            np.asarray(stress),np.asarray(time),dx,dx)
    return history,interface,used


def observable_summary(history, interface_width_m, criteria):
    qualifying = []
    run_start = None
    maximum_run = 0.0
    for snapshot in history:
        passed = bool(
            snapshot.active_fraction <= criteria.maximum_active_fraction
            and snapshot.temperature_excess_K
            >= criteria.minimum_temperature_excess_K
            and snapshot.softening_fraction
            >= criteria.minimum_softening_fraction
            and snapshot.effective_width_m
            >= criteria.minimum_width_to_interface*interface_width_m)
        if passed:
            qualifying.append(snapshot.time_s)
            if run_start is None:
                run_start = snapshot.time_s
            maximum_run = max(maximum_run, snapshot.time_s-run_start)
        else:
            run_start = None
    return {
        "physical_time_start_s": history[0].time_s,
        "physical_time_end_s": history[-1].time_s,
        "minimum_active_fraction": min(x.active_fraction for x in history),
        "minimum_effective_width_m": min(x.effective_width_m for x in history),
        "maximum_temperature_excess_K": max(
            x.temperature_excess_K for x in history),
        "maximum_post_peak_softening_fraction": max(
            x.softening_fraction for x in history),
        "maximum_softening_after_50K_fraction": max(
            (x.softening_fraction for x in history
             if x.temperature_excess_K >= 50.0), default=None),
        "simultaneous_qualifying_snapshot_times_s": qualifying,
        "maximum_conjunctive_run_s": maximum_run,
        "required_persistence_s": criteria.minimum_persistence_s,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--coarse-adiabatic",type=Path,required=True)
    parser.add_argument("--coarse-control",type=Path,required=True)
    parser.add_argument("--fine-adiabatic",type=Path,required=True)
    parser.add_argument("--fine-control",type=Path,required=True)
    parser.add_argument("--seed-adiabatic",type=Path,required=True)
    parser.add_argument("--seed-control",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    criteria=ASBCriteria(.25,50.,.20,2.,1e-6,.05)
    ch,ci,cs=matched_pair(args.coarse_adiabatic,args.coarse_control)
    fh,fi,fs=matched_pair(args.fine_adiabatic,args.fine_control)
    sh,si,ss=matched_pair(args.seed_adiabatic,args.seed_control)
    c0=classify(ch,ci,criteria,True); f0=classify(fh,fi,criteria,True)
    if c0.classified and f0.classified:
        cw=next(x.effective_width_m for x in ch if x.time_s>=c0.onset_time_s)
        fw=next(x.effective_width_m for x in fh if x.time_s>=f0.onset_time_s)
        refined=refinement_passes(c0.onset_time_s,f0.onset_time_s,cw,fw,.05)
    else: refined=False
    decisions={"coarse":classify(ch,ci,criteria,refined),
               "fine":classify(fh,fi,criteria,refined),
               "second_seed":classify(sh,si,criteria,refined)}
    passed=all(d.classified for d in decisions.values())
    result={"schema":"asb-drx/v22-strict-asb/v1",
            "classification":"STRICT_ASB_QUALIFIED" if passed else "STRICT_ASB_NOT_OBSERVED",
            "criteria":asdict(criteria),"refinement_passed":refined,
            "decisions":{k:asdict(v) for k,v in decisions.items()},
            "observable_summaries": {
                "coarse": observable_summary(ch, ci, criteria),
                "fine": observable_summary(fh, fi, criteria),
                "second_seed": observable_summary(sh, si, criteria)},
            "matched_histories": {
                "coarse": [asdict(x) for x in ch],
                "fine": [asdict(x) for x in fh],
                "second_seed": [asdict(x) for x in sh]},
            "matched_steps":{"coarse":cs,"fine":fs,"second_seed":ss},
            "fixture_passed": bool(cs and fs and ss),
            "scientific_gate_passed":passed,
            "claim_boundary": (
                "Classification is conjunctive and specific to these matched "
                "fields; localization or heating alone is not strict ASB.")}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(result["classification"])


if __name__ == "__main__": main()
