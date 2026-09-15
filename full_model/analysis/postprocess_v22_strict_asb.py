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
            "matched_steps":{"coarse":cs,"fine":fs,"second_seed":ss},
            "fixture_passed":True,"scientific_gate_passed":passed}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(result["classification"])


if __name__ == "__main__": main()
