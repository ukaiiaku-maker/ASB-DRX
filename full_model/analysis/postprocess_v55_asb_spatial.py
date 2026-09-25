#!/usr/bin/env python3
"""Supplement strict ASB metrics with band/background and strain histories."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.postprocess_v53_asb_mechanism import field_metrics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def band_background(field):
    value = np.maximum(np.asarray(field, dtype=float), 0.0)
    threshold = float(value.mean()+value.std())
    band = value > threshold
    background = ~band
    total = float(value.sum())
    return {
        "threshold": threshold,
        "band_area_fraction": float(np.mean(band)),
        "band_power_fraction": 0.0 if total == 0.0 else float(
            value[band].sum()/total),
        "band_mean": None if not np.any(band) else float(value[band].mean()),
        "background_mean": None if not np.any(background) else float(
            value[background].mean()),
        "band_to_background_mean_ratio": (
            None if not np.any(band) or not np.any(background)
            or float(value[background].mean()) == 0.0
            else float(value[band].mean()/value[background].mean())),
    }


def trajectory(rows):
    history=[]; accumulated=None; previous_time=0.0; spacing=None
    for row in rows:
        with np.load(Path(row["checkpoint"]), allow_pickle=True) as raw:
            power=np.asarray(raw["asb_last_plastic_power_W_m3"],dtype=float)
            rate=np.asarray(raw["asb_last_gdot_abs"],dtype=float)
            parameters=json.loads(str(raw["P_json"].item()))
            spacing=float(parameters["L_phys"])/int(parameters["Nx"])
        dt=float(row["physical_time_s"])-previous_time
        accumulated=rate*dt if accumulated is None else accumulated+rate*dt
        accumulated_metrics,_=field_metrics(accumulated,spacing)
        history.append({
            "step": int(row["step"]),
            "physical_time_s": float(row["physical_time_s"]),
            "audited_applied_strain": float(row["audited_applied_strain"]),
            "instantaneous_power": band_background(power),
            "sampled_accumulated_absolute_slip": accumulated_metrics,
            "sampled_accumulation_semantics": (
                "right-endpoint integration of saved |slip-rate| snapshots; supplementary, not the production slip state"),
        })
        previous_time=float(row["physical_time_s"])
    return history


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--decision",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); decision=json.loads(args.decision.read_text())
    histories={name:trajectory(decision["trajectories"][name])
               for name in ("baseline","control")}
    payload={
        "schema":"asb-drx/v55/asb-spatial-supplement/v1",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "source_decision":str(args.decision.resolve()),
        "source_decision_sha256":digest(args.decision),
        "strict_criteria_unchanged":True,
        "background_subtracted_for_strict_classification":False,
        "trajectories":histories,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(args.output),"sha256":digest(args.output)},
                     sort_keys=True))


if __name__=="__main__":main()
