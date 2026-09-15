#!/usr/bin/env python3
"""Strict high-cadence seed-43 ASB persistence and refinement decision."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import numpy as np

from full_model.production.asb_classifier import (
    ASBCriteria, classify, matched_history, refinement_passes,
)


def paths(directory):
    return sorted(directory.glob("drx_v25_restart_*.npz"), key=lambda path: int(
        re.search(r"(\d+)$", path.stem).group(1)))


def matched_pair(adiabatic, isothermal):
    controls = {int(re.search(r"(\d+)$", path.stem).group(1)): path
                for path in paths(isothermal)}
    rates=[]; hot=[]; cold=[]; stress=[]; times=[]; used=[]
    for path in paths(adiabatic):
        step = int(re.search(r"(\d+)$", path.stem).group(1))
        if step not in controls:
            continue
        with np.load(path, allow_pickle=True) as a, np.load(
                controls[step], allow_pickle=True) as c:
            rates.append(np.asarray(a["asb_last_gdot_abs"]))
            hot.append(np.asarray(a["T"])); cold.append(np.asarray(c["T"]))
            stress.append(float(a["sigma_bar"])); times.append(float(a["sim_time"]))
            used.append(step)
    if len(used) < 3:
        raise ValueError("high-cadence decision needs at least three matched fields")
    with np.load(paths(adiabatic)[-1], allow_pickle=True) as data:
        p = json.loads(str(data["P_json"].item()))
        dx = float(p["L_phys"])/int(p["Nx"])
        width = float(np.sqrt(float(p["kappa_eta"])/float(p["W_eta"])))
    history = matched_history(
        np.asarray(rates), np.asarray(hot), np.asarray(cold), np.asarray(stress),
        np.asarray(times), dx, dx)
    return history, width, used


def onset_width(history, decision):
    if not decision.classified:
        return None
    return next(item.effective_width_m for item in history
                if item.time_s >= decision.onset_time_s)


def summary(history):
    return {
        "start_time_s": history[0].time_s, "end_time_s": history[-1].time_s,
        "field_snapshot_count": len(history),
        "minimum_active_fraction": min(x.active_fraction for x in history),
        "maximum_temperature_excess_K": max(x.temperature_excess_K for x in history),
        "maximum_softening_fraction": max(x.softening_fraction for x in history),
    }


def conjunctive_persistence_diagnostic(history, interface_width_m, criteria):
    """Report the longest raw qualifying interval before refinement gating."""
    qualifies = [
        item.active_fraction <= criteria.maximum_active_fraction
        and item.temperature_excess_K >= criteria.minimum_temperature_excess_K
        and item.softening_fraction >= criteria.minimum_softening_fraction
        and item.effective_width_m >= (
            criteria.minimum_width_to_interface*interface_width_m)
        for item in history]
    intervals = []; start = None
    for index, accepted in enumerate(qualifies):
        if accepted and start is None:
            start = index
        if start is not None and (not accepted or index == len(qualifies)-1):
            end = index if accepted else index-1
            intervals.append((start, end))
            start = None
    if not intervals:
        return {"qualifying_snapshot_count": 0,
                "longest_conjunctive_persistence_s": 0.0,
                "longest_interval_start_s": None,
                "longest_interval_end_s": None}
    longest = max(intervals, key=lambda pair:
                  history[pair[1]].time_s-history[pair[0]].time_s)
    return {
        "qualifying_snapshot_count": int(sum(qualifies)),
        "longest_conjunctive_persistence_s": float(
            history[longest[1]].time_s-history[longest[0]].time_s),
        "longest_interval_start_s": history[longest[0]].time_s,
        "longest_interval_end_s": history[longest[1]].time_s,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--refined-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    criteria = ASBCriteria(.25, 50.0, .20, 2.0, 1e-6, .05)
    coarse, cw, cs = matched_pair(
        args.coarse_root/"cases"/"adiabatic_64_seed43_high_cadence",
        args.coarse_root/"cases"/"isothermal_64_seed43_high_cadence")
    base, bw, bs = matched_pair(
        args.base_root/"cases"/"adiabatic_128_seed43_base",
        args.base_root/"cases"/"isothermal_128_seed43_base")
    refined, rw, rs = matched_pair(
        args.refined_root/"cases"/"adiabatic_128_seed43_dt_refined",
        args.refined_root/"cases"/"isothermal_128_seed43_dt_refined")
    provisional = {
        "coarse64": classify(coarse, cw, criteria, True),
        "base128": classify(base, bw, criteria, True),
        "dt_refined128": classify(refined, rw, criteria, True),
    }
    grid_refined = False; timestep_refined = False
    if provisional["coarse64"].classified and provisional["base128"].classified:
        grid_refined = refinement_passes(
            provisional["coarse64"].onset_time_s,
            provisional["base128"].onset_time_s,
            onset_width(coarse, provisional["coarse64"]),
            onset_width(base, provisional["base128"]), criteria.refinement_tolerance)
    if provisional["base128"].classified and provisional["dt_refined128"].classified:
        timestep_refined = refinement_passes(
            provisional["base128"].onset_time_s,
            provisional["dt_refined128"].onset_time_s,
            onset_width(base, provisional["base128"]),
            onset_width(refined, provisional["dt_refined128"]),
            criteria.refinement_tolerance)
    refinement = bool(grid_refined and timestep_refined)
    decisions = {
        "coarse64": classify(coarse, cw, criteria, refinement),
        "base128": classify(base, bw, criteria, refinement),
        "dt_refined128": classify(refined, rw, criteria, refinement),
    }
    passed = refinement and all(item.classified for item in decisions.values())
    result = {
        "schema": "asb-drx/v24-strict-asb-persistence/v1",
        "criteria": asdict(criteria),
        "grid_refinement_passed": grid_refined,
        "timestep_refinement_passed": timestep_refined,
        "refinement_passed": refinement,
        "decisions": {key: asdict(value) for key, value in decisions.items()},
        "summaries": {"coarse64": summary(coarse), "base128": summary(base),
                      "dt_refined128": summary(refined)},
        "raw_conjunctive_persistence": {
            "coarse64": conjunctive_persistence_diagnostic(
                coarse, cw, criteria),
            "base128": conjunctive_persistence_diagnostic(
                base, bw, criteria),
            "dt_refined128": conjunctive_persistence_diagnostic(
                refined, rw, criteria),
        },
        "matched_steps": {"coarse64": cs, "base128": bs,
                          "dt_refined128": rs},
        "fixture_passed": True,
        "scientific_gate_passed": bool(passed),
        "classification": "STRICT_ASB_QUALIFIED" if passed else "STRICT_ASB_NOT_OBSERVED",
        "claim_boundary": "All localization, heating, softening, one-microsecond persistence, grid, and timestep requirements remain conjunctive.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(result["classification"])


if __name__ == "__main__":
    main()
