#!/usr/bin/env python3
"""Compute the immutable Hall--Petch nucleation capacity ceiling."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.arrhenius_dislocation_sources import (
    ArrheniusSourceParameters,
)


EV_J = 1.602176634e-19
KB_J_K = 1.380649e-23


def hall_petch_capacity_ceiling_s(temperature_K, parameters, families=4):
    """Return hazard and shear-rate ceilings with every family accessible.

    At the immutable EXP floor, the nucleation hazard is
    ``nu*(b/d)^2*exp(-Q_floor/kT)`` and one event supplies ``b/d`` slip.
    Their product is therefore an analytical upper bound, not a fitted rate.
    """
    temperature = float(temperature_K)
    count = int(families)
    if not math.isfinite(temperature) or temperature <= 0.0 or count <= 0:
        raise ValueError("positive temperature and family count required")
    b_over_d = parameters.burgers_m/parameters.grain_size_m
    boltzmann = math.exp(
        -parameters.barrier_floor_eV*EV_J/(KB_J_K*temperature))
    hazard = parameters.attempt_frequency_Hz*b_over_d**2*boltzmann
    per_family = hazard*b_over_d
    return {
        "maximum_hazard_per_family_s": hazard,
        "event_slip_increment": b_over_d,
        "maximum_shear_rate_per_family_s": per_family,
        "maximum_all_family_shear_rate_s": count*per_family,
        "family_count": count,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root: Path) -> dict:
    checkpoint = root/"checkpoint_latest.npz"
    result = json.loads((root/"result.json").read_text())
    state, _, step, load, _, config, provenance = _load_checkpoint(checkpoint)
    source = ArrheniusSourceParameters(grain_size_m=2.4e-6)
    temperature = np.asarray([owner.temperature_K for owner in state.owners])
    maximum_temperature = float(np.max(temperature))
    ceiling = hall_petch_capacity_ceiling_s(maximum_temperature, source)
    imposed = float(config["shear_rate_s"])
    ceiling_fraction = ceiling["maximum_all_family_shear_rate_s"]/imposed
    observed = result["flow_regime"][
        "final_plastic_to_imposed_power_fraction"]
    classification = (
        "HALL_PETCH_NUCLEATION_CAPACITY_CAN_CARRY_IMPOSED_RATE"
        if ceiling_fraction >= 1.0 else
        "IMPOSED_RATE_EXCEEDS_HP_NUCLEATION_CAPACITY_RETAINED_CARRIERS_REQUIRED")
    return {
        "directory": str(root.resolve()),
        "checkpoint_sha256": _sha(checkpoint),
        "parent_provenance": provenance,
        "step": step,
        "load": load,
        "temperature_maximum_K": maximum_temperature,
        "imposed_shear_rate_s": imposed,
        **ceiling,
        "maximum_nucleation_capacity_fraction_of_imposed_rate": ceiling_fraction,
        "observed_terminal_plastic_to_imposed_power_fraction": observed,
        "observed_to_nucleation_ceiling_ratio": observed/max(
            ceiling_fraction, np.finfo(float).tiny),
        "classification": classification,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    case = analyze(args.case)
    payload = {
        "schema": "v67-hall-petch-capacity-ceiling-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case": case,
        "decision": {"classification": case["classification"]},
        "claim_boundary": (
            "This is the generous all-four-families-accessible EXP-floor upper "
            "bound from the immutable analytical nucleation law. It does not "
            "limit glide of retained carriers and is not a fitted source rate."),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
