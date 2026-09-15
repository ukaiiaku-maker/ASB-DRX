#!/usr/bin/env python3
"""Sequential V23 moving-front transfer/storage/recovery handoff audit."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.production.moving_front import (
    DefectState, advance_front, apply_common_constitutive_increment,
    initialize_sparse_front, reconstruct_mixture, state_arrays,
    state_from_checkpoint, state_metadata_json,
)


def parent_state(n=8):
    return DefectState(
        np.full((n, n, 4), 4e14), np.full((n, n, 4), 3e14),
        np.full((n, n, 4), 2e14), np.full((n, n), 1e14))


def run_stage(name, *, transmission, storage, sink, common_increment=False):
    parent = parent_state()
    state = initialize_sparse_front(
        parent, np.zeros((8, 8)), 8e14, 0, 1)
    if common_increment:
        mixture = reconstruct_mixture(state)
        updated = DefectState(
            mixture.rp*1.01, mixture.rm*1.01,
            mixture.forest*1.02, mixture.wall*1.02)
        state = apply_common_constitutive_increment(state, updated)
    fraction = np.zeros((8, 8)); fraction[:, :2] = .25
    state, _ = advance_front(
        state, fraction, cell_area_m2=1e-14,
        represented_thickness_m=1e-6, line_energy_J_m=1.5e-9,
        transmission_fraction=transmission,
        boundary_storage_fraction=storage, sink_fraction=sink,
        newly_swept_fraction=fraction)
    restored = state_from_checkpoint(state_metadata_json(state), state_arrays(state))
    restart_exact = all(np.array_equal(value, state_arrays(restored)[key])
                        for key, value in state_arrays(state).items())
    ledger = state.ledger.__dict__
    scale = max(abs(ledger["parent_line_processed_m"]), 1e-300)
    passed = (
        abs(ledger["line_closure_m"])/scale < 1e-14
        and ledger["signed_burgers_change_m2"] == 0.0
        and ledger["line_energy_released_J"] == ledger["heat_released_J"]
        and restart_exact)
    return {
        "stage": name,
        "transmission_fraction": transmission,
        "boundary_storage_fraction": storage,
        "sink_fraction": sink,
        "common_constitutive_increment": common_increment,
        "ledger": ledger,
        "restart_bitwise_exact": restart_exact,
        "passed": bool(passed),
    }


def main():
    stages = [
        run_stage("front_transfer_only", transmission=1.0, storage=0.0, sink=0.0),
        run_stage("boundary_storage", transmission=.5, storage=.2, sink=0.0),
        run_stage("recovery_and_heat", transmission=.5, storage=0.0, sink=0.0),
        run_stage("all_channels", transmission=.5, storage=.2, sink=.1,
                  common_increment=True),
    ]
    semantic_checks = {
        "transfer_only_has_no_released_heat":
            stages[0]["ledger"]["heat_released_J"] == 0.0,
        "storage_stage_stores_boundary_line":
            stages[1]["ledger"]["boundary_line_stored_m"] > 0.0,
        "recovery_stage_releases_heat":
            stages[2]["ledger"]["heat_released_J"] > 0.0,
        "all_stage_uses_every_requested_channel":
            (stages[3]["ledger"]["child_line_transmitted_m"] > 0
             and stages[3]["ledger"]["boundary_line_stored_m"] > 0
             and stages[3]["ledger"]["sink_line_m"] > 0
             and stages[3]["ledger"]["heat_released_J"] > 0),
    }
    passed = all(stage["passed"] for stage in stages) and all(semantic_checks.values())
    result = {
        "schema": "v23-sibm-sequential-handoff-audit-1",
        "stages": stages,
        "semantic_checks": semantic_checks,
        "no_phase_or_label_allocation": True,
        "fixture_passed": bool(passed),
        "scientific_gate_passed": False,
        "classification": ("SIBM_SEQUENTIAL_HANDOFF_PASSED_FULL_DYNAMIC_REQUALIFICATION_PENDING"
                           if passed else "SIBM_SEQUENTIAL_HANDOFF_FAILED"),
    }
    output = ROOT/"full_model"/"verification"/"v23_sibm_handoff_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
