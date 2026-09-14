#!/usr/bin/env python3
"""Run deterministic local SIBM/DDRX hard-invariant fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from full_model.production.arrhenius_kinetics import ActivatedProcess
from full_model.production.moving_front import (
    DefectState, advance_front, initialize_sparse_front, state_arrays,
    state_from_checkpoint, state_metadata_json)
from full_model.production.sibm_boundary import (
    BoundaryGraphState, SIBMParameters, advance_boundary,
    circular_bulge_pressure_Pa, state_from_json, state_json)


def _parameters(mobility=2e-17):
    return SIBMParameters(
        boundary_energy_J_m2=0.5,
        mobility_prefactor_m4_J_s=mobility,
        process=ActivatedProcess("existing-HAGB", 1e8, 0.0, 1e8),
        activation_enthalpy_0_J=0.35*1.602176634e-19,
        critical_pressure_Pa=1e9, exp_a=2.0, exp_n=1.5,
        exp_floor=0.1, represented_thickness_m=2*2.48e-10)


def _state(height=None, labels=(3, 8), angles=(-0.2, 0.35)):
    return BoundaryGraphState(
        np.zeros(64) if height is None else np.asarray(height, float),
        labels[0], labels[1], angles[0], angles[1])


def _step(state, drive, params=None):
    return advance_boundary(
        state, spacing_m=1e-7, dt_s=1e-5, temperature_K=1100.0,
        parameters=params or _parameters(), stored_difference_Pa=drive)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    flat, flat_v = _step(_state(), 0.0)
    favorable, favorable_v = _step(_state(), 2e7)
    reversed_state, reversed_v = _step(_state(), -2e7)
    disabled, disabled_v = _step(_state(), 2e7, _parameters(0.0))
    x = np.arange(64)*2*np.pi/64
    curved0 = _state(2e-8*np.cos(x))
    curved, _ = _step(curved0, 0.0)
    params = _parameters(); drive = 2e6
    critical = params.boundary_energy_J_m2/drive
    sub_pressure = circular_bulge_pressure_Pa(0.8*critical, drive, 0.0, params)
    super_pressure = circular_bulge_pressure_Pa(1.2*critical, drive, 0.0, params)

    parent = DefectState(
        np.full((8, 8, 2), 4e14), np.full((8, 8, 2), 3e14),
        np.full((8, 8, 2), 2e14), np.full((8, 8), 1e14))
    front = initialize_sparse_front(parent, np.zeros((8, 8)), 5e14, 3, 8)
    chi = np.zeros((8, 8)); chi[:, :4] = 1.0
    front1, _ = advance_front(
        front, chi, cell_area_m2=1e-14, represented_thickness_m=1e-6,
        line_energy_J_m=2e-9, boundary_storage_fraction=0.1,
        sink_fraction=0.05)
    ledger_once = front1.ledger
    front2, _ = advance_front(
        front1, np.zeros_like(chi), cell_area_m2=1e-14,
        represented_thickness_m=1e-6, line_energy_J_m=2e-9,
        boundary_storage_fraction=0.1, sink_fraction=0.05)
    front3, _ = advance_front(
        front2, chi, cell_area_m2=1e-14, represented_thickness_m=1e-6,
        line_energy_J_m=2e-9, boundary_storage_fraction=0.1,
        sink_fraction=0.05)
    restored_front = state_from_checkpoint(
        state_metadata_json(front1), state_arrays(front1))

    restored_before = state_from_json(state_json(_state()), _state().height_m)
    restored_during = state_from_json(state_json(favorable), favorable.height_m)
    after, _ = _step(favorable, 2e7)
    restored_after = state_from_json(state_json(after), after.height_m)
    during_a, _ = _step(favorable, 2e7)
    during_b, _ = _step(restored_during, 2e7)
    perm_a, _ = _step(_state(labels=(3, 8), angles=(-0.2, 0.35)), 2e7)
    perm_b, _ = _step(_state(labels=(8, 3), angles=(0.35, -0.2)), -2e7)

    checks = dict(
        flat_equal_stationary=bool(np.array_equal(flat.height_m, _state().height_m)),
        favorable_moves_correct_direction=bool(np.mean(favorable_v) > 0.0),
        reversed_drive_reverses=bool(np.mean(reversed_v) < 0.0),
        mobility_disabled_stationary=bool(np.array_equal(disabled.height_m, _state().height_m)),
        curvature_only_relaxes=bool(np.ptp(curved.height_m) < np.ptp(curved0.height_m)),
        subcritical_bulge_retracts=bool(sub_pressure < 0.0),
        supercritical_bulge_grows=bool(super_pressure > 0.0),
        line_ledger_closed=bool(abs(front1.ledger.line_closure_m) <= 1e-20),
        signed_burgers_closed=bool(front1.ledger.signed_burgers_change_m2 == 0.0),
        energy_heat_ledger_closed=bool(
            front1.ledger.line_energy_released_J == front1.ledger.heat_released_J
            and abs(favorable.ledger.energy_closure_J) <= 1e-25),
        phase_area_ledger_nonzero=bool(favorable.ledger.signed_area_change_m2 > 0.0),
        advance_retreat_no_double_processing=bool(front3.ledger == ledger_once),
        front_restart_exact=bool(
            restored_front.ledger == front1.ledger and all(
                np.array_equal(state_arrays(restored_front)[k], v)
                for k, v in state_arrays(front1).items())),
        restart_before_exact=bool(np.array_equal(restored_before.height_m, _state().height_m)),
        restart_during_exact=bool(
            np.array_equal(during_a.height_m, during_b.height_m)
            and during_a.ledger == during_b.ledger),
        restart_after_exact=bool(
            np.array_equal(restored_after.height_m, after.height_m)
            and restored_after.ledger == after.ledger),
        label_permutation_common_equation=bool(np.array_equal(perm_a.height_m, -perm_b.height_m)),
        orientation_inherited=bool(
            favorable.left_orientation_rad == -0.2
            and favorable.right_orientation_rad == 0.35),
        no_new_label_allocated=bool(
            (favorable.left_label, favorable.right_label) == (3, 8)))
    asb_path = Path(__file__).resolve().parents[1]/"verification"/"v10_matched_asb_smoke.json"
    asb = json.loads(asb_path.read_text()) if asb_path.exists() else {"passed": False}
    checks["matched_v32_full_v34_asb_source_regression"] = bool(asb["passed"])
    result = dict(
        schema="full-v34-v10-sibm-local-fixtures/v1",
        common_functional="gamma*integral(ds)-(psi_left-psi_right-P_FB)*integral(h dx)",
        pressure_law="negative variational derivative: Delta_psi-gamma*kappa-P_FB-P_drag",
        kinetics="EXP-floor DeltaG=DeltaH-TDeltaS; sign supplied only by common pressure",
        labels=[3, 8], orientations_rad=[-0.2, 0.35],
        critical_bulge=dict(
            analytical_radius_m=critical, subcritical_radius_m=0.8*critical,
            subcritical_pressure_Pa=sub_pressure,
            supercritical_radius_m=1.2*critical,
            supercritical_pressure_Pa=super_pressure,
            coefficient="Rc=gamma/(Delta_psi-P_FB-P_drag) for local circular curvature"),
        ledgers=dict(sibm=favorable.ledger.__dict__, front=front1.ledger.__dict__),
        checks=checks, fixture_passed=all(checks.values()),
        scientific_gate_passed=False,
        full_model_experiment_status="pending_single_HPC3_existing_HAGB_run",
        asb_regression_status=("passed_short_source_regression"
                               if asb["passed"] else "pending_v10_rerun"))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    if not result["fixture_passed"]:
        raise SystemExit("SIBM fixture failure")


if __name__ == "__main__":
    main()
