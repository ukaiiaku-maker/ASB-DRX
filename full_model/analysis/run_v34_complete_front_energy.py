#!/usr/bin/env python3
"""Run the non-vacuous V34 complete common-front energy qualification."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = ROOT/"full_model"/"production"
if str(PRODUCTION) not in sys.path:
    sys.path.insert(0, str(PRODUCTION))

from common_front_state import (  # noqa: E402
    initialize_common_front, state_arrays, state_from_checkpoint,
    state_metadata_json)
from common_tensorial_wall import CommonWallParameters, CommonWallState  # noqa: E402
from complete_front_energy import (  # noqa: E402
    evaluate_common_front_transaction, evaluate_complete_directional_kinetics,
    independently_assemble_product_rule_nye)
from moving_front import DefectState, initialize_declared_boundary_front  # noqa: E402


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def common_state(n=32, nj=2):
    shape = (n, n, 4)
    beta = np.zeros((n, n, 3, 3))
    # Deliberately signed, finite, and unequal reservoirs.
    return CommonWallState(
        np.full(shape, 2.1e14), np.full(shape, 1.4e14),
        np.full(shape, 1.3e14), np.full(shape, .7e14),
        np.full(shape, .55e14), np.full(shape, .25e14),
        np.full((n, n, nj), .4e14), np.full((n, n), .2),
        np.full((n, n), .1), np.zeros(shape), beta,
        np.zeros(shape+(3,)), np.zeros(shape+(3, 3)),
        np.zeros((n, n)), np.full((n, n), 900.0))


def fixture(n=32):
    common = common_state(n)
    defect = DefectState(
        common.mobile_plus_m2.copy(), common.mobile_minus_m2.copy(),
        common.forest_plus_m2+common.forest_minus_m2,
        np.sum(common.wall_plus_m2+common.wall_minus_m2, axis=2))
    chi = np.zeros((n, n))
    front = initialize_declared_boundary_front(defect, defect, chi, 0, 1)
    return initialize_common_front(front, common)


def eta_field(n):
    x = np.arange(n)[:, None]
    b = .5*(1.0+np.tanh((x-.45*n)/2.5))*np.ones((1, n))
    return np.stack((1.0-b, b), axis=2)


def accepted_front(state, fraction=.35):
    chi = np.zeros_like(state.front.chi)
    chi[:, :5] = fraction
    return replace(state.front, chi=chi, processed_max=chi.copy(),
                   cleanup_max=chi.copy())


def run(output):
    n = 32
    spacing = 2.0e-8
    thickness = 5.0e-10
    cell_volume = spacing*spacing*thickness
    state = fixture(n)
    eta = eta_field(n)
    parameters = CommonWallParameters(spacing_m=spacing)

    def energy_options(boundary_energy=1.0e-9):
        return dict(
            wall_parameters=parameters, mean_strain=np.zeros((2, 2)),
            phase_barrier_J_m3=5.0e6, phase_gradient_J_m=5.0e-7,
            boundary_line_energy_J_m=boundary_energy,
            boundary_junction_energy_J_m=2.0e-10,
            reference_temperature_K=900.0)

    common = dict(
        spacing_m=spacing, cell_volume_m3=cell_volume,
        represented_thickness_m=thickness,
        transmission_fraction=.6, boundary_storage_fraction=.05,
        neutral_sink_fraction=.05)
    zero = evaluate_common_front_transaction(
        state, state.front, eta, eta, energy_kwargs=energy_options(), **common)
    downhill = evaluate_common_front_transaction(
        state, accepted_front(state), eta, eta,
        energy_kwargs=energy_options(), **common)
    uphill = evaluate_common_front_transaction(
        state, accepted_front(state), eta, eta,
        spacing_m=spacing, cell_volume_m3=cell_volume,
        represented_thickness_m=thickness, transmission_fraction=0.0,
        boundary_storage_fraction=1.0, neutral_sink_fraction=0.0,
        energy_kwargs=energy_options(1.0e-4))

    # Explicit kinematic sensitivity at unchanged geometry.
    beta = state.parent.beta_p.copy()
    beta[..., 0, 0] = .02
    beta_state = replace(state, parent=replace(state.parent, beta_p=beta))
    beta_trial = evaluate_common_front_transaction(
        beta_state, accepted_front(beta_state), eta, eta,
        energy_kwargs=energy_options(), **common)

    # Independently assembled product-rule checks.
    x = np.arange(n)[:, None]
    chi = .5+.25*np.sin(2*np.pi*x/n)*np.ones((1, n))
    support_front = replace(state.front, chi=chi, processed_max=chi.copy(),
                            cleanup_max=chi.copy())
    support_state = replace(state, front=support_front)
    identical_beta = support_state.parent.beta_p.copy()
    identical_beta[..., 0, 2] = .03
    identical = replace(
        support_state,
        parent=replace(support_state.parent, beta_p=identical_beta),
        child=replace(support_state.child, beta_p=identical_beta),
        wake=replace(support_state.wake, beta_p=identical_beta))
    identical_audit = independently_assemble_product_rule_nye(identical, spacing)
    jump_beta = support_state.child.beta_p.copy()
    jump_beta[..., 1, 2] = .02
    jump = replace(support_state,
                   child=replace(support_state.child, beta_p=jump_beta))
    jump_audit = independently_assemble_product_rule_nye(jump, spacing)

    # Finite actual forward/opposite trials for kinetic normalization.  The
    # opposite is evaluated from state, never inferred as minus the forward.
    kinetic_chi = np.full_like(state.front.chi, .4)
    kinetic_state = replace(state, front=replace(
        state.front, chi=kinetic_chi,
        processed_max=np.full_like(kinetic_chi, .7),
        cleanup_max=np.full_like(kinetic_chi, .7)))
    forward_chi = kinetic_chi.copy(); forward_chi[:, :5] += .1
    directional = evaluate_complete_directional_kinetics(
        kinetic_state, replace(kinetic_state.front, chi=forward_chi),
        eta, eta_field(n), event_volume_m3=cell_volume,
        spacing_m=spacing, cell_volume_m3=cell_volume,
        represented_thickness_m=thickness, transmission_fraction=.6,
        boundary_storage_fraction=.05, neutral_sink_fraction=.05,
        energy_kwargs=energy_options())

    restored = state_from_checkpoint(
        state_metadata_json(downhill.published_state),
        state_arrays(downhill.published_state), downhill.published_state.front)
    restart_exact = all(np.array_equal(value, state_arrays(restored)[name])
                        for name, value in state_arrays(
                            downhill.published_state).items())
    processed = downhill.published_state.ledger.processed_line_m
    finite_sweep = float(np.sum(
        downhill.published_state.front.chi-state.front.chi,
        dtype=np.longdouble)*cell_volume)
    record = {
        "schema": "asb-drx/v34-complete-front-energy/v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "parameters": {
            "grid": n, "spacing_m": spacing,
            "represented_thickness_m": thickness,
            "cell_volume_m3": cell_volume,
            "transmission_fraction": .6,
            "boundary_storage_fraction": .05,
            "neutral_sink_fraction": .05,
            "phase_barrier_J_m3": 5.0e6,
            "phase_gradient_J_m": 5.0e-7,
        },
        "zero_event": zero.decision.as_dict(),
        "downhill": downhill.decision.as_dict(),
        "uphill": uphill.decision.as_dict(),
        "beta_contrast": beta_trial.decision.as_dict(),
        "finite_signed_swept_volume_m3": finite_sweep,
        "junction_energy_response_J": (
            downhill.decision.candidate.signed_junction_storage_J
            -downhill.decision.before.signed_junction_storage_J),
        "beta_elastic_response_J": (
            beta_trial.decision.candidate.recoverable_elastic_J
            -downhill.decision.candidate.recoverable_elastic_J),
        "line_closure_relative": (
            downhill.published_state.ledger.maximum_line_closure_m/processed
            if processed else 0.0),
        "signed_closure_m2": downhill.published_state.ledger.maximum_signed_closure_m2,
        "product_rule": {
            "identical_owner_interface_max_m1": float(np.max(np.abs(
                identical_audit.support_interface_m1))),
            "jump_interface_norm_m1": float(np.linalg.norm(
                jump_audit.support_interface_m1)),
            "jump_representation_residual_relative": float(
                np.linalg.norm(jump_audit.discrete_representation_residual_m1)
                /max(np.linalg.norm(jump_audit.exact_reconstructed_m1), 1e-300)),
            "assembly_is_independent": True,
        },
        "directional_kinetics": {
            "a_to_b_event_J": directional.a_to_b_event_J,
            "b_to_a_event_J": directional.b_to_a_event_J,
            "forward_signed_volume_m3": directional.forward_signed_volume_m3,
            "opposite_signed_volume_m3": directional.opposite_signed_volume_m3,
            "opposite_evaluated_from_actual_state": True,
            "opposite_is_algebraic_negative": bool(
                directional.a_to_b_event_J == -directional.b_to_a_event_J),
            "minimum_normalization_fraction_of_cell": 1.0e-6
        },
        "restart_bitwise_exact": restart_exact,
        "fixture_passed": bool(
            zero.published_state is state
            and zero.decision.classification == "EXACT_ZERO_EVENT_IDENTITY"
            and downhill.decision.accepted and not uphill.decision.accepted
            and finite_sweep > 0.0 and restart_exact
            and downhill.published_state.ledger.boundary_line_m > 0.0
            and downhill.published_state.ledger.annihilated_line_m > 0.0
            and processed > 0.0),
        "scientific_gate_passed": False,
        "classification": "I2_OPERATOR_QUALIFIED_PRODUCTION_TRAJECTORY_PENDING",
        "scope": "Manufactured finite common-state transaction; not spontaneous migration.",
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path) for path in (
                PRODUCTION/"complete_front_energy.py",
                PRODUCTION/"common_front_state.py",
                PRODUCTION/"coupled_front_production.py",
                PRODUCTION/"drx_full_v34_recovery.py")},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["fixture_passed"] else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.output))


if __name__ == "__main__":
    main()
