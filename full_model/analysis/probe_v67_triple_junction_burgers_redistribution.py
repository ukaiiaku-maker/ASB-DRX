#!/usr/bin/env python3
"""Price a local conservative triple-junction Burgers reaction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.analyze_v67_existing_boundary_growth import (
    _directional_bias,
)
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.analysis.run_v67_growth_reversal_audit import _directional_audit
from full_model.analysis.run_v67_lattice_structural_continuation import (
    _args_from_configuration,
)
from full_model.production.complete_multigrain_energy import (
    evaluate_complete_multigrain_energy,
)
from full_model.production.coupled_multigrain_interval import make_endpoint
from full_model.production.structural_boundary import StructuralBoundaryParameters
from full_model.production.structural_interface_network import (
    JunctionDisclinationParameters,
)
from full_model.production.triple_junction_burgers_redistribution import (
    project_triple_junction_burgers,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _energy(state, runtime, load, context):
    strain = np.array([[0.0, .5*load], [.5*load, 0.0]])
    options = dict(context.energy_kwargs)
    options.update(
        structural_interfaces=runtime.structural_interfaces,
        structural_burgers_m=context.wall_parameters.burgers_m,
        structural_parameters=StructuralBoundaryParameters(
            burgers_m=context.wall_parameters.burgers_m),
        junction_disclination_parameters=JunctionDisclinationParameters())
    return evaluate_complete_multigrain_energy(
        state, spacing_m=context.spacing_m,
        represented_thickness_m=context.represented_thickness_m,
        wall_parameters=context.wall_parameters, mean_strain=strain, **options)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--grain-id", type=int, default=20)
    args = parser.parse_args()
    state, runtime, _, load, _, configuration, provenance = _load_checkpoint(
        args.checkpoint)
    physics_args = _args_from_configuration(configuration)
    *_, context = _physics(physics_args)
    original_energy = _energy(state, runtime, load, context)
    baseline = project_triple_junction_burgers(
        state, burgers_m=context.wall_parameters.burgers_m,
        reaction_fraction=0.0, relax_scalar_line_to_capacity=True)
    baseline_energy = _energy(
        baseline.candidate_state, runtime, load, context)
    baseline_audit = _directional_audit(
        make_endpoint(baseline.candidate_state, runtime, load, context),
        context, fraction=2e-4)
    baseline_bias = _directional_bias(baseline_audit, args.grain_id)[
        "net_growth_minus_shrink_Pa"]
    rows = []
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        projection = project_triple_junction_burgers(
            state, burgers_m=context.wall_parameters.burgers_m,
            reaction_fraction=fraction, relax_scalar_line_to_capacity=True)
        energy = _energy(projection.candidate_state, runtime, load, context)
        audit = _directional_audit(
            make_endpoint(projection.candidate_state, runtime, load, context),
            context, fraction=2e-4)
        bias = _directional_bias(audit, args.grain_id)[
            "net_growth_minus_shrink_Pa"]
        rows.append({
            "reaction_fraction": fraction,
            "projection": {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in asdict(projection).items()
                if key != "candidate_state"},
            "delta_helmholtz_from_capacity_baseline_J": (
                energy.helmholtz_J-baseline_energy.helmholtz_J),
            "directional_growth_minus_shrink_Pa": bias,
            "directional_bias_change_from_capacity_baseline_Pa": (
                bias-baseline_bias),
        })
    best = min(rows, key=lambda row: row[
        "delta_helmholtz_from_capacity_baseline_J"])
    hard = bool(all(row["projection"]["feasible"] for row in rows))
    downhill = best["delta_helmholtz_from_capacity_baseline_J"] < 0.0
    source_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    source_clean = not subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=ROOT, text=True).strip()
    payload = {
        "schema": "v67-triple-junction-burgers-redistribution-probe-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "source_tree_clean": source_clean,
        "parent_checkpoint": str(args.checkpoint.resolve()),
        "parent_checkpoint_sha256": _sha(args.checkpoint),
        "parent_provenance": provenance,
        "original_to_capacity_baseline_delta_helmholtz_J": (
            baseline_energy.helmholtz_J-original_energy.helmholtz_J),
        "capacity_baseline_directional_growth_minus_shrink_Pa": baseline_bias,
        "rows": rows,
        "decision": {
            "hard_vector_balance_passed": hard,
            "best_fraction": best["reaction_fraction"],
            "best_delta_helmholtz_from_capacity_baseline_J": best[
                "delta_helmholtz_from_capacity_baseline_J"],
            "best_directional_growth_minus_shrink_Pa": best[
                "directional_growth_minus_shrink_Pa"],
            "downhill_beyond_vector_neutral_capacity_relaxation": downhill,
            "restores_positive_RX_bias": bool(
                best["directional_growth_minus_shrink_Pa"] > 0.0),
            "classification": (
                "UNQUALIFIED_VECTOR_BALANCE" if not hard else
                ("DOWNHILL_LOCAL_TJ_BURGERS_REACTION" if downhill else
                 "NO_DOWNHILL_LOCAL_TJ_BURGERS_REACTION")),
        },
        "claim_boundary": (
            "Symmetric local three-edge upper-bound probe. The fraction-zero "
            "capacity-relaxed state isolates ordinary vector-neutral line "
            "removal; only the additional energy change is attributed to "
            "conservative Burgers redistribution. No production kinetics, "
            "plastic distortion, Nye update, support motion, or label change "
            "is asserted."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
