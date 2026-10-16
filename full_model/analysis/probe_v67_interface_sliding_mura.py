#!/usr/bin/env python3
"""Price the Mura-compatible tangential GB-sliding upper bound."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
from full_model.production.interface_sliding_mura import (
    project_tangential_interface_burgers_to_sliding,
)
from full_model.production.structural_boundary import StructuralBoundaryParameters
from full_model.production.structural_interface_network import (
    JunctionDisclinationParameters,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _energy(state, runtime, load, context, extra):
    strain = np.array([[0.0, .5*load], [.5*load, 0.0]])
    options = dict(context.energy_kwargs)
    options.update(
        structural_interfaces=runtime.structural_interfaces,
        structural_burgers_m=context.wall_parameters.burgers_m,
        structural_parameters=StructuralBoundaryParameters(
            burgers_m=context.wall_parameters.burgers_m),
        junction_disclination_parameters=JunctionDisclinationParameters(),
        additional_plastic_distortion=extra)
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
    *_, context = _physics(_args_from_configuration(configuration))
    if configuration.get("wall_nye_match_coefficient_J_m") is not None:
        context = replace(context, wall_parameters=replace(
            context.wall_parameters,
            wall_nye_match_coefficient_J_m=float(
                configuration["wall_nye_match_coefficient_J_m"])))
    zero = np.zeros_like(state.owners[0].beta_p)
    baseline_energy = _energy(state, runtime, load, context, zero)
    baseline_bias = _directional_bias(
        _directional_audit(make_endpoint(state, runtime, load, context),
                           context, fraction=2e-4), args.grain_id)[
        "net_growth_minus_shrink_Pa"]
    authoritative_result = args.checkpoint.parent/"result.json"
    authoritative_bias = None
    if authoritative_result.exists():
        authoritative_bias = _directional_bias(
            json.loads(authoritative_result.read_text())["after_directional_audit"],
            args.grain_id)["net_growth_minus_shrink_Pa"]
    baseline_matches = bool(
        authoritative_bias is None
        or abs(baseline_bias-authoritative_bias) <= 1.0)
    rows = []
    for fraction in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
        projection = project_tangential_interface_burgers_to_sliding(
            state, spacing_m=context.spacing_m,
            burgers_m=context.wall_parameters.burgers_m,
            reaction_fraction=fraction)
        energy = _energy(
            projection.candidate_state, runtime, load, context,
            projection.additional_plastic_distortion)
        probe_context = replace(
            context, energy_kwargs=dict(
                context.energy_kwargs,
                additional_plastic_distortion=(
                    projection.additional_plastic_distortion)))
        audit = _directional_audit(
            make_endpoint(projection.candidate_state, runtime, load,
                          probe_context), probe_context, fraction=2e-4)
        bias = _directional_bias(audit, args.grain_id)[
            "net_growth_minus_shrink_Pa"]
        record = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in asdict(projection).items()
            if key not in {"candidate_state", "additional_plastic_distortion",
                           "sliding_nye_m1", "removed_interface_burgers_m1"}}
        rows.append({
            "reaction_fraction": fraction,
            "projection": record,
            "additional_beta_rms": float(np.sqrt(np.mean(
                projection.additional_plastic_distortion**2))),
            "sliding_nye_rms_m1": float(np.sqrt(np.mean(
                projection.sliding_nye_m1**2))),
            "delta_helmholtz_J": energy.helmholtz_J-baseline_energy.helmholtz_J,
            "directional_growth_minus_shrink_Pa": bias,
            "directional_bias_change_Pa": bias-baseline_bias,
        })
    feasible = [row for row in rows if row["projection"]["feasible"]]
    best = min(feasible, key=lambda row: row["delta_helmholtz_J"])
    source_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    source_clean = not subprocess.check_output(
        ("git", "status", "--porcelain", "--untracked-files=no"),
        cwd=ROOT, text=True).strip()
    payload = {
        "schema": "v67-interface-sliding-mura-probe-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit, "source_tree_clean": source_clean,
        "parent_checkpoint": str(args.checkpoint.resolve()),
        "parent_checkpoint_sha256": _sha(args.checkpoint),
        "parent_provenance": provenance,
        "baseline_directional_growth_minus_shrink_Pa": baseline_bias,
        "authoritative_directional_growth_minus_shrink_Pa": authoritative_bias,
        "rows": rows,
        "decision": {
            "all_tested_fractions_feasible": len(feasible) == len(rows),
            "baseline_context_matches_authoritative_result": baseline_matches,
            "best_fraction": best["reaction_fraction"],
            "best_delta_helmholtz_J": best["delta_helmholtz_J"],
            "best_directional_growth_minus_shrink_Pa": best[
                "directional_growth_minus_shrink_Pa"],
            "best_directional_bias_change_Pa": best[
                "directional_bias_change_Pa"],
            "downhill_Mura_compatible_sliding_path": bool(
                best["delta_helmholtz_J"] < 0.0),
            "restores_positive_RX_bias": bool(
                baseline_bias <= 0.0
                < best["directional_growth_minus_shrink_Pa"]),
            "classification": (
                "UNQUALIFIED_BASELINE_CONTEXT" if not baseline_matches else
                "DOWNHILL_MURA_SLIDING_UPPER_BOUND"
                if best["delta_helmholtz_J"] < 0.0 else
                "NO_DOWNHILL_MURA_SLIDING_UPPER_BOUND"),
        },
        "claim_boundary": (
            "Zero-integral tangential content only. The topological mean "
            "remains on the interface and the debited fluctuation equals the "
            "curl-derived sliding Nye field. This is an energetic upper bound, "
            "not a production kinetic law or an inferred lattice-family "
            "transfer kernel."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
