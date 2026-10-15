#!/usr/bin/env python3
"""Retained-state production discriminator for lattice->GB->structure flow."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.analysis.run_v67_growth_reversal_audit import _directional_audit
from full_model.analysis.run_v67_interface_structural_bridge import (
    _args_from_configuration,
)
from full_model.production.coupled_multigrain_interval import (
    advance_complete_multigrain_interval, make_endpoint,
)
from full_model.production.interface_structural_absorption import (
    InterfaceStructuralAbsorptionParameters,
)
from full_model.production.lattice_interface_absorption import (
    LatticeInterfaceAbsorptionParameters,
)


ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--intervals", type=int, default=50)
    parser.add_argument("--capture-distance", type=float, default=5e-7)
    args = parser.parse_args()
    source_status = subprocess.check_output(
        ("git", "status", "--porcelain"), cwd=ROOT, text=True)
    args.out.mkdir(parents=True, exist_ok=True)
    state, runtime, _, load, _, configuration, provenance = _load_checkpoint(
        args.checkpoint)
    physics_args = _args_from_configuration(configuration)
    spacing, _, _, _, _, _, context = _physics(physics_args)
    context = replace(
        context, shear_rate_s=0.0, front_enabled=False,
        source_mode="disabled", sink_mode="disabled",
        interface_sink_mode="disabled", interface_sink_parameters=None,
        structural_absorption_mode="common_axis",
        structural_absorption_parameters=InterfaceStructuralAbsorptionParameters(),
        lattice_absorption_mode="one_flux_mura",
        lattice_absorption_parameters=LatticeInterfaceAbsorptionParameters(
            capture_distance_m=args.capture_distance))
    endpoint = make_endpoint(state, runtime, load, context)
    before = _directional_audit(endpoint, context, fraction=2e-4)
    records = []
    dt = float(configuration["dt_s"])
    for index in range(1, args.intervals+1):
        interval = advance_complete_multigrain_interval(endpoint, dt, context)
        endpoint = interval.endpoint
        records.append({
            "interval": index,
            "lattice_classification": interval.audit[
                "lattice_absorption_classification"],
            "lattice_fraction": interval.audit["lattice_absorption_fraction"],
            "lattice_edge_fractions": interval.audit[
                "lattice_absorption_edge_fractions"],
            "lattice_edge_affinities_J_m": interval.audit[
                "lattice_absorption_edge_affinities_J_m"],
            "lattice_removed_line_m": interval.audit[
                "lattice_absorption_removed_line_m"],
            "lattice_mura_relative_residual": interval.audit[
                "lattice_absorption_mura_relative_residual"],
            "lattice_first_law_relative_residual": interval.audit[
                "lattice_absorption_relative_first_law_residual"],
            "structural_classification": interval.audit[
                "structural_absorption_classification"],
            "structural_content_change": interval.audit[
                "structural_absorption_content_change"],
        })
    after = _directional_audit(endpoint, context, fraction=2e-4)
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    accepted_lattice = sum(row["lattice_classification"].startswith("ACCEPTED")
                           for row in records)
    accepted_structural = sum(
        row["structural_classification"].startswith("ACCEPTED")
        for row in records)
    payload = {
        "schema": "v67-one-flux-lattice-structural-continuation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "source_tree_clean": not bool(source_status),
        "parent_checkpoint": str(args.checkpoint.resolve()),
        "parent_checkpoint_sha256": _sha(args.checkpoint),
        "parent_provenance": provenance,
        "configuration": {
            "intervals": args.intervals, "dt_s": dt,
            "capture_distance_m": args.capture_distance,
            "shear_rate_s": 0.0, "front_enabled": False,
            "lattice_absorption": "one_flux_mura",
            "structural_absorption": "common_axis",
        },
        "before_directional_audit": before,
        "after_directional_audit": after,
        "interval_records": records,
        "ledger": asdict(endpoint.runtime.ledger),
        "decision": {
            "accepted_lattice_intervals": accepted_lattice,
            "accepted_structural_intervals": accepted_structural,
            "mura_hard_balance_passed": bool(all(
                row["lattice_mura_relative_residual"] < 2e-11
                for row in records)),
            "first_law_hard_balance_passed": bool(all(
                row["lattice_first_law_relative_residual"] < .05
                for row in records
                if row["lattice_classification"].startswith("ACCEPTED"))),
            "supports_unchanged": bool(np.array_equal(
                endpoint.state.supports, state.supports)),
            "grain_ids_unchanged": endpoint.state.grain_ids == state.grain_ids,
        },
        "claim_boundary": (
            "Retained-state stationary production discriminator. It qualifies "
            "the coupled absorption path but not sustained dynamic DRX."),
    }
    (args.out/"result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
