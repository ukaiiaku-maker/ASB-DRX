#!/usr/bin/env python3
"""Create a checksum-bound deterministic V64 analytic-origin checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.run_v58_three_grain_production import (
    _load_checkpoint, _save_checkpoint, initialize_network_state,
    network_interfaces,
)
from full_model.production.multigrain_common_state import multigrain_state_digest
from full_model.production.multigrain_production import MultiGrainProductionRuntime


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--expected-template-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if digest(args.template) != args.expected_template_sha256:
        raise ValueError("template checkpoint SHA-256 mismatch")
    head = subprocess.check_output(("git", "rev-parse", "HEAD"),
                                   text=True).strip()
    declared = subprocess.check_output(
        ("git", "rev-parse", args.source_commit), text=True).strip()
    if head != declared:
        raise ValueError("declared source commit does not match executing HEAD")
    _, _, _, _, _, configuration, template_provenance = _load_checkpoint(
        args.template)
    configuration = dict(configuration or {})
    if configuration.get("schema") == "asb-drx-v64-adaptive-production-v1":
        configuration = dict(configuration["physical_configuration"])
    n = int(configuration["n"])
    spacing = float(configuration["length_m"])/n
    thickness = 2.0*2.48e-10
    state = initialize_network_state(
        n, float(configuration["temperature_K"]),
        int(configuration["grain_count"]), spacing_m=spacing,
        interface_width_m=float(configuration["interface_width_m"]),
        equal_density=configuration.get("case") == "equal_density",
        initial_temperature_band_K=float(
            configuration.get("initial_temperature_band_K", 0.0)),
        initial_temperature_band_width_m=float(
            configuration.get("initial_temperature_band_width_m", 3.125e-7)),
        single_crystal_band_normal=configuration.get(
            "single_crystal_band_normal", "x"),
        initial_density_band_fraction=float(
            configuration.get("initial_density_band_fraction", 0.0)),
        initial_density_band_width_m=float(
            configuration.get("initial_density_band_width_m", 3.125e-7)))
    runtime = MultiGrainProductionRuntime(network_interfaces(state))
    initial_volume = (np.sum(state.supports, axis=(1, 2))*spacing**2*thickness)
    provenance = {
        "source_commit": head,
        "origin": "deterministic_analytic_prepared_network",
        "template_checkpoint_sha256": args.expected_template_sha256,
        "template_source_commit": (template_provenance or {}).get(
            "source_commit"),
        "physical_kernels_changed": False,
        "initialization_implementation_unchanged_from_physical_source": (
            "1c0cb4ca0e8a9a30b0c1a320817a0b35b4d14499"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _save_checkpoint(
        args.output, state, runtime, 0, .012, initial_volume,
        configuration, provenance)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": digest(args.output),
        "state_digest": multigrain_state_digest(state),
        "n": n, "physical_time_s": 0.0, "applied_shear_strain": .012,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
