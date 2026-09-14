#!/usr/bin/env python3
"""Map an equilibrated pinned cap into a zero-ledger common front state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PRODUCTION = Path(__file__).resolve().parents[1] / "production"
if str(PRODUCTION) not in sys.path:
    sys.path.insert(0, str(PRODUCTION))

from moving_front import (  # noqa: E402
    DefectState, initialize_existing_boundary_front,
    state_arrays, state_metadata_json,
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def build(source: Path, destination: Path, record_path: Path) -> dict:
    with np.load(source, allow_pickle=True) as data:
        arrays = {name: np.asarray(data[name]).copy() for name in data.files}
    experiment = json.loads(str(arrays["sibm_experiment_json"].item()))
    eta = arrays["eta"]
    phases = int(np.asarray(arrays["Ng"]).item())
    hphase = eta[..., :phases]**2*(3.0-2.0*eta[..., :phases])
    child = int(experiment["child_label"])
    chi = hphase[..., child]/np.maximum(np.sum(hphase, axis=2), 1e-300)
    state = initialize_existing_boundary_front(
        DefectState(arrays["rp"], arrays["rm"], arrays["rho_forest"],
                    arrays["rho_wall"]),
        chi, float(experiment["child_mean_density_m2"]),
        int(experiment["parent_label"]), child)
    for key in tuple(arrays):
        if key.startswith("sparse_front__"):
            del arrays[key]
    arrays.update({f"sparse_front__{key}": value
                   for key, value in state_arrays(state).items()})
    arrays["sparse_front_metadata_json"] = np.array(state_metadata_json(state))
    arrays["sibm_initial_child_fraction"] = chi.copy()
    experiment.update({
        "schema": "full-v34-sibm-existing-HAGB/v3",
        "common_pinned_state_equilibrated": True,
        "common_state_front_ledger_zeroed": True,
        "common_state_chi_equals_processed_max": True,
        "common_state_source_checkpoint": str(source),
        "common_state_source_sha256": digest(source),
    })
    arrays["sibm_experiment_json"] = np.array(
        json.dumps(experiment, sort_keys=True, separators=(",", ":")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    result = {
        "schema": "full-v34-v14-common-front-checkpoint/v1",
        "classification": "SIBM_EQUILIBRATED_COMMON_STATE_BUILT",
        "source_checkpoint": str(source),
        "source_sha256": digest(source),
        "output_checkpoint": str(destination),
        "output_sha256": digest(destination),
        "grid": list(chi.shape),
        "minimum_child_fraction": float(np.min(chi)),
        "maximum_child_fraction": float(np.max(chi)),
        "maximum_chi_processed_difference": float(np.max(np.abs(
            state.chi-state.processed_max))),
        "front_ledger": state.ledger.__dict__,
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.destination, args.record), sort_keys=True))


if __name__ == "__main__":
    main()
