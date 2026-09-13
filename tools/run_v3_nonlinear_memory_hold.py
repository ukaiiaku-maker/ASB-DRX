#!/usr/bin/env python3
"""Restartable frozen-state nonlinear delayed-memory hold and cost pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from asb_drx.arrhenius_v3 import ArrheniusMechanism, BoundedActivationEntropy, ExpFloorEnthalpy
from asb_drx.nonlinear_memory_cdd_v3 import (
    FrozenMemoryParameters, FrozenMemoryState, advance_frozen_memory,
    homogeneous_equilibrium_state,
)
from asb_drx.physical_noise import periodic_physical_noise
from asb_drx.vector_topology_cdd_v3 import (
    JunctionReaction, VectorTopologyNetwork, arrhenius_forest_linearization,
    reaction_transport_memory_symbol_s_inv,
)


EV_J = 1.602176634e-19
CASES = {
    "slow": dict(temperature=1000.0, stress=0.7e9, diffusion=1.0e-12,
                 relaxation=0.01, wavelength=1.0226458409032313e-6),
    "intermediate": dict(temperature=800.0, stress=0.5e9, diffusion=1.0e-12,
                         relaxation=0.01, wavelength=5.615731903931205e-7),
    "fast": dict(temperature=1000.0, stress=0.5e9, diffusion=1.0e-12,
                 relaxation=0.003, wavelength=3.076354820150677e-7),
}


def build(case: str, cells: int, wavelengths: int) -> tuple[FrozenMemoryParameters, FrozenMemoryState, int]:
    config = CASES[case]
    mechanism = ArrheniusMechanism(
        ExpFloorEnthalpy(0.45 * EV_J, 0.8e9, 900.0, 0.25, 1.2, 2.0),
        BoundedActivationEntropy(reference_kB=0.2), 2.0e7,
        validity_temperature_K=(500.0, 1400.0), validity_stress_Pa=(0.0, 2.0e9),
    )
    network = VectorTopologyNetwork(
        2.86e-10,
        (JunctionReaction((0, 1), "sessile", mechanism, 0.02 * EV_J, 5.0e14),),
    )
    parameters = FrozenMemoryParameters(
        network, (mechanism,) * 4, np.full(4, 1.0e-9),
        np.full(4, config["stress"]), np.full((4, 1), 1.0e-6),
        np.full(8, config["diffusion"]), np.asarray([config["relaxation"]]),
        config["temperature"], wavelengths * config["wavelength"],
    )
    return parameters, homogeneous_equilibrium_state(cells, 2.5e14, parameters), wavelengths


def eigenmode(parameters: FrozenMemoryParameters, state: FrozenMemoryState, mode: int):
    k = 2.0 * np.pi * mode / parameters.domain_m
    mobile = state.mobile_m2[:, 0]
    junction = state.junction_m2[:, 0]
    velocity, derivative = arrhenius_forest_linearization(
        parameters.glide, parameters.event_lengths_m, parameters.resolved_stress_Pa,
        junction, parameters.forest_coefficients_Pa_m2, parameters.temperature_K,
    )
    symbol = reaction_transport_memory_symbol_s_inv(
        np.asarray([k, 0.0]), mobile, junction, parameters.network,
        parameters.temperature_K, np.asarray([1.0, 0.0]), velocity, derivative,
        parameters.mobile_diffusivity_m2_s, parameters.memory_relaxation_s,
    )
    values, vectors = np.linalg.eig(symbol)
    selected = int(np.argmax(values.real))
    return values[selected], vectors[:, selected], np.linalg.inv(vectors)[selected]


def modal_amplitude(state: FrozenMemoryState, mode: int, left: np.ndarray) -> float:
    fields = np.concatenate((state.mobile_m2, state.junction_m2, state.memory_m2), axis=0)
    coefficient = np.fft.fft(fields - np.mean(fields, axis=1)[:, None], axis=1)[:, mode]
    coefficient /= fields.shape[1]
    return float(abs(left @ coefficient))


def save_checkpoint(
    path: Path, case: str, state: FrozenMemoryState,
    history: list[dict], initial_amplitude_m2: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, case=np.asarray(case), mobile_m2=state.mobile_m2,
                 junction_m2=state.junction_m2, memory_m2=state.memory_m2,
                 time_s=np.asarray(state.time_s), history=np.asarray(json.dumps(history)),
                 initial_amplitude_m2=np.asarray(initial_amplitude_m2))
    temporary.replace(path)


def main(args: argparse.Namespace) -> None:
    parameters, equilibrium, mode = build(args.case, args.cells, args.wavelengths)
    eigenvalue, right, left = eigenmode(parameters, equilibrium, mode)
    if eigenvalue.real <= 0.0:
        raise RuntimeError("selected nonlinear hold is not linearly unstable")
    history: list[dict] = []
    if args.checkpoint.exists():
        with np.load(args.checkpoint, allow_pickle=False) as archive:
            if str(archive["case"]) != args.case:
                raise ValueError("checkpoint case does not match request")
            state = FrozenMemoryState(
                archive["mobile_m2"], archive["junction_m2"], archive["memory_m2"],
                float(archive["time_s"]),
            )
            history = json.loads(str(archive["history"]))
            initial_amplitude = float(archive["initial_amplitude_m2"])
    else:
        if args.seed_type == "eigenvector":
            right = right / np.max(np.abs(right))
            phase = np.exp(2j * np.pi * mode * np.arange(args.cells) / args.cells)
            perturbation = (
                args.amplitude * 2.5e14
                * np.real(right[:, None] * phase[None, :])
            )
        else:
            base_fields = np.concatenate((
                equilibrium.mobile_m2, equilibrium.junction_m2,
                equilibrium.memory_m2,
            ), axis=0)
            perturbation = np.empty_like(base_fields)
            correlation_m = 0.35 * CASES[args.case]["wavelength"]
            for species in range(base_fields.shape[0]):
                noise = periodic_physical_noise(
                    args.cells, parameters.domain_m, correlation_m,
                    args.seed + 104729 * species,
                )
                perturbation[species] = (
                    args.amplitude * base_fields[species] * noise
                )
        state = FrozenMemoryState(
            equilibrium.mobile_m2 + perturbation[:8],
            equilibrium.junction_m2 + perturbation[8:9],
            equilibrium.memory_m2 + perturbation[9:10],
        )
        initial_amplitude = modal_amplitude(state, mode, left)
    start_time = state.time_s
    wall_start = time.perf_counter()
    target_time = args.target_efolds / eigenvalue.real
    validity_stop = None
    while state.time_s < target_time - 1.0e-15:
        duration = min(args.chunk_s, target_time - state.time_s)
        before = time.perf_counter()
        try:
            state, diagnostics = advance_frozen_memory(state, parameters, duration)
        except RuntimeError as error:
            validity_stop = str(error)
            break
        amplitude = modal_amplitude(state, mode, left)
        history.append({
            "time_s": state.time_s,
            "cumulative_efolds": float(eigenvalue.real * state.time_s),
            "modal_amplitude_m2": amplitude,
            "measured_log_amplification": float(np.log(amplitude / initial_amplitude)),
            "wall_seconds": time.perf_counter() - before,
            **diagnostics,
        })
        save_checkpoint(args.checkpoint, args.case, state, history, initial_amplitude)
    elapsed = time.perf_counter() - wall_start
    achieved = float(eigenvalue.real * state.time_s)
    result = {
        "schema": "asb-drx-nonlinear-memory-hold/v1",
        "source_commit": os.environ.get("SOURCE_COMMIT", "local-working-tree"),
        "case": args.case,
        "cells": args.cells,
        "domain_m": parameters.domain_m,
        "selected_mode": mode,
        "seed_type": args.seed_type,
        "seed": args.seed,
        "predicted_growth_s_inv": float(eigenvalue.real),
        "predicted_wavelength_m": parameters.domain_m / mode,
        "simulated_time_s": state.time_s,
        "cumulative_efolds": achieved,
        "measured_log_amplification": (
            history[-1]["measured_log_amplification"] if history else 0.0
        ),
        "wall_seconds_this_invocation": elapsed,
        "simulated_seconds_this_invocation": state.time_s - start_time,
        "projected_wall_seconds_to_10_efolds": elapsed / max(achieved - eigenvalue.real * start_time, 1e-30) * 10.0,
        "minimum_mobile_density_m2": float(np.min(state.mobile_m2)),
        "validity_stop": validity_stop,
        "classification": (
            "HARD_NONNEGATIVITY_VALIDITY_STOP_AFTER_HORIZON"
            if validity_stop is not None and achieved >= 10.0
            else "HARD_NONNEGATIVITY_VALIDITY_STOP_BEFORE_HORIZON"
            if validity_stop is not None
            else "AMPLIFICATION_HORIZON_REACHED" if achieved >= 10.0
            else "INSUFFICIENT_PHYSICAL_AMPLIFICATION_HORIZON"
        ),
        "scientific_gate_passed": False,
        "physical_CDD_wall_gate_passed": False,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), default="fast")
    parser.add_argument("--cells", type=int, default=128)
    parser.add_argument("--wavelengths", type=int, default=8)
    parser.add_argument("--target-efolds", type=float, default=2.0)
    parser.add_argument("--chunk-s", type=float, default=1.0e-3)
    parser.add_argument("--amplitude", type=float, default=1.0e-6)
    parser.add_argument("--seed-type", choices=("eigenvector", "broadband"),
                        default="eigenvector")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
