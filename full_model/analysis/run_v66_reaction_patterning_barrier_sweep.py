#!/usr/bin/env python3
"""Full-operator barrier-timescale discriminator for concurrent patterning."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v66_production_groma_tangent import candidate_parameters
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallState, fourier_symbol,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT/"full_model"/"verification"/
          "v66_homogeneous_reaction_slow_state"/"slow_state.npz")
OUT = ROOT/"full_model"/"verification"/"v66_reaction_patterning_barrier_sweep"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = 16
    spacing = 3.0e-7
    with np.load(SOURCE) as data:
        state = CommonWallState(**{
            item.name: np.broadcast_to(
                data[item.name], (n, n)+data[item.name].shape[2:]).copy()
            for item in fields(CommonWallState)})
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    base, gb2 = candidate_parameters(spacing)
    # This strain places active families just above their unchanged Taylor
    # threshold on the reaction-relaxed state. It is selected from the
    # analytical flow condition, not from a nonlinear PF outcome.
    engineering_shear = 5.35e-3
    driving = CommonWallDriving(mean_strain=np.array(
        [[0.0, .5*engineering_shear], [.5*engineering_shear, 0.0]]))
    offsets = (0.0, .25, .50, .75, 1.0)
    conditions = {}
    barrier_names = (
        "lock_barrier_eV", "unlock_barrier_eV", "wall_barrier_eV",
        "annihilation_barrier_eV", "junction_barrier_eV", "order_barrier_eV")
    for offset in offsets:
        parameters = replace(base, **{
            name: getattr(base, name)+offset for name in barrier_names})
        records = []
        for mode in range(1, n//2):
            symbol = fourier_symbol(
                state, driving, systems, topologies, parameters, (mode, 0),
                relative_step=1e-6)
            eigenvalues = symbol["eigenvalues_s_inv"]
            leading = int(np.argmax(eigenvalues.real))
            vector = symbol["right_eigenvectors"][:, leading]
            dominant = int(np.argmax(np.abs(vector)))
            records.append({
                "mode_index": mode,
                "wavelength_m": symbol["wavelength_m"],
                "leading_real_rate_s": float(eigenvalues[leading].real),
                "leading_imaginary_rate_s": float(eigenvalues[leading].imag),
                "dominant_coordinate": str(symbol["layout"][dominant][:2]),
                "dominant_weight": float(abs(vector[dominant])),
            })
        conditions[str(offset)] = records

    maxima = {
        offset: max(records, key=lambda row: row["leading_real_rate_s"])
        for offset, records in conditions.items()}
    finite = {
        offset: bool(row["leading_real_rate_s"] > 0.0
                     and row["mode_index"] not in (1, n//2-1)
                     and "mobile" in row["dominant_coordinate"])
        for offset, row in maxima.items()}
    qualifying = [float(offset) for offset, passed in finite.items() if passed]
    selected = min(qualifying) if qualifying else None

    figure = OUT/"barrier_dispersion.png"
    fig, ax = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    for offset, records in conditions.items():
        ax.plot([1e-6/row["wavelength_m"] for row in records],
                [row["leading_real_rate_s"] for row in records], marker="o",
                label=f"+{float(offset):g} eV")
    ax.axhline(0.0, color="black", linewidth=.7)
    ax.set(xlabel=r"wave number (cycles $\mu$m$^{-1}$)",
           ylabel=r"full-operator leading rate (s$^{-1}$)")
    ax.legend(frameon=False)
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    result = {
        "schema": "v66-reaction-patterning-barrier-sweep-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha(SOURCE),
        "grid": [n, n], "spacing_m": spacing,
        "fixed_engineering_shear": engineering_shear,
        "fixed_parameters": {"A": 1.25, "D": .5,
                             "reduced_modulus_b2_J_m": gb2,
                             "glide_barrier_eV": base.glide_barrier_eV,
                             "taylor_alpha": base.taylor_alpha,
                             "exp_floor": base.exp_floor},
        "barrier_offset_eV": list(offsets),
        "conditions": conditions,
        "maximum_by_offset": maxima,
        "finite_interior_mobile_mode": finite,
        "decision": {
            "minimum_qualifying_uniform_offset_eV": selected,
            "baseline_concurrent_patterning_supported": finite["0.0"],
            "candidate_concurrent_patterning_supported": bool(
                selected is not None),
            "selection_rule": (
                "minimum predeclared uniform barrier offset whose full "
                "reaction-plus-transport tangent has a positive interior "
                "mobile-dominated mode"),
        },
        "claim_boundary": (
            "generic 0--1 eV uncertainty sweep; no material calibration and "
            "no nonlinear wall or RX claim"),
        "artifacts": {str(figure.relative_to(ROOT)): _sha(figure)},
    }
    path = OUT/"result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"maximum_by_offset": maxima,
                      "finite_interior_mobile_mode": finite,
                      "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
