#!/usr/bin/env python3
"""Fast-wall-order quasi-equilibrium discriminator on the valid H3 prefix."""

from __future__ import annotations

from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from full_model.analysis.run_v66_production_groma_tangent import candidate_parameters
from full_model.production.common_tensorial_wall import (
    CommonWallState, _divergence, _spectral_gradient,
    wall_free_energy_derivatives, wall_total_free_energy_density_J_m3,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, make_junction_topology,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT/"full_model"/"verification"/
          "v66_concurrent_pattern_absorption_frozen_order"/"final_state.npz")
OUT = ROOT/"full_model"/"verification"/"v66_wall_order_quasiequilibrium"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE) as data:
        state = CommonWallState(**{
            item.name: np.asarray(data[item.name]).copy()
            for item in fields(CommonWallState)})
    systems = bcc_four_family_systems()
    topologies = (make_junction_topology(
        systems, 0, 1, 1, -1, line_tension_J_m=0.0),)
    parameters, _ = candidate_parameters(2.0e-7)
    parameters = replace(
        parameters, polarized_wall_absorption_enabled=True,
        wall_absorption_polarization_power=1.0,
        wall_order_enabled=True)
    shape = state.wall_order.shape
    scale = parameters.wall_order_barrier_J_m3
    reference = float(np.mean(wall_total_free_energy_density_J_m3(
        state, parameters, topologies, systems)))

    def evaluate(flat):
        candidate = replace(state, wall_order=np.asarray(flat).reshape(shape))
        density = wall_total_free_energy_density_J_m3(
            candidate, parameters, topologies, systems)
        chemical = wall_free_energy_derivatives(
            candidate, parameters, topologies, systems)
        qx, qy = _spectral_gradient(
            candidate.wall_order, parameters.spacing_m)
        lap = _divergence(np.stack((qx, qy), axis=-1), parameters.spacing_m)
        gradient = (chemical["wall_order_derivative_J_m3"]
                    -parameters.wall_order_gradient_J_m*lap)
        return ((float(np.mean(density))-reference)/scale,
                np.asarray(gradient).ravel()/(scale*np.prod(shape)))

    starts = (0.0, 0.01, 0.1, 0.5, 0.9)
    candidates = []
    for start in starts:
        result = minimize(
            evaluate, np.full(shape, start).ravel(), method="L-BFGS-B",
            jac=True, bounds=[(0.0, 1.0)]*int(np.prod(shape)),
            options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000,
                     "maxls": 40})
        q = result.x.reshape(shape)
        energy = float(np.mean(wall_total_free_energy_density_J_m3(
            replace(state, wall_order=q), parameters, topologies, systems)))
        candidates.append({
            "start": start, "success": bool(result.success),
            "status": int(result.status), "message": str(result.message),
            "iterations": int(result.nit), "function_evaluations": int(result.nfev),
            "energy_J_m3": energy, "energy_change_J_m3": energy-reference,
            "q_mean": float(np.mean(q)), "q_min": float(np.min(q)),
            "q_max": float(np.max(q)), "q_std": float(np.std(q)),
            "projected_gradient_max_scaled": float(np.max(np.abs(result.jac))),
            "q": q,
        })
    selected = min(candidates, key=lambda item: item["energy_J_m3"])
    selected_q = selected.pop("q")
    for item in candidates:
        if "q" in item:
            item.pop("q")
    energy_spread = max(item["energy_J_m3"] for item in candidates)-min(
        item["energy_J_m3"] for item in candidates)
    ordered = replace(state, wall_order=selected_q)
    checkpoint = OUT/"ordered_state.npz"
    np.savez_compressed(checkpoint, **{
        item.name: getattr(ordered, item.name) for item in fields(ordered)})
    figure = OUT/"wall_order_equilibrium.png"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)
    im = axes[0].imshow(selected_q.T, origin="lower", cmap="magma")
    fig.colorbar(im, ax=axes[0], label="wall order q")
    wall_signed = np.sqrt(np.sum(
        (state.wall_plus_m2-state.wall_minus_m2)**2, axis=2))
    im = axes[1].imshow(wall_signed.T, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=axes[1], label=r"signed wall norm (m$^{-2}$)")
    for ax in axes:
        ax.set(xlabel="x index", ylabel="y index")
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    result = {
        "schema": "v66-wall-order-quasiequilibrium-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(SOURCE.relative_to(ROOT)),
        "source_sha256": _sha(SOURCE),
        "method": "bounded L-BFGS-B minimization of authoritative wall free energy",
        "starts": list(starts), "candidates": candidates,
        "selected": {key: value for key, value in selected.items() if key != "q"},
        "energy_spread_across_starts_J_m3": energy_spread,
        "generated_heat_if_instantaneously_relaxed_J_m3": max(
            reference-selected["energy_J_m3"], 0.0),
        "decision": {
            "all_optimizers_converged": all(item["success"] for item in candidates),
            "unique_energy_to_1e_6_J_m3": bool(energy_spread < 1e-6),
            "downhill_ordered_state": bool(selected["energy_J_m3"] < reference),
            "production_multirate_promotion_allowed": False,
        },
        "claim_boundary": (
            "fixed-reservoir asymptotic discriminator; no kinetic pathway, "
            "physical wall, grain allocation, or RX qualification"),
        "artifacts": {str(figure.relative_to(ROOT)): _sha(figure),
                      str(checkpoint.relative_to(ROOT)): _sha(checkpoint)},
    }
    (OUT/"result.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key: result[key] for key in (
        "candidates", "energy_spread_across_starts_J_m3",
        "generated_heat_if_instantaneously_relaxed_J_m3", "decision")}, indent=2))


if __name__ == "__main__":
    main()
