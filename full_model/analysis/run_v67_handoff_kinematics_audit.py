#!/usr/bin/env python3
"""Read-only causal audit of plastic-history ownership during RX sweep."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from full_model.analysis.audit_v63_drx_event_energy import energy_delta
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.complete_front_energy import CompleteFrontEnergy
from full_model.production.complete_multigrain_energy import (
    MultiGrainDissipation, evaluate_joint_multigrain_transaction,
)
from full_model.production.multigrain_common_state import (
    PhysicalTransferLaw, derive_physical_transfer_proposal,
    joint_material_transaction,
)
from full_model.production.multigrain_production import (
    MultiGrainFrontKinetics, _geometric_sweep_weight,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, plastic_distortion_from_slip,
    rotated_system_fields,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V66 = Path(
    "/Users/sdillon/HPC3/worktrees/asb-drx-full-v66-20261013/"
    "full_model/verification")
DEFAULT_OUT = ROOT / "full_model" / "verification" / "v67_handoff_kinematics"
ENERGY_FIELDS = tuple(item.name for item in fields(CompleteFrontEnergy))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _physics(n: int, length: float, temperature: float):
    spacing = length / n
    thickness = 2.0 * 2.48e-10
    systems = bcc_four_family_systems()
    wall = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=2,
        mobile_correlation_diffusivity_m2_s=0.0,
        transport_scheme="upwind", maximum_fraction_per_step=.10,
        volumetric_heat_capacity_J_m3_K=3.8e6,
        thermal_diffusivity_m2_s=.15 / 3.8e6, bath_rate_s=0.0,
        bath_temperature_K=temperature)
    kinetics = MultiGrainFrontKinetics(
        ActivatedProcess("V67 handoff audit", 1e10, 0.0, 1e10),
        PhysicalTransferLaw(.55, .08, .02), .35 * EV_J, 1e9,
        2.0, 1.5, .10, wall.burgers_m**3, wall.burgers_m,
        virtual_fraction=2e-4, maximum_fraction_per_step=.01,
        closure_fraction=.05, maximum_backtracks=14,
        temperature_resolution="local_interface")
    energy = {
        "topologies": (), "systems": systems,
        "phase_barrier_J_m3": 5e6, "phase_gradient_J_m": 5e-7,
        "boundary_line_energy_J_m": wall.line_energy_J_m,
        "boundary_junction_energy_J_m": wall.junction_energy_J_m,
        "reference_temperature_K": temperature,
    }
    return spacing, thickness, systems, wall, kinetics, energy


def _receiver_basis_projection(donor_beta, product, systems):
    """Project donor beta onto the receiver's four declared slip dyads."""
    _, directions, normals = rotated_system_fields(
        systems, product.orientation_rad)
    basis = np.einsum("...ai,...aj->...aij", directions, normals)
    matrix = np.moveaxis(basis.reshape(basis.shape[:3] + (9,)), 2, -1)
    target = np.asarray(donor_beta).reshape(donor_beta.shape[:2] + (9,))
    slip = np.einsum("...ai,...i->...a", np.linalg.pinv(matrix), target)
    projected = plastic_distortion_from_slip(
        slip, systems, product.orientation_rad)
    residual = np.linalg.norm(projected - donor_beta) / max(
        np.linalg.norm(donor_beta), 1e-300)
    return replace(product, slip=slip, beta_p=projected), float(residual)


def _evaluate(state, capacity, *, spacing, thickness, wall, energy, dt):
    result = evaluate_joint_multigrain_transaction(
        state, capacity, spacing_m=spacing,
        represented_thickness_m=thickness, wall_parameters=wall,
        interval_s=dt, dissipation=MultiGrainDissipation(),
        energy_kwargs=energy, bind_state_identity=False)
    decision = result.decision
    extent = next(iter(capacity.accepted_fraction_by_interface.values()))
    swept_volume = float(np.sum(extent, dtype=np.longdouble)
                         * spacing**2 * thickness)
    return {
        "swept_volume_m3": swept_volume,
        "cold_delta_helmholtz_J": decision.delta_helmholtz_J,
        "directional_energy_density_Pa": (
            decision.delta_helmholtz_J / max(swept_volume, 1e-300)),
        "energy_component_change_J": energy_delta(
            decision.before, decision.candidate),
    }


def audit_checkpoint(path: Path, *, fraction: float,
                     target_receiver_id: int | None = 20) -> dict:
    state, runtime, step, gamma, _, configuration, provenance = _load_checkpoint(
        path)
    n = int(configuration.get("n", state.supports.shape[1]))
    length = float(configuration.get("length_m", 4.8e-6))
    temperature = float(configuration.get(
        "temperature_K", configuration.get("temperature_parameter_K", 1000.0)))
    spacing, thickness, systems, wall, kinetics, energy0 = _physics(
        n, length, temperature)
    history_mode = configuration.get("plastic_history_mode", "receiver_dormant")
    kinetics = replace(
        kinetics,
        transfer_law=replace(
            kinetics.transfer_law, plastic_history_mode=history_mode))
    energy = dict(
        energy0, mean_strain=np.array([[0.0, .5 * gamma],
                                      [.5 * gamma, 0.0]]))
    index = {grain_id: position
             for position, grain_id in enumerate(state.grain_ids)}
    rows = []
    for interface in runtime.interfaces:
        for donor_id, receiver_id, direction in (
                (interface.grain_a_id, interface.grain_b_id, "a_to_b"),
                (interface.grain_b_id, interface.grain_a_id, "b_to_a")):
            if (target_receiver_id is not None
                    and receiver_id != target_receiver_id):
                continue
            donor = state.owners[index[donor_id]]
            weight = _geometric_sweep_weight(
                state, index[donor_id], index[receiver_id], spacing)
            proposal = derive_physical_transfer_proposal(
                state, interface_id=interface.component_id,
                donor_id=donor_id, receiver_id=receiver_id,
                requested_fraction=fraction * weight,
                law=kinetics.transfer_law, interval_s=float(
                    configuration.get("dt_s", configuration.get(
                        "hold_dt_s", 2e-9))), systems=systems)
            configured_variant = f"authoritative_{history_mode}"
            variants = {
                configured_variant: (
                    proposal.product_owner, None),
                "exact_donor_beta_energy_control": (
                    replace(proposal.product_owner,
                            beta_p=np.asarray(donor.beta_p).copy()), None),
            }
            projected, residual = _receiver_basis_projection(
                np.asarray(donor.beta_p), proposal.product_owner, systems)
            variants["receiver_slip_basis_projection"] = (projected, residual)
            evaluated = {}
            for name, (product, projection_residual) in variants.items():
                candidate_proposal = replace(proposal, product_owner=product)
                capacity = joint_material_transaction(
                    state, (candidate_proposal,))
                record = _evaluate(
                    state, capacity, spacing=spacing, thickness=thickness,
                    wall=wall, energy=energy,
                    dt=float(configuration.get(
                        "dt_s", configuration.get("hold_dt_s", 2e-9))))
                record["donor_beta_projection_relative_residual"] = (
                    projection_residual)
                evaluated[name] = record
            rows.append({
                "interface": interface.component_id,
                "direction": direction,
                "donor_id": donor_id,
                "receiver_id": receiver_id,
                "variants": evaluated,
            })
    return {
        "checkpoint": str(path.resolve()),
        "checkpoint_sha256": _sha(path),
        "parent_source_commit": (provenance or {}).get("source_commit"),
        "step": step, "applied_shear_strain": gamma,
        "configuration_schema": configuration.get("schema"),
        "configured_plastic_history_mode": history_mode,
        "target_growth_directions": rows,
    }


def run(checkpoints: tuple[Path, ...], output: Path, *, fraction: float) -> dict:
    cases = {path.parent.name: audit_checkpoint(path, fraction=fraction)
             for path in checkpoints}
    result = {
        "schema": "v67-plastic-history-handoff-causal-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_source_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "virtual_fraction": fraction,
        "cases": cases,
        "control_semantics": {
            "authoritative_receiver_dormant_history": (
                "unchanged production transfer; newly swept support receives "
                "the receiver owner's dormant beta_p at that point"),
            "exact_donor_beta_energy_control": (
                "read-only upper compatibility control; preserves donor beta_p "
                "without asserting a receiver-slip reaction path"),
            "receiver_slip_basis_projection": (
                "read-only accessible-space test; least-squares projects donor "
                "beta_p onto the four receiver-oriented slip dyads"),
        },
        "claim_boundary": (
            "These are immutable same-state energy prices. A favorable control "
            "identifies a missing kinematic/accommodation path but cannot be "
            "published as a physical production event until slip, Nye, work, "
            "heat, and structural compatibility close together."),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v66-root", type=Path, default=DEFAULT_V66)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fraction", type=float, default=2e-4)
    args = parser.parse_args()
    names = (
        "v66_coupled_rx_asb", "v66_regime_T1000_r1e4",
        "v66_high_rate_unload_static", "v66_zero_mean_stress")
    checkpoints = tuple(
        args.v66_root / name / "checkpoint_latest.npz" for name in names)
    result = run(checkpoints, args.out, fraction=args.fraction)
    print(json.dumps({
        "sha256": _sha(args.out / "result.json"),
        "cases": list(result["cases"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
