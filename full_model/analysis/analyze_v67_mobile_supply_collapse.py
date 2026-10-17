#!/usr/bin/env python3
"""Diagnose loss of polycrystal flow exposure without retuning resistance."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.analysis.run_v66_coupled_rx_asb import _physics
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, resolved_driving_components,
)
from full_model.production.multigrain_production import (
    _owner_drivings_from_common_stress,
)


def load_case(root: Path) -> dict:
    result = json.loads((root / "result.json").read_text())
    state, _, step, load, _, config, _ = _load_checkpoint(
        root / "checkpoint_latest.npz")
    args = SimpleNamespace(
        n=config["n"], length=config["length_m"],
        temperature=config["temperature_K"],
        shear_rate=config["shear_rate_s"], dt=config["dt_s"],
        thermal_control=config["thermal_control"],
        bath_rate=config["thermal_bath_rate_s"])
    _, _, systems, wall, _, _, _ = _physics(args)
    wall = replace(
        wall, wall_nye_match_coefficient_J_m=wall.c44_Pa * (1.5e-7) ** 2)
    mean_strain = np.array([[0.0, .5 * load], [.5 * load, 0.0]])
    drivings = _owner_drivings_from_common_stress(
        state, CommonWallDriving(mean_strain=mean_strain), systems, wall)

    extensive = {name: 0.0 for name in ("mobile", "forest", "wall")}
    weighted = {name: 0.0 for name in (
        "raw_abs_Pa", "effective_abs_Pa", "taylor_Pa", "speed_abs_m_s")}
    weight_sum = 0.0
    owners = {}
    for grain, support, owner, driving in zip(
            state.grain_ids, state.supports, state.owners, drivings):
        eta = np.asarray(support)[..., None]
        drive = resolved_driving_components(
            owner, driving, systems, (), wall)
        mobile = owner.mobile_plus_m2 + owner.mobile_minus_m2
        forest = owner.forest_plus_m2 + owner.forest_minus_m2
        wall_line = owner.wall_plus_m2 + owner.wall_minus_m2
        for name, value in (
                ("mobile", mobile), ("forest", forest), ("wall", wall_line)):
            extensive[name] += float(np.sum(eta * value, dtype=np.longdouble))
        local_weight = np.broadcast_to(eta, mobile.shape)
        count = float(np.sum(local_weight))
        values = {
            "raw_abs_Pa": np.abs(drive["raw_stress_Pa"]),
            "effective_abs_Pa": np.abs(drive["effective_stress_Pa"]),
            "taylor_Pa": drive["taylor_resistance_Pa"],
            "speed_abs_m_s": np.abs(drive["speed_m_s"]),
        }
        for name, value in values.items():
            weighted[name] += float(np.sum(
                local_weight * value, dtype=np.longdouble))
        weight_sum += count
        core = np.asarray(support) > .5
        owners[str(grain)] = {
            "core_mobile_density_m2": float(np.mean(np.sum(mobile, axis=2)[core])),
            "core_taylor_resistance_Pa": float(np.mean(
                drive["taylor_resistance_Pa"][core])),
            "core_effective_stress_abs_Pa": float(np.mean(
                np.abs(drive["effective_stress_Pa"])[core])),
            "core_glide_speed_abs_m_s": float(np.mean(
                np.abs(drive["speed_m_s"])[core])),
        }
    ledger = result["ledger"]
    neutral_boundary_pairs = float(sum(
        np.sum(np.minimum(
            np.asarray(component.boundary_plus_m2)[..., 0, :],
            np.asarray(component.boundary_minus_m2)[..., 0, :]),
            dtype=np.longdouble)
        for component in state.interfaces))
    return {
        "directory": str(root.resolve()),
        "source_commit": result["source_commit"],
        "source_tree_clean": result["source_tree_clean"],
        "step": step,
        "load": load,
        "flow_exposure_fraction": result["flow_regime"][
            "final_plastic_to_imposed_power_fraction"],
        "extensive_cell_inventory_m2": extensive,
        "support_family_weighted_means": {
            name: value / weight_sum for name, value in weighted.items()},
        "source_created_line_cell_sum_m2": ledger[
            "source_created_line_cell_sum_m2"],
        "neutral_mobile_channel_interface_pair_cell_sum_m2": (
            neutral_boundary_pairs),
        "neutral_mobile_channel_interface_line_cell_sum_m2": (
            2.0*neutral_boundary_pairs),
        "lattice_absorbed_line_length_m": ledger[
            "lattice_absorbed_line_length_m"],
        "owners": owners,
        "hard_audits_passed": bool(
            result["decision"]["all_hard_absorption_balances_passed"]
            and result["decision"]["mura_hard_balance_passed"]
            and result["decision"]["first_law_hard_balance_passed"]),
    }


def classify(early: dict, late: dict) -> dict:
    mobile_early = early["extensive_cell_inventory_m2"]["mobile"]
    mobile_late = late["extensive_cell_inventory_m2"]["mobile"]
    mobile_loss = max(mobile_early - mobile_late, 0.0)
    source_increment = max(
        late["source_created_line_cell_sum_m2"]
        - early["source_created_line_cell_sum_m2"], 0.0)
    ratios = {
        "mobile_inventory_retention": mobile_late / max(mobile_early, 1e-300),
        "flow_exposure_retention": late["flow_exposure_fraction"] / max(
            early["flow_exposure_fraction"], 1e-300),
        "taylor_resistance_ratio": late["support_family_weighted_means"][
            "taylor_Pa"] / max(early["support_family_weighted_means"][
                "taylor_Pa"], 1e-300),
        "glide_speed_ratio": late["support_family_weighted_means"][
            "speed_abs_m_s"] / max(early["support_family_weighted_means"][
                "speed_abs_m_s"], 1e-300),
        "source_increment_to_mobile_loss": source_increment / max(
            mobile_loss, 1e-300),
        "late_neutral_interface_line_to_mobile_inventory": (
            late["neutral_mobile_channel_interface_line_cell_sum_m2"]
            /max(mobile_late, 1e-300)),
    }
    starvation = bool(
        early["hard_audits_passed"] and late["hard_audits_passed"]
        and ratios["mobile_inventory_retention"] < .25
        and ratios["flow_exposure_retention"] < .01
        and ratios["taylor_resistance_ratio"] < 1.10
        and ratios["glide_speed_ratio"] > .5
        and ratios["source_increment_to_mobile_loss"] < .01)
    return {
        "schema": "v67-mobile-supply-collapse-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": {"early": early, "late": late},
        "ratios": ratios,
        "decision": {
            "mobile_supply_collapse_not_taylor_locking": starvation,
            "classification": (
                "FLOW_ARRESTED_BY_MOBILE_SUPPLY_COLLAPSE"
                if starvation else "FLOW_COLLAPSE_CAUSE_NOT_RESOLVED"),
        },
        "next_discriminator": (
            "Move the identical Hall-Petch source opportunities from GB/TJ "
            "sites to the matched-random geography on the same retained state; "
            "do not reduce Taylor resistance or retune source kinetics."),
        "claim_boundary": (
            "This identifies the proximate carrier-supply loss over the tested "
            "trajectory. It does not establish that source geography is the "
            "remedy until the checksum-matched random-site fork is executed."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("early", type=Path)
    parser.add_argument("late", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    payload = classify(load_case(args.early), load_case(args.late))
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**payload["ratios"], **payload["decision"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
