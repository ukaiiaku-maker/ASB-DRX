#!/usr/bin/env python3
"""Classify the V67 arrival-limited carrier production discriminator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.run_v58_three_grain_production import _load_checkpoint


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _result(directory):
    return json.loads((Path(directory)/"result.json").read_text())


def _checkpoint_metrics(directory):
    path = Path(directory)/"checkpoint_latest.npz"
    state, _, step, load, _, config, provenance = _load_checkpoint(path)
    spacing = float(config["length_m"])/int(config["n"])
    thickness = 2.0*2.48e-10
    mobile = sum(float(np.sum(
        np.asarray(support)[..., None]
        *(np.asarray(owner.mobile_plus_m2)+np.asarray(owner.mobile_minus_m2)),
        dtype=np.longdouble))*spacing**2*thickness
        for support, owner in zip(state.supports, state.owners))
    area = np.sum(np.asarray(state.supports), axis=(1, 2))*spacing**2
    return {
        "path": str(path.resolve()), "sha256": _sha(path),
        "step": int(step), "load": float(load),
        "mobile_line_m": mobile, "owner_area_m2": area.tolist(),
        "source_provenance": provenance,
    }


def _hard(result):
    decision = result["decision"]
    return bool(
        decision["all_hard_absorption_balances_passed"]
        and decision["mura_hard_balance_passed"]
        and decision["first_law_hard_balance_passed"]
        and decision["grain_ids_unchanged"])


def _endpoint(result):
    row = result["response_history"][-1]
    return {
        "load": row["load"],
        "plastic_to_imposed_power_fraction": row[
            "plastic_to_imposed_power_fraction"],
        "plastic_power_mean_W_m3": row["plastic_power_mean_W_m3"],
        "mean_shear_stress_Pa": row["mean_shear_stress_Pa"],
        "temperature_mean_K": row["temperature_mean_K"],
        "temperature_range_K": row["temperature_range_K"],
        "connected_physical_band": row["connected_physical_band"],
        "terminal_band_persistence_intervals": result[
            "response_summary"]["terminal_band_persistence_intervals"],
        "endpoint_post_peak_softening_fraction": result[
            "response_summary"]["endpoint_post_peak_softening_fraction"],
        "hard_valid": _hard(result),
    }


def classify(payload):
    base = payload["high_rate"]["reaction_only_ad600"]["endpoint"]
    candidate = payload["high_rate"]["arrival_limited_ad600"]["endpoint"]
    dynamic = payload["lower_rate_dynamic"]["result"]
    static = payload["retained_static"]["result"]
    hard = all((base["hard_valid"], candidate["hard_valid"],
                dynamic["hard_valid"], static["hard_valid"]))
    preserves_rx = (
        dynamic["grain20_relative_area_change"] > 0.0
        and static["grain20_relative_area_change"] > 0.0)
    prevents_collapse = (
        candidate["plastic_to_imposed_power_fraction"] >= .01
        and base["plastic_to_imposed_power_fraction"] < .01)
    qualifies_asb = (
        candidate["terminal_band_persistence_intervals"] >= 20
        and candidate["endpoint_post_peak_softening_fraction"] >= .05)
    if not hard:
        label = "HARD_INVALID_ARRIVAL_EXCHANGE_CANDIDATE"
    elif prevents_collapse and preserves_rx and not qualifies_asb:
        label = "CARRIER_COLLAPSE_PREVENTED_RX_PRESERVED_ASB_NOT_YET_QUALIFIED"
    elif prevents_collapse and preserves_rx and qualifies_asb:
        label = "ARRIVAL_EXCHANGE_SUPPORTS_COUPLED_DRX_ASB_CANDIDATE"
    else:
        label = "ARRIVAL_EXCHANGE_INSUFFICIENT"
    return {
        "classification": label, "all_hard_audits_passed": hard,
        "prevents_high_rate_flow_collapse": prevents_collapse,
        "preserves_lower_rate_dynamic_and_static_rx": preserves_rx,
        "qualifies_high_rate_asb": qualifies_asb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--baseline-one", type=Path, required=True)
    parser.add_argument("--candidate-one", type=Path, required=True)
    parser.add_argument("--baseline-ad600", type=Path, required=True)
    parser.add_argument("--candidate-ad600", type=Path, required=True)
    parser.add_argument("--lower-dynamic", type=Path, required=True)
    parser.add_argument("--retained-static", type=Path, required=True)
    args = parser.parse_args()
    parent = _result(args.parent)
    base_one = _result(args.baseline_one)
    candidate_one = _result(args.candidate_one)
    baseline = _result(args.baseline_ad600)
    candidate = _result(args.candidate_ad600)
    dynamic = _result(args.lower_dynamic)
    static = _result(args.retained_static)
    parent_absorbed = parent["ledger"]["lattice_absorbed_line_length_m"]

    def one_interval(result):
        return {
            "endpoint": _endpoint(result),
            "absorbed_line_increment_m": (
                result["ledger"]["lattice_absorbed_line_length_m"]
                -parent_absorbed),
            "grain20_relative_area_change": result[
                "owner_results"]["20"]["relative_area_change"],
            "source_commit": result["source_commit"],
            "source_tree_clean": result["source_tree_clean"],
        }

    parent_checkpoint = _checkpoint_metrics(args.parent)
    candidate_checkpoint = _checkpoint_metrics(args.candidate_ad600)
    dynamic_row = {
        **_endpoint(dynamic),
        "grain20_relative_area_change": dynamic[
            "owner_results"]["20"]["relative_area_change"],
        "source_commit": dynamic["source_commit"],
    }
    static_row = {
        **_endpoint(static),
        "grain20_relative_area_change": static[
            "owner_results"]["20"]["relative_area_change"],
        "source_commit": static["source_commit"],
    }
    payload = {
        "schema": "v67-arrival-limited-exchange-candidate-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "glide-supplied lattice capture is limited by signed normal "
            "arrival and shares one serial clock with the unchanged "
            "energy-qualified Arrhenius interface reaction"),
        "immutable_laws": (
            "Peierls-Taylor-Hall-Petch EXP-floor mechanics and Hall-Petch "
            "source event measure unchanged"),
        "one_interval_exact_parent": {
            "reaction_only": one_interval(base_one),
            "arrival_limited": one_interval(candidate_one),
        },
        "high_rate": {
            "parent_ad400": parent_checkpoint,
            "reaction_only_ad600": {
                "endpoint": _endpoint(baseline),
                "checkpoint": _checkpoint_metrics(args.baseline_ad600),
                "absorbed_line_since_ad400_m": (
                    baseline["ledger"]["lattice_absorbed_line_length_m"]
                    -parent_absorbed),
            },
            "arrival_limited_ad600": {
                "endpoint": _endpoint(candidate),
                "checkpoint": candidate_checkpoint,
                "absorbed_line_since_ad400_m": (
                    candidate["ledger"]["lattice_absorbed_line_length_m"]
                    -parent_absorbed),
                "grain20_relative_area_change_latest_segment": candidate[
                    "owner_results"]["20"]["relative_area_change"],
            },
            "ratios": {
                "candidate_to_baseline_flow": (
                    _endpoint(candidate)["plastic_to_imposed_power_fraction"]
                    /max(_endpoint(baseline)[
                        "plastic_to_imposed_power_fraction"], 1e-300)),
                "candidate_to_baseline_mobile_line": (
                    candidate_checkpoint["mobile_line_m"]
                    /max(_checkpoint_metrics(args.baseline_ad600)[
                        "mobile_line_m"], 1e-300)),
            },
        },
        "lower_rate_dynamic": {"result": dynamic_row},
        "retained_static": {"result": static_row},
        "criteria": {
            "physical_flow_floor": .01,
            "asb_softening_floor": .05,
            "asb_terminal_persistence_intervals": 20,
        },
        "claim_boundary": (
            "This exact-parent result tests carrier collapse and preservation "
            "of existing-boundary RX. It does not qualify ASB without the "
            "matched bath branch, persistent localization and post-peak "
            "softening, and it does not establish endogenous grain birth."),
    }
    payload["decision"] = classify(payload)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
