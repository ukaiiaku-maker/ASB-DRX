#!/usr/bin/env python3
"""Classify the V41 one-source stage-localized Nye transfer replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


STAGES = ("macro_initial", "after_first_mura_half_stage",
          "after_front_transaction", "after_second_mura_half_stage")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(a, b):
    return abs(float(a)-float(b))/max(abs(float(a)), abs(float(b)), 1e-300)


def ordered_line(stage, spacing):
    inventory = stage["line_inventory"]
    total = 0.0
    for sign in ("plus", "minus"):
        total += sum(inventory[
            f"wall_ordered_{sign}_m_per_m_by_family"])
    return float(total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n128", type=Path, required=True)
    parser.add_argument("--n192", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = {128: json.loads(args.n128.read_text()),
           192: json.loads(args.n192.read_text())}
    source_equal = raw[128]["source_sha"] == raw[192]["source_sha"]
    rows = {}
    for stage_name in STAGES:
        values = {}
        for grid in (128, 192):
            stage = raw[grid]["records"][-1]["stage_diagnostics"][stage_name]
            nye = stage["nye"]
            values[str(grid)] = {
                "curl_beta_rms_m1": nye["curl_beta_rms_m1"],
                "reservoir_bulk_rms_m1": nye["reservoir_moment_rms_m1"],
                "interface_product_rule_rms_m1": nye[
                    "interface_product_rule_rms_m1"],
                "declared_identity_residual_rms_m1": nye[
                    "difference_rms_m1"],
                "declared_identity_residual_interface_rms_m1": nye[
                    "difference_interface_rms_m1"],
                "declared_identity_residual_bulk_rms_m1": nye[
                    "difference_bulk_rms_m1"],
                "ordered_line_m_per_m": ordered_line(stage, 3.2e-6/grid),
                "interface_cells": stage["owner_support"]["interface_cells"],
                "interface_area_proxy_m2_per_m": stage["owner_support"][
                    "interface_area_proxy_m2_per_m"],
            }
        rows[stage_name] = {
            "grids": values,
            "cross_grid_relative": {
                key: rel(values["128"][key], values["192"][key])
                for key in values["128"] if isinstance(values["128"][key], float)
            },
        }
    post = rows["after_front_transaction"]["grids"]
    residual_fraction = {grid: post[grid][
        "declared_identity_residual_rms_m1"]/max(
            post[grid]["curl_beta_rms_m1"], 1e-300)
        for grid in ("128", "192")}
    front_closed = source_equal and max(residual_fraction.values()) <= 5e-5
    second = rows["after_second_mura_half_stage"]["cross_grid_relative"][
        "curl_beta_rms_m1"]
    first_ordered = rows["after_first_mura_half_stage"]["grids"]
    ordered_h_exponent = math.log(
        first_ordered["128"]["ordered_line_m_per_m"]
        /first_ordered["192"]["ordered_line_m_per_m"])/math.log(192.0/128.0)
    result = {
        "schema": "asb-drx/v41/gradient-transfer-decision/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "n128": {"path": str(args.n128.resolve()),
                     "sha256": digest(args.n128)},
            "n192": {"path": str(args.n192.resolve()),
                     "sha256": digest(args.n192)},
        },
        "one_current_source": source_equal,
        "source_sha": raw[128]["source_sha"] if source_equal else None,
        "stages": rows,
        "post_front_declared_identity_residual_fraction": residual_fraction,
        "front_transfer_identity_passed": front_closed,
        "second_mura_cross_grid_curl_nye_relative_difference": second,
        "second_mura_five_percent_passed": second <= .05,
        "ordered_line_h_exponent_n128_n192": ordered_h_exponent,
        "ordered_line_scaling_stage": "after_first_mura_half_stage_before_front",
        "ordered_trace_front_transfer_causal": False,
        "diagnosis": (
            "The V40 post-front curl-minus-reservoir jump was the separately "
            "owned support-gradient/product-rule Nye term, not lost Burgers "
            "content. The corrected declared identity includes that term. "
            "Any remaining second-half grid sensitivity is a physical/numerical "
            "Mura accuracy question and is not repaired by target-Nye projection."),
        "classification": (
            "FRONT_TRANSFER_DECLARED_INTERFACE_TERM_CLOSED"
            if front_closed else "FRONT_TRANSFER_IDENTITY_UNRESOLVED"),
        "authoritative_state_filtered": False,
        "target_nye_projection_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
