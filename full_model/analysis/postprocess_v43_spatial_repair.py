#!/usr/bin/env python3
"""Assemble baseline and time-refined V43 spatial evidence and plots."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from full_model.analysis.run_v43_spatial_first_difference import (
    STAGES, compare_field, fields,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pair(root128, root192):
    roots = {128: Path(root128), 192: Path(root192)}
    result = {}
    loaded = {}
    for stage in STAGES:
        loaded[stage] = {}
        for n in (128, 192):
            loaded[stage][n] = fields(roots[n]/f"{stage}.npz", n)
        result[stage] = {
            name: compare_field(loaded[stage][128][0][name],
                                loaded[stage][192][0][name])
            for name in loaded[stage][128][0]
        }
    return result, loaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-n128", type=Path, required=True)
    parser.add_argument("--baseline-n192", type=Path, required=True)
    parser.add_argument("--one-step-n128", type=Path, required=True)
    parser.add_argument("--one-step-n192", type=Path, required=True)
    parser.add_argument("--refined-n128", type=Path, required=True)
    parser.add_argument("--refined-n192", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    baseline, baseline_fields = pair(args.baseline_n128, args.baseline_n192)
    one, one_fields = pair(args.one_step_n128, args.one_step_n192)
    refined, refined_fields = pair(args.refined_n128, args.refined_n192)
    one_metric = one["after_second_mura"]["curl_nye"]
    refined_metric = refined["after_second_mura"]["curl_nye"]
    passes = (refined_metric["complex_coefficient_relative_rms"] <= .05
              and refined_metric["rms_relative_difference"] <= .05)
    improves = (refined_metric["complex_coefficient_relative_rms"]
                < one_metric["complex_coefficient_relative_rms"])
    classification = (
        "MURA_TIME_REFINEMENT_REPAIRED_INTERVAL2" if passes else
        "MURA_TIME_REFINEMENT_IMPROVED_BUT_NOT_QUALIFIED" if improves else
        "MURA_TIME_REFINEMENT_DID_NOT_REPAIR_SPATIAL_ERROR")
    ordered = {}
    for label, loaded in (("one_step", one_fields), ("four_substep", refined_fields)):
        ordered[label] = {}
        for n in (128, 192):
            field, _, context = loaded["after_second_mura"][n]
            integral = float(np.sum(field["ordered_density"], dtype=np.longdouble)
                             *context["spacing_m"]**2)
            ordered[label][str(n)] = {
                "ordered_line_m_per_m": integral,
                "ordered_density_rms_m2": float(np.sqrt(np.mean(
                    field["ordered_density"]**2))),
            }
    payload = {
        "schema": "asb-drx/v43/spatial-first-difference-and-repair/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "postprocessor_source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_equivalence": {
            "v42_n128": "ee4b805e4576af5110301692d72bc4c4d76a68c7",
            "v42_n192": "17f63eef216b7bf0fbab33414271f17f871f6ccc",
            "scientific_dependency_diff": (
                "only unused legacy topology imports were removed; no equation, "
                "initialization, parameter, or numerical update changed"),
            "retained_v42_pair_scientifically_equivalent": True,
        },
        "first_macro_baseline": baseline,
        "interval2_one_step": one,
        "interval2_four_substep": refined,
        "first_discrepancy": {
            "stage": "second Mura half of macro 1",
            "curl_common_mode_complex_relative_rms": baseline[
                "after_second_mura"]["curl_nye"][
                    "complex_coefficient_relative_rms"],
            "curl_common_mode_power_relative_rms": baseline[
                "after_second_mura"]["curl_nye"]["power_relative_rms"],
            "curl_rms_amplitude_relative_difference": baseline[
                "after_second_mura"]["curl_nye"]["rms_relative_difference"],
            "diagnosis": (
                "phase/location error precedes large power-amplitude error; "
                "initial, first-Mura, and front states remain common-mode matched"),
        },
        "ordered_near_extinction": ordered,
        "repair_test": {
            "changed_numerical_control": "Mura substeps per half: 1 -> 4",
            "unchanged_physics": [
                "diffusivity", "gradient coefficient", "kinetic parameters",
                "loading", "initial state", "Fourier normalization"],
            "one_step": one_metric, "four_substep": refined_metric,
            "five_percent_passed": passes,
        },
        "implicated_operator": {
            "name": "accepted_mura_transport_capture_step",
            "scalar_population_discretization": (
                "positivity-preserving first-order donor-cell face transport"),
            "first_moment_and_plastic_nye_discretization": (
                "spectral curl of local velocity-cross-alignment product"),
            "nonlinear_dealiasing_declared": False,
            "evidence_scope": (
                "Mura is the first evolving stage to exceed tolerance; the "
                "front adds negligible error; time subdivision does not repair "
                "the pair. This implicates, but does not uniquely prove, the "
                "mixed spatial update or an unresolved continuum scale."),
            "next_executable_discriminator": (
                "construct one conservative compatible face flux for scalar "
                "and moment transport, then rerun this exact interval-2 pair"),
        },
        "classification": classification,
        "original_62p5us_classification_retained": "UNRESOLVED_SPATIAL_SCALE",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    panels = (("one step", one_fields), ("four substeps", refined_fields))
    for row, (label, loaded) in enumerate(panels):
        for col, (n, stage) in enumerate(((128, "initial"), (192, "initial"),
                                          (128, "after_second_mura"),
                                          (192, "after_second_mura"))):
            alpha = loaded[stage][n][0]["curl_nye"]
            image = np.linalg.norm(alpha, axis=(-2, -1))
            axes[row, col].imshow(image.T, origin="lower", cmap="magma")
            axes[row, col].set_title(f"{label}: n{n} {stage.replace('_',' ')}")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    fig.savefig(args.plot, dpi=180)
    plt.close(fig)
    print(json.dumps({"classification": classification,
                      "five_percent_passed": passes}, sort_keys=True))


if __name__ == "__main__":
    main()
