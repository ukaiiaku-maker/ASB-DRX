#!/usr/bin/env python3
"""Physical-clock, stage-consistent ASB audit for V59 multi-grain runs.

The V58 three-record flag is retained only as an exploratory screen.  This
module reconstructs accepted post-front states, follows the same periodic
power component, and applies the conservative conjunction registered for V59.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from full_model.analysis.postprocess_v53_asb_mechanism import (
    field_metrics, overlap, periodic_components,
)
from full_model.analysis.run_v37_conduction_localization import weighted_width
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, CommonWallParameters,
)
from full_model.production.complete_multigrain_energy import _mean_mechanical_stress
from full_model.production.multigrain_common_state import reconstruct_multigrain_common
from full_model.production.multigrain_production import (
    multigrain_instantaneous_dissipation_fields,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


@dataclass(frozen=True)
class PhysicalASBCriteria:
    minimum_persistence_s: float = 1.0e-6
    maximum_power_participation: float = 0.25
    minimum_temperature_peak_minus_mean_K: float = 50.0
    minimum_matched_temperature_excess_K: float = 50.0
    minimum_softening_fraction: float = 0.20
    minimum_width_to_interface: float = 2.0
    maximum_width_to_domain: float = 0.25
    minimum_aspect_ratio: float = 3.0
    minimum_heat_power_overlap: float = 0.25
    minimum_component_identity_overlap: float = 0.25


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_map(directories: list[Path]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for directory in directories:
        for path in sorted(directory.glob("checkpoint_*.npz")):
            step = int(path.stem.rsplit("_", 1)[-1])
            result[step] = path
    return result


def _history_map(directories: list[Path]) -> dict[int, dict]:
    result = {}
    for directory in directories:
        path = directory/"history.json"
        if path.is_file():
            for row in json.loads(path.read_text()):
                result[int(row["step"])] = row
    return result


_TEMPERATURE_INTERVENTION_KEYS = (
    "flow_temperature_mode", "recovery_temperature_mode",
    "front_temperature_mode",
)


def temperature_intervention_certificate(
        baseline_configuration: dict, control_configuration: dict,
        baseline_provenance: dict | None, control_provenance: dict | None,
        *, baseline_step: int, control_step: int,
        baseline_time_s: float, control_time_s: float,
        baseline_gamma: float, control_gamma: float,
        baseline_initial_volume: np.ndarray,
        control_initial_volume: np.ndarray,
        baseline_grain_ids: tuple[int, ...],
        control_grain_ids: tuple[int, ...],
        intervention_scope: str = "all_arrhenius") -> dict:
    """Certify the declared channel-resolved temperature intervention.

    Fresh runs are analytically co-initialized by the deterministic production
    initializer when source, non-intervention configuration, initial volumes,
    and grain identities agree.  Resumed members retain those same bound
    fields and their explicit parent edges in checkpoint provenance.
    """
    baseline = dict(baseline_configuration or {})
    control = dict(control_configuration or {})
    baseline_modes = {
        key: baseline.pop(key, "physical") for key in _TEMPERATURE_INTERVENTION_KEYS}
    control_modes = {
        key: control.pop(key, "physical") for key in _TEMPERATURE_INTERVENTION_KEYS}
    expected_control_modes = {
        "all_arrhenius": {
            "flow_temperature_mode": "frozen",
            "recovery_temperature_mode": "frozen",
            "front_temperature_mode": "frozen",
        },
        "flow_recovery": {
            "flow_temperature_mode": "frozen",
            "recovery_temperature_mode": "frozen",
            "front_temperature_mode": "physical",
        },
    }
    if intervention_scope not in expected_control_modes:
        raise ValueError("unknown temperature-intervention scope")
    checks = {
        "same_source_commit": bool(
            baseline_provenance and control_provenance
            and baseline_provenance.get("source_commit")
            == control_provenance.get("source_commit")),
        "same_nonintervention_configuration": baseline == control,
        "baseline_all_temperature_channels_physical": all(
            value == "physical" for value in baseline_modes.values()),
        "control_matches_declared_temperature_intervention": (
            control_modes == expected_control_modes[intervention_scope]),
        "same_step": int(baseline_step) == int(control_step),
        "same_physical_time": bool(np.isclose(
            baseline_time_s, control_time_s, rtol=0.0, atol=1e-18)),
        "same_loading_coordinate": bool(np.isclose(
            baseline_gamma, control_gamma, rtol=0.0, atol=1e-14)),
        "same_initial_grain_volumes": bool(np.array_equal(
            np.asarray(baseline_initial_volume),
            np.asarray(control_initial_volume))),
        "same_grain_identities": tuple(baseline_grain_ids)
            == tuple(control_grain_ids),
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "intervention_scope": intervention_scope,
        "baseline_modes": baseline_modes, "control_modes": control_modes,
        "semantics": (
            "same deterministic analytic origin and loading; only the "
            f"declared {intervention_scope} Arrhenius temperature routes "
            "differ while the physical heat equation evolves in both"),
    }


def _largest_component(field: np.ndarray) -> np.ndarray:
    value = np.asarray(field, dtype=float)
    threshold = float(value.mean()+value.std())
    components = periodic_components(value > threshold)
    return (max(components, key=np.count_nonzero) if components else
            np.zeros(value.shape, dtype=bool))


def component_morphology(field: np.ndarray, component: np.ndarray,
                         threshold: float, spacing_m: float) -> dict:
    """Measure the tracked hot component without diffuse-background bias.

    The earlier V59 draft evaluated second moments of ``field-field.min()``
    over the whole periodic box.  A narrow intense band on a nonzero plastic-
    power background could therefore acquire a nearly box-sized, isotropic
    width.  Here the weight is the positive excess above the same threshold
    that defines the connected component.  ``weighted_width`` supplies a
    periodic-translation-invariant covariance.
    """
    value = np.asarray(field, dtype=float)
    mask = np.asarray(component, dtype=bool)
    if value.shape != mask.shape or value.ndim != 2:
        raise ValueError("component morphology requires matching 2-D arrays")
    weight = np.where(mask, np.maximum(value-float(threshold), 0.0), 0.0)
    if not np.any(weight > 0.0):
        return {
            "status": "UNDEFINED_NO_POSITIVE_COMPONENT_EXCESS",
            "area_fraction": float(mask.mean()), "second_moment_widths": None,
            "aspect_ratio": None,
        }
    widths = weighted_width(weight, float(spacing_m))
    minor = float(widths["minor_gaussian_fwhm_m"])
    major = float(widths["major_gaussian_fwhm_m"])
    return {
        "status": "DEFINED_THRESHOLD_COMPONENT_EXCESS",
        "weight_semantics": "max(field-(mean+std),0) on largest periodic component",
        "area_fraction": float(mask.mean()),
        "second_moment_widths": widths,
        "aspect_ratio": float(major/max(minor, 1e-300)),
    }


def periodic_identity(left: np.ndarray, right: np.ndarray,
                      spacing_m: float) -> dict:
    """Best periodic Jaccard overlap and the associated physical displacement."""
    a = np.asarray(left, dtype=bool); b = np.asarray(right, dtype=bool)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("periodic components must have the same 2-D shape")
    if not np.any(a) or not np.any(b):
        return {"overlap": 0.0, "shift_cells": [0, 0],
                "displacement_m": [0.0, 0.0]}
    correlation = np.fft.ifftn(
        np.fft.fftn(a.astype(float))*np.conj(np.fft.fftn(b.astype(float)))).real
    index = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    shift = np.asarray(index, dtype=int)
    for axis, count in enumerate(a.shape):
        if shift[axis] > count//2:
            shift[axis] -= count
    aligned = np.roll(b, tuple(shift), axis=(0, 1))
    return {
        "overlap": overlap(a, aligned),
        "shift_cells": shift.tolist(),
        "displacement_m": (shift.astype(float)*spacing_m).tolist(),
    }


def _wall_parameters(configuration: dict, spacing_m: float) -> CommonWallParameters:
    return CommonWallParameters(
        spacing_m=spacing_m, elastic_iterations=2,
        mobile_correlation_diffusivity_m2_s=0.0,
        transport_scheme="upwind",
        maximum_fraction_per_step=float(configuration["maximum_fraction_per_step"]),
        flow_temperature_override_K=(
            float(configuration["temperature_K"])
            if configuration.get("flow_temperature_mode") == "frozen" else None),
        recovery_temperature_override_K=(
            float(configuration["temperature_K"])
            if configuration.get("recovery_temperature_mode") == "frozen"
            else None),
        volumetric_heat_capacity_J_m3_K=3.8e6,
        thermal_diffusivity_m2_s=float(configuration["thermal_diffusivity_m2_s"]),
        bath_rate_s=0.0)


def checkpoint_snapshot(path: Path, pre_front_row: dict | None = None) -> tuple[dict, np.ndarray]:
    state, runtime, step, gamma, _, configuration, provenance = _load_checkpoint(path)
    if configuration is None:
        raise ValueError(f"checkpoint lacks bound configuration: {path}")
    spacing = float(configuration["length_m"])/int(configuration["n"])
    wall = _wall_parameters(configuration, spacing)
    strain = np.array([[0.0, .5*gamma], [.5*gamma, 0.0]])
    dissipation = multigrain_instantaneous_dissipation_fields(
        state, driving=CommonWallDriving(mean_strain=strain),
        systems=bcc_four_family_systems(), topologies=(), wall_parameters=wall)
    power_field = np.asarray(dissipation["plastic_power_W_m3"], dtype=float)
    heat_field = np.asarray(dissipation["irreversible_heat_rate_W_m3"], dtype=float)
    power, power_component = field_metrics(power_field, spacing)
    heat, heat_component = field_metrics(heat_field, spacing)
    morphology = component_morphology(
        power_field, power_component, power["threshold"], spacing)
    common, _ = reconstruct_multigrain_common(state, spacing)
    temperature = np.asarray(common.temperature_K, dtype=float)
    accumulated_slip = np.sum(np.abs(np.asarray(common.slip)), axis=2)
    slip_metrics, slip_component = field_metrics(accumulated_slip, spacing)
    nye_magnitude = np.sqrt(np.sum(
        np.asarray(common.family_nye_m1, dtype=float)**2,
        axis=(2, 3, 4)))
    nye_metrics, nye_component = field_metrics(nye_magnitude, spacing)
    total_line_density = sum(
        np.sum(np.asarray(getattr(common, name), dtype=float), axis=2)
        for name in ("mobile_plus_m2", "mobile_minus_m2",
                     "forest_plus_m2", "forest_minus_m2",
                     "wall_plus_m2", "wall_minus_m2"))
    density_metrics, density_component = field_metrics(
        total_line_density, spacing)
    widths = morphology["second_moment_widths"]
    post_front_stress = float(_mean_mechanical_stress(
        state, spacing, wall, strain)[0, 1])
    row = {
        "step": step, "physical_time_s": runtime.ledger.physical_time_s,
        "applied_shear_strain": gamma,
        "checkpoint": str(path.resolve()), "checkpoint_sha256": _digest(path),
        "source_commit": (provenance or {}).get("source_commit"),
        "spacing_m": spacing,
        "interface_width_m": float(configuration["interface_width_m"]),
        "domain_length_m": float(configuration["length_m"]),
        "pre_front_interval_stress_Pa": (
            None if pre_front_row is None else float(pre_front_row["shear_stress_Pa"])),
        "post_front_equilibrated_stress_Pa": post_front_stress,
        "stress_stage_semantics": (
            "pre-front value is the accepted mechanical interval endpoint; "
            "post-front value is reconstructed without advancing physics"),
        "temperature_mean_K": float(temperature.mean()),
        "temperature_max_minus_mean_K": float(temperature.max()-temperature.mean()),
        "temperature_max_minus_min_K": float(temperature.max()-temperature.min()),
        "accumulated_absolute_slip": slip_metrics,
        "accumulated_slip_power_component_overlap": overlap(
            slip_component, power_component),
        "nye_magnitude_m1": nye_metrics,
        "nye_power_component_overlap": overlap(nye_component, power_component),
        "total_line_density_m2": density_metrics,
        "density_power_component_overlap": overlap(
            density_component, power_component),
        "orientation_range_rad": float(
            np.max(common.orientation_rad)-np.min(common.orientation_rad)),
        "plastic_power": power,
        "plastic_power_component_morphology": morphology,
        "irreversible_heat_rate": heat,
        "heat_power_component_overlap": overlap(power_component, heat_component),
        "power_width_minor_m": (None if widths is None else
                                 float(widths["minor_gaussian_fwhm_m"])),
        "power_width_major_m": (None if widths is None else
                                 float(widths["major_gaussian_fwhm_m"])),
        "power_aspect_ratio": morphology["aspect_ratio"],
        "whole_field_power_width_diagnostic": power["second_moment_widths"],
        "strict_width_semantics": (
            "positive excess above mean+std on the largest periodic component; "
            "whole-field shifted width is retained as a nonqualifying diagnostic"),
        "instantaneous_rate_semantics": "reconstructed from exact accepted post-front state",
        "interval_average_semantics": (
            "saved runner localization fields summarize the preceding accepted mechanical interval"),
    }
    return row, power_component


def classify_physical_episode(rows: list[dict], components: dict[int, np.ndarray],
                              criteria: PhysicalASBCriteria,
                              *, matched_control_available: bool,
                              refinement_passed: bool) -> dict:
    if not rows:
        raise ValueError("physical ASB history is empty")
    peak = -math.inf; peak_time = None; episode = None; episodes = []
    previous_component = None; previous_time = None
    nominal_dt = min(np.diff([row["physical_time_s"] for row in rows]), default=math.inf)
    for row_index, row in enumerate(rows):
        stress = abs(float(row["post_front_equilibrated_stress_Pa"]))
        if stress > peak:
            peak = stress; peak_time = row["physical_time_s"]
        softening = 0.0 if peak <= 0.0 else (peak-stress)/peak
        width = row["power_width_minor_m"]
        identity = (None if previous_component is None else periodic_identity(
            previous_component, components[row["step"]], row["spacing_m"]))
        matched = row.get("matched_temperature_excess_K")
        checks = {
            "power_participation": row["plastic_power"][
                "inverse_participation_fraction"] <= criteria.maximum_power_participation,
            "temperature_peak_minus_mean": row[
                "temperature_max_minus_mean_K"] >= criteria.minimum_temperature_peak_minus_mean_K,
            "matched_temperature_excess": bool(
                matched_control_available and matched is not None
                and matched >= criteria.minimum_matched_temperature_excess_K),
            "post_peak_softening": bool(
                peak_time is not None and row["physical_time_s"] > peak_time
                and softening >= criteria.minimum_softening_fraction),
            "resolved_width": bool(width is not None and
                width >= criteria.minimum_width_to_interface*row["interface_width_m"]),
            "narrow_width": bool(width is not None and
                width <= criteria.maximum_width_to_domain*row["domain_length_m"]),
            "elongated_morphology": bool(row["power_aspect_ratio"] is not None and
                row["power_aspect_ratio"] >= criteria.minimum_aspect_ratio),
            "heat_power_colocation": row[
                "heat_power_component_overlap"] >= criteria.minimum_heat_power_overlap,
        }
        identity_ok = (identity is None or identity["overlap"] >=
                       criteria.minimum_component_identity_overlap)
        no_gap = (previous_time is None or not math.isfinite(nominal_dt)
                  or row["physical_time_s"]-previous_time <= 1.5*nominal_dt)
        qualifies = bool(all(checks.values()) and identity_ok and no_gap)
        row.update({"preceding_peak_stress_Pa": peak,
                    "preceding_peak_time_s": peak_time,
                    "softening_fraction": softening,
                    "component_identity": identity,
                    "strict_snapshot_checks": checks,
                    "strict_snapshot_qualifies": qualifies})
        if qualifies:
            if episode is None:
                episode = {"start_s": row["physical_time_s"],
                           "start_step": row["step"]}
        elif episode is not None:
            episode.update({"end_s": previous_time,
                            "end_step": rows[row_index-1]["step"]})
            episode["duration_s"] = episode["end_s"]-episode["start_s"]
            episodes.append(episode); episode = None
        previous_component = components[row["step"]]
        previous_time = row["physical_time_s"]
    if episode is not None:
        episode.update({"end_s": rows[-1]["physical_time_s"],
                        "end_step": rows[-1]["step"]})
        episode["duration_s"] = episode["end_s"]-episode["start_s"]
        episodes.append(episode)
    maximum = max((item["duration_s"] for item in episodes), default=0.0)
    strict = bool(maximum >= criteria.minimum_persistence_s and refinement_passed)
    missing = []
    if not matched_control_available:
        missing.append("matched temperature field at the same accepted times")
    if maximum < criteria.minimum_persistence_s:
        missing.append("one microsecond continuous same-component conjunctive episode")
    if not refinement_passed:
        missing.append("localization onset/width/persistence refinement")
    return {
        "criteria": asdict(criteria), "episodes": episodes,
        "maximum_episode_duration_s": maximum,
        "matched_control_available": matched_control_available,
        "localization_refinement_passed": refinement_passed,
        "inherited_strict_asb": strict,
        "missing_or_failed_requirements": missing,
    }


def analyze(baseline_dirs: list[Path], control_dirs: list[Path],
            refinement_passed: bool = False,
            intervention_scope: str = "all_arrhenius") -> dict:
    baseline = _checkpoint_map(baseline_dirs); history = _history_map(baseline_dirs)
    control = _checkpoint_map(control_dirs) if control_dirs else {}
    common = sorted(set(baseline) & set(control))
    rows = []; components = {}
    control_temperature = {}; pair_certificates = {}
    for step in common:
        (state, control_runtime, control_step, control_gamma,
         control_initial_volume, configuration,
         control_provenance) = _load_checkpoint(control[step])
        spacing = float(configuration["length_m"])/int(configuration["n"])
        reconstructed, _ = reconstruct_multigrain_common(state, spacing)
        control_temperature[step] = np.asarray(reconstructed.temperature_K)
        (baseline_state, baseline_runtime, baseline_step, baseline_gamma,
         baseline_initial_volume, baseline_configuration,
         baseline_provenance) = _load_checkpoint(baseline[step])
        pair_certificates[step] = temperature_intervention_certificate(
            baseline_configuration, configuration,
            baseline_provenance, control_provenance,
            baseline_step=baseline_step, control_step=control_step,
            baseline_time_s=baseline_runtime.ledger.physical_time_s,
            control_time_s=control_runtime.ledger.physical_time_s,
            baseline_gamma=baseline_gamma, control_gamma=control_gamma,
            baseline_initial_volume=baseline_initial_volume,
            control_initial_volume=control_initial_volume,
            baseline_grain_ids=baseline_state.grain_ids,
            control_grain_ids=state.grain_ids,
            intervention_scope=intervention_scope)
    for step, path in sorted(baseline.items()):
        row, component = checkpoint_snapshot(path, history.get(step))
        if (step in control_temperature
                and pair_certificates[step]["passed"]):
            state, *_ = _load_checkpoint(path)
            reconstructed, _ = reconstruct_multigrain_common(state, row["spacing_m"])
            row["matched_temperature_excess_K"] = float(np.max(
                np.asarray(reconstructed.temperature_K)-control_temperature[step]))
        else:
            row["matched_temperature_excess_K"] = None
        rows.append(row); components[step] = component
    valid_matched_rows = [
        row for row in rows if row["matched_temperature_excess_K"] is not None]
    # Retain the complete baseline stress history so the preceding peak is
    # physical even when an attributable control starts later.  Unmatched
    # rows fail the conjunctive matched-excess check and therefore cannot
    # enter or bridge an episode.
    episode_rows = rows
    episode_components = {row["step"]: components[row["step"]]
                          for row in episode_rows}
    matched_available = bool(control_dirs and valid_matched_rows)
    episode = classify_physical_episode(
        episode_rows, episode_components, PhysicalASBCriteria(),
        matched_control_available=matched_available,
        refinement_passed=refinement_passed)
    return {
        "schema": "asb-drx-v59-physical-asb-v1",
        "exploratory_v58_three_record_flag_is_strict": False,
        "temperature_intervention_scope": intervention_scope,
        "criterion_provenance": {
            "retained_recent": (
                "postprocess_v53_asb_mechanism.py: 1 us, power participation "
                "<=0.25, Tmax-Tmean >=50 K, component overlap >=0.25"),
            "retained_older_strict": (
                "production/asb_classifier.py and its tests: 20% true "
                "preceding-peak softening and width >=2 interface widths"),
            "prospective_v59_morphology": (
                "registered before V59 production classification: minor width "
                "<=0.25 domain and aspect ratio >=3; prevents isotropic hot "
                "patches from being called bands"),
        },
        "baseline_directories": [str(path.resolve()) for path in baseline_dirs],
        "control_directories": [str(path.resolve()) for path in control_dirs],
        "common_control_steps": common,
        "temperature_intervention_certificate": {
            "passed": bool(common and all(
                pair_certificates[step]["passed"] for step in common)),
            "steps": {str(step): pair_certificates[step] for step in common},
        },
        "rows": rows, "physical_episode": episode,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", action="append", type=Path, required=True)
    parser.add_argument("--control-dir", action="append", type=Path, default=[])
    parser.add_argument("--refinement-passed", action="store_true")
    parser.add_argument("--intervention-scope",
                        choices=("all_arrhenius", "flow_recovery"),
                        default="all_arrhenius")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        args.baseline_dir, args.control_dir, args.refinement_passed,
        args.intervention_scope)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["physical_episode"], indent=2))


if __name__ == "__main__":
    main()
