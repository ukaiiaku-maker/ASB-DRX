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
    maximum_component_speed_m_s: float = 2.0
    maximum_sampling_gap_s: float = 1.0e-7


def _primitive_lattice_vector(vector: np.ndarray) -> tuple[int, int]:
    value = np.asarray(vector, dtype=int)
    divisor = math.gcd(abs(int(value[0])), abs(int(value[1])))
    if divisor == 0:
        return (0, 0)
    value //= divisor
    first = next((int(item) for item in value if item != 0), 1)
    if first < 0:
        value *= -1
    return int(value[0]), int(value[1])


def _lift_component(mask: np.ndarray) -> tuple[dict, list[tuple[int, int]]]:
    """Lift one periodic component to Z² and recover its torus winding."""
    points = np.argwhere(mask)
    if len(points) == 0:
        return {}, []
    shape = np.asarray(mask.shape, dtype=int)
    start = tuple(int(x) for x in points[0])
    lifted = {start: np.asarray(start, dtype=int)}
    queue = [start]
    windings: set[tuple[int, int]] = set()
    steps = tuple((di, dj) for di in (-1, 0, 1)
                  for dj in (-1, 0, 1) if di or dj)
    while queue:
        point = queue.pop()
        origin = lifted[point]
        for step in steps:
            neighbor = ((point[0]+step[0]) % shape[0],
                        (point[1]+step[1]) % shape[1])
            if not mask[neighbor]:
                continue
            proposed = origin+np.asarray(step, dtype=int)
            if neighbor not in lifted:
                lifted[neighbor] = proposed
                queue.append(neighbor)
            else:
                loop = proposed-lifted[neighbor]
                if np.any(loop):
                    winding = np.rint(loop/shape).astype(int)
                    if np.any(winding):
                        windings.add(_primitive_lattice_vector(winding))
    independent = []
    for winding in sorted(windings):
        if winding == (0, 0):
            continue
        if not independent:
            independent.append(winding)
        elif abs(np.linalg.det(np.asarray([independent[0], winding]))) > 0:
            independent.append(winding)
            break
    return lifted, independent


def topology_aware_width(weight: np.ndarray, spacing_m: float) -> dict:
    """Measure a weighted component on the periodic torus.

    Contractible components are measured in a consistent universal-cover
    lift.  A rank-one winding band is measured in its periodic transverse
    phase, avoiding the invalid independent x/y unwrap used previously.
    """
    values = np.asarray(weight, dtype=float)
    if values.ndim != 2 or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("topology width requires a finite nonnegative 2-D field")
    mask = values > 0.0
    lifted, windings = _lift_component(mask)
    total = float(np.sum(values, dtype=np.longdouble))
    if total <= 0.0:
        raise ValueError("topology width requires positive weight")
    gaussian = 2.0*math.sqrt(2.0*math.log(2.0))
    if len(lifted) != int(mask.sum()):
        raise ValueError("topology width requires one connected component")
    if not windings:
        coordinates = np.asarray([lifted[tuple(point)] for point in np.argwhere(mask)],
                                 dtype=float)*float(spacing_m)
        weights = values[mask]
        center = np.average(coordinates, axis=0, weights=weights)
        centered = coordinates-center
        covariance = ((centered*weights[:, None]).T@centered)/total
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        minor_rms, major_rms = np.sqrt(eigenvalues)
        topology = "contractible"
        winding = None
    elif len(windings) == 1:
        winding = np.asarray(windings[0], dtype=int)
        normal = np.asarray((winding[1], -winding[0]), dtype=int)
        indices = np.indices(values.shape)
        phase = 2.0*math.pi*(normal[0]*indices[0]/values.shape[0]
                             +normal[1]*indices[1]/values.shape[1])
        resultant = np.sum(values*np.exp(1j*phase), dtype=np.clongdouble)/total
        center_phase = float(np.angle(complex(resultant)))
        wrapped_phase = np.angle(np.exp(1j*(phase-center_phase)))
        lengths = np.asarray(values.shape, dtype=float)*float(spacing_m)
        wave_number = 2.0*math.pi*np.linalg.norm(normal/lengths)
        transverse = wrapped_phase/max(wave_number, 1e-300)
        minor_rms = math.sqrt(float(
            np.sum(values*transverse*transverse, dtype=np.longdouble)/total))
        # A winding component has no finite longitudinal covariance on the
        # torus.  Its shortest homology representative is the declared extent.
        major_extent = float(np.linalg.norm(winding*lengths))
        major_rms = major_extent/gaussian
        topology = "rank_one_winding"
        winding = winding.tolist()
    else:
        # A component winding independently around both cycles is a network,
        # not a single band. Retain the legacy moment only as a diagnostic.
        legacy = weighted_width(values, float(spacing_m))
        return {**legacy, "topology": "rank_two_winding_network",
                "winding_vectors": [list(value) for value in windings],
                "width_semantics": "network diagnostic; not a single-band width"}
    return {
        "minor_rms_width_m": float(minor_rms),
        "major_rms_width_m": float(major_rms),
        "minor_gaussian_fwhm_m": float(gaussian*minor_rms),
        "major_gaussian_fwhm_m": float(gaussian*major_rms),
        "topology": topology,
        "winding_vectors": ([] if winding is None else [winding]),
        "width_semantics": ("universal-cover covariance" if winding is None else
                            "periodic transverse phase and shortest homology extent"),
    }


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


def _sampling_segment_metadata(directories: list[Path]) -> dict[str, dict]:
    """Recover the actual saved cadence independently for each directory."""
    result = {}
    for directory in directories:
        paths = sorted(directory.glob("checkpoint_*.npz"))
        segment_id = str(directory.resolve())
        steps = [int(path.stem.rsplit("_", 1)[-1]) for path in paths]
        if not paths:
            continue
        *_, configuration, provenance = _load_checkpoint(paths[0])
        differences = np.diff(steps)
        cadence = (None if len(differences) == 0 else
                   float(np.median(differences))*float(configuration["dt_s"]))
        result[segment_id] = {
            "declared_sampling_cadence_s": cadence,
            "step_differences": differences.tolist(),
            "source_commit": (provenance or {}).get("source_commit"),
            "operator_id": configuration.get(
                "front_heat_deposition", "uniform_legacy_front_heat"),
        }
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
    widths = topology_aware_width(weight, float(spacing_m))
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
                      spacing_m: float, maximum_displacement_m=None) -> dict:
    """Periodic identity with an optional physically admissible motion cone."""
    a = np.asarray(left, dtype=bool); b = np.asarray(right, dtype=bool)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("periodic components must have the same 2-D shape")
    if not np.any(a) or not np.any(b):
        return {"overlap": 0.0, "shift_cells": [0, 0],
                "displacement_m": [0.0, 0.0]}
    correlation = np.fft.ifftn(
        np.fft.fftn(a.astype(float))*np.conj(np.fft.fftn(b.astype(float)))).real
    unrestricted_index = np.unravel_index(
        int(np.argmax(correlation)), correlation.shape)
    candidate = correlation.copy()
    if maximum_displacement_m is not None:
        shifts = [np.where(np.arange(count) <= count//2,
                           np.arange(count), np.arange(count)-count)
                  for count in a.shape]
        sy, sx = np.meshgrid(shifts[0], shifts[1], indexing="ij")
        admissible = np.hypot(sy, sx)*float(spacing_m) <= (
            float(maximum_displacement_m)+1e-15*float(spacing_m))
        candidate = np.where(admissible, candidate, -np.inf)
    index = np.unravel_index(int(np.argmax(candidate)), correlation.shape)
    shift = np.asarray(index, dtype=int)
    for axis, count in enumerate(a.shape):
        if shift[axis] > count//2:
            shift[axis] -= count
    aligned = np.roll(b, tuple(shift), axis=(0, 1))
    unrestricted_shift = np.asarray(unrestricted_index, dtype=int)
    for axis, count in enumerate(a.shape):
        if unrestricted_shift[axis] > count//2:
            unrestricted_shift[axis] -= count
    unrestricted_aligned = np.roll(
        b, tuple(unrestricted_shift), axis=(0, 1))
    return {
        "overlap": overlap(a, aligned),
        "shift_cells": shift.tolist(),
        "displacement_m": (shift.astype(float)*spacing_m).tolist(),
        "maximum_admissible_displacement_m": maximum_displacement_m,
        "unrestricted_alignment_diagnostic": {
            "overlap": overlap(a, unrestricted_aligned),
            "shift_cells": unrestricted_shift.tolist(),
            "displacement_m": (unrestricted_shift.astype(float)*spacing_m).tolist(),
        },
    }


def source_association(source_field: np.ndarray, candidate: np.ndarray) -> dict:
    """Associate a source with one candidate without largest-mask aliasing."""
    source = np.asarray(source_field, dtype=float)
    region = np.asarray(candidate, dtype=bool)
    if source.shape != region.shape or source.ndim != 2:
        raise ValueError("source association requires matching 2-D fields")
    positive = np.maximum(source, 0.0)
    threshold = float(source.mean()+source.std())
    source_components = periodic_components(source > threshold)
    overlaps = [overlap(region, item) for item in source_components]
    largest = (max(source_components, key=np.count_nonzero)
               if source_components else np.zeros(source.shape, dtype=bool))
    total_positive = float(np.sum(positive, dtype=np.longdouble))
    local_positive = float(np.sum(positive[region], dtype=np.longdouble))
    return {
        "threshold": threshold,
        "source_component_count": len(source_components),
        "independently_largest_component_overlap": overlap(region, largest),
        "maximum_overlap_with_any_source_component": max(overlaps, default=0.0),
        "overlap_with_each_source_component": overlaps,
        "positive_source_cell_sum_on_candidate": local_positive,
        "positive_source_fraction_on_candidate": (
            local_positive/max(total_positive, 1e-300)),
        "candidate_positive_cell_fraction": (
            float(np.mean(positive[region] > 0.0)) if np.any(region) else 0.0),
        "semantics": (
            "direct positive-source integral on this power component plus "
            "all thresholded source-component associations; the independently "
            "largest overlap is retained only as a historical diagnostic"),
    }


def component_temperature_excess(
        baseline_temperature: np.ndarray, control_temperature: np.ndarray,
        baseline_power_component: np.ndarray) -> dict:
    """Return causal thermal excess locally on the candidate structure."""
    baseline = np.asarray(baseline_temperature, dtype=float)
    control = np.asarray(control_temperature, dtype=float)
    component = np.asarray(baseline_power_component, dtype=bool)
    if baseline.shape != control.shape or baseline.shape != component.shape:
        raise ValueError("matched temperatures and component must share a grid")
    difference = baseline-control
    return {
        "component_maximum_K": (None if not np.any(component) else
                                  float(np.max(difference[component]))),
        "whole_field_maximum_diagnostic_K": float(np.max(difference)),
        "component_mean_K": (None if not np.any(component) else
                               float(np.mean(difference[component]))),
        "semantics": (
            "Eulerian physical-minus-control temperature on the baseline "
            "accepted-state plastic-power component; the whole-field maximum "
            "is diagnostic only"),
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


def _signed_budget(field: np.ndarray, component: np.ndarray,
                   cell_volume_m3: float) -> dict:
    value = np.asarray(field, dtype=float)
    mask = np.asarray(component, dtype=bool)
    return {
        "whole_domain_signed_W": float(
            np.sum(value, dtype=np.longdouble)*cell_volume_m3),
        "whole_domain_positive_W": float(
            np.sum(np.maximum(value, 0.0), dtype=np.longdouble)*cell_volume_m3),
        "whole_domain_negative_W": float(
            np.sum(np.minimum(value, 0.0), dtype=np.longdouble)*cell_volume_m3),
        "component_signed_W": float(
            np.sum(value[mask], dtype=np.longdouble)*cell_volume_m3),
        "component_positive_W": float(
            np.sum(np.maximum(value[mask], 0.0), dtype=np.longdouble)
            *cell_volume_m3),
        "component_negative_W": float(
            np.sum(np.minimum(value[mask], 0.0), dtype=np.longdouble)
            *cell_volume_m3),
    }


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
    thickness = float(configuration.get(
        "represented_thickness_m", 2.0*2.48e-10))
    cell_volume = spacing*spacing*thickness
    with np.load(path, allow_pickle=False) as checkpoint_data:
        front_source_J_by_cell = (
            np.asarray(checkpoint_data[
                "diagnostic_front_heat_source_J_by_cell"], dtype=float)
            if "diagnostic_front_heat_source_J_by_cell" in checkpoint_data.files
            else None)
        interval_mechanical_heat = (
            np.asarray(checkpoint_data[
                "diagnostic_mechanical_heat_J_m3_cells"], dtype=float)
            if "diagnostic_mechanical_heat_J_m3_cells" in checkpoint_data.files
            else None)
        interval_conduction = (
            np.asarray(checkpoint_data[
                "diagnostic_thermal_conduction_J_m3_cells"], dtype=float)
            if "diagnostic_thermal_conduction_J_m3_cells" in checkpoint_data.files
            else None)
        interval_bath_exchange = (
            np.asarray(checkpoint_data[
                "diagnostic_thermal_bath_exchange_J_m3_cells"], dtype=float)
            if "diagnostic_thermal_bath_exchange_J_m3_cells" in checkpoint_data.files
            else None)
    interval_s = float(configuration["dt_s"])
    front_source_rate = (None if front_source_J_by_cell is None else
                         front_source_J_by_cell/max(
                             interval_s*cell_volume, 1e-300))
    mechanical_interval_heat_rate = (
        None if interval_mechanical_heat is None else
        interval_mechanical_heat/max(interval_s, 1e-300))
    budget_fields = {
        "plastic_power": power_field,
        "irreversible_heat": heat_field,
        "reversible_defect_transport_divergence": dissipation[
            "reversible_defect_transport_divergence_W_m3"],
        "thermal_conduction": dissipation["thermal_conduction_W_m3"],
        "thermal_bath_exchange": dissipation["thermal_bath_exchange_W_m3"],
        "instantaneous_local_thermal_storage": dissipation[
            "instantaneous_local_thermal_storage_W_m3"],
        **{
            name.removesuffix("_W_m3"): value
            for name, value in dissipation["dissipation_channels_W_m3"].items()
        },
    }
    if front_source_rate is not None:
        budget_fields["preceding_interval_front_dissipation"] = front_source_rate
    if mechanical_interval_heat_rate is not None:
        budget_fields["preceding_interval_mechanical_heat"] = (
            mechanical_interval_heat_rate)
    if interval_conduction is not None:
        budget_fields["preceding_interval_thermal_conduction"] = (
            interval_conduction/max(interval_s, 1e-300))
    if interval_bath_exchange is not None:
        budget_fields["preceding_interval_thermal_bath_exchange"] = (
            interval_bath_exchange/max(interval_s, 1e-300))
    finite_interval_terms = [value for value in (
        mechanical_interval_heat_rate,
        None if interval_conduction is None else
        interval_conduction/max(interval_s, 1e-300),
        None if interval_bath_exchange is None else
        interval_bath_exchange/max(interval_s, 1e-300),
        front_source_rate) if value is not None]
    if finite_interval_terms:
        budget_fields["preceding_interval_thermal_storage"] = sum(
            finite_interval_terms, start=np.zeros_like(power_field))
    eligible_components = periodic_components(power_field > power["threshold"])
    eligible_components.sort(key=np.count_nonzero, reverse=True)
    component_budgets = []
    for component_index, eligible in enumerate(eligible_components):
        component_budgets.append({
            "component_index_by_descending_area": component_index,
            "cell_count": int(np.count_nonzero(eligible)),
            "area_fraction": float(np.mean(eligible)),
            "channels": {
                name: _signed_budget(value, eligible, cell_volume)
                for name, value in budget_fields.items()
            },
            "source_association": {
                name: source_association(value, eligible)
                for name, value in budget_fields.items()
            },
        })
    front_interval = None
    if pre_front_row and pre_front_row.get("front"):
        front_interval = {
            "generated_heat_J": pre_front_row["front"].get("generated_heat_J"),
            "heat_source": pre_front_row["front"].get("heat_source"),
            "deposition_mode": pre_front_row["front"].get(
                "heat_deposition_mode"),
            "semantics": "accepted preceding macro-interval, not instantaneous rate",
        }
    front_source_metrics = (None if front_source_rate is None else
                            field_metrics(front_source_rate, spacing))
    front_source_overlap = (None if front_source_metrics is None else overlap(
        power_component, front_source_metrics[1]))
    row = {
        "step": step, "physical_time_s": runtime.ledger.physical_time_s,
        "applied_shear_strain": gamma,
        "checkpoint": str(path.resolve()), "checkpoint_sha256": _digest(path),
        "source_commit": (provenance or {}).get("source_commit"),
        "restart_transition": (provenance or {}).get("restart_transition"),
        "parent_checkpoint_sha256": (provenance or {}).get(
            "parent_checkpoint_sha256"),
        "operator_id": configuration.get(
            "front_heat_deposition", "uniform_legacy_front_heat"),
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
        "signed_channel_budget": {
            "instantaneous_stage": "accepted post-front state",
            "represented_cell_volume_m3": cell_volume,
            "eligible_component_rule": (
                "every periodic plastic-power component above mean+std; "
                "ordered by descending cell count"),
            "whole_domain_and_largest_component": {
                name: _signed_budget(value, power_component, cell_volume)
                for name, value in budget_fields.items()
            },
            "all_eligible_components": component_budgets,
            "preceding_interval_front_dissipation": front_interval,
            "front_source_power_component_overlap": front_source_overlap,
            "largest_power_component_source_association": {
                name: source_association(value, power_component)
                for name, value in budget_fields.items()
            },
            "front_source_field_metrics": (
                None if front_source_metrics is None else front_source_metrics[0]),
            "stage_warning": (
                "instantaneous accepted-state rates and preceding accepted-"
                "interval sources are reported separately and are not summed "
                "as if simultaneous"),
        },
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
                              refinement_passed: bool,
                              qualifying_operator_id: str | None = None) -> dict:
    if not rows:
        raise ValueError("physical ASB history is empty")
    # Sort physical time and deterministically collapse duplicate records.
    # A later entry for the same (time, step) wins, matching directory overlay
    # semantics without creating fictitious persistence.
    unique = {}
    for row in rows:
        key = (float(row["physical_time_s"]), int(row["step"]))
        unique[key] = row
    ordered_rows = [unique[key] for key in sorted(unique)]
    if any(right["physical_time_s"] <= left["physical_time_s"]
           for left, right in zip(ordered_rows, ordered_rows[1:])):
        raise ValueError("physical ASB records must have strictly increasing time")
    duplicate_count = len(rows)-len(ordered_rows)
    peak = -math.inf; peak_time = None; peak_row = None
    operator_peak = -math.inf; operator_peak_time = None
    episode = None; episodes = []
    previous_component = None; previous_time = None
    legacy_nominal_dt = min(np.diff(
        [row["physical_time_s"] for row in ordered_rows]), default=math.inf)
    previous_row = None
    for row_index, row in enumerate(ordered_rows):
        stress = abs(float(row["post_front_equilibrated_stress_Pa"]))
        if stress > peak:
            peak = stress; peak_time = row["physical_time_s"]; peak_row = row
        operator_eligible = (qualifying_operator_id is None
                             or row.get("operator_id") == qualifying_operator_id)
        if operator_eligible and stress > operator_peak:
            operator_peak = stress; operator_peak_time = row["physical_time_s"]
        whole_softening = 0.0 if peak <= 0.0 else (peak-stress)/peak
        operator_softening = (None if operator_peak <= 0.0 else
                              (operator_peak-stress)/operator_peak)
        width = row["power_width_minor_m"]
        elapsed = (None if previous_time is None else
                   row["physical_time_s"]-previous_time)
        identity = (None if previous_component is None else periodic_identity(
            previous_component, components[row["step"]], row["spacing_m"],
            maximum_displacement_m=(criteria.maximum_component_speed_m_s
                                    *max(float(elapsed), 0.0)
                                    +row["spacing_m"])))
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
                and whole_softening >= criteria.minimum_softening_fraction),
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
        if previous_row is None:
            no_gap = True; gap_limit = None; seam = "initial"
        else:
            same_segment = (row.get("sampling_segment_id")
                            == previous_row.get("sampling_segment_id"))
            declared = row.get("declared_sampling_cadence_s")
            previous_declared = previous_row.get("declared_sampling_cadence_s")
            if declared is None or previous_declared is None:
                gap_limit = 1.5*legacy_nominal_dt
                seam = "legacy_inferred_global_minimum"
            else:
                gap_limit = min(
                    criteria.maximum_sampling_gap_s,
                    1.5*max(float(declared), float(previous_declared)))
                seam = ("same_declared_segment" if same_segment else
                        "verified_restart_seam" if row.get(
                            "source_seam_verified", False) else
                        "unverified_source_seam")
            no_gap = bool(
                row["physical_time_s"]-previous_time <= gap_limit
                and (same_segment or row.get("source_seam_verified", False)
                     or declared is None or previous_declared is None))
        qualifies = bool(operator_eligible and all(checks.values())
                         and identity_ok and no_gap)
        row.update({"preceding_peak_stress_Pa": peak,
                    "preceding_peak_time_s": peak_time,
                    "preceding_peak_checkpoint": (
                        None if peak_row is None else peak_row.get("checkpoint")),
                    "preceding_peak_source_commit": (
                        None if peak_row is None else peak_row.get("source_commit")),
                    "whole_history_softening_fraction": whole_softening,
                    "operator_segment_peak_stress_Pa": (
                        None if operator_peak <= 0.0 else operator_peak),
                    "operator_segment_peak_time_s": operator_peak_time,
                    "operator_segment_softening_fraction": operator_softening,
                    "softening_fraction": whole_softening,
                    "component_identity": identity,
                    "sampling_continuity": {
                        "passed": no_gap, "gap_limit_s": gap_limit,
                        "classification": seam,
                    },
                    "qualifying_operator_eligible": operator_eligible,
                    "strict_snapshot_checks": checks,
                    "strict_snapshot_qualifies": qualifies})
        if qualifies:
            if episode is None:
                episode = {"start_s": row["physical_time_s"],
                           "start_step": row["step"]}
        elif episode is not None:
            episode.update({"end_s": previous_time,
                            "end_step": ordered_rows[row_index-1]["step"]})
            episode["duration_s"] = episode["end_s"]-episode["start_s"]
            episodes.append(episode); episode = None
        previous_component = components[row["step"]]
        previous_time = row["physical_time_s"]
        previous_row = row
    if episode is not None:
        episode.update({"end_s": ordered_rows[-1]["physical_time_s"],
                        "end_step": ordered_rows[-1]["step"]})
        episode["duration_s"] = episode["end_s"]-episode["start_s"]
        episodes.append(episode)
    maximum = max((item["duration_s"] for item in episodes), default=0.0)
    strict = bool(maximum >= criteria.minimum_persistence_s and refinement_passed)
    maximum_whole_softening = max(
        float(row["whole_history_softening_fraction"])
        for row in ordered_rows)
    maximum_operator_softening = max(
        (float(row["operator_segment_softening_fraction"])
         for row in ordered_rows
         if row["operator_segment_softening_fraction"] is not None),
        default=None)
    missing = []
    if not matched_control_available:
        missing.append("matched temperature field at the same accepted times")
    if maximum < criteria.minimum_persistence_s:
        missing.append("one microsecond continuous same-component conjunctive episode")
    if not refinement_passed:
        missing.append("localization onset/width/persistence refinement")
    return {
        "criteria": asdict(criteria), "episodes": episodes,
        "record_count_input": len(rows),
        "record_count_unique": len(ordered_rows),
        "duplicate_record_count": duplicate_count,
        "qualifying_operator_id": qualifying_operator_id,
        "whole_history_peak": {
            "stress_Pa": peak, "physical_time_s": peak_time,
            "checkpoint": None if peak_row is None else peak_row.get("checkpoint"),
            "source_commit": None if peak_row is None else peak_row.get("source_commit"),
            "stress_stage": "post_front_equilibrated_stress_Pa",
        },
        "qualifying_operator_peak": {
            "stress_Pa": None if operator_peak <= 0.0 else operator_peak,
            "physical_time_s": operator_peak_time,
        },
        "softening_summary": {
            "maximum_whole_history_fraction": maximum_whole_softening,
            "maximum_qualifying_operator_segment_fraction": (
                maximum_operator_softening),
            "final_whole_history_fraction": ordered_rows[-1][
                "whole_history_softening_fraction"],
            "final_qualifying_operator_segment_fraction": ordered_rows[-1][
                "operator_segment_softening_fraction"],
            "semantics": (
                "whole-history softening may inherit a peak established by an "
                "earlier operator; qualifying-operator softening measures only "
                "stress loss from the maximum reached under the declared "
                "operator and is the relevant persistence diagnostic"),
        },
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
    segment_metadata = _sampling_segment_metadata(baseline_dirs)
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
        segment_id = str(path.parent.resolve())
        row["sampling_segment_id"] = segment_id
        row["declared_sampling_cadence_s"] = segment_metadata.get(
            segment_id, {}).get("declared_sampling_cadence_s")
        row["source_seam_verified"] = bool(
            row.get("restart_transition") and row.get(
                "parent_checkpoint_sha256"))
        if (step in control_temperature
                and pair_certificates[step]["passed"]):
            state, *_ = _load_checkpoint(path)
            reconstructed, _ = reconstruct_multigrain_common(state, row["spacing_m"])
            causal_temperature = component_temperature_excess(
                reconstructed.temperature_K, control_temperature[step], component)
            row["matched_temperature_excess_K"] = causal_temperature[
                "component_maximum_K"]
            row["matched_temperature_excess"] = causal_temperature
        else:
            row["matched_temperature_excess_K"] = None
            row["matched_temperature_excess"] = None
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
    qualifying_operator = (None if not rows else rows[-1].get("operator_id"))
    episode = classify_physical_episode(
        episode_rows, episode_components, PhysicalASBCriteria(),
        matched_control_available=matched_available,
        refinement_passed=refinement_passed,
        qualifying_operator_id=qualifying_operator)
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
        "sampling_segments": segment_metadata,
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
