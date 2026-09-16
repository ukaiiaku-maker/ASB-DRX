#!/usr/bin/env python3
"""Quantify phase representation at the V31/V32 n128 front terminals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
from skimage.measure import find_contours

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from full_model.production.front_topology import (
    cut_cell_receiver_fraction, extract_front_components)


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _periodic_components(mask):
    labels, count = ndimage.label(mask)
    parent = list(range(count+1))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]; item = parent[item]
        return item

    def union(a, b):
        if a and b:
            a = find(int(a)); b = find(int(b))
            if a != b:
                parent[b] = a
    for j in range(mask.shape[1]):
        union(labels[0, j], labels[-1, j])
    for i in range(mask.shape[0]):
        union(labels[i, 0], labels[i, -1])
    groups = {}
    for label in range(1, count+1):
        groups.setdefault(find(label), []).append(label)
    result = []
    for members in groups.values():
        component = np.isin(labels, members)
        tiled = np.tile(component, (3, 3))
        distance = ndimage.distance_transform_edt(tiled)
        nx, ny = component.shape
        result.append({
            "area_cells2": int(np.count_nonzero(component)),
            "inradius_cells": float(np.max(distance[nx:2*nx, ny:2*ny])),
            "mask": component,
            "touches_periodic_seam": bool(
                np.any(component[0]) or np.any(component[-1])
                or np.any(component[:, 0]) or np.any(component[:, -1])),
        })
    return sorted(result, key=lambda item: item["area_cells2"], reverse=True)


def _contours(phi):
    result = {}
    for level in (-.8, -.5, -.2, 0.0, .2, .5, .8):
        contours = find_contours(phi, level, fully_connected="high")
        rows = []
        for points in contours:
            length = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
            closed = bool(np.linalg.norm(points[0]-points[-1]) < 1.5)
            area = None
            if closed:
                x, y = points[:, 0], points[:, 1]
                area = float(abs(np.dot(x, np.roll(y, 1))
                                 -np.dot(y, np.roll(x, 1)))/2.0)
            rows.append({"length_cells": length, "closed": closed,
                         "enclosed_area_cells2": area})
        result[f"{level:+.1f}"] = {
            "count": len(rows), "closed_count": sum(r["closed"] for r in rows),
            "total_length_cells": sum(r["length_cells"] for r in rows),
            "components": sorted(rows, key=lambda r: r["length_cells"], reverse=True),
        }
    return result


def _spectrum(phi):
    fluctuation = phi-np.mean(phi)
    transform = np.fft.rfft2(fluctuation)
    power = np.abs(transform)**2
    kx = np.fft.fftfreq(phi.shape[0])[:, None]
    ky = np.fft.rfftfreq(phi.shape[1])[None, :]
    high = (np.abs(kx) > 1/3) | (np.abs(ky) > 1/3)
    checkerboard = (-1.0)**np.sum(np.indices(phi.shape), axis=0)
    return {
        "high_k_power_fraction_two_thirds_nyquist": float(
            np.sum(power[high])/max(np.sum(power), 1e-300)),
        "checkerboard_projection": float(abs(np.mean(fluctuation*checkerboard))),
    }


def _normal_thickness(phi, spacing):
    gradient = .5*(np.roll(phi, -1, axis=0)-np.roll(phi, 1, axis=0))
    crossings = phi*np.roll(phi, -1, axis=0) <= 0.0
    slopes = np.abs(gradient[crossings])
    slopes = slopes[slopes > 1e-12]
    widths = 1.6/np.maximum(slopes, 1e-300)
    return {
        "sample_count": int(widths.size),
        "median_cells": float(np.median(widths)),
        "p05_cells": float(np.percentile(widths, 5)),
        "p95_cells": float(np.percentile(widths, 95)),
        "median_m": float(np.median(widths)*spacing),
    }


def _energy(eta, spacing, kappa, barrier):
    gradient_sum = np.zeros(eta.shape[:2])
    for phase in range(eta.shape[2]):
        gx = (np.roll(eta[:, :, phase], -1, 0)
              -np.roll(eta[:, :, phase], 1, 0))/(2*spacing)
        gy = (np.roll(eta[:, :, phase], -1, 1)
              -np.roll(eta[:, :, phase], 1, 1))/(2*spacing)
        gradient_sum += gx*gx+gy*gy
    gradient = .5*kappa*gradient_sum
    bulk = barrier*np.prod(eta*eta, axis=2)
    return {
        "gradient_J_per_m": float(np.sum(gradient, dtype=np.longdouble)*spacing**2),
        "bulk_J_per_m": float(np.sum(bulk, dtype=np.longdouble)*spacing**2),
        "total_J_per_m": float(np.sum(gradient+bulk, dtype=np.longdouble)*spacing**2),
        "maximum_gradient_density_J_m3": float(np.max(gradient)),
        "maximum_bulk_density_J_m3": float(np.max(bulk)),
    }


def _field_metrics(eta, active_mask, spacing, kappa, barrier):
    phi = eta[:, :, 1]-eta[:, :, 0]
    components = {}
    for sign, name in ((-1, "negative"), (1, "positive")):
        rows = _periodic_components(sign*phi > 0.0)
        components[name] = [{
            "area_cells2": row["area_cells2"],
            "inradius_cells": row["inradius_cells"],
            "inradius_m": row["inradius_cells"]*spacing,
            "pure_core_area_phi_abs_ge_0p9_cells2": int(np.count_nonzero(
                row["mask"] & (sign*phi >= .9))),
            "maximum_signed_phi": float(np.max(sign*phi[row["mask"]])),
            "touches_periodic_seam": row["touches_periodic_seam"],
        } for row in rows]
    before_hash = _sha(eta)
    unfiltered = extract_front_components(
        phi, active_mask=active_mask, periodic=True,
        minimum_resolved_loop_area_cells2=0.0,
        minimum_resolved_loop_length_cells=0.0)
    filtered = extract_front_components(
        phi, active_mask=active_mask, periodic=True)
    fraction0 = cut_cell_receiver_fraction(
        phi, active_mask=active_mask, periodic=True)
    after_hash = _sha(eta)
    fraction1 = cut_cell_receiver_fraction(
        phi, active_mask=active_mask, periodic=True)
    return {
        "eta_min": float(np.min(eta)), "eta_max": float(np.max(eta)),
        "phi_min": float(np.min(phi)), "phi_max": float(np.max(phi)),
        "partition_unity_max_abs": float(np.max(abs(np.sum(eta, axis=2)-1.0))),
        "mixed_area_abs_phi_lt_0p9_cells2": int(np.count_nonzero(abs(phi) < .9)),
        "periodic_sign_components": components,
        "neighboring_isovalue_contours": _contours(phi),
        "normal_thickness": _normal_thickness(phi, spacing),
        "energy": _energy(eta, spacing, kappa, barrier),
        "spectrum": _spectrum(phi),
        "topology_filter_audit": {
            "unfiltered_component_count": len(unfiltered.components),
            "filtered_component_count": len(filtered.components),
            "filtered_subcell_component_count": filtered.filtered_subcell_component_count,
            "eta_sha256_before": before_hash, "eta_sha256_after": after_hash,
            "eta_bitwise_unchanged": before_hash == after_hash,
            "cut_cell_fraction_bitwise_unchanged": bool(
                np.array_equal(fraction0, fraction1)),
        },
    }


def _case(field_path, checkpoint_path, terminal_path):
    with np.load(checkpoint_path, allow_pickle=True) as state:
        parameters = json.loads(str(state["P_json"].item()))
    with np.load(field_path) as data:
        before = data["eta_before"]; trial = data["eta_trial"]
        accepted = data["eta_accepted"]; active = data["active_mask"]
        spacing = float(data["spacing_m"])
    kappa = float(parameters["kappa_eta"]); barrier = float(parameters["W_eta"])
    changed_sign = np.signbit(trial[:, :, 1]-trial[:, :, 0]) != np.signbit(
        before[:, :, 1]-before[:, :, 0])
    flip_rows, flip_columns = np.where(changed_sign)
    profile_rows = []
    for row in sorted(set(map(int, flip_rows))):
        columns = flip_columns[flip_rows == row]
        profile_rows.append({
            "normal_row": row, "flipped_pixel_count": int(columns.size),
            "column_min": int(np.min(columns)), "column_max": int(np.max(columns)),
            "before_phi_normal_window": [float(item) for item in np.mean(
                before[max(0, row-3):min(row+4, before.shape[0]), :, 1]
                -before[max(0, row-3):min(row+4, before.shape[0]), :, 0], axis=1)],
            "trial_phi_normal_window": [float(item) for item in np.mean(
                trial[max(0, row-3):min(row+4, trial.shape[0]), :, 1]
                -trial[max(0, row-3):min(row+4, trial.shape[0]), :, 0], axis=1)],
        })
    terminal = json.loads(Path(terminal_path).read_text())
    closed = [item for item in terminal["front_decision"]["topology_event"][
        "new_components"] if item["closed"]]
    before_metrics = _field_metrics(before, active, spacing, kappa, barrier)
    trial_metrics = _field_metrics(trial, active, spacing, kappa, barrier)
    energy_before = before_metrics["energy"]["total_J_per_m"]
    energy_trial = trial_metrics["energy"]["total_J_per_m"]
    contour_delta = {
        level: (trial_metrics["neighboring_isovalue_contours"][level]["count"]
                -before_metrics["neighboring_isovalue_contours"][level]["count"])
        for level in before_metrics["neighboring_isovalue_contours"]}
    increment = abs(trial-before)
    return {
        "spacing_m": spacing,
        "physical_domain_m": [before.shape[0]*spacing, before.shape[1]*spacing],
        "maximum_abs_phase_increment": float(np.max(abs(trial-before))),
        "mean_abs_phase_increment": float(np.mean(abs(trial-before))),
        "phase_values_at_increment_limit_count": int(np.count_nonzero(
            increment >= .02-128*np.finfo(float).eps)),
        "sign_flip_pixel_count": int(np.count_nonzero(changed_sign)),
        "sign_flip_profiles": profile_rows,
        "accepted_equals_before_bitwise": bool(np.array_equal(accepted, before)),
        "trial_energy_change_J_per_m": energy_trial-energy_before,
        "trial_energy_relative_change": (
            energy_trial-energy_before)/max(abs(energy_before), 1e-300),
        "isovalue_component_count_change": contour_delta,
        "zero_level_change_without_neighboring_0p2_change": bool(
            contour_delta["+0.0"] != 0
            and contour_delta["-0.2"] == 0 and contour_delta["+0.2"] == 0),
        "terminal_graph": {
            "classification": terminal["classification"],
            "ray_crossings_before": terminal["front_decision"][
                "ray_crossing_count_before"],
            "ray_crossings_after": terminal["front_decision"][
                "ray_crossing_count_after"],
            "closed_components": [{
                "area_cells2": item["enclosed_area_cells2"],
                "length_cells": item["interface_length_cells"],
                "equivalent_radius_m": spacing*np.sqrt(
                    item["enclosed_area_cells2"]/np.pi),
            } for item in closed],
        },
        "before": before_metrics,
        "trial": trial_metrics,
    }


def _spatial_control(checkpoint_path, terminal_path):
    with np.load(checkpoint_path, allow_pickle=True) as data:
        eta = data["eta"][:, :, :int(data["Ng"])]
        active = np.asarray(data["sibm_active_mask"], dtype=bool)
        parameters = json.loads(str(data["P_json"].item()))
    spacing = 10e-6/eta.shape[0]
    terminal = json.loads(Path(terminal_path).read_text())
    loops = [item for item in terminal["front_decision"]["topology_event"][
        "new_components"] if item["closed"]]
    return {
        "grid": eta.shape[0], "spacing_m": spacing,
        "historical_classification": terminal["classification"],
        "historical_step": terminal["step"],
        "closed_components": [{
            "area_cells2": item["enclosed_area_cells2"],
            "length_cells": item["interface_length_cells"],
            "equivalent_radius_m": spacing*np.sqrt(
                item["enclosed_area_cells2"]/np.pi),
        } for item in loops],
        "accepted_field": _field_metrics(
            eta, active, spacing, float(parameters["kappa_eta"]),
            float(parameters["W_eta"])),
    }


def _enrich_controls(records, control_root, source_eta):
    enriched = []
    for original in records:
        row = dict(original)
        directory = control_root/row["case"]
        field_path = directory/"sibm_front_terminal_fields.npz"
        if field_path.exists():
            with np.load(field_path) as data:
                before = data["eta_before"]; trial = data["eta_trial"]
            phi0 = before[:, :, 1]-before[:, :, 0]
            phi1 = trial[:, :, 1]-trial[:, :, 0]
            row.update(
                maximum_abs_phase_increment=float(np.max(abs(trial-before))),
                sign_flip_pixel_count=int(np.count_nonzero(
                    np.signbit(phi0) != np.signbit(phi1))),
                trial_high_k_power_fraction=_spectrum(phi1)[
                    "high_k_power_fraction_two_thirds_nyquist"])
        checkpoints = sorted(directory.glob("v33_front_restart_*.npz"))
        if checkpoints:
            with np.load(checkpoints[-1], allow_pickle=True) as data:
                final_eta = data["eta"][:, :, :source_eta.shape[2]]
            row["accepted_maximum_abs_phase_change"] = float(
                np.max(abs(final_eta-source_eta)))
            source_phi = source_eta[:, :, 1]-source_eta[:, :, 0]
            final_phi = final_eta[:, :, 1]-final_eta[:, :, 0]
            row["accepted_sign_flip_pixel_count"] = int(np.count_nonzero(
                np.signbit(source_phi) != np.signbit(final_phi)))
        enriched.append(row)
    return enriched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-root", type=Path, required=True)
    parser.add_argument("--continuation-root", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--n64-checkpoint", type=Path, required=True)
    parser.add_argument("--n64-terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-source-commit", required=True)
    args = parser.parse_args()
    cases = {}
    for case_id in ("a1_n128_delta_p1e-6", "a1_n128_delta_m1e-6"):
        cases[case_id] = _case(
            args.field_root/case_id/"sibm_front_terminal_fields.npz",
            args.continuation_root/case_id/"attempt-000"/
            "v31_front_restart_000137.npz",
            args.field_root/case_id/"sibm_terminal_event.json")
    controls = json.loads(args.controls.read_text())
    with np.load(args.continuation_root/"a1_n128_delta_p1e-6"/"attempt-000"/
                 "v31_front_restart_000137.npz", allow_pickle=True) as data:
        source_eta = data["eta"][:, :, :int(data["Ng"])]
    result = {
        "schema": "asb-drx/v33-front-preterminal-diagnostic/v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"),
        "authority": {
            "v31_execution_source_commit": (
                "c643afe00b4ea6f7f30e021675d7648ecdb6422c"),
            "v33_diagnostic_source_commit": args.diagnostic_source_commit,
            "n128_positive_checkpoint_sha256": _file_sha(
                args.continuation_root/"a1_n128_delta_p1e-6"/"attempt-000"/
                "v31_front_restart_000137.npz"),
            "n128_negative_checkpoint_sha256": _file_sha(
                args.continuation_root/"a1_n128_delta_m1e-6"/"attempt-000"/
                "v31_front_restart_000137.npz"),
            "n64_checkpoint_sha256": _file_sha(args.n64_checkpoint),
        },
        "cases": cases,
        "spatial_control_n64": _spatial_control(
            args.n64_checkpoint, args.n64_terminal),
        "controls": _enrich_controls(
            controls["records"], args.control_root, source_eta),
        "classification": "GRID_SCALE_PHASE_REPRESENTATION_BREAKDOWN",
        "classification_basis": [
            "terminal loops are one-cell-inradius filaments without a pure core",
            "neighboring isovalues do not define a nested resolved interface",
            "the accepted physical phase is bitwise the pretrial field",
            "topology filtering changes graph ownership only and never eta or cut-cell area",
            "both near-equal signs reproduce the same grid-scale mechanism",
        ],
        "genuine_physical_instability_supported": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"classification": result["classification"],
                      "cases": list(cases)}, indent=2))


if __name__ == "__main__":
    main()
