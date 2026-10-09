#!/usr/bin/env python3
"""Track one physical localization candidate and its fixed-axis profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from full_model.analysis.audit_v63_phase_sensitive_fields import checkpoint_fields, digest
from full_model.analysis.postprocess_v53_asb_mechanism import periodic_components


def periodic_delta(value: np.ndarray, origin: float, period: float) -> np.ndarray:
    return (value-origin+.5*period) % period-.5*period


def component_at_anchor(field: np.ndarray, anchor_xy: tuple[float, float],
                        length_m: float) -> np.ndarray:
    positive = np.maximum(np.asarray(field, dtype=float), 0.0)
    components = periodic_components(positive > 2.0*float(positive.mean()))
    if not components:
        return np.zeros(positive.shape, dtype=bool)
    n = positive.shape[0]
    ii, jj = np.indices(positive.shape)
    x = ii*length_m/n; y = jj*length_m/n
    distance = periodic_delta(x, anchor_xy[0], length_m)**2 + periodic_delta(
        y, anchor_xy[1], length_m)**2
    return min(components, key=lambda c: float(np.min(distance[c])))


def anchor_geometry(field: np.ndarray, component: np.ndarray,
                    length_m: float) -> dict:
    positive = np.maximum(np.asarray(field, dtype=float), 0.0)
    masked = positive*component
    peak = np.unravel_index(int(np.argmax(masked)), masked.shape)
    n = positive.shape[0]
    center = np.asarray(peak, dtype=float)*length_m/n
    ii, jj = np.indices(positive.shape)
    coordinates = np.stack((periodic_delta(ii*length_m/n, center[0], length_m),
                            periodic_delta(jj*length_m/n, center[1], length_m)), axis=-1)
    weights = masked
    covariance = np.einsum("abi,abj,ab->ij", coordinates, coordinates, weights)
    covariance /= max(float(weights.sum()), 1e-300)
    values, vectors = np.linalg.eigh(covariance)
    tangent = vectors[:, int(np.argmax(values))]
    if tangent[0] < 0.0:
        tangent = -tangent
    normal = np.asarray((-tangent[1], tangent[0]))
    tangent_coordinate = np.einsum("...i,i->...", coordinates, tangent)
    tangent_half_width = max(float(np.max(np.abs(tangent_coordinate[component]))),
                             length_m/n)
    return {"center_m": center.tolist(), "tangent": tangent.tolist(),
            "normal": normal.tolist(), "tangent_half_width_m": tangent_half_width,
            "anchor_peak_index": list(map(int, peak))}


def profile(field: np.ndarray, geometry: dict, length_m: float,
            bins: int) -> dict:
    value = np.maximum(np.asarray(field, dtype=float), 0.0)
    n = value.shape[0]; ii, jj = np.indices(value.shape)
    center = np.asarray(geometry["center_m"])
    displacement = np.stack((periodic_delta(ii*length_m/n, center[0], length_m),
                             periodic_delta(jj*length_m/n, center[1], length_m)), axis=-1)
    tangent = np.einsum("...i,i->...", displacement, geometry["tangent"])
    normal = np.einsum("...i,i->...", displacement, geometry["normal"])
    selected = np.abs(tangent) <= geometry["tangent_half_width_m"]
    edges = np.linspace(-.5*length_m, .5*length_m, bins+1)
    sums = np.histogram(normal[selected], edges, weights=value[selected])[0]
    counts = np.histogram(normal[selected], edges)[0]
    values = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    centers = .5*(edges[:-1]+edges[1:])
    total = float(values.sum())
    mean = float(np.sum(centers*values)/max(total, 1e-300))
    width = float(np.sqrt(np.sum((centers-mean)**2*values)/max(total, 1e-300)))
    half = .5*float(values.max())
    above = np.flatnonzero(values >= half)
    fwhm = (0.0 if above.size == 0 else
            float(edges[above[-1]+1]-edges[above[0]]))
    return {"bin_centers_m": centers.tolist(), "power_W_m3": values.tolist(),
            "second_moment_width_m": width, "fwhm_m": fwhm,
            "fwhm_native_cells": fwhm/(length_m/n),
            "native_spacing_m": length_m/n}


def record(path: Path, anchor_xy: tuple[float, float], geometry: dict,
           bins: int) -> tuple[dict, np.ndarray]:
    fields, metadata = checkpoint_fields(path)
    power = fields["instantaneous_plastic_power_W_m3"]
    length_m = metadata["spacing_m"]*metadata["n"]
    component = component_at_anchor(power, anchor_xy, length_m)
    masked = np.maximum(power, 0.0)*component
    peak = np.unravel_index(int(np.argmax(masked)), masked.shape)
    peak_xy = np.asarray(peak)*metadata["spacing_m"]
    row = {**metadata, "component_cells": int(component.sum()),
           "component_fraction": float(component.mean()),
           "component_peak_xy_m": peak_xy.tolist(),
           "component_peak_distance_from_anchor_m": float(np.linalg.norm([
               periodic_delta(peak_xy[0], anchor_xy[0], length_m),
               periodic_delta(peak_xy[1], anchor_xy[1], length_m)])),
           "profile": profile(power*component, geometry, length_m, bins)}
    return row, component


def nearest_resample(mask: np.ndarray, q: int) -> np.ndarray:
    indices = np.floor(np.arange(q)*mask.shape[0]/q).astype(int)
    return mask[np.ix_(indices, indices)]


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return 0.0 if union == 0 else float(np.count_nonzero(a & b)/union)


def run(left: list[Path], right: list[Path], output: Path,
        anchor_step: int, bins: int) -> dict:
    if len(left) != len(right) or not left:
        raise ValueError("equal nonempty checkpoint lists are required")
    left_data = [checkpoint_fields(path) for path in left]
    anchor_candidates = [(fields, meta) for fields, meta in left_data
                         if meta["step"] == anchor_step]
    if len(anchor_candidates) != 1:
        raise ValueError("anchor step must occur exactly once on left")
    anchor_fields, anchor_meta = anchor_candidates[0]
    power = anchor_fields["instantaneous_plastic_power_W_m3"]
    length_m = anchor_meta["spacing_m"]*anchor_meta["n"]
    positive = np.maximum(power, 0.0)
    components = periodic_components(positive > 2.0*float(positive.mean()))
    if not components:
        raise ValueError("anchor contains no threshold component")
    anchor_component = max(components, key=np.count_nonzero)
    geometry = anchor_geometry(power, anchor_component, length_m)
    anchor_xy = tuple(geometry["center_m"])
    trajectories = {"left": [], "right": []}; masks = {"left": [], "right": []}
    for name, paths in (("left", left), ("right", right)):
        for path in paths:
            row, mask = record(path, anchor_xy, geometry, bins)
            row["jaccard_with_prior_sample"] = (None if not masks[name] else
                                                   jaccard(masks[name][-1], mask))
            trajectories[name].append(row); masks[name].append(mask)
    comparisons = []
    for i, (a, b) in enumerate(zip(trajectories["left"], trajectories["right"])):
        if a["step"] != b["step"]:
            raise ValueError("paired steps differ")
        q = np.lcm(masks["left"][i].shape[0], masks["right"][i].shape[0])
        pa = np.asarray(a["profile"]["power_W_m3"]); pb = np.asarray(b["profile"]["power_W_m3"])
        comparisons.append({"step": a["step"], "component_jaccard_common_grid": jaccard(
            nearest_resample(masks["left"][i], q), nearest_resample(masks["right"][i], q)),
            "profile_relative_l2": float(np.linalg.norm(pa-pb)/max(np.linalg.norm(pa),
                                                                    np.linalg.norm(pb), 1e-300))})
    result = {"schema": "asb-drx-v63-fixed-candidate-profile-v1",
              "anchor_step": anchor_step, "anchor_geometry": geometry,
              "selection": "component above twice the positive spatial mean nearest the fixed anchor peak",
              "profile_semantics": "positive power in the tracked threshold component, averaged in fixed normal-coordinate bins within the anchor tangent span",
              "trajectories": trajectories, "paired_comparisons": comparisons,
              "limitations": "threshold masks identify continuity; widths below two native cells are reported but not called resolved"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, nargs="+", required=True)
    parser.add_argument("--right", type=Path, nargs="+", required=True)
    parser.add_argument("--anchor-step", type=int, default=210)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.left, args.right, args.output, args.anchor_step, args.bins)
    print(json.dumps({"sha256": digest(args.output),
                      "comparisons": result["paired_comparisons"]}, sort_keys=True))


if __name__ == "__main__":
    main()
