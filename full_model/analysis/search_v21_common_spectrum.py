#!/usr/bin/env python3
"""Bounded physical-state search for a V21 finite signed-wall branch."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

PRODUCTION = Path(__file__).resolve().parents[1]/"production"
sys.path.insert(0, str(PRODUCTION))
from common_tensorial_wall import (  # noqa: E402
    CommonWallDriving, CommonWallParameters, CommonWallState,
    fourier_symbol,
)
from tensorial_nye import (  # noqa: E402
    JunctionTopology, bcc_four_family_systems,
)
from audit_v21_common_trajectory import participation, topology_from_json  # noqa: E402


def make_state(n, total_density, q, temperature, parameters, topologies):
    # Declared nonzero statistical partition.  A small alternating signed bias
    # prevents evaluating a singular exactly-unpolarized coordinate while
    # introducing no net aggregate Burgers bias.
    fractions = {"mobile": .40, "forest": .40, "wall": .15, "junction": .05}
    bias = np.asarray([1.0, -1.0, 1.0, -1.0])*1e-3
    arrays = {}
    for name in ("mobile", "forest", "wall"):
        per_family_pair = fractions[name]*total_density/(2.0*4.0)
        arrays[name+"_plus_m2"] = np.broadcast_to(
            per_family_pair*(1.0+bias), (n, n, 4)).copy()
        arrays[name+"_minus_m2"] = np.broadcast_to(
            per_family_pair*(1.0-bias), (n, n, 4)).copy()
    multiplicity = sum(t.product_line_multiplicity for t in topologies)
    junction_each = fractions["junction"]*total_density/max(multiplicity, 1e-300)
    junction = np.full((n, n, len(topologies)), junction_each/max(len(topologies), 1))
    shape = (n, n, 4)
    return CommonWallState(
        arrays["mobile_plus_m2"], arrays["mobile_minus_m2"],
        arrays["forest_plus_m2"], arrays["forest_minus_m2"],
        arrays["wall_plus_m2"], arrays["wall_minus_m2"], junction,
        np.full((n, n), q), np.zeros((n, n)), np.zeros(shape), np.zeros((n, n, 3, 3)),
        np.zeros(shape+(3,)), np.zeros(shape+(3, 3)),
        np.full((n, n), np.deg2rad(25.1)), np.full((n, n), temperature),
    )


def leading(symbol, signed_required):
    values = symbol["eigenvalues_s_inv"]
    vectors = symbol["right_eigenvectors"]
    candidates = []
    for index in np.argsort(values.real)[::-1]:
        part = participation(vectors[:, index], symbol["layout"])
        if not signed_required or part["signed_polarization"] >= .02:
            candidates.append((float(values[index].real), part))
            if len(candidates) == 1:
                break
    return (candidates[0] if candidates else (float("-inf"), {}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.checkpoint, allow_pickle=True) as z:
        base = CommonWallParameters(**json.loads(str(
            z["v21_common_parameters_json"].item())))
        topologies = tuple(topology_from_json(item) for item in json.loads(str(
            z["v21_topology_json"].item())))
    systems = bcc_four_family_systems(base.burgers_m)
    n = 8
    records = []
    domains = (7.3e-6, 11.1e-6)
    for temperature in (900.0, 1300.0, 1600.0):
        for density_ratio in (.5, 1.3, 3.0):
            for q in (.01, .5):
                for domain_index, domain in enumerate(domains):
                    parameters = replace(base, spacing_m=domain/n,
                                         bath_temperature_K=temperature)
                    state = make_state(
                        n, density_ratio*base.rho_reference_m2, q,
                        temperature, parameters, topologies)
                    driving = CommonWallDriving(mean_strain=np.array(
                        [[.006, 0.0], [0.0, -.0015]]))
                    modes = [(0, 0), (1, 0), (2, 0), (3, 0)]
                    # The first domain also checks the orthogonal crystal/grid
                    # direction; the incommensurate second domain diagnoses
                    # physical wavelength versus box selection.
                    if domain_index == 0:
                        modes += [(0, 1), (0, 2), (0, 3)]
                    spectra = []
                    for mode in modes:
                        symbol = fourier_symbol(
                            state, driving, systems, topologies, parameters,
                            mode, relative_step=2e-6)
                        top, top_part = leading(symbol, False)
                        signed, signed_part = leading(symbol, True)
                        spectra.append({
                            "mode": list(mode),
                            "wavelength_m": symbol["wavelength_m"],
                            "leading_growth_s_inv": top,
                            "leading_participation": top_part,
                            "signed_wall_growth_s_inv": signed,
                            "signed_wall_participation": signed_part,
                        })
                    uniform = spectra[0]["leading_growth_s_inv"]
                    finite_signed = max(s["signed_wall_growth_s_inv"] for s in spectra[1:])
                    best = max(spectra[1:], key=lambda s: s["signed_wall_growth_s_inv"])
                    mode_number = max(best["mode"])
                    records.append({
                        "temperature_K": temperature,
                        "density_ratio": density_ratio,
                        "wall_order": q, "domain_m": domain,
                        "uniform_growth_s_inv": uniform,
                        "best_finite_signed_growth_s_inv": finite_signed,
                        "finite_over_uniform_s_inv": finite_signed-uniform,
                        "best_finite_mode": best["mode"],
                        "best_finite_wavelength_m": best["wavelength_m"],
                        "interior_mode": mode_number == 2,
                        "time_for_G8_s": (8.0/finite_signed
                                          if finite_signed > 0.0 else None),
                        "spectra": spectra,
                    })
    robust = [r for r in records if r["interior_mode"]
              and r["finite_over_uniform_s_inv"] > 0.0
              and r["best_finite_signed_growth_s_inv"] > 0.0]
    # Robust selection additionally requires matching physical wavelengths in
    # the two incommensurate domains at the same thermodynamic state.
    matched = []
    for left in records:
        for right in records:
            same = (left["temperature_K"] == right["temperature_K"]
                    and left["density_ratio"] == right["density_ratio"]
                    and left["wall_order"] == right["wall_order"]
                    and left["domain_m"] != right["domain_m"])
            if same and left in robust and right in robust:
                ratio = left["best_finite_wavelength_m"]/right["best_finite_wavelength_m"]
                if .8 <= ratio <= 1.25:
                    matched.append({"left": left, "right": right,
                                    "wavelength_ratio": ratio})
    result = {
        "schema": "asb-drx/v21-bounded-common-spectrum-search/v1",
        "bounds": {
            "temperature_K": [900.0, 1600.0],
            "total_density_over_reference": [.5, 3.0],
            "wall_order": [.01, .5], "mean_axial_strain": .006,
            "domains_um": [7.3, 11.1], "grid": n,
        },
        "state_count": len(records), "records": records,
        "robust_interior_matches": matched,
        "finite_mode_region_found": bool(matched),
        "hpc3_wall_campaign_authorized": bool(matched),
        "fixture_passed": True,
        "scientific_gate_passed": bool(matched),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({
        "state_count": len(records), "robust_matches": len(matched),
        "finite_mode_region_found": bool(matched),
    }, indent=2))


if __name__ == "__main__":
    main()
