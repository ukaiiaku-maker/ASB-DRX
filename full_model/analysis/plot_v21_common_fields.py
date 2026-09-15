#!/usr/bin/env python3
"""Render decision fields and signed-wall structure factor for V21."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRODUCTION = Path(__file__).resolve().parents[1]/"production"
sys.path.insert(0, str(PRODUCTION))
from common_tensorial_wall import (  # noqa: E402
    CommonWallDriving, CommonWallParameters, CommonWallState,
    resolved_driving_fields,
)
from tensorial_nye import bcc_four_family_systems  # noqa: E402
from audit_v21_common_trajectory import fixed_eigenstrain, topology_from_json  # noqa: E402


def render(checkpoint, output, title):
    with np.load(checkpoint, allow_pickle=True) as z:
        p = json.loads(str(z["P_json"].item()))
        parameters = CommonWallParameters(**json.loads(str(
            z["v21_common_parameters_json"].item())))
        topologies = tuple(topology_from_json(item) for item in json.loads(str(
            z["v21_topology_json"].item())))
        systems = bcc_four_family_systems(parameters.burgers_m)
        state = CommonWallState(
            z["rp"], z["rm"], z["rho_forest_plus"], z["rho_forest_minus"],
            z["rho_wall_plus"], z["rho_wall_minus"], z["v21_junction_m2"],
            z["q_wall_v19"], z["v20_slip"], z["v20_beta_p"],
            z["v20_alignment_m2"], z["v20_family_nye_m1"], z["psi_lat"], z["T"])
        c11, c12 = float(p["C11"]), float(p["C12"])
        effective = (c11*c11-c12*c12)/c11; poisson = c12/c11
        beta2 = state.beta_p[..., :2, :2]
        epsp = .5*(beta2+np.swapaxes(beta2, -1, -2)); epmean = epsp.mean((0, 1))
        stress = float(z["sigma_bar"])
        mean_strain = np.array([
            [stress/effective+epmean[0, 0], epmean[0, 1]],
            [epmean[0, 1], -poisson*stress/effective+epmean[1, 1]]])
        _, resolved = resolved_driving_fields(
            state, CommonWallDriving(
                mean_strain=mean_strain,
                fixed_eigenstrain=fixed_eigenstrain(p, state.wall_order.shape)),
            systems, parameters)
        total = np.asarray(z["rho"])
        wall_signed_family = state.wall_plus_m2-state.wall_minus_m2
        wall_signed = np.sum(wall_signed_family, axis=2)
        nye = np.sum(state.family_nye_m1, axis=2)
        nye_norm = np.linalg.norm(nye, axis=(-2, -1))
        slip = np.sum(np.abs(state.slip), axis=2)
        orientation = np.rad2deg(state.orientation_rad-state.orientation_rad.mean())
        power = np.abs(np.fft.fftshift(np.fft.fft2(wall_signed-wall_signed.mean())))**2
        panels = (
            (total/1e14, "total density", r"$10^{14}$ m$^{-2}$"),
            (wall_signed/1e10, "signed wall density", r"$10^{10}$ m$^{-2}$"),
            (wall_signed_family[..., 0]/1e10, "family-0 signed wall", r"$10^{10}$ m$^{-2}$"),
            (nye_norm/1e5, "tensorial Nye norm", r"$10^5$ m$^{-1}$"),
            (slip, "accumulated |slip|", ""),
            (orientation, "orientation relative to mean", "degree"),
            (np.max(np.abs(resolved), axis=2)/1e6, "max resolved internal stress", "MPa"),
            (np.log10(power+1.0), "signed-wall structure factor", "log10 power"),
        )
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for axis, (field, name, unit) in zip(axes.flat, panels):
        image = axis.imshow(field.T, origin="lower", cmap="coolwarm", interpolation="nearest")
        axis.set_title(name); axis.set_xticks([]); axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=.75, label=unit)
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    render(args.checkpoint, args.output, args.title)


if __name__ == "__main__":
    main()
