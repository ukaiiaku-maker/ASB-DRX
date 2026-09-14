#!/usr/bin/env python3
"""Finite-difference the full frozen-state phase functional at v14 caps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PRODUCTION = Path(__file__).resolve().parents[1] / "production"
sys.path.insert(0, str(PRODUCTION))
from moving_front import phase_total_line_densities, state_from_checkpoint  # noqa: E402
from stored_energy_coupling import (  # noqa: E402
    common_variational_stored_energy, signed_pair_pressure_offsets,
)


def _front(state):
    keys = [name.removeprefix("sparse_front__") for name in state.files
            if name.startswith("sparse_front__")]
    return state_from_checkpoint(
        str(state["sparse_front_metadata_json"].item()),
        {key: state[f"sparse_front__{key}"] for key in keys})


def _gb_support(eta, parameters):
    order = np.sort(eta, axis=2)
    largest, second = order[..., -1], order[..., -2]
    soft = np.clip(parameters.get("gb_pair_support_scale", 4.0)
                   * largest * second, 0.0, 1.0)
    labels = np.argmax(eta, axis=2)
    pure = ((largest >= parameters.get("gb_hard_eta_min", 0.65))
            & (second <= parameters.get("gb_hard_second_frac_max", 0.35)
               * np.maximum(largest, 1.0e-30)))
    hard = np.zeros_like(largest)
    for shift in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        edge = ((labels != np.roll(labels, shift, (0, 1))) & pure
                & np.roll(pure, shift, (0, 1)))
        hard = np.maximum(hard, edge)
    support = np.maximum(soft, hard)
    support = np.maximum(support, 0.25 * (
        np.roll(support, 1, 0) + np.roll(support, -1, 0)
        + np.roll(support, 1, 1) + np.roll(support, -1, 1)))
    return np.clip(support, 0.0, 1.0)


def audit(common_path: Path, sub_path: Path, super_path: Path) -> dict:
    with np.load(common_path, allow_pickle=True) as z:
        arrays = {name: np.asarray(z[name]).copy() for name in z.files}
    p = json.loads(str(arrays["P_json"].item()))
    eta = arrays["eta"][..., :int(arrays["Ng"])]
    ngrid = eta.shape[0]
    dx = float(p["L_phys"]) / ngrid
    wave = np.fft.fftfreq(ngrid, d=dx/(2.0*np.pi))
    kx, ky = np.meshgrid(wave, wave, indexing="ij")
    k2 = kx*kx + ky*ky
    def lap(field):
        return np.real(np.fft.ifft2(-k2*np.fft.fft2(field)))
    def derivative(field, direction):
        return np.real(np.fft.ifft2(1j*direction*np.fft.fft2(field)))
    def grad(field):
        return np.hypot(derivative(field, kx), derivative(field, ky))

    front = _front(type("Checkpoint", (), {"files": tuple(arrays),
        "__getitem__": lambda _, key: arrays[key]})())
    parent, child = front.parent_label, front.child_label
    with np.load(sub_path, allow_pickle=True) as sub, np.load(super_path, allow_pickle=True) as sup:
        child_direction = (sup["eta"][..., child] - sub["eta"][..., child])
    pair_interior = ((eta[..., child] > 1.0e-6) & (eta[..., child] < 1.0-1.0e-6)
                     & (eta[..., parent] > 1.0e-6) & (eta[..., parent] < 1.0-1.0e-6))
    child_direction *= pair_interior
    direction = np.zeros_like(eta)
    direction[..., child] = child_direction
    direction[..., parent] = -child_direction
    direction /= np.max(np.abs(direction))

    gb = _gb_support(eta, p)
    nonchild_rho, child_rho = phase_total_line_densities(front)
    mu = np.maximum(float(p.get("Estar_mu0", 82.0e9))
                    + float(p.get("Estar_dmu_dT", -3.7e7))*(arrays["T"]-300.0),
                    float(p.get("Estar_mu_floor", 5.0e9)))
    line_energy = 0.5*mu*float(p["b"])**2
    kappa = np.sum(arrays["rp"]-arrays["rm"], axis=2)
    grad_psi = grad(arrays["psi_lat"])
    compatibility = line_energy*(
        np.abs(np.abs(kappa)-float(p["c_alpha"])*grad_psi/float(p["b"]))
        + np.abs(arrays["rho_GB"]-float(p["c_GB"])*grad_psi/float(p["b"]))) * gb
    rho = np.maximum(arrays["rho"], float(p["rho_min"]))
    structural = np.sum(arrays["rho_forest"], axis=2)+arrays["rho_wall"]
    r = structural/float(p["_rho_state_struct_ref_runtime"])
    r /= max(float(np.nanmedian(r)), 1.0e-12)
    lo, hi = float(p["rho_eta_r_lo"]), float(p["rho_eta_r_hi"])
    x = np.clip((r-lo)/(hi-lo), 0.0, 1.0)
    high_rho = x*x*(3.0-2.0*x)
    signed = np.clip(np.abs(kappa)/rho/float(p["nuc_min_kappa_frac"]), 0.0, 1.0)
    grad_r = grad(r)
    grad_scaled = np.clip(grad_r/max(float(np.nanquantile(grad_r, 0.95)), 1.0e-30), 0.0, 1.0)
    precursor = np.clip(
        float(p["rho_eta_kappa_weight"])*signed
        + float(p["rho_eta_grad_weight"])*grad_scaled
        + float(p["rho_eta_gb_weight"])*gb,
        float(p["rho_eta_precursor_floor"]), 1.0)
    rho_eta_drive = high_rho*precursor

    def phase_energy(applied_pressure):
        energy = np.broadcast_to((line_energy*nonchild_rho)[..., None], eta.shape).copy()
        energy[..., child] = line_energy*child_rho
        applied = applied_pressure*gb
        parent_offset, child_offset = signed_pair_pressure_offsets(applied)
        energy += parent_offset[..., None]
        energy[..., child] += compatibility+child_offset-parent_offset
        return energy

    def functional(fields, pressure):
        stored, _ = common_variational_stored_energy(fields, phase_energy(pressure))
        gradient = sum(0.5*float(p["kappa_eta"])
                       * np.sum(fields[..., i]*(-lap(fields[..., i])))
                       for i in range(fields.shape[2]))
        sum2 = np.sum(fields*fields, axis=2)
        barrier = 0.5*float(p["W_eta"])*np.sum(
            sum2*sum2-np.sum(fields**4, axis=2))
        coupling = -float(p["rho_eta_ac_strength"])*np.sum(
            rho_eta_drive[..., None]*fields*(1.0-fields))
        return float((gradient+barrier+np.sum(stored)+coupling)*dx*dx)

    def analytical(pressure):
        _, stored_derivative = common_variational_stored_energy(
            eta, phase_energy(pressure))
        sum2 = np.sum(eta*eta, axis=2)
        derivative_field = np.empty_like(eta)
        for i in range(eta.shape[2]):
            derivative_field[..., i] = (
                -float(p["kappa_eta"])*lap(eta[..., i])
                + 2.0*float(p["W_eta"])*eta[..., i]
                *(sum2-eta[..., i]**2) + stored_derivative[..., i]
                - float(p["rho_eta_ac_strength"])*rho_eta_drive
                *(1.0-2.0*eta[..., i]))
        return float(np.sum(derivative_field*direction)*dx*dx)

    epsilon = 1.0e-6
    def finite_difference(pressure):
        return ((functional(eta+epsilon*direction, pressure)
                 - functional(eta-epsilon*direction, pressure))/(2.0*epsilon))
    a0, a1 = analytical(0.0), analytical(1.0e6)
    f0, f1 = finite_difference(0.0), finite_difference(1.0e6)
    critical_a = -a0*1.0e6/(a1-a0)
    critical_f = -f0*1.0e6/(f1-f0)
    derivative_error = max(abs(a0-f0), abs(a1-f1))/max(abs(a0), abs(a1), 1.0e-30)
    passed = (derivative_error <= 2.0e-6 and -4.5e6 < critical_f < 5.5e6)
    return {
        "grid": ngrid, "direction": "supercritical-minus-subcritical pair-field mode",
        "perturbation": epsilon, "analytical_derivative_at_0Pa_J": a0,
        "finite_difference_derivative_at_0Pa_J": f0,
        "analytical_derivative_at_1MPa_J": a1,
        "finite_difference_derivative_at_1MPa_J": f1,
        "maximum_relative_derivative_error": derivative_error,
        "analytical_diffuse_critical_pressure_Pa": critical_a,
        "finite_difference_diffuse_critical_pressure_Pa": critical_f,
        "tested_motion_bracket_Pa": [-4.5e6, 5.5e6],
        "critical_inside_observed_sign_bracket": bool(-4.5e6 < critical_f < 5.5e6),
        "passed": bool(passed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--grid", action="append", nargs=4, metavar=("COMMON", "SUB", "SUPER", "LABEL"), required=True)
    args = parser.parse_args()
    results = [audit(Path(common), Path(sub), Path(sup))
               for common, sub, sup, _ in args.grid]
    passed = all(item["passed"] for item in results)
    payload = {"schema": "full-v34-v14-full-functional-fd/v1",
               "classification": ("V14_FULL_FUNCTIONAL_FD_PASSED" if passed
                                  else "V14_FULL_FUNCTIONAL_FD_FAILED"),
               "results": results, "passed": passed}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(payload["classification"])


if __name__ == "__main__":
    main()
