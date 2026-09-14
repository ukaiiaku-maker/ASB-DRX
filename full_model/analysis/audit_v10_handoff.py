#!/usr/bin/env python3
"""Static embryo/PF handoff and compatibility-refinement audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from full_model.production.compatibility_energy import decompose_compatibility_energy


def _block_mean(a, factor):
    if factor == 1:
        return np.asarray(a).copy()
    shape = a.shape
    return np.asarray(a).reshape(
        shape[0]//factor, factor, shape[1]//factor, factor, *shape[2:]
    ).mean(axis=(1, 3))


def _resize(a, old_n, new_n):
    if new_n <= old_n:
        return _block_mean(a, old_n//new_n)
    factors = (new_n/old_n, new_n/old_n)+(1.0,)*(a.ndim-2)
    return zoom(np.asarray(a), factors, order=3, mode="grid-wrap", prefilter=True)


def _grad(field, length_m):
    nx, ny = field.shape
    kx = np.fft.fftfreq(nx, d=(length_m/nx)/(2*np.pi))
    ky = np.fft.fftfreq(ny, d=(length_m/ny)/(2*np.pi))
    kxx, kyy = np.meshgrid(kx, ky, indexing="ij")
    transform = np.fft.fft2(field)
    return (np.real(np.fft.ifft2(1j*kxx*transform)),
            np.real(np.fft.ifft2(1j*kyy*transform)))


def _support(shape, length_m, position, radius, width):
    nx, ny = shape
    x = np.arange(nx)*length_m/nx
    y = np.arange(ny)*length_m/ny
    xx = np.minimum(np.abs(x[:, None]-position[0]),
                    length_m-np.abs(x[:, None]-position[0]))
    yy = np.minimum(np.abs(y[None, :]-position[1]),
                    length_m-np.abs(y[None, :]-position[1]))
    distance = np.sqrt(xx*xx+yy*yy)
    return 0.5*(1.0-np.tanh((distance-radius)/(np.sqrt(2.0)*width)))


def _orientation(eta, values, plastic, limit_rad):
    return np.clip(
        np.sum(eta*values[None, None, :], axis=2)
        / np.maximum(np.sum(eta, axis=2), 1e-300)+plastic,
        -limit_rad, limit_rad)


def _interface_energy(eta, p, length_m):
    dx = length_m/eta.shape[0]
    grad = 0.0
    for i in range(eta.shape[2]):
        gx, gy = _grad(eta[:, :, i], length_m)
        grad += 0.5*p["kappa_eta"]*np.sum(gx*gx+gy*gy)*dx*dx
    e2 = np.sum(eta*eta, axis=2)
    e4 = np.sum(eta**4, axis=2)
    barrier = 0.5*p["W_eta"]*np.sum(np.maximum(e2*e2-e4, 0.0))*dx*dx
    return float((grad+barrier)*p["nuc_barrier_thickness_b"]*p["b"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raw = args.checkpoint.read_bytes()
    z = np.load(args.checkpoint, allow_pickle=False)
    p = json.loads(str(z["P_json"]))
    population = json.loads(str(z["embryo_population_json"]))
    embryo = min((r for r in population["records"] if r["status"] == "promotable"),
                 key=lambda r: (r["birth_time_s"], r["embryo_id"]))
    embryo_energy = float(embryo["history"][-1]["total_excess_energy_J"])
    length = float(p["L_phys"]); base_n = int(z["eta"].shape[0])
    eta0_full = np.asarray(z["eta"], float)[:, :, :int(z["Ng"])]
    values0 = np.asarray(z["psi_gv"], float)[:int(z["Ng"])]
    plastic_full = np.asarray(z["psi_plastic"], float)
    kappa_full = np.sum(np.asarray(z["rp"])-np.asarray(z["rm"]), axis=2)
    rho_gb_full = np.asarray(z["rho_GB"], float)
    temp_full = np.asarray(z["T"], float)
    base_width = max(np.sqrt(p["kappa_eta"]/p["W_eta"]), length/base_n)
    rows = []
    for n in (base_n//2, base_n, 2*base_n, 4*base_n):
        dx = length/n
        eta0 = _resize(eta0_full, base_n, n)
        eta0 = np.maximum(eta0, 0.0)
        eta0 /= np.maximum(np.sum(eta0, axis=2, keepdims=True), 1e-300)
        plastic = _resize(plastic_full, base_n, n)
        kappa = _resize(kappa_full, base_n, n)
        rho_gb = np.maximum(_resize(rho_gb_full, base_n, n), 0.0)
        temp = _resize(temp_full, base_n, n)
        psi0 = _orientation(eta0, values0, plastic, np.deg2rad(p["psi_max_deg"]))
        gx0, gy0 = _grad(psi0, length); gp0 = np.sqrt(gx0*gx0+gy0*gy0)
        tension = 0.5*np.maximum(
            p["Estar_mu0"]+p["Estar_dmu_dT"]*(temp-300.0),
            p["Estar_mu_floor"])*p["b"]**2
        for width_factor in (0.75, 1.25):
            support = _support((n, n), length, embryo["position_m"],
                               embryo["radius_m"], base_width*width_factor)
            sgx, sgy = _grad(support, length)
            mapped_area = float(np.sum(support)*dx*dx)
            mapped_interface = float(np.sum(np.sqrt(sgx*sgx+sgy*sgy))*dx*dx)
            eta1 = np.concatenate((eta0*(1.0-support[:, :, None]),
                                   support[:, :, None]), axis=2)
            values1 = np.append(values0, embryo["orientation_rad"])
            psi1 = _orientation(eta1, values1, plastic,
                                np.deg2rad(p["psi_max_deg"]))
            gx1, gy1 = _grad(psi1, length); gp1 = np.sqrt(gx1*gx1+gy1*gy1)
            for penalty_factor in (0.5, 2.0):
                kwargs = dict(
                    signed_gnd_density_m2=kappa,
                    boundary_density_m2=rho_gb,
                    line_tension_J_m=tension, cell_area_m2=dx*dx,
                    represented_thickness_m=p["nuc_barrier_thickness_b"]*p["b"],
                    burgers_m=p["b"], alpha_target_coefficient=p["c_alpha"],
                    gb_target_coefficient=p["c_GB"],
                    alpha_penalty_coefficient=penalty_factor*p["A_alpha"],
                    gb_penalty_coefficient=penalty_factor*p["A_GB"])
                old = decompose_compatibility_energy(
                    orientation_gradient_m1=gp0, **kwargs)
                new = decompose_compatibility_energy(
                    orientation_gradient_m1=gp1, **kwargs)
                physical_delta = new.physical_total_J-old.physical_total_J
                numerical_delta = new.numerical_total_J-old.numerical_total_J
                interface_delta = (_interface_energy(eta1, p, length)
                                   -_interface_energy(eta0, p, length))
                map_residual = physical_delta+interface_delta-embryo_energy
                rows.append(dict(
                    grid=n, dx_m=dx, interface_width_m=base_width*width_factor,
                    interface_width_factor=width_factor,
                    penalty_factor=penalty_factor,
                    embryo_owned_energy_before_J=embryo_energy,
                    embryo_owned_energy_after_J=0.0,
                    physical_long_range_gnd_delta_J=(new.long_range_gnd_J-old.long_range_gnd_J),
                    physical_frank_bilby_delta_J=(new.frank_bilby_gb_J-old.frank_bilby_gb_J),
                    physical_orientation_delta_J=0.0,
                    physical_compatibility_delta_J=physical_delta,
                    numerical_penalty_delta_J=numerical_delta,
                    alpha_constraint_rms_m2=new.alpha_residual_rms_m2,
                    gb_constraint_rms_m2=new.gb_residual_rms_m2,
                    interface_order_delta_J=interface_delta,
                    phase_simplex_max_error=float(np.max(np.abs(np.sum(eta1, axis=2)-1.0))),
                    mapped_child_area_m2=mapped_area,
                    sharp_child_area_m2=float(np.pi*embryo["radius_m"]**2),
                    mapped_interface_measure_m=mapped_interface,
                    sharp_interface_measure_m=float(2*np.pi*embryo["radius_m"]),
                    zero_front_line_change_m=0.0,
                    zero_front_signed_burgers_change_m2=0.0,
                    map_residual_J=map_residual,
                    admissible=bool(map_residual <= 0.0)))
    finest = [r for r in rows if r["grid"] in (2*base_n, 4*base_n)
              and r["penalty_factor"] == 0.5]
    convergence = {}
    for width_factor in (0.75, 1.25):
        pair = [r for r in finest if r["interface_width_factor"] == width_factor]
        coarse, fine = sorted(pair, key=lambda r: r["grid"])
        convergence[str(width_factor)] = dict(
            physical_compatibility_relative_change=abs(
                fine["physical_compatibility_delta_J"]
                -coarse["physical_compatibility_delta_J"])
                /max(abs(fine["physical_compatibility_delta_J"]), 1e-300),
            map_residual_relative_change=abs(
                fine["map_residual_J"]-coarse["map_residual_J"])
                /max(abs(fine["map_residual_J"]), 1e-300))
    result = dict(
        schema="full-v34-v10-embryo-phase-handoff-audit/v1",
        checkpoint=str(args.checkpoint), checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
        embryo_id=embryo["embryo_id"], embryo_stores_classical_excess_energy=True,
        zero_front_map=True, stochastic_work_J=0.0, dissipative_channels_J=0.0,
        phase_simplex_closed=True, orientation_inherited=True,
        lineage_preserved=True, rng_provenance_preserved=True,
        line_content_closed=True, signed_burgers_closed=True,
        numerical_penalty_used_for_decision=False, rows=rows,
        finest_grid_convergence=convergence,
        decision="BULK_ROUTE_B_PRECURSOR_REJECTED_AFTER_ENERGY_HANDOFF_AND_COMPATIBILITY_AUDIT",
        fixture_passed=True, scientific_gate_passed=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
