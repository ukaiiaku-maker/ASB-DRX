"""Physical grid-scaling and work/heat audits for V25 ASB decisions."""

from __future__ import annotations

import numpy as np


def manufactured_periodic_kernel(n, length_m, sigma_m, *, source_fraction=(.5, .5),
                                 minimum_sigma_pixels=0.0):
    """Return a cell-integrated, periodic Gaussian deposition kernel.

    The array sums to one, so multiplying it by an energy increment deposits
    exactly that energy.  Coordinates use minimum-image distances from a
    source at the same physical fractional location on every grid.
    """
    n = int(n); length_m = float(length_m); sigma_m = float(sigma_m)
    if n < 4 or length_m <= 0.0 or sigma_m < 0.0:
        raise ValueError("kernel grid, length, and width are invalid")
    dx = length_m/n
    sigma_eff = max(sigma_m, float(minimum_sigma_pixels)*dx)
    if sigma_eff <= 0.0:
        raise ValueError("effective kernel width must be positive")
    sx, sy = (float(source_fraction[0])*length_m,
              float(source_fraction[1])*length_m)
    x = (np.arange(n)+.5)*dx
    rx = (x-sx+.5*length_m) % length_m-.5*length_m
    ry = (x-sy+.5*length_m) % length_m-.5*length_m
    radius2 = rx[:, None]**2+ry[None, :]**2
    kernel = np.exp(-.5*radius2/sigma_eff**2)
    kernel /= np.sum(kernel, dtype=np.longdouble)
    return kernel, rx[:, None], ry[None, :], sigma_eff


def kernel_moment_audit(kernel, rx_m, ry_m, cell_size_m):
    """Measure normalization, centroid, covariance, peak, and transfer data."""
    kernel = np.asarray(kernel, float)
    rx = np.broadcast_to(np.asarray(rx_m, float), kernel.shape)
    ry = np.broadcast_to(np.asarray(ry_m, float), kernel.shape)
    if kernel.shape != rx.shape or kernel.shape != ry.shape:
        raise ValueError("kernel coordinates must be grid matched")
    mass = float(np.sum(kernel, dtype=np.longdouble))
    cx = float(np.sum(kernel*rx, dtype=np.longdouble))/mass
    cy = float(np.sum(kernel*ry, dtype=np.longdouble))/mass
    second = float(np.sum(kernel*((rx-cx)**2+(ry-cy)**2),
                          dtype=np.longdouble))/mass
    transfer = np.abs(np.fft.rfft2(kernel))
    transfer /= max(float(transfer[0, 0]), 1e-300)
    return {
        "normalization": mass,
        "centroid_m": [cx, cy],
        "radial_second_moment_m2": second,
        "peak_cell_fraction": float(np.max(kernel)),
        "cell_size_m": float(cell_size_m),
        "fourier_axis_transfer": transfer[:, 0].tolist(),
    }


def compare_kernel_audits(coarse, fine, *, moment_tolerance=.05,
                          peak_tolerance=.10, transfer_tolerance=.10):
    """Compare physical moments and common Fourier modes across two grids."""
    cn = np.asarray(coarse["fourier_axis_transfer"], float)
    fn = np.asarray(fine["fourier_axis_transfer"], float)
    # The first axis is a full FFT axis.  Compare only its nonnegative branch;
    # indices beyond the coarse Nyquist represent wrapped negative modes and
    # do not correspond to the same signed wavenumber on unequal grids.
    common = min(cn.size, fn.size)//2+1
    # Same physical domain gives the same mode wavenumbers for equal indices.
    transfer_error = float(np.max(np.abs(cn[:common]-fn[:common])))
    moment_error = relative_difference(
        coarse["radial_second_moment_m2"],
        fine["radial_second_moment_m2"])
    peak_error = relative_difference(
        coarse["peak_cell_fraction"]/(coarse["cell_size_m"]**2),
        fine["peak_cell_fraction"]/(fine["cell_size_m"]**2))
    passed = (abs(coarse["normalization"]-1.0) <= 1e-12
              and abs(fine["normalization"]-1.0) <= 1e-12
              and moment_error <= moment_tolerance
              and peak_error <= peak_tolerance
              and transfer_error <= transfer_tolerance)
    return {
        "second_moment_relative_difference": moment_error,
        "physical_peak_relative_difference": peak_error,
        "common_mode_max_transfer_difference": transfer_error,
        "passed": bool(passed),
    }


def physical_scales(parameters):
    p = dict(parameters)
    n = int(p["Nx"]); length = float(p["L_phys"]); dx = length/n
    intrinsic_interface = float(np.sqrt(float(p["kappa_eta"])/float(p["W_eta"])))
    heat_sigma = float(p.get("heat_process_zone_sigma_um", 0.0))*1e-6
    heat_min_pixels = float(p.get("heat_process_zone_min_sigma_px", 0.0))
    return {
        "grid": n,
        "domain_size_m": length,
        "cell_size_m": dx,
        "phase_interface_intrinsic_m": intrinsic_interface,
        "phase_interface_resolved_m": max(intrinsic_interface, dx),
        "heat_kernel_requested_m": heat_sigma,
        "heat_kernel_effective_m": max(heat_sigma, heat_min_pixels*dx),
        "thermal_diffusivity_m2_s": float(p["k_thermal"])/float(p["cp_rho_vol"]),
        "heat_capacity_J_m3_K": float(p["cp_rho_vol"]),
        "conductivity_W_m_K": float(p["k_thermal"]),
        "mechanical_correlation_length_m": float(
            p.get("collective_activity_smooth_um", 0.0))*1e-6,
        "gb_transmission_width_m": (None if p.get("gb_band_um") is None
                                    else float(p["gb_band_um"])*1e-6),
        "hotspot_process_zone_width_m": heat_sigma,
        "represented_thickness_m": float(
            p.get("represented_thickness_m",
                  float(p.get("nuc_barrier_thickness_b", 2.0))*float(p["b"]))),
        "cell_area_m2": dx*dx,
        "cell_volume_m3": dx*dx*float(
            p.get("represented_thickness_m",
                  float(p.get("nuc_barrier_thickness_b", 2.0))*float(p["b"]))),
        "noise_spectrum_declared": p.get("initial_noise_spectrum"),
        "noise_amplitude_declared": p.get("initial_noise_amplitude"),
        "classifier_width_declared_m": (None if p.get("asb_classifier_width_um") is None
                                        else float(p["asb_classifier_width_um"])*1e-6),
    }


def relative_difference(a, b):
    return abs(float(a)-float(b))/max(abs(float(a)), abs(float(b)), 1e-300)


def compare_physical_scales(coarse, fine, tolerance=.05):
    keys = (
        "domain_size_m", "phase_interface_intrinsic_m",
        "phase_interface_resolved_m", "heat_kernel_requested_m",
        "heat_kernel_effective_m", "thermal_diffusivity_m2_s",
        "heat_capacity_J_m3_K", "conductivity_W_m_K",
        "mechanical_correlation_length_m", "hotspot_process_zone_width_m",
        "represented_thickness_m",
    )
    comparisons = {}
    for key in keys:
        if coarse[key] is None or fine[key] is None:
            comparisons[key] = {"declared_on_both_grids": False}
        else:
            difference = relative_difference(coarse[key], fine[key])
            comparisons[key] = {
                "coarse": coarse[key], "fine": fine[key],
                "relative_difference": difference,
                "passed": bool(difference <= tolerance),
            }
    return comparisons


def restrict_categorical_fine_to_coarse(fine_labels, coarse_shape):
    fine = np.asarray(fine_labels)
    nx, ny = map(int, coarse_shape)
    if fine.shape[0] % nx or fine.shape[1] % ny:
        raise ValueError("fine categorical grid must be an integer refinement")
    rx, ry = fine.shape[0]//nx, fine.shape[1]//ny
    blocks = fine.reshape(nx, rx, ny, ry).transpose(0, 2, 1, 3).reshape(nx, ny, -1)
    result = np.empty((nx, ny), dtype=fine.dtype)
    for i in range(nx):
        for j in range(ny):
            values, counts = np.unique(blocks[i, j], return_counts=True)
            result[i, j] = values[np.argmax(counts)]
    return result


def categorical_restriction_mismatch(coarse_labels, fine_labels):
    coarse = np.asarray(coarse_labels)
    restricted = restrict_categorical_fine_to_coarse(fine_labels, coarse.shape)
    # Compare partitions, not arbitrary numeric labels: equality of same-grain
    # relations to right/up neighbors is invariant under label permutation.
    def edges(field):
        return np.stack((field == np.roll(field, -1, axis=0),
                         field == np.roll(field, -1, axis=1)), axis=-1)
    mismatch = float(np.mean(edges(coarse) != edges(restricted)))
    return {"edge_relation_mismatch_fraction": mismatch,
            "restriction_consistent": bool(mismatch <= .05)}


def integrate_work_heat(history, edot_app_s, cp_J_m3_K, t0_K,
                        start_s=None, end_s=None):
    """Integrate volume densities over a declared common physical interval."""
    t = np.asarray(history["time_s"], float)
    mask = np.ones(t.shape, bool)
    if start_s is not None: mask &= t >= float(start_s)
    if end_s is not None: mask &= t <= float(end_s)
    if np.count_nonzero(mask) < 2:
        raise ValueError("work/heat audit needs at least two samples")
    t = t[mask]
    stress = np.asarray(history["stress_Pa"], float)[mask]
    plastic = np.asarray(history["plastic_power_W_m3"], float)[mask]
    heat = np.asarray(history["deposited_heat_W_m3"], float)[mask]
    temperature = np.asarray(history["mean_temperature_K"], float)[mask]
    trap = getattr(np, "trapezoid", np.trapz)
    return {
        "start_time_s": float(t[0]), "end_time_s": float(t[-1]),
        "external_work_J_m3": float(trap(stress*float(edot_app_s), t)),
        "plastic_work_J_m3": float(trap(plastic, t)),
        "deposited_heat_J_m3": float(trap(heat, t)),
        "stored_thermal_energy_change_J_m3": float(
            cp_J_m3_K*(temperature[-1]-temperature[0])),
        "final_temperature_excess_from_T0_K": float(temperature[-1]-t0_K),
    }


def first_law_budget(*, external_work_J_m3, elastic_change_J_m3,
                     plastic_work_J_m3, defect_free_energy_change_J_m3,
                     taylor_quinney_heat_J_m3, conducted_out_J_m3,
                     thermal_energy_change_J_m3, other_dissipation_J_m3=0.0):
    """Return an explicit mechanical/thermal ledger without hiding residuals.

    Plastic work is reported as a diagnostic intermediate.  It is partitioned
    into defect storage, Taylor--Quinney heat, and other dissipation; the final
    system first law then accounts for elastic/defect/thermal storage and heat
    conducted out.
    """
    values = {name: float(value) for name, value in locals().items()}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("first-law terms must be finite")
    plastic_partition_residual = (
        values["plastic_work_J_m3"]
        -values["defect_free_energy_change_J_m3"]
        -values["taylor_quinney_heat_J_m3"]
        -values["other_dissipation_J_m3"])
    system_residual = (
        values["external_work_J_m3"]
        -values["elastic_change_J_m3"]
        -values["defect_free_energy_change_J_m3"]
        -values["thermal_energy_change_J_m3"]
        -values["conducted_out_J_m3"]
        -values["other_dissipation_J_m3"])
    scale = max(*(abs(x) for x in values.values()), 1e-300)
    return {
        **values,
        "plastic_partition_residual_J_m3": plastic_partition_residual,
        "system_residual_J_m3": system_residual,
        "plastic_partition_relative_residual": abs(plastic_partition_residual)/scale,
        "system_relative_residual": abs(system_residual)/scale,
    }
