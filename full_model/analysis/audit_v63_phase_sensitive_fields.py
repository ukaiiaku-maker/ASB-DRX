#!/usr/bin/env python3
"""Phase-sensitive, owner-preserving comparison of V58/V62 checkpoints.

The two grids share the periodic coordinate origin and physical domain.  The
comparison therefore uses their normalized complex Fourier coefficients
directly; it never optimizes a translation.  Scalar radial-power distances are
retained only as secondary descriptors.  A band-limited reconstruction on a
declared common quadrature grid supplies a checked physical-space comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from full_model.analysis.postprocess_v59_physical_asb import _wall_parameters
from full_model.analysis.run_v58_three_grain_production import _load_checkpoint
from full_model.production.common_tensorial_wall import (
    CommonWallDriving, resolved_driving_components, wall_residual,
)
from full_model.production.multigrain_common_state import (
    audit_multigrain_nye, reconstruct_multigrain_common,
)
from full_model.production.multigrain_production import (
    _owner_drivings_from_common_stress,
    multigrain_instantaneous_dissipation_fields,
)
from full_model.production.tensorial_nye import (
    bcc_four_family_systems, nye_from_plastic_distortion,
)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_coefficients(field: np.ndarray) -> np.ndarray:
    """Return coefficients on the source's node-centered periodic origin."""
    value = np.asarray(field, dtype=float)
    return np.fft.fftshift(
        np.fft.fftn(value, axes=(0, 1))/(value.shape[0]*value.shape[1]),
        axes=(0, 1))


def centered_band(coefficients: np.ndarray, half_width: int) -> np.ndarray:
    nx, ny = coefficients.shape[:2]
    if not (0 <= half_width < min(nx, ny)//2):
        raise ValueError("half width must exclude both Nyquist lines")
    cx, cy = nx//2, ny//2
    return coefficients[cx-half_width:cx+half_width+1,
                        cy-half_width:cy+half_width+1]


def norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value).ravel()))


def rms(value: np.ndarray) -> float:
    array = np.asarray(value)
    return float(np.sqrt(np.mean(np.abs(array)**2)))


def remove_spatial_mean(field: np.ndarray) -> np.ndarray:
    value = np.asarray(field, dtype=float)
    return value-np.mean(value, axis=(0, 1), keepdims=True)


def tail_power_fraction(coefficients: np.ndarray, half_width: int) -> float:
    total = norm(coefficients)**2
    retained = norm(centered_band(coefficients, half_width))**2
    return float(max(total-retained, 0.0)/max(total, 1e-300))


def radial_power_distribution(coefficients: np.ndarray) -> np.ndarray:
    n = coefficients.shape[0]
    center = n//2
    x = np.arange(n)-center
    radius = np.rint(np.sqrt(x[:, None]**2+x[None, :]**2)).astype(int)
    trailing = tuple(range(2, coefficients.ndim))
    power = np.abs(coefficients)**2
    if trailing:
        power = np.sum(power, axis=trailing)
    shell = np.bincount(radius.ravel(), weights=power.ravel())
    shell[0] = 0.0
    return shell/max(float(np.sum(shell)), 1e-300)


def common_quadrature(band: np.ndarray, quadrature_n: int) -> np.ndarray:
    """Evaluate one centered coefficient square on a common periodic grid."""
    width = band.shape[0]
    if band.shape[1] != width or width > quadrature_n:
        raise ValueError("invalid common band or quadrature grid")
    spectrum = np.zeros((quadrature_n, quadrature_n)+band.shape[2:], complex)
    center = quadrature_n//2
    half = width//2
    spectrum[center-half:center+half+1,
             center-half:center+half+1] = band
    return np.real(np.fft.ifftn(
        np.fft.ifftshift(spectrum, axes=(0, 1))*(quadrature_n**2),
        axes=(0, 1)))


def comparison(field_a: np.ndarray, field_b: np.ndarray,
               half_widths: tuple[int, ...], quadrature_n: int) -> dict:
    a = np.asarray(field_a, dtype=float)
    b = np.asarray(field_b, dtype=float)
    if a.shape[2:] != b.shape[2:]:
        raise ValueError("field components do not match")
    af = remove_spatial_mean(a)
    bf = remove_spatial_mean(b)
    ca = normalized_coefficients(af)
    cb = normalized_coefficients(bf)
    radial_a = radial_power_distribution(ca)
    radial_b = radial_power_distribution(cb)
    radial_n = min(len(radial_a), len(radial_b))
    rows = {}
    for half in half_widths:
        ba = centered_band(ca, half)
        bb = centered_band(cb, half)
        scale = max(norm(ba), norm(bb), 1e-300)
        amplitude_scale = max(norm(np.abs(ba)), norm(np.abs(bb)), 1e-300)
        active = np.maximum(np.abs(ba), np.abs(bb)) > (
            1e-8*max(float(np.max(np.abs(ba))),
                     float(np.max(np.abs(bb))), 1e-300))
        phase = np.angle(ba[active]*np.conj(bb[active]))
        qa = common_quadrature(ba, quadrature_n)
        qb = common_quadrature(bb, quadrature_n)
        rows[str(half)] = {
            "complex_coefficient_relative_l2": norm(ba-bb)/scale,
            "amplitude_only_relative_l2": norm(np.abs(ba)-np.abs(bb))/amplitude_scale,
            "active_phase_rms_rad": (0.0 if not np.any(active) else rms(phase)),
            "physical_projection_relative_l2": norm(qa-qb)/max(
                norm(qa), norm(qb), 1e-300),
            "a_coefficient_l2_absolute": norm(ba),
            "b_coefficient_l2_absolute": norm(bb),
            "a_discarded_power_fraction": tail_power_fraction(ca, half),
            "b_discarded_power_fraction": tail_power_fraction(cb, half),
        }
    mean_a = np.mean(a, axis=(0, 1))
    mean_b = np.mean(b, axis=(0, 1))
    return {
        "component_shape": list(a.shape[2:]),
        "a_grid": list(a.shape[:2]), "b_grid": list(b.shape[:2]),
        "a_mean_l2_absolute": norm(mean_a),
        "b_mean_l2_absolute": norm(mean_b),
        "mean_difference_l2_absolute": norm(mean_a-mean_b),
        "a_field_rms_absolute": rms(a), "b_field_rms_absolute": rms(b),
        "a_fluctuation_rms_absolute": rms(af),
        "b_fluctuation_rms_absolute": rms(bf),
        "radial_power_total_variation": float(
            .5*np.sum(np.abs(radial_a[:radial_n]-radial_b[:radial_n]))),
        "bands": rows,
    }


def projection_checks(fields: dict, metadata: dict, half_width: int,
                      quadrature_n: int) -> dict:
    """Check invariants of the analysis-only spectral projection."""
    def project(name):
        return common_quadrature(centered_band(
            normalized_coefficients(fields[name]), half_width), quadrature_n)

    supports = project("supports")
    beta = project("common_beta_p")
    projected_nye = project("exact_reconstructed_nye_m1")
    curl = nye_from_plastic_distortion(
        beta, float(metadata["spacing_m"])*int(metadata["n"])/quadrature_n)
    source_support_mean = np.mean(fields["supports"], axis=(0, 1))
    projected_support_mean = np.mean(supports, axis=(0, 1))
    source_signed_mean = np.mean(
        fields["signed_total_density_m2"], axis=(0, 1))
    projected_signed_mean = np.mean(
        project("signed_total_density_m2"), axis=(0, 1))
    signed_mean_error = projected_signed_mean-source_signed_mean
    return {
        "half_width": half_width, "quadrature_n": quadrature_n,
        "support_partition_maximum_absolute_error": float(
            np.max(np.abs(np.sum(supports, axis=2)-1.0))),
        "support_mean_maximum_absolute_error": float(
            np.max(np.abs(projected_support_mean-source_support_mean))),
        "support_projected_minimum": float(np.min(supports)),
        "support_projected_maximum": float(np.max(supports)),
        "signed_density_mean_maximum_absolute_error_m2": float(
            np.max(np.abs(signed_mean_error))),
        "signed_density_mean_relative_l2_error": norm(signed_mean_error)/max(
            norm(source_signed_mean), norm(projected_signed_mean), 1e-300),
        "nye_curl_commutation_relative_l2": norm(curl-projected_nye)/max(
            norm(curl), norm(projected_nye), 1e-300),
        "semantics": (
            "linear analysis-only Fourier projection preserves means and "
            "partition identity; projected support bounds are reported and "
            "the projection is never published as a material state"),
    }


def checkpoint_fields(path: Path) -> tuple[dict, dict]:
    state, runtime, step, gamma, _, configuration, provenance = _load_checkpoint(path)
    if configuration is None:
        raise ValueError("checkpoint lacks bound configuration")
    n = int(configuration["n"])
    spacing = float(configuration["length_m"])/n
    systems = bcc_four_family_systems()
    wall = _wall_parameters(configuration, spacing)
    strain = np.array([[0.0, .5*gamma], [.5*gamma, 0.0]])
    driving = CommonWallDriving(mean_strain=strain)
    common, _ = reconstruct_multigrain_common(state, spacing)
    nye = audit_multigrain_nye(state, spacing)
    owner_drivings = _owner_drivings_from_common_stress(
        state, driving, systems, wall)
    owner_fields = {name: [] for name in (
        "raw_stress_Pa", "chemical_backstress_Pa", "effective_stress_Pa",
        "taylor_resistance_Pa", "speed_m_s", "signed_slip_rate_s",
        "plastic_power_W_m3")}
    for support, owner, owner_driving in zip(
            state.supports, state.owners, owner_drivings):
        resolved = resolved_driving_components(
            owner, owner_driving, systems, (), wall)
        residual = wall_residual(owner, owner_driving, systems, (), wall)
        values = {**{name: resolved[name] for name in (
            "raw_stress_Pa", "chemical_backstress_Pa", "effective_stress_Pa",
            "taylor_resistance_Pa", "speed_m_s")},
            "signed_slip_rate_s": residual.state_rate.slip,
            "plastic_power_W_m3": residual.plastic_power_W_m3}
        for name, value in values.items():
            weight = np.asarray(support)[(...,)+(None,)*(np.ndim(value)-2)]
            owner_fields[name].append(weight*np.asarray(value))
    owner_fields = {name: np.stack(values, axis=2)
                    for name, values in owner_fields.items()}
    instantaneous = multigrain_instantaneous_dissipation_fields(
        state, driving=driving, systems=systems, topologies=(),
        wall_parameters=wall)
    signed_mobile = sum(
        state.supports[i][..., None]
        *(state.owners[i].mobile_plus_m2-state.owners[i].mobile_minus_m2)
        for i in range(len(state.owners)))
    signed_total = sum(
        state.supports[i][..., None]*sum(
            getattr(state.owners[i], plus)-getattr(state.owners[i], minus)
            for plus, minus in (("mobile_plus_m2", "mobile_minus_m2"),
                                ("forest_plus_m2", "forest_minus_m2"),
                                ("wall_plus_m2", "wall_minus_m2")))
        for i in range(len(state.owners)))
    fields = {
        "supports": np.moveaxis(np.asarray(state.supports), 0, 2),
        "common_temperature_K": common.temperature_K,
        "common_orientation_rad": common.orientation_rad,
        "common_slip": common.slip,
        "common_beta_p": common.beta_p,
        "signed_mobile_polarization_m2": signed_mobile,
        "signed_total_density_m2": signed_total,
        "family_nye_reservoir_m1": common.family_nye_m1,
        "exact_reconstructed_nye_m1": nye.exact_reconstructed_m1,
        "owner_curl_weighted_nye_m1": nye.owner_curl_weighted_m1,
        "owner_reservoir_weighted_nye_m1": nye.owner_reservoir_weighted_m1,
        "support_gradient_nye_m1": nye.support_gradient_m1,
        "owner_reservoir_mismatch_nye_m1": nye.owner_reservoir_mismatch_m1,
        "instantaneous_plastic_power_W_m3": instantaneous["plastic_power_W_m3"],
        "instantaneous_signed_slip_rate_s": instantaneous["signed_slip_rate_s"],
        "instantaneous_irreversible_heat_rate_W_m3": instantaneous[
            "irreversible_heat_rate_W_m3"],
        "common_stress_tensor_Pa": owner_drivings[0].fixed_stress_tensor_Pa,
        **{"owner_weighted_"+name: value for name, value in owner_fields.items()},
    }
    with np.load(path, allow_pickle=False) as data:
        for key, label in (
            ("diagnostic_front_heat_source_J_by_cell",
             "accepted_interval_front_heat_J_by_cell"),
            ("diagnostic_mechanical_heat_J_m3_cells",
             "accepted_interval_mechanical_heat_J_m3"),
            ("diagnostic_mechanical_work_J_m3_cells",
             "accepted_interval_mechanical_work_J_m3"),
            ("diagnostic_thermal_conduction_J_m3_cells",
             "accepted_interval_conduction_J_m3")):
            if key in data.files:
                fields[label] = np.asarray(data[key], dtype=float)
    metadata = {
        "path": str(path.resolve()), "sha256": digest(path), "step": step,
        "physical_time_s": runtime.ledger.physical_time_s, "gamma": gamma,
        "n": n, "spacing_m": spacing, "source_commit": (provenance or {}).get(
            "source_commit"), "grain_ids": list(state.grain_ids),
        "owner_nye_identity_relative_l2": norm(
            nye.owner_reservoir_mismatch_m1)/max(
                norm(nye.owner_curl_weighted_m1), 1e-300),
    }
    return fields, metadata


def run(left: Path, right: Path, output: Path,
        half_widths: tuple[int, ...], quadrature_n: int) -> dict:
    left_fields, left_meta = checkpoint_fields(left)
    right_fields, right_meta = checkpoint_fields(right)
    if left_meta["step"] != right_meta["step"] or not np.isclose(
            left_meta["physical_time_s"], right_meta["physical_time_s"],
            rtol=0.0, atol=1e-18):
        raise ValueError("checkpoints do not share one physical state time")
    if left_meta["grain_ids"] != right_meta["grain_ids"]:
        raise ValueError("owner identity differs")
    if set(left_fields) != set(right_fields):
        raise ValueError("checkpoint field sets differ")
    result = {
        "schema": "asb-drx-v63-phase-sensitive-field-comparison-v1",
        "coordinate_convention": (
            "same periodic node-centered [0,L)^2 origin; coefficients are "
            "normalized by nx*ny; no translation or phase optimization"),
        "left": left_meta, "right": right_meta,
        "half_widths": list(half_widths), "common_quadrature_n": quadrature_n,
        "projection_operator_checks": {
            "left": projection_checks(left_fields, left_meta,
                                      max(half_widths), quadrature_n),
            "right": projection_checks(right_fields, right_meta,
                                       max(half_widths), quadrature_n),
        },
        "field_comparisons": {
            name: comparison(left_fields[name], right_fields[name],
                             half_widths, quadrature_n)
            for name in sorted(left_fields)},
        "scope": (
            "analysis-only common-band projection; both trajectories remain "
            "independently evolved and no projected field enters production"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--half-widths", type=int, nargs="+", default=(7, 15, 23))
    parser.add_argument("--quadrature-n", type=int, default=96)
    args = parser.parse_args()
    result = run(args.left, args.right, args.output,
                 tuple(args.half_widths), args.quadrature_n)
    selected = result["field_comparisons"]
    print(json.dumps({
        "output_sha256": digest(args.output),
        "step": result["left"]["step"],
        "temperature_band_15": selected["common_temperature_K"]["bands"]["15"],
        "power_band_15": selected["instantaneous_plastic_power_W_m3"]["bands"]["15"],
        "nye_band_15": selected["exact_reconstructed_nye_m1"]["bands"]["15"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
