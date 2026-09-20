"""V23 extensive ordered/disordered wall thermodynamics and kinetics.

The wall state is line density, not a phase fraction.  For every sign and BCC
family, ``rho_w = rho_tangle + rho_ordered``.  Ordering transfers line between
the two reservoirs with detailed balance and cannot create density or Burgers
content.  ``q = rho_ordered/rho_w`` is a derived diagnostic only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import LinearOperator, expm_multiply, gmres

try:
    from .arrhenius_kinetics import (
        ActivatedProcess, activated_rate_array_s, exp_floor_enthalpy_j,
    )
    from .density_state_map import DensityInventory, derived_density_fields
    from .tensorial_nye import rotation_z, rotated_system_fields, spectral_derivatives
except ImportError:
    from arrhenius_kinetics import (
        ActivatedProcess, activated_rate_array_s, exp_floor_enthalpy_j,
    )
    from density_state_map import DensityInventory, derived_density_fields
    from tensorial_nye import rotation_z, rotated_system_fields, spectral_derivatives

KB_J_K = 1.380649e-23
EV_J = 1.602176634e-19


@dataclass(frozen=True)
class ExtensiveWallParameters:
    spacing_m: float
    rho_reference_m2: float = 5e14
    line_energy_J_m: float = 1.5e-9
    rho_log_coefficient_J_m: float = 8e-11
    disordered_excess_J_m: float = 2e-10
    ordered_excess_J_m: float = 3e-10
    forest_excess_J_m: float = 1e-10
    junction_excess_J_m: float = 2e-10
    nye_match_coefficient_J_m: float = 2e-5
    ordered_gradient_J_m3: float = 1e-33
    ordering_attempt_frequency_s: float = 1e7
    ordering_entropy_over_kB: float = 0.0
    ordering_drag_rate_s: float = np.inf
    negative_barrier_mode: str = "drag"
    ordering_barrier_eV: float = 0.70
    event_length_m: float = 2.5e-9
    exp_floor: float = 0.05
    exp_a: float = 2.2
    exp_n: float = 2.5
    critical_stress_Pa: float = 1.5e9
    maximum_fraction_per_step: float = 0.15
    # Generic fixtures retain one-step behavior unless a production campaign
    # explicitly qualifies an internal physical-time resolution.
    ordering_internal_substep_s: float = 1.0
    ordering_internal_max_substeps: int = 8192
    ordering_stationary_remainder_relative_tolerance: float = 1e-8
    ordering_integration_method: str = "complete_time_explicit"
    ordering_finite_time_backend: str = "dense_bdf_oracle"
    ordering_matrix_free_max_attempt_exposure: float = 0.05
    ordering_finite_relative_tolerance: float = 2e-4
    ordering_finite_absolute_tolerance: float = 2e-8
    ordering_finite_max_rejections: int = 24
    ordering_implicit_residual_tolerance: float = 2e-9
    ordering_implicit_max_nfev: int = 100
    ordering_asymptotic_minimum_attempt_exposure: float = 50.0
    ordering_asymptotic_maximum_endpoint_distance_relative: float = 1.0
    ordering_asymptotic_certificate_mode: str = "legacy_euclidean_heuristic"

    def __post_init__(self):
        positive = (
            self.spacing_m, self.rho_reference_m2, self.line_energy_J_m,
            self.rho_log_coefficient_J_m, self.disordered_excess_J_m,
            self.ordered_excess_J_m, self.forest_excess_J_m,
            self.junction_excess_J_m,
            self.ordering_attempt_frequency_s, self.ordering_barrier_eV,
            self.event_length_m, self.exp_a, self.exp_n,
            self.critical_stress_Pa, self.maximum_fraction_per_step,
            self.ordering_internal_substep_s,
        )
        if any((not np.isfinite(x)) or x <= 0 for x in positive):
            raise ValueError("V23 wall parameters must be finite and positive")
        if (self.ordered_gradient_J_m3 < 0
                or self.nye_match_coefficient_J_m < 0
                or not 0 <= self.exp_floor <= 1):
            raise ValueError("invalid gradient coefficient or EXP floor")
        if self.maximum_fraction_per_step > 1:
            raise ValueError("step fraction cannot exceed one")
        if int(self.ordering_internal_max_substeps) <= 0:
            raise ValueError("ordering internal substep limit must be positive")
        if (not np.isfinite(self.ordering_stationary_remainder_relative_tolerance)
                or self.ordering_stationary_remainder_relative_tolerance <= 0.0):
            raise ValueError("ordering stationary tolerance must be positive")
        if self.ordering_integration_method not in (
                "complete_time_explicit", "implicit_backward_euler",
                "finite_time_bdf"):
            raise ValueError("unknown ordering integration method")
        if self.ordering_finite_time_backend not in (
                "dense_bdf_oracle", "matrix_free_backward_euler",
                "matrix_free_projected_rk2", "matrix_free_adaptive_dop853",
                "matrix_free_rosenbrock_euler",
                "matrix_free_exponential_rosenbrock",
                "matrix_free_adaptive_backward_euler",
                "matrix_free_adaptive_rosenbrock_euler"):
            raise ValueError("unknown finite-time ordering backend")
        if (not np.isfinite(self.ordering_implicit_residual_tolerance)
                or self.ordering_implicit_residual_tolerance <= 0.0):
            raise ValueError("ordering implicit tolerance must be positive")
        if (not np.isfinite(self.ordering_matrix_free_max_attempt_exposure)
                or self.ordering_matrix_free_max_attempt_exposure <= 0.0):
            raise ValueError("matrix-free attempt exposure must be positive")
        if int(self.ordering_implicit_max_nfev) <= 0:
            raise ValueError("ordering implicit evaluation limit must be positive")
        if (not np.isfinite(self.ordering_finite_relative_tolerance)
                or self.ordering_finite_relative_tolerance <= 0.0
                or not np.isfinite(self.ordering_finite_absolute_tolerance)
                or self.ordering_finite_absolute_tolerance <= 0.0):
            raise ValueError("finite-time error tolerances must be positive")
        if int(self.ordering_finite_max_rejections) < 0:
            raise ValueError("finite-time rejection limit must be nonnegative")
        if (not np.isfinite(self.ordering_asymptotic_minimum_attempt_exposure)
                or self.ordering_asymptotic_minimum_attempt_exposure <= 0.0):
            raise ValueError("ordering asymptotic exposure must be positive")
        if (not np.isfinite(
                self.ordering_asymptotic_maximum_endpoint_distance_relative)
                or not 0.0 <
                self.ordering_asymptotic_maximum_endpoint_distance_relative
                <= 1.0):
            raise ValueError("ordering asymptotic endpoint-distance tolerance invalid")
        if self.ordering_asymptotic_certificate_mode not in (
                "disabled", "legacy_euclidean_heuristic"):
            raise ValueError("unknown asymptotic certificate mode")
        # Validate entropy, drag limit, and negative-barrier validity policy in
        # the campaign-wide Arrhenius representation.
        ActivatedProcess(
            "extensive-wall-ordering", self.ordering_attempt_frequency_s,
            self.ordering_entropy_over_kB, self.ordering_drag_rate_s,
            self.negative_barrier_mode)


def _laplacian(field, spacing_m):
    gx, gy = spectral_derivatives(field, spacing_m)
    gxx, _ = spectral_derivatives(gx, spacing_m)
    _, gyy = spectral_derivatives(gy, spacing_m)
    return gxx + gyy


def ordered_wall_nye_m1(inventory, systems, orientation_rad,
                        line_direction_crystal=(0.0, 0.0, 1.0)):
    """Nye tensor from only the polarized extensive ordered-line content."""
    burgers, _, _ = rotated_system_fields(systems, orientation_rad)
    line0 = np.asarray(line_direction_crystal, dtype=float)
    if line0.shape != (3,) or not np.isfinite(line0).all() or np.linalg.norm(line0) == 0:
        raise ValueError("ordered boundary line must be a finite three-vector")
    line0 = line0 / np.linalg.norm(line0)
    rotation = rotation_z(orientation_rad)
    line = np.einsum("...ij,j->...i", rotation, line0)
    lines = np.broadcast_to(line[..., None, :], burgers.shape)
    basis = np.einsum("...ai,...aj->...aij", burgers, lines)
    signed = (inventory.wall_ordered_plus_m2
              - inventory.wall_ordered_minus_m2)
    return np.einsum("...a,...aij->...ij", signed, basis), basis


def planar_frank_bilby_target_m1(shape, spacing_m, left_angle_rad,
                                 right_angle_rad, *, normal_axis=0,
                                 line_axis=2, width_cells=3):
    """Independent planar Frank--Bilby target distributed over grid cells."""
    nx, ny = map(int, shape)
    if normal_axis not in (0, 1) or line_axis not in (0, 1, 2):
        raise ValueError("invalid planar wall axes")
    width = int(width_cells)
    if width <= 0 or width >= shape[normal_axis]:
        raise ValueError("wall width must fit inside the grid")
    tangent = np.zeros(3); tangent[1-normal_axis] = 1.0
    closure = (rotation_z(float(right_angle_rad))
               - rotation_z(float(left_angle_rad))) @ tangent
    target = np.zeros((nx, ny, 3, 3))
    center = shape[normal_axis] // 2
    start = center - width // 2
    indices = np.arange(start, start + width)
    if normal_axis == 0:
        target[indices, :, :, line_axis] = closure[None, None, :] / (width * spacing_m)
    else:
        target[:, indices, :, line_axis] = closure[None, None, :] / (width * spacing_m)
    return target, closure


def orientation_gradient_frank_bilby_target_m1(orientation_rad, spacing_m,
                                                line_axis=2):
    """Smooth local Frank--Bilby density derived only from lattice rotation.

    Integrating the x-gradient term through an x-normal planar wall returns
    ``(R_right-R_left)e_y``; the y-gradient term uses the consistently oriented
    x tangent.  No dislocation population enters this construction.
    """
    theta = np.asarray(orientation_rad, dtype=float)
    tx, ty = spectral_derivatives(theta, spacing_m)
    c, s = np.cos(theta), np.sin(theta)
    dR_ey = np.stack((-c, -s, np.zeros_like(theta)), axis=-1)
    dR_ex = np.stack((-s, c, np.zeros_like(theta)), axis=-1)
    closure_density = tx[..., None]*dR_ey - ty[..., None]*dR_ex
    target = np.zeros(theta.shape+(3, 3))
    target[..., :, int(line_axis)] = closure_density
    return target


def manufacture_ordered_inventory(inventory, systems, orientation_rad,
                                  target_nye_m1,
                                  line_direction_crystal=(0.0, 0.0, 1.0)):
    """Least-squares signed BCC line inventory for an independent Nye target."""
    zero = replace(
        inventory,
        wall_ordered_plus_m2=np.zeros_like(inventory.wall_ordered_plus_m2),
        wall_ordered_minus_m2=np.zeros_like(inventory.wall_ordered_minus_m2))
    _, basis = ordered_wall_nye_m1(
        zero, systems, orientation_rad, line_direction_crystal)
    target = np.asarray(target_nye_m1)
    signed = np.zeros_like(inventory.wall_ordered_plus_m2)
    for index in np.ndindex(target.shape[:2]):
        matrix = np.stack([basis[index + (a,)].ravel()
                           for a in range(len(systems))], axis=1)
        signed[index] = np.linalg.lstsq(matrix, target[index].ravel(), rcond=None)[0]
    result = replace(
        inventory,
        wall_ordered_plus_m2=np.maximum(signed, 0.0),
        wall_ordered_minus_m2=np.maximum(-signed, 0.0))
    reconstructed, _ = ordered_wall_nye_m1(
        result, systems, orientation_rad, line_direction_crystal)
    scale = max(float(np.linalg.norm(target)), 1.0)
    return result, float(np.linalg.norm(reconstructed-target)/scale)


def extensive_wall_energy_components_J_m3(inventory, systems, topologies,
                                           orientation_rad, target_nye_m1,
                                           parameters):
    inventory.validate(len(systems), len(topologies))
    fields = derived_density_fields(inventory, topologies)
    rho = fields["rho_total_m2"]
    safe = np.maximum(rho, 1e-30 * parameters.rho_reference_m2)
    common_line = (parameters.line_energy_J_m * rho
                   + parameters.rho_log_coefficient_J_m * rho
                   * np.log(safe / parameters.rho_reference_m2))
    tangle = parameters.disordered_excess_J_m * fields["rho_wall_tangle_m2"]
    ordered = parameters.ordered_excess_J_m * fields["rho_wall_ordered_m2"]
    forest = parameters.forest_excess_J_m * np.sum(
        inventory.forest_plus_m2+inventory.forest_minus_m2, axis=2)
    alpha, _ = ordered_wall_nye_m1(inventory, systems, orientation_rad)
    target = np.asarray(target_nye_m1, dtype=float)
    if target.shape != alpha.shape:
        raise ValueError("target Nye tensor must have grid x 3 x 3 layout")
    mismatch = alpha - target
    matching = 0.5 * parameters.nye_match_coefficient_J_m * np.sum(
        mismatch * mismatch, axis=(-2, -1))
    gradient = np.zeros_like(rho)
    if parameters.ordered_gradient_J_m3:
        for name in ("wall_ordered_plus_m2", "wall_ordered_minus_m2"):
            gx, gy = spectral_derivatives(getattr(inventory, name), parameters.spacing_m)
            gradient += 0.5 * parameters.ordered_gradient_J_m3 * np.sum(
                gx * gx + gy * gy, axis=2)
    topology = np.zeros_like(rho)
    for index, item in enumerate(topologies):
        topology += inventory.junction_m2[..., index] * (
            item.product_line_multiplicity * parameters.junction_excess_J_m
            + item.delta_free_energy_J_m)
    return {
        "common_line": common_line,
        "disordered_wall": tangle,
        "ordered_boundary": ordered,
        "forest_excess": forest,
        "nye_mismatch": matching,
        "ordered_gradient": gradient,
        "junction_topology": topology,
        "total": (common_line+forest+tangle+ordered+matching+gradient
                  +topology),
    }


def ordered_gradient_increment_J_m3_cells(before_inventory, after_inventory,
                                           parameters):
    """Evaluate the quadratic ordered-gradient increment without subtraction.

    This is the exact discrete polarization identity for the same spectral
    derivative used by :func:`extensive_wall_energy_components_J_m3`.  Its
    units are the historical ledger units, J m^-3 summed over grid cells;
    callers apply their physical cell volume exactly once.
    """
    kappa = float(parameters.ordered_gradient_J_m3)
    if kappa == 0.0:
        return 0.0
    result = np.longdouble(0.0)
    for name in ("wall_ordered_plus_m2", "wall_ordered_minus_m2"):
        before = np.asarray(getattr(before_inventory, name), dtype=float)
        delta = np.asarray(getattr(after_inventory, name), dtype=float)-before
        before_x, before_y = spectral_derivatives(
            before, parameters.spacing_m)
        delta_x, delta_y = spectral_derivatives(
            delta, parameters.spacing_m)
        cross = before_x*delta_x+before_y*delta_y
        quadratic = delta_x*delta_x+delta_y*delta_y
        result += np.sum(kappa*(cross+0.5*quadratic),
                         dtype=np.longdouble)
    return float(result)


def extensive_wall_chemical_potentials_J_m(inventory, systems, topologies,
                                            orientation_rad, target_nye_m1,
                                            parameters):
    fields = derived_density_fields(inventory, topologies)
    rho = fields["rho_total_m2"]
    safe = np.maximum(rho, 1e-30 * parameters.rho_reference_m2)
    common = (parameters.line_energy_J_m
              + parameters.rho_log_coefficient_J_m
              * (np.log(safe / parameters.rho_reference_m2) + 1.0))
    alpha, basis = ordered_wall_nye_m1(inventory, systems, orientation_rad)
    mismatch = alpha - np.asarray(target_nye_m1, dtype=float)
    match_derivative = parameters.nye_match_coefficient_J_m * np.einsum(
        "...ij,...aij->...a", mismatch, basis)
    gradient_plus = -parameters.ordered_gradient_J_m3 * _laplacian(
        inventory.wall_ordered_plus_m2, parameters.spacing_m)
    gradient_minus = -parameters.ordered_gradient_J_m3 * _laplacian(
        inventory.wall_ordered_minus_m2, parameters.spacing_m)
    tangle_mu = common[..., None] + parameters.disordered_excess_J_m
    return {
        "tangle_plus": tangle_mu,
        "tangle_minus": tangle_mu,
        "ordered_plus": (common[..., None] + parameters.ordered_excess_J_m
                         + match_derivative + gradient_plus),
        "ordered_minus": (common[..., None] + parameters.ordered_excess_J_m
                          - match_derivative + gradient_minus),
    }


def _attempt_rate_s(stress_Pa, temperature_K, parameters):
    enthalpy = exp_floor_enthalpy_j(
        np.abs(np.asarray(stress_Pa, dtype=float)),
        parameters.ordering_barrier_eV*EV_J,
        parameters.critical_stress_Pa, parameters.exp_a,
        parameters.exp_n, parameters.exp_floor)
    process = ActivatedProcess(
        "extensive-wall-ordering", parameters.ordering_attempt_frequency_s,
        parameters.ordering_entropy_over_kB, parameters.ordering_drag_rate_s,
        parameters.negative_barrier_mode)
    return activated_rate_array_s(process, enthalpy, temperature_K)


def ordering_residual(inventory, systems, topologies, orientation_rad,
                      target_nye_m1, stress_Pa, temperature_K, parameters,
                      *, enforce_availability=True):
    """Return conservative tangle->ordered transfer rates [m^-2 s^-1]."""
    mu = extensive_wall_chemical_potentials_J_m(
        inventory, systems, topologies, orientation_rad, target_nye_m1, parameters)
    stress = np.asarray(stress_Pa, dtype=float)
    if stress.ndim == 3:
        stress = np.max(np.abs(stress), axis=2)
    rate = _attempt_rate_s(stress, temperature_K, parameters)[..., None]
    result = {}
    turnover = {}
    for sign in ("plus", "minus"):
        source = getattr(inventory, f"wall_tangle_{sign}_m2")
        target = getattr(inventory, f"wall_ordered_{sign}_m2")
        event_bias = ((mu[f"ordered_{sign}"] - mu[f"tangle_{sign}"])
                      * parameters.event_length_m
                      / (2 * KB_J_K * np.asarray(temperature_K)[..., None]))
        bias = np.tanh(event_bias)
        # A shared extensive line pool makes the gross-rate ratio exactly
        # exp(-Delta_mu*event_length/kT), while their difference always has
        # the opposite sign to Delta_mu.  Reservoir availability is enforced
        # by the accepted-step limiter below, not hidden in the rate ratio.
        pool = 0.5 * (source + target)
        forward = rate * pool * (1 - bias)
        reverse = rate * pool * (1 + bias)
        net = forward - reverse
        # Exact invariant-set projection at an exhausted reservoir.  This is
        # a physical availability condition; no finite pool may transfer out
        # of a state containing zero line.
        if enforce_availability:
            net = np.where((source <= 0) & (net > 0), 0.0, net)
            net = np.where((target <= 0) & (net < 0), 0.0, net)
        result[sign] = net
        turnover[sign] = forward + reverse
    return result, turnover, mu


def _accepted_ordering_substep(inventory, systems, topologies, orientation_rad,
                               target_nye_m1, stress_Pa, temperature_K,
                               parameters, dt_s):
    transfer, turnover, mu = ordering_residual(
        inventory, systems, topologies, orientation_rad, target_nye_m1,
        stress_Pa, temperature_K, parameters)
    updates = {}
    effective_rates = {}
    local_scales = []
    for sign in ("plus", "minus"):
        raw_extent = dt_s * transfer[sign]
        tangle0 = getattr(inventory, f"wall_tangle_{sign}_m2")
        ordered0 = getattr(inventory, f"wall_ordered_{sign}_m2")
        donor = np.where(raw_extent >= 0, tangle0, ordered0)
        demand = np.divide(np.abs(raw_extent), donor, out=np.zeros_like(raw_extent),
                           where=donor > 0)
        fraction = np.minimum(-np.expm1(-demand),
                              parameters.maximum_fraction_per_step)
        extent = np.sign(raw_extent) * donor * fraction
        ratio = np.divide(np.abs(extent), np.abs(raw_extent),
                          out=np.ones_like(extent), where=raw_extent != 0)
        local_scales.append(ratio)
        effective_rates[sign] = extent/max(float(dt_s), 1e-300)
        updates[f"wall_tangle_{sign}_m2"] = (
            tangle0 - extent)
        updates[f"wall_ordered_{sign}_m2"] = (
            ordered0 + extent)
    updated = replace(inventory, **updates)
    updated.validate(len(systems), len(topologies))
    dissipation_W_m3 = np.sum(
        (mu["ordered_plus"] - mu["tangle_plus"]) * effective_rates["plus"]
        + (mu["ordered_minus"] - mu["tangle_minus"]) * effective_rates["minus"],
        axis=2)
    scale = float(np.min(np.stack(local_scales)))
    return updated, {"transfer_m2_s": transfer, "turnover_m2_s": turnover,
                     "accepted_transfer_m2_s": effective_rates,
                     "chemical_potential_J_m": mu,
                     "free_energy_rate_W_m3": dissipation_W_m3}, scale


def _ordering_affinity_linear_action(delta_plus, delta_minus, systems,
                                     orientation_rad, parameters):
    """Derivative of ordered-minus-tangle chemical affinity [J/m]."""
    shape = np.asarray(delta_plus).shape
    # Only the ordered Nye basis is needed; all density arrays are dummy.
    # Constructing it through the authoritative routine avoids introducing a
    # second crystallographic convention in the implicit solver.
    dummy_inventory = type("_OrderingBasisInventory", (), {})()
    # ordered_wall_nye_m1 accesses only the two ordered arrays.
    dummy_inventory.wall_ordered_plus_m2 = np.zeros(shape)
    dummy_inventory.wall_ordered_minus_m2 = np.zeros(shape)
    _, basis = ordered_wall_nye_m1(
        dummy_inventory, systems, orientation_rad)
    signed_delta = np.asarray(delta_plus)-np.asarray(delta_minus)
    delta_alpha = np.einsum("...a,...aij->...ij", signed_delta, basis)
    match = parameters.nye_match_coefficient_J_m*np.einsum(
        "...ij,...aij->...a", delta_alpha, basis)
    return (
        match-parameters.ordered_gradient_J_m3*_laplacian(
            np.asarray(delta_plus), parameters.spacing_m),
        -match-parameters.ordered_gradient_J_m3*_laplacian(
            np.asarray(delta_minus), parameters.spacing_m),
    )


def _accepted_ordering_implicit(inventory, systems, topologies,
                                orientation_rad, target_nye_m1, stress_Pa,
                                temperature_K, parameters, dt_s,
                                alignment=None, force_finite_time=False,
                                finite_time_endpoint=None):
    """Bounded backward-Euler solve of the declared ordering rate.

    Ordered fractions are the nonlinear coordinates, so every trial state
    conserves tangle+ordered line and remains inside the physical reservoir
    bounds.  The matrix-free Jacobian differentiates the existing EXP-floor,
    tanh-affinity rate; it does not introduce an equilibrium replacement.
    """
    total_dt = float(dt_s)
    if not np.isfinite(total_dt) or total_dt <= 0.0:
        raise ValueError("ordering timestep must be finite and positive")
    stress_for_rate = np.asarray(stress_Pa, dtype=float)
    if stress_for_rate.ndim == 3:
        stress_for_rate = np.max(np.abs(stress_for_rate), axis=2)
    attempt_exposure = total_dt*float(np.max(_attempt_rate_s(
        stress_for_rate, temperature_K, parameters)))
    if (not force_finite_time and (
            parameters.ordering_asymptotic_certificate_mode == "disabled"
            or attempt_exposure
            < parameters.ordering_asymptotic_minimum_attempt_exposure)):
        # Below the independently verified finite/asymptotic overlap, use the
        # selected finite-time backend itself.  The former capped explicit
        # map is not a finite-time oracle for the stiff spectral-gradient
        # fixture and is deliberately not dispatched here.
        result = _accepted_ordering_implicit(
            inventory, systems, topologies, orientation_rad, target_nye_m1,
            stress_Pa, temperature_K, parameters, total_dt,
            alignment=alignment, force_finite_time=True)
        ledger_index = 2 if alignment is not None else 1
        result[ledger_index]["stiff_dispatch"] = (
            "finite_time_"+parameters.ordering_finite_time_backend)
        result[ledger_index]["maximum_attempt_exposure"] = attempt_exposure
        if parameters.ordering_asymptotic_certificate_mode == "disabled":
            result[ledger_index]["asymptotic_dispatch_disabled_reason"] = (
                "Euclidean endpoint distance is a dispatch heuristic, not a "
                "proved contraction/error metric for the coupled tanh flow")
        return result
    totals = {
        sign: (np.asarray(getattr(inventory, f"wall_tangle_{sign}_m2"))
               +np.asarray(getattr(inventory, f"wall_ordered_{sign}_m2")))
        for sign in ("plus", "minus")}
    density_scale = max(float(np.max(totals["plus"])),
                        float(np.max(totals["minus"])), 1.0)
    diagnostic_density_floor = 1e-12*density_scale
    active = {sign: totals[sign] > 0.0 for sign in ("plus", "minus")}
    active_count = sum(int(np.count_nonzero(active[s]))
                       for s in ("plus", "minus"))
    if active_count == 0:
        transfer, turnover, mu = ordering_residual(
            inventory, systems, topologies, orientation_rad, target_nye_m1,
            stress_Pa, temperature_K, parameters)
        aggregate = {
            "transfer_m2_s": transfer, "turnover_m2_s": turnover,
            "accepted_transfer_m2_s": {s: np.zeros_like(transfer[s])
                                        for s in ("plus", "minus")},
            "chemical_potential_J_m": mu,
            "free_energy_rate_W_m3": np.zeros_like(np.asarray(temperature_K)),
            "internal_substeps": 1, "complete_elapsed_time_s": total_dt,
            "discarded_reaction_time_s": 0.0,
            "stationary_remainder_s": 0.0,
            "integration_method": "bounded_backward_euler",
            "implicit_success": True, "implicit_max_scaled_residual": 0.0,
            "implicit_nfev": 0, "implicit_njev": 0,
        }
        if alignment is not None:
            from .wall_topology_supply import apply_signed_ordering_extent
            updated, updated_alignment, topology = apply_signed_ordering_extent(
                inventory, alignment, np.zeros_like(totals["plus"]),
                np.zeros_like(totals["minus"]), systems, orientation_rad,
                topologies)
            aggregate["topology_subcycled"] = False
            aggregate["topology_ledger"] = topology
            return updated, updated_alignment, aggregate, 1.0
        return inventory, aggregate, 1.0

    shapes = totals["plus"].shape
    slices = {}; offset = 0
    for sign in ("plus", "minus"):
        size = int(np.count_nonzero(active[sign]))
        slices[sign] = slice(offset, offset+size); offset += size

    def unpack(vector):
        vector = np.asarray(vector, dtype=float).reshape(-1)
        result = {}
        for sign in ("plus", "minus"):
            q = np.zeros(shapes, dtype=float)
            q[active[sign]] = vector[slices[sign]]
            result[sign] = q
        return result

    q0_fields = {
        sign: np.divide(
            getattr(inventory, f"wall_ordered_{sign}_m2"), totals[sign],
            out=np.zeros_like(totals[sign]), where=active[sign])
        for sign in ("plus", "minus")}
    q0 = np.concatenate([q0_fields[s][active[s]]
                         for s in ("plus", "minus")])

    def state_from_q(vector):
        q = unpack(vector); updates = {}
        for sign in ("plus", "minus"):
            ordered = totals[sign]*q[sign]
            updates[f"wall_ordered_{sign}_m2"] = ordered
            updates[f"wall_tangle_{sign}_m2"] = totals[sign]-ordered
        return replace(inventory, **updates)

    energy_scale = max(
        sum(float(np.sum(totals[s], dtype=np.longdouble))
            for s in ("plus", "minus"))
        *max(abs(parameters.ordered_excess_J_m
                 -parameters.disordered_excess_J_m), 1e-12), 1.0)

    def equilibrium_objective(vector):
        candidate = state_from_q(vector)
        energy = extensive_wall_energy_components_J_m3(
            candidate, systems, topologies, orientation_rad, target_nye_m1,
            parameters)["total"]
        mu = extensive_wall_chemical_potentials_J_m(
            candidate, systems, topologies, orientation_rad, target_nye_m1,
            parameters)
        gradient = []
        for sign in ("plus", "minus"):
            affinity = mu[f"ordered_{sign}"]-mu[f"tangle_{sign}"]
            gradient.append((affinity*totals[sign])[active[sign]]/energy_scale)
        return (float(np.sum(energy, dtype=np.longdouble))/energy_scale,
                np.concatenate(gradient))

    if (not force_finite_time
            and parameters.ordering_asymptotic_certificate_mode != "disabled"
            and attempt_exposure
            >= parameters.ordering_asymptotic_minimum_attempt_exposure):
        # The ordering energy is a convex quadratic in the ordered extents:
        # linear reservoir excess plus positive Nye-mismatch and spectral
        # gradient terms.  A projected accelerated-gradient solve therefore
        # follows the unique constrained basin without selecting an arbitrary
        # nonconvex stationary root.
        dummy = type("_OrderingBasisInventory", (), {})()
        dummy.wall_ordered_plus_m2 = np.zeros(shapes)
        dummy.wall_ordered_minus_m2 = np.zeros(shapes)
        _, basis = ordered_wall_nye_m1(dummy, systems, orientation_rad)
        maximum_gram = 0.0
        if parameters.nye_match_coefficient_J_m:
            for index in np.ndindex(shapes[:2]):
                matrix = basis[index].reshape(shapes[-1], -1)
                maximum_gram = max(maximum_gram, float(np.linalg.norm(
                    matrix@matrix.T, ord=2)))
        spectral_lipschitz = (parameters.ordered_gradient_J_m3*2.0
                              *(np.pi/parameters.spacing_m)**2)
        nye_lipschitz = (2.0*parameters.nye_match_coefficient_J_m
                         *maximum_gram)
        lipschitz_J_m3 = spectral_lipschitz+nye_lipschitz
        y = {s: np.asarray(getattr(
            inventory, f"wall_ordered_{s}_m2"), dtype=float).copy()
             for s in ("plus", "minus")}
        z = {s: value.copy() for s, value in y.items()}
        acceleration = 1.0
        equilibrium_iterations = 0
        projected_change = np.inf
        if lipschitz_J_m3 <= 0.0:
            delta = (parameters.ordered_excess_J_m
                     -parameters.disordered_excess_J_m)
            bound = 1.0 if delta < 0.0 else 0.0
            y = {s: bound*totals[s] for s in ("plus", "minus")}
            projected_change = 0.0
        elif (parameters.nye_match_coefficient_J_m == 0.0
              and active_count < 0.25*sum(
                  value.size for value in totals.values())):
            # Production capture occupies a thin support.  With C_FB=0 every
            # sign/family is an independent convex spectral obstacle problem.
            # Solve those O(n_support) KKT systems separately instead of
            # constructing one dense 8*n_support system.
            delta_excess = (parameters.ordered_excess_J_m
                            -parameters.disordered_excess_J_m)
            y = {s: np.zeros_like(totals[s]) for s in ("plus", "minus")}
            projected_change = 0.0; equilibrium_iterations = 0
            for sign in ("plus", "minus"):
                for family in range(shapes[-1]):
                    upper_field = totals[sign][..., family]
                    indices = np.flatnonzero(upper_field.ravel() > 0.0)
                    if indices.size == 0:
                        continue
                    hessian = np.empty((indices.size, indices.size))
                    for column, flat_index in enumerate(indices):
                        direction = np.zeros(shapes[:2])
                        direction.ravel()[flat_index] = 1.0
                        hessian[:, column] = (
                            -parameters.ordered_gradient_J_m3
                            *_laplacian(direction, parameters.spacing_m)
                        ).ravel()[indices]
                    hessian = 0.5*(hessian+hessian.T)
                    upper = upper_field.ravel()[indices]
                    value = upper.copy()
                    status = np.ones(indices.size, dtype=np.int8)
                    kkt_tolerance = max(1e-18, 1e-8*abs(delta_excess))
                    iterations = 0
                    for iteration in range(max(4*indices.size, 32)):
                        gradient = hessian@value+delta_excess
                        release = (status == 1) & (gradient > kkt_tolerance)
                        release |= ((status == -1)
                                    &(gradient < -kkt_tolerance))
                        if np.any(release):
                            status[release] = 0
                        free = status == 0; fixed = ~free
                        if np.any(free):
                            rhs = np.full(np.count_nonzero(free), -delta_excess)
                            if np.any(fixed):
                                rhs -= hessian[np.ix_(free, fixed)]@value[fixed]
                            value[free] = np.linalg.lstsq(
                                hessian[np.ix_(free, free)], rhs,
                                rcond=1e-13)[0]
                        below = free & (value < 0.0)
                        above = free & (value > upper)
                        if np.any(below) or np.any(above):
                            value[below] = 0.0; status[below] = -1
                            value[above] = upper[above]; status[above] = 1
                            continue
                        gradient = hessian@value+delta_excess
                        violation = gradient.copy()
                        violation[status == -1] = np.minimum(
                            violation[status == -1], 0.0)
                        violation[status == 1] = np.maximum(
                            violation[status == 1], 0.0)
                        iterations = iteration+1
                        if float(np.max(np.abs(violation))) <= kkt_tolerance:
                            break
                    component = y[sign][..., family].copy().reshape(-1)
                    component[indices] = value
                    y[sign][..., family] = component.reshape(shapes[:2])
                    equilibrium_iterations += iterations
                    projected_change = max(projected_change, float(
                        np.max(np.abs(violation))/max(abs(delta_excess), 1e-30)))
        elif active_count < 0.25*sum(value.size for value in totals.values()):
            # A front/capture support can leave only a thin set of cells with
            # any wall line.  In that case optimize only those true degrees of
            # freedom; the zero-capacity exterior remains exactly zero and the
            # spectral gradient is still evaluated on the complete grid.
            # Build the exact small Hessian of the convex quadratic and solve
            # its box-constrained KKT system by an active set.  This avoids the
            # false relative-function convergence of a generic optimizer when
            # the ordered inventory is many decades below the total line.
            hessian = np.empty((active_count, active_count), dtype=float)
            for column in range(active_count):
                direction = np.zeros(active_count); direction[column] = 1.0
                dq = unpack(direction)
                dy = {s: totals[s]*dq[s] for s in ("plus", "minus")}
                dmu_plus, dmu_minus = _ordering_affinity_linear_action(
                    dy["plus"], dy["minus"], systems, orientation_rad,
                    parameters)
                hessian[:, column] = np.concatenate([
                    (dmu_plus*totals["plus"])[active["plus"]],
                    (dmu_minus*totals["minus"])[active["minus"]],
                ])/energy_scale
            hessian = 0.5*(hessian+hessian.T)
            linear = equilibrium_objective(np.zeros(active_count))[1]
            q_sparse = np.zeros(active_count)
            status = np.zeros(active_count, dtype=np.int8)
            maximum_iterations = max(4*active_count, 32)
            projected_change = np.inf
            for iteration in range(maximum_iterations):
                free = status == 0
                fixed = ~free
                if np.any(free):
                    rhs = -linear[free]
                    if np.any(fixed):
                        rhs -= hessian[np.ix_(free, fixed)]@q_sparse[fixed]
                    block = hessian[np.ix_(free, free)]
                    q_sparse[free] = np.linalg.lstsq(
                        block, rhs, rcond=1e-13)[0]
                below = free & (q_sparse < 0.0)
                above = free & (q_sparse > 1.0)
                if np.any(below) or np.any(above):
                    q_sparse[below] = 0.0; status[below] = -1
                    q_sparse[above] = 1.0; status[above] = 1
                    continue
                gradient = hessian@q_sparse+linear
                lower_violation = (status == -1) & (gradient < -1e-13)
                upper_violation = (status == 1) & (gradient > 1e-13)
                if np.any(lower_violation) or np.any(upper_violation):
                    status[lower_violation | upper_violation] = 0
                    continue
                kkt = gradient.copy()
                kkt[status == -1] = np.minimum(kkt[status == -1], 0.0)
                kkt[status == 1] = np.maximum(kkt[status == 1], 0.0)
                projected_change = float(np.max(np.abs(kkt)))
                equilibrium_iterations = iteration+1
                break
            else:
                equilibrium_iterations = maximum_iterations
            fields = unpack(q_sparse)
            y = {s: totals[s]*fields[s] for s in ("plus", "minus")}
        elif parameters.nye_match_coefficient_J_m == 0.0:
            # With no Frank--Bilby penalty (the production V34/V39 setting),
            # the convex obstacle problem has a diagonal Fourier solve.  ADMM
            # treats its spatially varying donor bounds without ill-conditioned
            # generic optimization.
            nx, ny = shapes[:2]
            kx = 2*np.pi*np.fft.fftfreq(nx, d=parameters.spacing_m)
            ky = 2*np.pi*np.fft.fftfreq(ny, d=parameters.spacing_m)
            if nx % 2 == 0:
                kx[nx//2] = 0.0
            if ny % 2 == 0:
                ky[ny//2] = 0.0
            wave_number_squared = (kx[:, None]**2+ky[None, :]**2)[..., None]
            # A small augmented penalty lets the linear excess term move the
            # zero Fourier mode on the same iteration scale as the bounded
            # high modes; rho~lambda_max would require O(10^4) iterations just
            # to traverse a typical reservoir.
            # The former L/2048 penalty left the active capture families far
            # from their obstacle KKT point at the production iteration cap.
            # L/128 is selected from the conditioning of this declared
            # quadratic operator (not from a PF outcome) and resolves both
            # the bounded interface modes and the zero mode.
            rho = max(spectral_lipschitz/128.0, 1e-40)
            denominator = (rho+parameters.ordered_gradient_J_m3
                           *wave_number_squared)
            dual = {s: np.zeros_like(y[s]) for s in ("plus", "minus")}
            delta_excess = (parameters.ordered_excess_J_m
                            -parameters.disordered_excess_J_m)
            maximum_iterations = max(
                4000, int(parameters.ordering_implicit_max_nfev)*40)
            for iteration in range(maximum_iterations):
                previous_z = {s: z[s].copy() for s in ("plus", "minus")}
                primal = dual_change = 0.0
                for sign in ("plus", "minus"):
                    rhs = rho*(z[sign]-dual[sign])-delta_excess
                    spectrum = np.fft.fftn(rhs, axes=(0, 1))
                    y[sign] = np.real(np.fft.ifftn(
                        spectrum/denominator, axes=(0, 1)))
                    z[sign] = np.clip(y[sign]+dual[sign], 0.0, totals[sign])
                    dual[sign] += y[sign]-z[sign]
                    scale = np.maximum(totals[sign], diagnostic_density_floor)
                    primal = max(primal, float(np.max(
                        np.abs(y[sign]-z[sign])/scale)))
                    dual_change = max(dual_change, float(np.max(
                        np.abs(z[sign]-previous_z[sign])/scale)))
                equilibrium_iterations = iteration+1
                projected_change = max(primal, dual_change)
                if projected_change <= 2e-12:
                    break
            y = z
            # ADMM's auxiliary primal residual can be dominated by cells whose
            # physical capacity is at the diagnostic density floor even after
            # the returned bounded state satisfies the convex obstacle KKT
            # conditions.  Qualify the state that will actually be published,
            # using the projected physical chemical-potential residual.
            trial_state = replace(
                inventory,
                wall_ordered_plus_m2=y["plus"],
                wall_tangle_plus_m2=totals["plus"]-y["plus"],
                wall_ordered_minus_m2=y["minus"],
                wall_tangle_minus_m2=totals["minus"]-y["minus"])
            mu = extensive_wall_chemical_potentials_J_m(
                trial_state, systems, topologies, orientation_rad,
                target_nye_m1, parameters)
            projected_change = 0.0
            gradient_scale = max(abs(delta_excess), 1e-30)
            for sign in ("plus", "minus"):
                gradient = (mu[f"ordered_{sign}"]
                            -mu[f"tangle_{sign}"])
                scale = np.maximum(totals[sign], diagnostic_density_floor)
                lower = y[sign] <= 64*np.finfo(float).eps*scale
                upper = y[sign] >= (
                    totals[sign]-64*np.finfo(float).eps*scale)
                violation = gradient.copy()
                violation[lower] = np.minimum(violation[lower], 0.0)
                violation[upper] = np.maximum(violation[upper], 0.0)
                projected_change = max(
                    projected_change,
                    float(np.max(np.abs(violation)))/gradient_scale)
        else:
            step = 0.9/lipschitz_J_m3
            maximum_iterations = max(
                4000, int(parameters.ordering_implicit_max_nfev)*40)
            for iteration in range(maximum_iterations):
                trial_state = replace(
                    inventory,
                    wall_ordered_plus_m2=z["plus"],
                    wall_tangle_plus_m2=totals["plus"]-z["plus"],
                    wall_ordered_minus_m2=z["minus"],
                    wall_tangle_minus_m2=totals["minus"]-z["minus"])
                mu = extensive_wall_chemical_potentials_J_m(
                    trial_state, systems, topologies, orientation_rad,
                    target_nye_m1, parameters)
                next_y = {}
                projected_change = 0.0
                for sign in ("plus", "minus"):
                    gradient = (mu[f"ordered_{sign}"]
                                -mu[f"tangle_{sign}"])
                    next_y[sign] = np.clip(
                        z[sign]-step*gradient, 0.0, totals[sign])
                    projected_change = max(projected_change, float(np.max(
                        np.abs(next_y[sign]-y[sign])
                        /np.maximum(totals[sign], 1.0))))
                next_acceleration = 0.5*(1.0+np.sqrt(
                    1.0+4.0*acceleration*acceleration))
                factor = (acceleration-1.0)/next_acceleration
                z = {s: np.clip(next_y[s]+factor*(next_y[s]-y[s]),
                                0.0, totals[s])
                     for s in ("plus", "minus")}
                y = next_y
                acceleration = next_acceleration
                equilibrium_iterations = iteration+1
                if projected_change <= 2e-12:
                    break
        equilibrium_vector = np.concatenate([
            np.divide(y[s], totals[s], out=np.zeros_like(y[s]),
                      where=active[s])[active[s]]
            for s in ("plus", "minus")])
        # An equilibrium endpoint is admissible only if every active local
        # fraction can reach it on the requested physical clock.  The declared
        # kinetics has |dq/dt| <= a(x), independently of the affinity, so a
        # global maximum attempt exposure cannot qualify a slow or inaccessible
        # pool.  This is a necessary state-dependent guard, not a sufficient
        # finite-time error estimate; the measured transient/asymptotic overlap
        # remains a separate qualification.
        local_attempt_exposure = np.broadcast_to(
            total_dt*np.asarray(_attempt_rate_s(
                stress_for_rate, temperature_K, parameters),
                dtype=float)[..., None], shapes)
        equilibrium_fields = unpack(equilibrium_vector)
        accessibility_violation = 0.0
        maximum_fraction_change = 0.0
        minimum_active_attempt_exposure = np.inf
        for sign in ("plus", "minus"):
            displacement = np.abs(equilibrium_fields[sign]-q0_fields[sign])
            tolerance = 64*np.finfo(float).eps*np.maximum(
                np.maximum(np.abs(equilibrium_fields[sign]),
                           np.abs(q0_fields[sign])), 1.0)
            violation = displacement-local_attempt_exposure-tolerance
            accessibility_violation = max(
                accessibility_violation,
                float(np.max(violation[active[sign]], initial=0.0)))
            maximum_fraction_change = max(
                maximum_fraction_change,
                float(np.max(displacement[active[sign]], initial=0.0)))
            minimum_active_attempt_exposure = min(
                minimum_active_attempt_exposure,
                float(np.min(local_attempt_exposure[active[sign]],
                             initial=np.inf)))
        if accessibility_violation > 0.0:
            result = _accepted_ordering_implicit(
                inventory, systems, topologies, orientation_rad,
                target_nye_m1, stress_Pa, temperature_K, parameters,
                total_dt, alignment=alignment, force_finite_time=True,
                finite_time_endpoint=(equilibrium_vector
                    if projected_change <= 2e-10 else None))
            ledger_index = 2 if alignment is not None else 1
            result[ledger_index].update({
                "stiff_dispatch": (
                    "finite_time_state_inaccessible_asymptotic_endpoint"),
                "maximum_attempt_exposure": attempt_exposure,
                "asymptotic_state_accessibility_passed": False,
                "asymptotic_accessibility_maximum_violation": (
                    accessibility_violation),
                "asymptotic_proposed_maximum_fraction_change": (
                    maximum_fraction_change),
                "minimum_active_local_attempt_exposure": (
                    minimum_active_attempt_exposure),
                "asymptotic_accessibility_semantics": (
                    "necessary componentwise |delta_q| <= integral a_i dt; "
                    "not a sufficient finite-time error certificate"),
            })
            return result
        endpoint_distance_relative = float(np.linalg.norm(
            equilibrium_vector-q0)/max(
                np.linalg.norm(equilibrium_vector), np.linalg.norm(q0),
                np.sqrt(equilibrium_vector.size)*1e-30))
        if (endpoint_distance_relative
                > parameters.ordering_asymptotic_maximum_endpoint_distance_relative):
            result = _accepted_ordering_implicit(
                inventory, systems, topologies, orientation_rad,
                target_nye_m1, stress_Pa, temperature_K, parameters,
                total_dt, alignment=alignment, force_finite_time=True,
                finite_time_endpoint=(equilibrium_vector
                    if projected_change <= 2e-10 else None))
            ledger_index = 2 if alignment is not None else 1
            result[ledger_index].update({
                "stiff_dispatch": (
                    "finite_time_asymptotic_endpoint_distance_unqualified"),
                "maximum_attempt_exposure": attempt_exposure,
                "asymptotic_state_accessibility_passed": True,
                "asymptotic_accessibility_maximum_violation": 0.0,
                "asymptotic_endpoint_distance_relative": (
                    endpoint_distance_relative),
                "asymptotic_endpoint_distance_tolerance_relative": (
                    parameters.
                    ordering_asymptotic_maximum_endpoint_distance_relative),
                "asymptotic_endpoint_distance_bound_passed": False,
                "asymptotic_endpoint_distance_bound_semantics": (
                    "for the declared convex projected gradient flow, distance "
                    "to its equilibrium is non-increasing; initial-to-endpoint "
                    "distance is a conservative finite-horizon error bound"),
            })
            return result
        boundary_tolerance = 128*np.finfo(float).eps
        equilibrium_vector[equilibrium_vector <= boundary_tolerance] = 0.0
        equilibrium_vector[equilibrium_vector >= 1.0-boundary_tolerance] = 1.0
        equilibrium_state = state_from_q(equilibrium_vector)
        equilibrium_transfer, _, _ = ordering_residual(
            equilibrium_state, systems, topologies, orientation_rad,
            target_nye_m1, stress_Pa, temperature_K, parameters)
        normalized_remainder = 0.0
        for sign in ("plus", "minus"):
            value = np.divide(
                total_dt*np.abs(equilibrium_transfer[sign]),
                np.maximum(totals[sign], diagnostic_density_floor),
                out=np.zeros_like(totals[sign]), where=active[sign])
            normalized_remainder = max(normalized_remainder, float(np.max(value)))
        asymptotic_tolerance = max(
            parameters.ordering_implicit_residual_tolerance, 1e-3)
        if (normalized_remainder > asymptotic_tolerance
                or projected_change > 2e-10):
            result = _accepted_ordering_implicit(
                inventory, systems, topologies, orientation_rad,
                target_nye_m1, stress_Pa, temperature_K, parameters,
                total_dt, alignment=alignment, force_finite_time=True)
            ledger_index = 2 if alignment is not None else 1
            result[ledger_index].update({
                "stiff_dispatch": "finite_time_asymptotic_solver_unqualified",
                "maximum_attempt_exposure": attempt_exposure,
                "asymptotic_solver_qualified": False,
                "asymptotic_trial_normalized_remainder": normalized_remainder,
                "asymptotic_trial_projected_change": projected_change,
            })
            return result
        # Reuse the common endpoint ledger below without presenting the
        # minimization as a backward-Euler root.
        class _AsymptoticResult:
            pass
        solution = _AsymptoticResult()
        solution.x = equilibrium_vector
        solution.cost = 0.5*normalized_remainder**2
        solution.optimality = projected_change
        solution.nfev = equilibrium_iterations
        solution.njev = equilibrium_iterations
        solution.success = True
        maximum_residual = normalized_remainder
        integration_method = "bounded_convex_asymptotic"
        solver_message = "projected convex energy convergence"
    else:
        # At intermediate exposure, integrate the actual nonlinear rate law
        # rather than treating an accurately solved stationary endpoint as a
        # finite-time certificate.  This branch is also the independently
        # selectable fallback used to overlap the asymptotic switch.
        def finite_time_rhs(_, vector):
            clipped = np.clip(vector, 0.0, 1.0)
            candidate = state_from_q(clipped)
            transfer, _, _ = ordering_residual(
                candidate, systems, topologies, orientation_rad,
                target_nye_m1, stress_Pa, temperature_K, parameters,
                enforce_availability=False)
            output = []
            q = unpack(clipped)
            for sign in ("plus", "minus"):
                rate = np.divide(
                    transfer[sign], totals[sign],
                    out=np.zeros_like(totals[sign]), where=active[sign])
                rate = rate[active[sign]]
                fraction = q[sign][active[sign]]
                rate[(fraction <= 0.0) & (rate < 0.0)] = 0.0
                rate[(fraction >= 1.0) & (rate > 0.0)] = 0.0
                output.append(rate)
            return np.concatenate(output)

        finite_time_error_control_assessed = False
        finite_time_error_tolerance_satisfied = False
        maximum_error_norm = None
        maximum_accepted_error_norm = None
        rejected_steps = 0
        maximum_accepted_step_s = None
        last_accepted_step_s = None

        if parameters.ordering_finite_time_backend == "matrix_free_adaptive_dop853":
            finite = solve_ivp(
                finite_time_rhs, (0.0, total_dt), q0, method="DOP853",
                rtol=max(parameters.ordering_implicit_residual_tolerance, 1e-7),
                atol=max(parameters.ordering_implicit_residual_tolerance*1e-2,
                         1e-10),
                max_step=(total_dt/max(16, int(np.ceil(
                    attempt_exposure
                    / parameters.ordering_matrix_free_max_attempt_exposure)))))
            if not finite.success:
                raise RuntimeError(
                    "adaptive matrix-free ordering solve failed: "
                    f"message={finite.message}")
            final_vector = np.clip(finite.y[:, -1], 0.0, 1.0)
            nfev = int(finite.nfev); njev = int(finite.njev)
            linear_iterations = 0; nonlinear_iterations = 0
            integration_method = "bounded_finite_time_matrix_free_dop853"
            solver_message = str(finite.message)
            internal_steps = max(int(finite.t.size-1), 1)
            finite_time_error_control_assessed = True
            finite_time_error_tolerance_satisfied = True
            maximum_accepted_step_s = float(np.max(np.diff(finite.t)))
            last_accepted_step_s = float(finite.t[-1]-finite.t[-2])
        elif parameters.ordering_finite_time_backend == "dense_bdf_oracle":
            finite = solve_ivp(
                finite_time_rhs, (0.0, total_dt), q0, method="BDF",
                rtol=max(parameters.ordering_implicit_residual_tolerance, 1e-8),
                atol=max(parameters.ordering_implicit_residual_tolerance*1e-2,
                         1e-11),
                max_step=total_dt/16.0)
            if not finite.success:
                raise RuntimeError(
                    "bounded finite-time ordering solve failed: "
                    f"message={finite.message}")
            final_vector = np.clip(finite.y[:, -1], 0.0, 1.0)
            nfev = int(finite.nfev); njev = int(finite.njev)
            linear_iterations = 0; nonlinear_iterations = 0
            integration_method = "bounded_finite_time_bdf"
            solver_message = str(finite.message)
            internal_steps = max(int(finite.t.size-1), 1)
            finite_time_error_control_assessed = True
            finite_time_error_tolerance_satisfied = True
            maximum_accepted_step_s = float(np.max(np.diff(finite.t)))
            last_accepted_step_s = float(finite.t[-1]-finite.t[-2])
        elif parameters.ordering_finite_time_backend == "matrix_free_projected_rk2":
            # Explicit midpoint/Heun integration of the declared finite-rate
            # ODE.  The rate already contains the exact invariant-set tangent
            # projection at q=0,1; endpoint projection only removes roundoff.
            # Work is O(N log N) per stage and no Jacobian is assembled.
            requested_finite_steps = max(16, int(np.ceil(
                attempt_exposure
                / parameters.ordering_matrix_free_max_attempt_exposure)))
            internal_steps = requested_finite_steps
            step_dt = total_dt/requested_finite_steps
            vector = q0.copy(); nfev = njev = 0
            linear_iterations = nonlinear_iterations = 0
            certified_endpoint_remainder_s = 0.0
            endpoint_switch_distance_relative = None
            executed_finite_steps = 0
            rk2_min_stage_cancellation_ratio = 1.0
            rk2_maximum_stage_norm = 0.0
            rk2_maximum_predictor_projection = 0.0
            rk2_maximum_corrector_projection = 0.0
            endpoint = (None if finite_time_endpoint is None else
                        np.asarray(finite_time_endpoint, dtype=float))
            if endpoint is not None and endpoint.shape != vector.shape:
                raise ValueError("finite-time endpoint shape mismatch")
            endpoint_scale = (None if endpoint is None else max(
                np.linalg.norm(endpoint), np.linalg.norm(q0),
                np.sqrt(endpoint.size)*1e-30))
            for index in range(requested_finite_steps):
                k1 = finite_time_rhs(0.0, vector)
                raw_predictor = vector+step_dt*k1
                predictor = np.clip(raw_predictor, 0.0, 1.0)
                k2 = finite_time_rhs(0.0, predictor)
                denominator = np.linalg.norm(k1)+np.linalg.norm(k2)
                if denominator > 0.0:
                    rk2_min_stage_cancellation_ratio = min(
                        rk2_min_stage_cancellation_ratio,
                        float(np.linalg.norm(k1+k2)/denominator))
                rk2_maximum_stage_norm = max(
                    rk2_maximum_stage_norm,
                    float(np.linalg.norm(k1)), float(np.linalg.norm(k2)))
                rk2_maximum_predictor_projection = max(
                    rk2_maximum_predictor_projection,
                    float(np.max(np.abs(predictor-raw_predictor))))
                raw_corrector = vector+0.5*step_dt*(k1+k2)
                next_vector = np.clip(raw_corrector, 0.0, 1.0)
                rk2_maximum_corrector_projection = max(
                    rk2_maximum_corrector_projection,
                    float(np.max(np.abs(next_vector-raw_corrector))))
                vector = next_vector
                executed_finite_steps = index+1
                if endpoint is not None:
                    endpoint_switch_distance_relative = float(
                        np.linalg.norm(vector-endpoint)/endpoint_scale)
                    if (endpoint_switch_distance_relative <= parameters.
                            ordering_asymptotic_maximum_endpoint_distance_relative):
                        certified_endpoint_remainder_s = (
                            total_dt-executed_finite_steps*step_dt)
                        vector = endpoint.copy()
                        break
            internal_steps = executed_finite_steps
            final_vector = vector
            if certified_endpoint_remainder_s > 0.0:
                integration_method = (
                    "bounded_finite_time_rk2_certified_endpoint_remainder")
                solver_message = (
                    "projected Heun reached the convex endpoint-distance "
                    "certificate; equilibrium represents the remaining clock")
            else:
                integration_method = "bounded_finite_time_matrix_free_rk2"
                solver_message = "projected Heun integration completed full clock"
        else:
            # Backward Euler with the exact global FFT Jacobian-vector action.
            # No Jacobian matrix or physical-space sparsity pattern is formed.
            # Keep the fastest local reaction exposure below the declared
            # backward-Euler solve. This is a numerical resolution rule, not
            # a kinetic cap; every substep is accumulated on the full clock.
            internal_steps = max(16, int(np.ceil(
                attempt_exposure
                / parameters.ordering_matrix_free_max_attempt_exposure)))
            step_dt = total_dt/internal_steps
            vector = q0.copy(); nfev = njev = 0
            linear_iterations = nonlinear_iterations = 0
            tolerance = max(parameters.ordering_implicit_residual_tolerance,
                            1e-8)

            def normalized_rate(vector):
                nonlocal nfev
                nfev += 1
                return finite_time_rhs(0.0, vector)

            def projected_residual(vector, previous):
                rate_value = normalized_rate(vector)
                predicted = previous+step_dt*rate_value
                return vector-np.clip(predicted, 0.0, 1.0)

            def exact_jacobian(vector, previous, duration=None):
                nonlocal njev
                njev += 1
                local_dt = step_dt if duration is None else float(duration)
                candidate = state_from_q(vector)
                mu = extensive_wall_chemical_potentials_J_m(
                    candidate, systems, topologies, orientation_rad,
                    target_nye_m1, parameters)
                stress = np.asarray(stress_Pa, dtype=float)
                if stress.ndim == 3:
                    stress = np.max(np.abs(stress), axis=2)
                attempt = _attempt_rate_s(
                    stress, temperature_K, parameters)[..., None]
                thermal = (parameters.event_length_m/(2*KB_J_K
                           *np.asarray(temperature_K)[..., None]))
                coefficients = {}; interior = {}
                fields = unpack(vector)
                trial_transfer = ordering_residual(
                    candidate, systems, topologies, orientation_rad,
                    target_nye_m1, stress_Pa, temperature_K, parameters,
                    enforce_availability=False)[0]
                previous_fields = unpack(previous)
                for sign in ("plus", "minus"):
                    affinity = ((mu[f"ordered_{sign}"]
                                 -mu[f"tangle_{sign}"])*thermal)
                    coefficients[sign] = (-attempt
                        *(1.0-np.tanh(affinity)**2)*thermal)
                    rate_value = np.divide(
                        trial_transfer[sign],
                        totals[sign], out=np.zeros_like(totals[sign]),
                        where=active[sign])
                    predicted = previous_fields[sign]+step_dt*rate_value
                    interior[sign] = (predicted > 0.0) & (predicted < 1.0)

                def matvec(direction):
                    dq = unpack(direction)
                    dy = {s: totals[s]*dq[s] for s in ("plus", "minus")}
                    dmu_plus, dmu_minus = _ordering_affinity_linear_action(
                        dy["plus"], dy["minus"], systems, orientation_rad,
                        parameters)
                    dmu = {"plus": dmu_plus, "minus": dmu_minus}
                    output = []
                    for sign in ("plus", "minus"):
                        drate = coefficients[sign]*dmu[sign]
                        field = dq[sign]-local_dt*interior[sign]*drate
                        output.append(field[active[sign]])
                    return np.concatenate(output)
                def rmatvec(direction):
                    # H = d(mu_ordered-mu_tangle)/d(rho_ordered)
                    # is the symmetric Hessian of the quadratic ordering
                    # energy.  Apply J^T without assembling either H or J:
                    # J = I-dt*D_active*C*H*T.
                    value = unpack(direction)
                    weighted = {
                        sign: coefficients[sign]*interior[sign]*value[sign]
                        for sign in ("plus", "minus")}
                    adjoint_plus, adjoint_minus = (
                        _ordering_affinity_linear_action(
                            weighted["plus"], weighted["minus"], systems,
                            orientation_rad, parameters))
                    adjoint = {
                        "plus": adjoint_plus, "minus": adjoint_minus}
                    output = []
                    for sign in ("plus", "minus"):
                        field = (value[sign]
                                 -local_dt*totals[sign]*adjoint[sign])
                        output.append(field[active[sign]])
                    return np.concatenate(output)
                return LinearOperator(
                    (active_count, active_count), matvec=matvec,
                    rmatvec=rmatvec, dtype=float)

            def backward_euler_trial(previous, duration):
                """One bounded BE trial; caller owns acceptance and the clock."""
                initial = np.clip(
                    previous+duration*normalized_rate(previous), 0.0, 1.0)

                def trial_residual(trial):
                    rate_value = normalized_rate(trial)
                    return trial-np.clip(
                        previous+duration*rate_value, 0.0, 1.0)

                def trial_jacobian(trial):
                    return exact_jacobian(trial, previous, duration)

                local = least_squares(
                    trial_residual, initial, jac=trial_jacobian,
                    bounds=(0.0, 1.0), ftol=None, xtol=tolerance,
                    gtol=tolerance,
                    max_nfev=int(parameters.ordering_implicit_max_nfev),
                    tr_solver="lsmr")
                trial = np.clip(local.x, 0.0, 1.0)
                local_residual = trial_residual(trial)
                return trial, local, float(np.max(np.abs(local_residual)))

            def rosenbrock_trial(previous, duration):
                """One linearly implicit Euler trial with the exact FFT JVP."""
                rate_value = normalized_rate(previous)
                counter = [0]
                def count_iteration(_):
                    counter[0] += 1
                delta, info = gmres(
                    exact_jacobian(previous, previous, duration),
                    duration*rate_value, rtol=1e-9, atol=1e-12,
                    restart=30, maxiter=100, callback=count_iteration,
                    callback_type="pr_norm")
                if info != 0 or np.any(~np.isfinite(delta)):
                    return None, counter[0], int(info)
                return np.clip(previous+delta, 0.0, 1.0), counter[0], 0

            if (parameters.ordering_finite_time_backend
                    == "matrix_free_adaptive_rosenbrock_euler"):
                requested_finite_steps = internal_steps
                step_dt = total_dt/requested_finite_steps
                time = 0.0
                accepted_steps = 0
                rejected_steps = 0
                consecutive_rejections = 0
                maximum_error_norm = 0.0
                maximum_accepted_error_norm = 0.0
                maximum_accepted_step_s = 0.0
                last_accepted_step_s = 0.0
                minimum_step = max(
                    32*np.finfo(float).eps*total_dt,
                    np.nextafter(0.0, 1.0))
                relative_tolerance = float(
                    parameters.ordering_finite_relative_tolerance)
                absolute_tolerance = float(
                    parameters.ordering_finite_absolute_tolerance)
                while time < total_dt:
                    duration = min(step_dt, total_dt-time)
                    full, full_iterations, full_info = rosenbrock_trial(
                        vector, duration)
                    half, half_iterations, half_info = rosenbrock_trial(
                        vector, 0.5*duration)
                    if half is None:
                        half2 = None; half2_iterations = 0; half2_info = half_info
                    else:
                        half2, half2_iterations, half2_info = rosenbrock_trial(
                            half, 0.5*duration)
                    linear_iterations += (
                        full_iterations+half_iterations+half2_iterations)
                    nonlinear_ok = bool(
                        full is not None and half is not None
                        and half2 is not None and full_info == 0
                        and half_info == 0 and half2_info == 0)
                    if nonlinear_ok:
                        scale = (absolute_tolerance
                                 +relative_tolerance*np.maximum(
                                     np.maximum(np.abs(half2), np.abs(vector)),
                                     1e-12))
                        error_norm = float(np.max(np.abs(half2-full)/scale))
                    else:
                        error_norm = np.inf
                    maximum_error_norm = max(maximum_error_norm, error_norm)
                    if nonlinear_ok and error_norm <= 1.0:
                        vector = half2
                        time += duration
                        accepted_steps += 1
                        consecutive_rejections = 0
                        maximum_accepted_error_norm = max(
                            maximum_accepted_error_norm, error_norm)
                        maximum_accepted_step_s = max(
                            maximum_accepted_step_s, duration)
                        last_accepted_step_s = duration
                        factor = (1.5 if error_norm == 0.0 else
                                  np.clip(0.9/np.sqrt(error_norm), 0.5, 1.5))
                        step_dt = min(total_dt-time, duration*factor)
                    else:
                        rejected_steps += 1
                        consecutive_rejections += 1
                        if consecutive_rejections > int(
                                parameters.ordering_finite_max_rejections):
                            raise RuntimeError(
                                "adaptive matrix-free Rosenbrock exceeded the "
                                "rejected-trial limit")
                        factor = (0.25 if not nonlinear_ok else
                                  np.clip(0.8/np.sqrt(max(error_norm, 1e-16)),
                                          0.1, 0.5))
                        step_dt = duration*factor
                        if step_dt < minimum_step:
                            raise RuntimeError(
                                "adaptive matrix-free Rosenbrock reached the "
                                "minimum physical timestep")
                internal_steps = accepted_steps
                final_vector = vector
                finite_time_error_control_assessed = True
                finite_time_error_tolerance_satisfied = bool(
                    maximum_accepted_error_norm <= 1.0)
                integration_method = (
                    "bounded_finite_time_matrix_free_adaptive_ros1")
                solver_message = (
                    "step-doubled exact-FFT-JVP Rosenbrock-Euler completed "
                    "the full clock with local error control")
            elif (parameters.ordering_finite_time_backend
                    == "matrix_free_adaptive_backward_euler"):
                # L-stable BE with step doubling.  The accepted state is the
                # two-half-step result; the full step is an independent local
                # truncation-error estimate.  Failed nonlinear/error trials do
                # not publish state or advance physical time.
                requested_finite_steps = internal_steps
                step_dt = total_dt/requested_finite_steps
                time = 0.0
                accepted_steps = 0
                rejected_steps = 0
                consecutive_rejections = 0
                maximum_error_norm = 0.0
                maximum_accepted_error_norm = 0.0
                maximum_accepted_step_s = 0.0
                last_accepted_step_s = 0.0
                minimum_step = max(
                    32*np.finfo(float).eps*total_dt,
                    np.nextafter(0.0, 1.0))
                relative_tolerance = float(
                    parameters.ordering_finite_relative_tolerance)
                absolute_tolerance = float(
                    parameters.ordering_finite_absolute_tolerance)
                while time < total_dt:
                    duration = min(step_dt, total_dt-time)
                    full, full_solve, full_residual = backward_euler_trial(
                        vector, duration)
                    half, half_solve, half_residual = backward_euler_trial(
                        vector, 0.5*duration)
                    half2, half2_solve, half2_residual = backward_euler_trial(
                        half, 0.5*duration)
                    nonlinear_iterations += int(
                        full_solve.nfev+half_solve.nfev+half2_solve.nfev)
                    linear_iterations += int(
                        (full_solve.njev or 0)+(half_solve.njev or 0)
                        +(half2_solve.njev or 0))
                    nonlinear_ok = bool(
                        full_solve.success and half_solve.success
                        and half2_solve.success
                        and max(full_residual, half_residual, half2_residual)
                        <= tolerance)
                    scale = (absolute_tolerance+relative_tolerance*np.maximum(
                        np.maximum(np.abs(half2), np.abs(vector)), 1e-12))
                    error_norm = float(np.max(np.abs(half2-full)/scale))
                    maximum_error_norm = max(maximum_error_norm, error_norm)
                    if nonlinear_ok and error_norm <= 1.0:
                        vector = half2
                        time += duration
                        accepted_steps += 1
                        consecutive_rejections = 0
                        maximum_accepted_error_norm = max(
                            maximum_accepted_error_norm, error_norm)
                        maximum_accepted_step_s = max(
                            maximum_accepted_step_s, duration)
                        last_accepted_step_s = duration
                        factor = (2.0 if error_norm == 0.0 else
                                  np.clip(0.9/np.sqrt(error_norm), 0.5, 2.0))
                        step_dt = min(total_dt-time, duration*factor)
                    else:
                        rejected_steps += 1
                        consecutive_rejections += 1
                        if consecutive_rejections > int(
                                parameters.ordering_finite_max_rejections):
                            raise RuntimeError(
                                "adaptive matrix-free backward-Euler exceeded "
                                "the rejected-trial limit")
                        factor = (0.25 if not nonlinear_ok else
                                  np.clip(0.8/np.sqrt(max(error_norm, 1e-16)),
                                          0.1, 0.5))
                        step_dt = duration*factor
                        if step_dt < minimum_step:
                            raise RuntimeError(
                                "adaptive matrix-free backward-Euler reached "
                                "the minimum physical timestep")
                internal_steps = accepted_steps
                final_vector = vector
                finite_time_error_control_assessed = True
                finite_time_error_tolerance_satisfied = bool(
                    maximum_accepted_error_norm <= 1.0)
                integration_method = (
                    "bounded_finite_time_matrix_free_adaptive_be")
                solver_message = (
                    "step-doubled exact-FFT-JVP backward Euler completed the "
                    "full clock with local error control")
            elif (parameters.ordering_finite_time_backend
                    == "matrix_free_exponential_rosenbrock"):
                for _ in range(internal_steps):
                    previous = vector.copy()
                    rate_value = normalized_rate(previous)
                    backward = exact_jacobian(previous, previous)
                    def augmented_matvec(value):
                        flat = np.asarray(value).reshape(-1)
                        state_direction = flat[:-1]
                        top = ((state_direction-backward.matvec(
                            state_direction))+step_dt*flat[-1]*rate_value)
                        return np.concatenate([top, [0.0]])
                    def augmented_rmatvec(value):
                        flat = np.asarray(value).reshape(-1)
                        state_direction = flat[:-1]
                        top = (state_direction-backward.rmatvec(
                            state_direction))
                        bottom = step_dt*np.dot(rate_value, state_direction)
                        return np.concatenate([top, [bottom]])
                    augmented = LinearOperator(
                        (active_count+1, active_count+1),
                        matvec=augmented_matvec,
                        rmatvec=augmented_rmatvec, dtype=float)
                    seed = np.zeros(active_count+1); seed[-1] = 1.0
                    evolved = expm_multiply(augmented, seed, traceA=0.0)
                    delta = evolved[:-1]
                    if np.any(~np.isfinite(delta)):
                        raise RuntimeError(
                            "matrix-free exponential Rosenbrock action failed")
                    vector = np.clip(previous+delta, 0.0, 1.0)
                    linear_iterations += 1
                final_vector = vector
                integration_method = "bounded_finite_time_matrix_free_exprb1"
                solver_message = "FFT-JVP exponential Rosenbrock completed full clock"
            elif (parameters.ordering_finite_time_backend
                    == "matrix_free_rosenbrock_euler"):
                for _ in range(internal_steps):
                    previous = vector.copy()
                    rate_value = normalized_rate(previous)
                    counter = [0]
                    def count_iteration(_):
                        counter[0] += 1
                    delta, info = gmres(
                        exact_jacobian(previous, previous),
                        step_dt*rate_value, rtol=1e-8, atol=1e-11,
                        restart=30, maxiter=80, callback=count_iteration,
                        callback_type="pr_norm")
                    linear_iterations += counter[0]
                    if info != 0 or np.any(~np.isfinite(delta)):
                        raise RuntimeError(
                            "matrix-free Rosenbrock linear solve failed: "
                            f"gmres_info={info}")
                    vector = np.clip(previous+delta, 0.0, 1.0)
                final_vector = vector
                integration_method = "bounded_finite_time_matrix_free_ros1"
                solver_message = "exact FFT JVP Rosenbrock-Euler completed full clock"
            else:
                for _ in range(internal_steps):
                    previous = vector.copy()
                    vector, local, local_residual = backward_euler_trial(
                        previous, step_dt)
                    nonlinear_iterations += int(local.nfev)
                    linear_iterations += int(local.njev or 0)
                    if not local.success or local_residual > tolerance:
                        raise RuntimeError(
                            "matrix-free ordering trust-region solve failed: "
                            f"success={local.success}, "
                            f"residual={local_residual:.17g}, "
                            f"message={local.message}")
                final_vector = vector
                integration_method = "bounded_finite_time_matrix_free_be"
                solver_message = "exact FFT JVP/JTV trust-region LSMR converged"

        class _FiniteTimeResult:
            pass
        solution = _FiniteTimeResult()
        solution.x = final_vector
        endpoint_scaled_rate = finite_time_rhs(total_dt, solution.x)*total_dt
        solution.cost = 0.5*float(np.dot(
            endpoint_scaled_rate, endpoint_scaled_rate))
        solution.optimality = float(np.max(np.abs(endpoint_scaled_rate)))
        solution.nfev = nfev
        solution.njev = njev
        solution.success = True
        maximum_residual = solution.optimality

    last = {}
    def residual(vector):
        candidate = state_from_q(vector)
        transfer, turnover, mu = ordering_residual(
            candidate, systems, topologies, orientation_rad, target_nye_m1,
            stress_Pa, temperature_K, parameters,
            enforce_availability=False)
        q = unpack(vector); values = []
        for sign in ("plus", "minus"):
            scaled_rate = np.divide(
                transfer[sign], totals[sign], out=np.zeros_like(totals[sign]),
                where=active[sign])
            implicit_trial = q0_fields[sign]+total_dt*scaled_rate
            field = q[sign]-np.clip(implicit_trial, 0.0, 1.0)
            values.append(field[active[sign]])
        last.update(candidate=candidate, transfer=transfer,
                    turnover=turnover, mu=mu)
        return np.concatenate(values)

    def jacobian(vector):
        candidate = state_from_q(vector)
        mu = extensive_wall_chemical_potentials_J_m(
            candidate, systems, topologies, orientation_rad, target_nye_m1,
            parameters)
        stress = np.asarray(stress_Pa, dtype=float)
        if stress.ndim == 3:
            stress = np.max(np.abs(stress), axis=2)
        attempt = _attempt_rate_s(stress, temperature_K, parameters)[..., None]
        thermal = (parameters.event_length_m/(2*KB_J_K
                   *np.asarray(temperature_K)[..., None]))
        coefficients = {}
        interior = {}
        for sign in ("plus", "minus"):
            affinity = ((mu[f"ordered_{sign}"]-mu[f"tangle_{sign}"])
                        *thermal)
            coefficients[sign] = (-attempt*totals[sign]
                                  *(1.0-np.tanh(affinity)**2)*thermal)
            raw_rate = (-attempt*totals[sign]*np.tanh(affinity))
            trial = q0_fields[sign]+total_dt*np.divide(
                raw_rate, totals[sign], out=np.zeros_like(raw_rate),
                where=active[sign])
            interior[sign] = (trial > 0.0) & (trial < 1.0)

        def matvec(direction):
            dq = unpack(direction)
            dy = {s: totals[s]*dq[s] for s in ("plus", "minus")}
            dmu_plus, dmu_minus = _ordering_affinity_linear_action(
                dy["plus"], dy["minus"], systems, orientation_rad, parameters)
            dmu = {"plus": dmu_plus, "minus": dmu_minus}
            output = []
            for sign in ("plus", "minus"):
                dr = coefficients[sign]*dmu[sign]
                scaled = np.divide(dr, totals[sign],
                                   out=np.zeros_like(dr), where=active[sign])
                field = dq[sign]-total_dt*interior[sign]*scaled
                output.append(field[active[sign]])
            return np.concatenate(output)

        def rmatvec(direction):
            v = unpack(direction)
            weighted = {}
            for sign in ("plus", "minus"):
                weighted[sign] = np.divide(
                    coefficients[sign]*interior[sign]*v[sign], totals[sign],
                    out=np.zeros_like(v[sign]), where=active[sign])
            lv_plus, lv_minus = _ordering_affinity_linear_action(
                weighted["plus"], weighted["minus"], systems,
                orientation_rad, parameters)
            lv = {"plus": lv_plus, "minus": lv_minus}
            output = []
            for sign in ("plus", "minus"):
                field = v[sign]-total_dt*totals[sign]*lv[sign]
                output.append(field[active[sign]])
            return np.concatenate(output)
        return LinearOperator((active_count, active_count), matvec=matvec,
                              rmatvec=rmatvec, dtype=float)

    if solution is None:
        epsilon = 8*np.finfo(float).eps
        projected_seed = q0-residual(q0)
        initial = np.minimum(np.maximum(projected_seed, epsilon), 1.0-epsilon)
        solution = least_squares(
            residual, initial, jac=jacobian, bounds=(0.0, 1.0),
            ftol=None, xtol=parameters.ordering_implicit_residual_tolerance,
            gtol=parameters.ordering_implicit_residual_tolerance,
            max_nfev=int(parameters.ordering_implicit_max_nfev),
            tr_solver="lsmr")
        final_residual = residual(solution.x)
        maximum_residual = float(np.max(np.abs(final_residual)))
        integration_method = "bounded_backward_euler"
        solver_message = str(solution.message)
        if (not solution.success or maximum_residual
                > parameters.ordering_implicit_residual_tolerance):
            raise RuntimeError(
                "bounded implicit ordering solve failed: "
                f"success={solution.success}, max_scaled_residual="
                f"{maximum_residual:.6e}, message={solution.message}")
    updated = state_from_q(solution.x)
    endpoint_transfer, endpoint_turnover, endpoint_mu = ordering_residual(
        updated, systems, topologies, orientation_rad, target_nye_m1,
        stress_Pa, temperature_K, parameters)
    extent = {s: (getattr(updated, f"wall_ordered_{s}_m2")
                  -getattr(inventory, f"wall_ordered_{s}_m2"))
              for s in ("plus", "minus")}
    topology = None; updated_alignment = alignment
    if alignment is not None:
        from .wall_topology_supply import apply_signed_ordering_extent
        updated, updated_alignment, topology = apply_signed_ordering_extent(
            inventory, alignment, extent["plus"], extent["minus"], systems,
            orientation_rad, topologies)
    before_energy = extensive_wall_energy_components_J_m3(
        inventory, systems, topologies, orientation_rad, target_nye_m1,
        parameters)["total"]
    after_energy = extensive_wall_energy_components_J_m3(
        updated, systems, topologies, orientation_rad, target_nye_m1,
        parameters)["total"]
    aggregate = {
        "transfer_m2_s": endpoint_transfer,
        "turnover_m2_s": endpoint_turnover,
        "accepted_transfer_m2_s": {s: extent[s]/total_dt
                                    for s in ("plus", "minus")},
        "chemical_potential_J_m": endpoint_mu,
        "free_energy_rate_W_m3": (after_energy-before_energy)/total_dt,
        "internal_substeps": int(locals().get("internal_steps", 1)),
        "maximum_internal_substep_s": (total_dt/max(
            int(locals().get("internal_steps", 1)), 1)
            if locals().get("maximum_accepted_step_s") is None
            else float(locals()["maximum_accepted_step_s"])),
        "last_internal_substep_s": (total_dt/max(
            int(locals().get("internal_steps", 1)), 1)
            if locals().get("last_accepted_step_s") is None
            else float(locals()["last_accepted_step_s"])),
        "complete_elapsed_time_s": total_dt,
        "discarded_reaction_time_s": 0.0,
        "stationary_remainder_s": 0.0,
        "certified_endpoint_remainder_s": float(locals().get(
            "certified_endpoint_remainder_s", 0.0)),
        "endpoint_switch_distance_relative": locals().get(
            "endpoint_switch_distance_relative"),
        "requested_internal_substeps": int(locals().get(
            "requested_finite_steps", 1)),
        "internal_resolution_limit_active": False,
        "integration_method": integration_method,
        "stiff_dispatch": "qualified_asymptotic" if integration_method.endswith(
            "asymptotic") else "finite_time_bdf",
        "maximum_attempt_exposure": attempt_exposure,
        "asymptotic_state_accessibility_passed": (
            True if integration_method.endswith("asymptotic") else None),
        "asymptotic_accessibility_maximum_violation": (
            accessibility_violation if integration_method.endswith("asymptotic")
            else None),
        "asymptotic_proposed_maximum_fraction_change": (
            maximum_fraction_change if integration_method.endswith("asymptotic")
            else None),
        "asymptotic_endpoint_distance_relative": (
            endpoint_distance_relative if integration_method.endswith(
                "asymptotic") else None),
        "asymptotic_endpoint_distance_tolerance_relative": (
            parameters.ordering_asymptotic_maximum_endpoint_distance_relative
            if integration_method.endswith("asymptotic") else None),
        "asymptotic_endpoint_distance_bound_passed": (
            bool(endpoint_distance_relative <= parameters.
                 ordering_asymptotic_maximum_endpoint_distance_relative)
            if integration_method.endswith("asymptotic") else None),
        "asymptotic_endpoint_distance_bound_semantics": (
            "for the declared convex projected gradient flow, distance to its "
            "equilibrium is non-increasing; initial-to-endpoint distance is a "
            "conservative finite-horizon error bound"
            if integration_method.endswith("asymptotic") else None),
        "minimum_active_local_attempt_exposure": (
            minimum_active_attempt_exposure
            if integration_method.endswith("asymptotic") else None),
        "asymptotic_accessibility_semantics": (
            "necessary componentwise |delta_q| <= integral a_i dt; not a "
            "sufficient finite-time error certificate"
            if integration_method.endswith("asymptotic") else None),
        "implicit_success": True,
        "implicit_max_scaled_residual": maximum_residual,
        "implicit_cost": float(solution.cost),
        "implicit_optimality": float(solution.optimality),
        "implicit_nfev": int(solution.nfev),
        "implicit_njev": int(solution.njev or 0),
        "linear_iterations": int(locals().get("linear_iterations", 0)),
        "nonlinear_iterations": int(locals().get("nonlinear_iterations", 0)),
        "active_degrees_of_freedom": int(active_count),
        "dense_jacobian_bytes_avoided": int(8*active_count*active_count),
        "finite_time_backend": parameters.ordering_finite_time_backend,
        "solver_message": solver_message,
        "asymptotic_endpoint_inventory_change_bound_relative": (
            maximum_residual if integration_method.endswith("asymptotic")
            else None),
        "asymptotic_endpoint_drift_over_requested_time_relative": (
            maximum_residual if integration_method.endswith("asymptotic")
            else None),
        "asymptotic_endpoint_diagnostic_semantics": (
            "H_times_endpoint_rate_over_local_inventory_floor; stationary_"
            "endpoint_drift_diagnostic_not_a_finite_time_kinetic_error_bound"
            if integration_method.endswith("asymptotic") else None),
        "finite_time_error_control_assessed": bool(locals().get(
            "finite_time_error_control_assessed", False)),
        "finite_time_error_tolerance_satisfied": bool(locals().get(
            "finite_time_error_tolerance_satisfied", False)),
        "finite_time_maximum_trial_error_norm": locals().get(
            "maximum_error_norm"),
        "finite_time_maximum_accepted_error_norm": locals().get(
            "maximum_accepted_error_norm"),
        "finite_time_rejected_trials": int(locals().get(
            "rejected_steps", 0)),
        "rk2_minimum_stage_cancellation_ratio": locals().get(
            "rk2_min_stage_cancellation_ratio"),
        "rk2_maximum_stage_norm_s-1": locals().get(
            "rk2_maximum_stage_norm"),
        "rk2_maximum_predictor_projection": locals().get(
            "rk2_maximum_predictor_projection"),
        "rk2_maximum_corrector_projection": locals().get(
            "rk2_maximum_corrector_projection"),
        "finite_time_relative_tolerance": (
            parameters.ordering_finite_relative_tolerance
            if locals().get("finite_time_error_control_assessed", False)
            else None),
        "finite_time_absolute_tolerance": (
            parameters.ordering_finite_absolute_tolerance
            if locals().get("finite_time_error_control_assessed", False)
            else None),
        # An embedded/local controller qualifies accepted local trials.  A
        # single solve does not independently certify its own global endpoint;
        # that requires a tighter solve or a distinct stable reference.
        "finite_time_local_error_control_passed": bool(
            locals().get("finite_time_error_control_assessed", False)
            and locals().get("finite_time_error_tolerance_satisfied", False)),
        "finite_time_kinetic_accuracy_certified_by_this_solve": False,
        "finite_time_accuracy_status": (
            "LOCAL_ERROR_CONTROL_PASSED_INDEPENDENT_ENDPOINT_COMPARISON_REQUIRED"
            if (locals().get("finite_time_error_control_assessed", False)
                and locals().get("finite_time_error_tolerance_satisfied", False))
            else "UNASSESSED_OR_LOCAL_ERROR_CONTROL_FAILED"),
        "asymptotic_relative_diagnostic_density_floor_m2": (
            diagnostic_density_floor if integration_method.endswith("asymptotic")
            else None),
        "turnover_quadrature": "backward_euler_endpoint",
        "free_energy_rate_source": "exact_discrete_endpoint_difference",
    }
    if alignment is not None:
        aggregate["topology_subcycled"] = False
        aggregate["topology_ledger"] = topology
        return updated, updated_alignment, aggregate, 1.0
    return updated, aggregate, 1.0


def accepted_ordering_step(inventory, systems, topologies, orientation_rad,
                           target_nye_m1, stress_Pa, temperature_K, parameters,
                           dt_s, alignment=None):
    """Advance the reversible ordering channel on its own physical clock.

    Before V38, a capped ordering extent was accepted once per outer Mura
    interval and the unconsumed reaction time was discarded.  The cap then
    acted as an outer-timestep-dependent kinetic coefficient.  Here the same
    conservative map is subcycled over the complete elapsed time and all
    reported rates are actual time averages.
    """
    if parameters.ordering_integration_method in (
            "implicit_backward_euler", "finite_time_bdf"):
        return _accepted_ordering_implicit(
            inventory, systems, topologies, orientation_rad, target_nye_m1,
            stress_Pa, temperature_K, parameters, dt_s, alignment=alignment,
            force_finite_time=(parameters.ordering_integration_method
                               == "finite_time_bdf"))
    total_dt = float(dt_s)
    if not np.isfinite(total_dt) or total_dt <= 0.0:
        raise ValueError("ordering timestep must be finite and positive")
    maximum_substep = float(parameters.ordering_internal_substep_s)
    requested_count = max(1, int(np.ceil(total_dt/maximum_substep)))
    count = min(requested_count, int(parameters.ordering_internal_max_substeps))
    current = inventory
    current_alignment = alignment
    topology_totals = None
    transfer_integral = turnover_integral = mu_integral = None
    accepted_extent = None
    energy_integral = None
    minimum_scale = 1.0
    executed = 0
    advanced_time = 0.0
    last_substep = 0.0
    stationary_remainder = 0.0
    for index in range(count):
        substep = min(maximum_substep, total_dt-advanced_time)
        last_substep = substep
        updated, ledger, scale = _accepted_ordering_substep(
            current, systems, topologies, orientation_rad, target_nye_m1,
            stress_Pa, temperature_K, parameters, substep)
        if current_alignment is not None:
            from .wall_topology_supply import apply_signed_ordering_extent
            extent_plus = (updated.wall_ordered_plus_m2
                           -current.wall_ordered_plus_m2)
            extent_minus = (updated.wall_ordered_minus_m2
                            -current.wall_ordered_minus_m2)
            updated, current_alignment, topology = apply_signed_ordering_extent(
                current, current_alignment, extent_plus, extent_minus, systems,
                orientation_rad, topologies)
            if topology_totals is None:
                topology_totals = {
                    "operator": "subcycled_signed_topology_ordering_extent",
                    "sign": {sign: {
                        name: np.zeros_like(value)
                        for name, value in topology["sign"][sign].items()}
                        for sign in ("plus", "minus")},
                    "total_nye_residual_m1": np.zeros_like(
                        topology["total_nye_residual_m1"]),
                }
            for sign in ("plus", "minus"):
                for name, value in topology["sign"][sign].items():
                    topology_totals["sign"][sign][name] += value
            topology_totals["total_nye_residual_m1"] += topology[
                "total_nye_residual_m1"]
        if transfer_integral is None:
            transfer_integral = {sign: np.zeros_like(ledger["transfer_m2_s"][sign])
                                 for sign in ("plus", "minus")}
            turnover_integral = {sign: np.zeros_like(ledger["turnover_m2_s"][sign])
                                 for sign in ("plus", "minus")}
            accepted_extent = {sign: np.zeros_like(
                ledger["accepted_transfer_m2_s"][sign])
                               for sign in ("plus", "minus")}
            mu_integral = {name: np.zeros_like(value)
                           for name, value in ledger["chemical_potential_J_m"].items()}
            energy_integral = np.zeros_like(ledger["free_energy_rate_W_m3"])
        for sign in ("plus", "minus"):
            transfer_integral[sign] += substep*ledger["transfer_m2_s"][sign]
            turnover_integral[sign] += substep*ledger["turnover_m2_s"][sign]
            accepted_extent[sign] += (
                substep*ledger["accepted_transfer_m2_s"][sign])
        for name in mu_integral:
            mu_integral[name] += substep*ledger["chemical_potential_J_m"][name]
        energy_integral += substep*ledger["free_energy_rate_W_m3"]
        minimum_scale = min(minimum_scale, float(scale))
        current = updated
        executed = index+1
        advanced_time += substep
        remaining = total_dt-advanced_time
        if remaining > 0.0:
            absolute_rate = sum(float(np.sum(np.abs(
                ledger["accepted_transfer_m2_s"][sign]),
                dtype=np.longdouble)) for sign in ("plus", "minus"))
            line_scale = sum(float(np.sum(
                getattr(current, f"wall_{kind}_{sign}_m2"),
                dtype=np.longdouble))
                for kind in ("tangle", "ordered")
                for sign in ("plus", "minus"))
            if (remaining*absolute_rate <=
                    parameters.ordering_stationary_remainder_relative_tolerance
                    *max(line_scale, 1.0)):
                stationary_remainder = remaining
                for sign in ("plus", "minus"):
                    transfer_integral[sign] += (
                        remaining*ledger["transfer_m2_s"][sign])
                    turnover_integral[sign] += (
                        remaining*ledger["turnover_m2_s"][sign])
                for name in mu_integral:
                    mu_integral[name] += (
                        remaining*ledger["chemical_potential_J_m"][name])
                break
    if advanced_time+stationary_remainder < total_dt-64*np.finfo(float).eps*total_dt:
        raise RuntimeError(
            "ordering reaction did not reach the declared stationary tolerance "
            "within the internal substep limit")
    aggregate = {
        "transfer_m2_s": {key: value/total_dt
                          for key, value in transfer_integral.items()},
        "turnover_m2_s": {key: value/total_dt
                          for key, value in turnover_integral.items()},
        "accepted_transfer_m2_s": {key: value/total_dt
                                   for key, value in accepted_extent.items()},
        "chemical_potential_J_m": {key: value/total_dt
                                   for key, value in mu_integral.items()},
        "free_energy_rate_W_m3": energy_integral/total_dt,
        "internal_substeps": executed,
        "maximum_internal_substep_s": maximum_substep,
        "last_internal_substep_s": last_substep,
        "complete_elapsed_time_s": total_dt,
        "discarded_reaction_time_s": 0.0,
        "stationary_remainder_s": stationary_remainder,
        "requested_internal_substeps": requested_count,
        "internal_resolution_limit_active": bool(count < requested_count),
        "integration_method": "complete_time_explicit_subcycling",
    }
    if alignment is not None:
        aggregate["topology_subcycled"] = True
        aggregate["topology_ledger"] = topology_totals
        return current, current_alignment, aggregate, minimum_scale
    return current, aggregate, minimum_scale


def accepted_ordering_step_jvp(inventory, direction, systems, topologies,
                               orientation_rad, target_nye_m1, stress_Pa,
                               temperature_K, parameters, dt_s,
                               relative_step=1e-7):
    """Centered derivative of the exact accepted extensive-state map."""
    scale = max(max(float(np.max(np.abs(getattr(inventory, name)))), 1.0)
                for name in inventory.__dict__)
    direction_scale = max(max(float(np.max(np.abs(getattr(direction, name)))), 0.0)
                          for name in direction.__dict__)
    if direction_scale == 0:
        return replace(direction)
    h = relative_step * scale / direction_scale
    plus = replace(inventory, **{
        name: getattr(inventory, name) + h * getattr(direction, name)
        for name in inventory.__dict__})
    minus = replace(inventory, **{
        name: getattr(inventory, name) - h * getattr(direction, name)
        for name in inventory.__dict__})
    mapped_plus = accepted_ordering_step(
        plus, systems, topologies, orientation_rad, target_nye_m1, stress_Pa,
        temperature_K, parameters, dt_s)[0]
    mapped_minus = accepted_ordering_step(
        minus, systems, topologies, orientation_rad, target_nye_m1, stress_Pa,
        temperature_K, parameters, dt_s)[0]
    return replace(direction, **{
        name: (getattr(mapped_plus, name) - getattr(mapped_minus, name)) / (2*h)
        for name in inventory.__dict__})


def wall_diagnostics(inventory, systems, topologies, orientation_rad,
                     target_nye_m1, *, minimum_ordered_density_m2=1e13,
                     minimum_polarization=0.2, maximum_relative_mismatch=0.25,
                     minimum_orientation_jump_rad=np.deg2rad(1.0)):
    fields = derived_density_fields(inventory, topologies)
    alpha, _ = ordered_wall_nye_m1(inventory, systems, orientation_rad)
    alpha_norm = np.linalg.norm(alpha, axis=(-2, -1))
    target_norm = np.linalg.norm(target_nye_m1, axis=(-2, -1))
    bmean = float(np.mean([system.burgers_m for system in systems]))
    polarization = alpha_norm / np.maximum(
        bmean * fields["rho_wall_ordered_m2"], 1e-300)
    mismatch = np.linalg.norm(alpha - target_nye_m1, axis=(-2, -1))
    relative_mismatch = mismatch / np.maximum(target_norm, bmean * minimum_ordered_density_m2)
    angle = np.unwrap(np.asarray(orientation_rad), axis=0)
    jump = float(np.max(angle) - np.min(angle))
    physical = ((fields["rho_wall_ordered_m2"] >= minimum_ordered_density_m2)
                & (polarization >= minimum_polarization)
                & (relative_mismatch <= maximum_relative_mismatch))
    return {
        **fields,
        "ordered_nye_m1": alpha,
        "target_nye_m1": np.asarray(target_nye_m1),
        "polarization": polarization,
        "relative_nye_mismatch": relative_mismatch,
        "orientation_jump_rad": jump,
        "has_required_orientation_jump": jump >= minimum_orientation_jump_rad,
        "physical_wall_mask": physical if jump >= minimum_orientation_jump_rad
                              else np.zeros_like(physical, dtype=bool),
    }
