"""V23 extensive ordered/disordered wall thermodynamics and kinetics.

The wall state is line density, not a phase fraction.  For every sign and BCC
family, ``rho_w = rho_tangle + rho_ordered``.  Ordering transfers line between
the two reservoirs with detailed balance and cannot create density or Burgers
content.  ``q = rho_ordered/rho_w`` is a derived diagnostic only.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

try:
    from .density_state_map import DensityInventory, derived_density_fields
    from .tensorial_nye import rotation_z, rotated_system_fields, spectral_derivatives
except ImportError:
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
    nye_match_coefficient_J_m: float = 2e-5
    ordered_gradient_J_m3: float = 1e-33
    ordering_attempt_frequency_s: float = 1e7
    ordering_barrier_eV: float = 0.70
    event_length_m: float = 2.5e-9
    exp_floor: float = 0.05
    exp_a: float = 2.2
    exp_n: float = 2.5
    critical_stress_Pa: float = 1.5e9
    maximum_fraction_per_step: float = 0.15

    def __post_init__(self):
        positive = (
            self.spacing_m, self.rho_reference_m2, self.line_energy_J_m,
            self.rho_log_coefficient_J_m, self.disordered_excess_J_m,
            self.ordered_excess_J_m, self.nye_match_coefficient_J_m,
            self.ordering_attempt_frequency_s, self.ordering_barrier_eV,
            self.event_length_m, self.exp_a, self.exp_n,
            self.critical_stress_Pa, self.maximum_fraction_per_step,
        )
        if any((not np.isfinite(x)) or x <= 0 for x in positive):
            raise ValueError("V23 wall parameters must be finite and positive")
        if self.ordered_gradient_J_m3 < 0 or not 0 <= self.exp_floor <= 1:
            raise ValueError("invalid gradient coefficient or EXP floor")
        if self.maximum_fraction_per_step > 1:
            raise ValueError("step fraction cannot exceed one")


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
            item.product_line_multiplicity * parameters.disordered_excess_J_m
            + item.delta_free_energy_J_m)
    return {
        "common_line": common_line,
        "disordered_wall": tangle,
        "ordered_boundary": ordered,
        "nye_mismatch": matching,
        "ordered_gradient": gradient,
        "junction_topology": topology,
        "total": common_line + tangle + ordered + matching + gradient + topology,
    }


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
    stress = np.abs(np.asarray(stress_Pa, dtype=float))
    temperature = np.asarray(temperature_K, dtype=float)
    barrier = parameters.ordering_barrier_eV * EV_J * (
        parameters.exp_floor + (1 - parameters.exp_floor) * np.exp(
            -parameters.exp_a * (stress / parameters.critical_stress_Pa)
            ** parameters.exp_n))
    return parameters.ordering_attempt_frequency_s * np.exp(np.clip(
        -barrier / (KB_J_K * temperature), -700, 40))


def ordering_residual(inventory, systems, topologies, orientation_rad,
                      target_nye_m1, stress_Pa, temperature_K, parameters):
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
        net = np.where((source <= 0) & (net > 0), 0.0, net)
        net = np.where((target <= 0) & (net < 0), 0.0, net)
        result[sign] = net
        turnover[sign] = forward + reverse
    return result, turnover, mu


def accepted_ordering_step(inventory, systems, topologies, orientation_rad,
                           target_nye_m1, stress_Pa, temperature_K, parameters,
                           dt_s):
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
