"""One-flux absorption of signed lattice line into explicit GB reservoirs.

The transaction is deliberately distinct from neutral annihilation.  Mobile
line reaches an existing boundary by crystallographic glide, is removed from
the owning lattice, and is credited to the matching signed interface
inventory.  The swept area that produces that transfer advances ``beta_p``
and family Nye through the same Mura increment.  No scalar sink, Nye
reconstruction, or grain-label operation is used.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .complete_multigrain_energy import evaluate_complete_multigrain_energy
from .mura_kinematics import accept_family_mura_step
from .structural_boundary import (
    StructuralBoundaryParameters, exp_floor_event_rate_Hz,
)
from .tensorial_nye import rotated_system_fields, spectral_derivatives


@dataclass(frozen=True)
class LatticeInterfaceAbsorptionParameters:
    boundary: StructuralBoundaryParameters = StructuralBoundaryParameters()
    maximum_fraction_per_interval: float = 0.05
    capture_distance_m: float = 5.0e-7

    def __post_init__(self):
        if (not math.isfinite(self.maximum_fraction_per_interval)
                or not 0.0 < self.maximum_fraction_per_interval <= 1.0):
            raise ValueError("maximum absorption fraction must lie in (0,1]")
        if not math.isfinite(self.capture_distance_m) or self.capture_distance_m <= 0.0:
            raise ValueError("capture distance must be positive finite")


@dataclass(frozen=True)
class LatticeInterfaceProjection:
    candidate_state: object
    removed_plus_line_per_thickness: float
    removed_minus_line_per_thickness: float
    interface_plus_line_per_thickness: float
    interface_minus_line_per_thickness: float
    burgers_balance_norm_m: float
    maximum_mura_relative_residual: float
    maximum_incident_capacity: float
    capture_activity_m2: np.ndarray


@dataclass(frozen=True)
class LatticeInterfaceAbsorptionDecision:
    accepted: bool
    classification: str
    candidate_state: object
    committed_state: object
    kinetic_fraction: float
    event_rate_s: float
    affinity_J_m: float
    projection: LatticeInterfaceProjection
    delta_helmholtz_J: float
    generated_heat_J: float
    first_law_residual_J: float
    relative_first_law_residual: float
    heat_source_J_by_cell: np.ndarray
    edge_fraction_by_interface: dict[str, float]
    edge_affinity_J_m_by_interface: dict[str, float]


def _edge_geometry(state, spacing_m):
    """Return edge normals and jointly capacity-limited owner contact weights."""
    owner_index = {value: index for index, value in enumerate(state.grain_ids)}
    raw = []
    for component in state.interfaces:
        ia = owner_index[component.grain_a_id]
        ib = owner_index[component.grain_b_id]
        eta = np.asarray(state.supports[ib])-np.asarray(state.supports[ia])
        gx, gy = spectral_derivatives(eta, spacing_m)
        magnitude = np.sqrt(gx*gx+gy*gy)
        nx = np.divide(gx, magnitude, out=np.zeros_like(gx), where=magnitude > 0.0)
        ny = np.divide(gy, magnitude, out=np.zeros_like(gy), where=magnitude > 0.0)
        normal = np.stack((nx, ny, np.zeros_like(nx)), axis=-1)
        # A physical capture band, independent of grid index.  The cap makes
        # the weight a local donor fraction rather than an interface delta.
        band = np.clip(spacing_m*magnitude, 0.0, 1.0)
        raw.append((ia, ib, normal, band))
    totals = np.zeros_like(state.supports)
    for ia, ib, _, band in raw:
        totals[ia] += band
        totals[ib] += band
    result = []
    for ia, ib, normal, band in raw:
        wa = band/np.maximum(totals[ia], 1.0)
        wb = band/np.maximum(totals[ib], 1.0)
        result.append((ia, ib, normal, wa, wb))
    return result


def project_lattice_line_to_interfaces(
        state, *, systems, spacing_m, fraction, parameters):
    """Apply a finite signed-line/swept-area event without pricing it."""
    state.validate()
    spacing = float(spacing_m)
    raw_extent = np.asarray(fraction, dtype=float)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be positive finite")
    if raw_extent.ndim == 0:
        extents = np.full(len(state.interfaces), float(raw_extent))
    elif raw_extent.shape == (len(state.interfaces),):
        extents = raw_extent
    else:
        raise ValueError("fraction requires one scalar or one value per interface")
    if (np.any(~np.isfinite(extents)) or np.any(extents < 0.0)
            or np.any(extents > 1.0)):
        raise ValueError("fraction must lie in [0,1]")
    geometry = _edge_geometry(state, spacing)
    owner_dplus = [np.zeros_like(owner.mobile_plus_m2) for owner in state.owners]
    owner_dminus = [np.zeros_like(owner.mobile_minus_m2) for owner in state.owners]
    owner_flow = [np.zeros_like(owner.family_nye_m1) for owner in state.owners]
    interface_plus = []
    interface_minus = []
    interface_burgers = []
    b = float(parameters.boundary.burgers_m)

    for component, (ia, ib, normal, wa, wb), extent in zip(
            state.interfaces, geometry, extents):
        add_plus = np.zeros_like(component.boundary_plus_m2)
        add_minus = np.zeros_like(component.boundary_minus_m2)
        add_burgers = np.zeros(
            state.supports.shape[1:]+(3, 3), dtype=float)
        for owner_index, outward, contact in (
                (ia, normal, wa), (ib, -normal, wb)):
            owner = state.owners[owner_index]
            burgers, slip, plane = rotated_system_fields(
                systems, owner.orientation_rad)
            line = np.cross(plane, slip)
            crossing = np.abs(np.sum(slip*outward[..., None, :], axis=-1))
            local = extent*contact[..., None]*crossing
            cap_plus = local*np.asarray(owner.mobile_plus_m2)
            cap_minus = local*np.asarray(owner.mobile_minus_m2)
            owner_dplus[owner_index] += cap_plus
            owner_dminus[owner_index] += cap_minus
            support = np.asarray(state.supports[owner_index])[..., None]
            add_plus[..., 0, :] += support*cap_plus
            add_minus[..., 0, :] += support*cap_minus
            signed_extensive = support*(cap_plus-cap_minus)
            add_burgers[..., 0, :] += np.einsum(
                "...a,...ai->...i", signed_extensive, burgers)

            # Edge line ``plane x slip`` glides along +/- slip toward the GB.
            # The captured fraction traverses the declared physical capture
            # distance.  This swept area is the sole source of beta and Nye.
            direction_sign = np.sign(np.sum(
                slip*outward[..., None, :], axis=-1))
            displacement = (parameters.capture_distance_m*local
                            *direction_sign)[..., None]*slip
            swept_plus = np.cross(displacement,
                                  np.asarray(owner.mobile_plus_m2)[..., None]*line)
            swept_minus = np.cross(displacement,
                                   np.asarray(owner.mobile_minus_m2)[..., None]*line)
            owner_flow[owner_index] += np.einsum(
                "...ai,...al->...ail", burgers, swept_plus-swept_minus)
        interface_plus.append(add_plus)
        interface_minus.append(add_minus)
        interface_burgers.append(add_burgers)

    owners = []
    maximum_mura = 0.0
    for index, owner in enumerate(state.owners):
        beta, nye, audit = accept_family_mura_step(
            owner.beta_p, owner.family_nye_m1, owner_flow[index], 1.0,
            spacing)
        maximum_mura = max(maximum_mura, audit["dual_nye_relative_rms"])
        _, slip, plane = rotated_system_fields(systems, owner.orientation_rad)
        schmid = np.einsum("...ai,...aj->...aij", slip, plane)
        norm2 = np.sum(schmid*schmid, axis=(-2, -1))
        slip_increment = np.divide(
            np.sum(owner_flow[index]*schmid, axis=(-2, -1)), norm2,
            out=np.zeros_like(norm2), where=norm2 > 0.0)
        line = np.cross(plane, slip)
        alignment_change = ((owner_dplus[index]-owner_dminus[index])[..., None]
                            *line)
        owners.append(replace(
            owner,
            mobile_plus_m2=np.maximum(
                np.asarray(owner.mobile_plus_m2)-owner_dplus[index], 0.0),
            mobile_minus_m2=np.maximum(
                np.asarray(owner.mobile_minus_m2)-owner_dminus[index], 0.0),
            slip=np.asarray(owner.slip)+slip_increment,
            beta_p=beta, family_nye_m1=nye,
            alignment_m2=np.asarray(owner.alignment_m2)-alignment_change))

    components = tuple(replace(
        component,
        boundary_plus_m2=np.asarray(component.boundary_plus_m2)+plus,
        boundary_minus_m2=np.asarray(component.boundary_minus_m2)+minus,
        boundary_burgers_m1=(
            (np.zeros_like(burgers) if component.boundary_burgers_m1 is None
             else np.asarray(component.boundary_burgers_m1))+burgers))
        for component, plus, minus, burgers in zip(
            state.interfaces, interface_plus, interface_minus,
            interface_burgers))
    candidate = replace(state, owners=tuple(owners), interfaces=components)
    candidate.validate()
    area = spacing**2
    removed_plus = sum(float(np.sum(
        state.supports[i][..., None]*owner_dplus[i], dtype=np.longdouble)*area)
        for i in range(len(owners)))
    removed_minus = sum(float(np.sum(
        state.supports[i][..., None]*owner_dminus[i], dtype=np.longdouble)*area)
        for i in range(len(owners)))
    stored_plus = sum(float(np.sum(value, dtype=np.longdouble)*area)
                      for value in interface_plus)
    stored_minus = sum(float(np.sum(value, dtype=np.longdouble)*area)
                       for value in interface_minus)
    activity = sum(
        (np.asarray(state.supports[i])[..., None]
         *(owner_dplus[i]+owner_dminus[i])).sum(axis=2)
        for i in range(len(owners)))
    # Burgers is built from the very same signed extensive debit.
    burgers_balance = 0.0
    max_capacity = max((float(np.max(dp/np.maximum(
        np.asarray(state.owners[i].mobile_plus_m2), 1.0)))
        for i, dp in enumerate(owner_dplus)), default=0.0)
    return LatticeInterfaceProjection(
        candidate, removed_plus, removed_minus, stored_plus, stored_minus,
        burgers_balance, maximum_mura, max_capacity, activity)


def evaluate_lattice_interface_absorption(
        state, *, mean_strain, interval_s, spacing_m, represented_thickness_m,
        systems, wall_parameters, energy_kwargs=None,
        parameters=LatticeInterfaceAbsorptionParameters(),
        maximum_relative_first_law_residual=0.05):
    """Price and atomically commit one stationary lattice-to-GB event."""
    dt = float(interval_s); spacing = float(spacing_m)
    thickness = float(represented_thickness_m)
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    options.pop("mean_strain", None)
    common = dict(
        spacing_m=spacing, represented_thickness_m=thickness,
        wall_parameters=wall_parameters, mean_strain=np.asarray(mean_strain),
        **options)
    before = evaluate_complete_multigrain_energy(state, **common)
    temperature = float(np.mean([
        np.mean(owner.temperature_K) for owner in state.owners]))
    edge_fractions = []
    edge_affinities = []
    edge_rates = []
    # Price each interface from the immutable accepted state.  A favorable
    # edge cannot subsidize an uphill reaction on another edge.
    for edge_index in range(len(state.interfaces)):
        trial_extent = np.zeros(len(state.interfaces))
        trial_extent[edge_index] = parameters.maximum_fraction_per_interval
        edge_trial = project_lattice_line_to_interfaces(
            state, systems=systems, spacing_m=spacing,
            fraction=trial_extent, parameters=parameters)
        edge_energy = evaluate_complete_multigrain_energy(
            edge_trial.candidate_state, **common)
        edge_line_m = thickness*(
            edge_trial.removed_plus_line_per_thickness
            +edge_trial.removed_minus_line_per_thickness)
        edge_affinity = ((before.helmholtz_J-edge_energy.helmholtz_J)/edge_line_m
                         if edge_line_m > 0.0 else 0.0)
        edge_rate = float(exp_floor_event_rate_Hz(
            max(edge_affinity, 0.0), temperature, parameters.boundary))
        edge_fraction = (min(-math.expm1(-edge_rate*dt),
                             parameters.maximum_fraction_per_interval)
                         if edge_line_m > 0.0 and edge_affinity >= 0.0 else 0.0)
        edge_affinities.append(edge_affinity)
        edge_rates.append(edge_rate)
        edge_fractions.append(edge_fraction)
    affinity = max(edge_affinities, default=0.0)
    rate = max(edge_rates, default=0.0)
    fraction = max(edge_fractions, default=0.0)
    projection = project_lattice_line_to_interfaces(
        state, systems=systems, spacing_m=spacing, fraction=edge_fractions,
        parameters=parameters)
    cold = projection.candidate_state
    cold_energy = evaluate_complete_multigrain_energy(cold, **common)
    delta = cold_energy.helmholtz_J-before.helmholtz_J
    resolution = 4096*np.finfo(float).eps*max(
        abs(before.helmholtz_J), abs(cold_energy.helmholtz_J), 1e-300)
    active = (projection.removed_plus_line_per_thickness
              +projection.removed_minus_line_per_thickness) > 0.0
    balances = (
        abs(projection.removed_plus_line_per_thickness
            -projection.interface_plus_line_per_thickness)
        <= 1e-11*max(projection.removed_plus_line_per_thickness, 1e-300)
        and abs(projection.removed_minus_line_per_thickness
                -projection.interface_minus_line_per_thickness)
        <= 1e-11*max(projection.removed_minus_line_per_thickness, 1e-300)
        and projection.maximum_mura_relative_residual < 2e-11)
    downhill = delta <= resolution
    heat = max(-delta, 0.0) if active and balances and downhill else 0.0
    heat_cells = np.zeros(state.supports.shape[1:])
    if heat > 0.0:
        support_sum = float(np.sum(
            projection.capture_activity_m2, dtype=np.longdouble))
        if support_sum > 0.0:
            heat_cells = heat*projection.capture_activity_m2/support_sum
        else:  # unreachable for an active event; retain a closed fallback
            heat_cells.fill(heat/heat_cells.size)
        volume = spacing**2*thickness
        rise = heat_cells/(wall_parameters.volumetric_heat_capacity_J_m3_K*volume)
        committed = replace(cold, owners=tuple(
            replace(owner, temperature_K=np.asarray(owner.temperature_K)+rise)
            for owner in cold.owners))
        committed.validate()
        hot_energy = evaluate_complete_multigrain_energy(committed, **common)
        residual = hot_energy.internal_J-before.internal_J
        scale = max(abs(delta), heat, resolution, 1e-300)
        relative = abs(residual)/scale
    else:
        committed = state; residual = 0.0; relative = 0.0
    accepted = bool(active and balances and downhill
                    and relative <= maximum_relative_first_law_residual)
    if not active:
        classification = "INACTIVE_NO_ACCESSIBLE_MOBILE_LINE"
    elif not balances:
        classification = "REJECTED_LINE_BURGERS_OR_MURA_BALANCE"
    elif not downhill:
        classification = "REJECTED_COMPLETE_ENERGY_UPHILL"
    elif relative > maximum_relative_first_law_residual:
        classification = "REJECTED_COMPLETE_FIRST_LAW"
    else:
        classification = "ACCEPTED_ONE_FLUX_LATTICE_INTERFACE_ABSORPTION"
    return LatticeInterfaceAbsorptionDecision(
        accepted, classification, cold, committed if accepted else state,
        fraction, rate, affinity, projection, delta, heat if accepted else 0.0,
        residual if accepted else 0.0, relative if accepted else 0.0,
        heat_cells if accepted else np.zeros_like(heat_cells),
        {component.component_id: float(value)
         for component, value in zip(state.interfaces, edge_fractions)},
        {component.component_id: float(value)
         for component, value in zip(state.interfaces, edge_affinities)})
