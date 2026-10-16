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

from .complete_multigrain_energy import (
    evaluate_complete_multigrain_energy,
    evaluate_complete_multigrain_local_fields,
)
from .common_tensorial_wall import CommonWallDriving, resolved_driving_components
from .mura_kinematics import accept_family_mura_step
from .multigrain_production import _owner_drivings_from_common_stress
from .structural_boundary import (
    StructuralBoundaryParameters, exp_floor_event_rate_Hz,
)
from .tensorial_nye import rotated_system_fields, spectral_derivatives


@dataclass(frozen=True)
class LatticeInterfaceAbsorptionParameters:
    boundary: StructuralBoundaryParameters = StructuralBoundaryParameters()
    maximum_fraction_per_interval: float = 0.05
    capture_distance_m: float = 5.0e-7
    complete_physical_clock: bool = False

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
    capture_activity_m2_by_interface: np.ndarray


@dataclass(frozen=True)
class LatticeInterfaceAbsorptionDecision:
    accepted: bool
    classification: str
    candidate_state: object
    committed_state: object
    kinetic_fraction: float
    event_rate_s: float
    maximum_uncapped_fraction: float
    selected_hazard_scale: float
    energy_search_evaluations: int
    affinity_J_m: float
    projection: LatticeInterfaceProjection
    delta_helmholtz_J: float
    generated_heat_J: float
    first_law_residual_J: float
    relative_first_law_residual: float
    heat_source_J_by_cell: np.ndarray
    edge_fraction_by_interface: dict[str, float]
    edge_affinity_J_m_by_interface: dict[str, float]
    edge_fraction_field_by_interface: dict[str, np.ndarray]
    edge_affinity_field_J_m_by_interface: dict[str, np.ndarray]
    local_kinetics_mode: str


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
        state, *, systems, spacing_m, fraction, parameters,
        stress_direction_by_owner=None, hazard_exposure_by_interface=None):
    """Apply a finite signed-line/swept-area event without pricing it."""
    state.validate()
    spacing = float(spacing_m)
    raw_extent = np.asarray(fraction, dtype=float)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be positive finite")
    grid_shape = state.supports.shape[1:]
    if raw_extent.ndim == 0:
        extents = np.full((len(state.interfaces),)+grid_shape,
                          float(raw_extent))
    elif raw_extent.shape == (len(state.interfaces),):
        extents = np.broadcast_to(
            raw_extent[:, None, None], (len(state.interfaces),)+grid_shape)
    elif raw_extent.shape == (len(state.interfaces),)+grid_shape:
        extents = raw_extent
    else:
        raise ValueError(
            "fraction requires a scalar, one value per interface, or one "
            "spatial field per interface")
    if (np.any(~np.isfinite(extents)) or np.any(extents < 0.0)
            or np.any(extents > 1.0)):
        raise ValueError("fraction must lie in [0,1]")
    geometry = _edge_geometry(state, spacing)
    if stress_direction_by_owner is not None:
        stress_directions = tuple(
            np.asarray(value, dtype=float) for value in stress_direction_by_owner)
        if (len(stress_directions) != len(state.owners)
                or any(value.shape != state.owners[0].mobile_plus_m2.shape
                       for value in stress_directions)):
            raise ValueError("stress directions require owner x grid x family")
    else:
        stress_directions = None
    hazard_exposures = None
    hazard_shares = None
    total_plus_hazard = total_minus_hazard = None
    if hazard_exposure_by_interface is not None:
        hazard_exposures = np.asarray(
            hazard_exposure_by_interface, dtype=float)
        if hazard_exposures.shape != (len(state.interfaces),)+grid_shape:
            raise ValueError(
                "hazard exposure requires one spatial field per interface")
        if np.any(~np.isfinite(hazard_exposures)) or np.any(
                hazard_exposures < 0.0):
            raise ValueError("hazard exposure must be finite and nonnegative")
        if stress_directions is None:
            raise ValueError("hazard capture requires signed stress directions")
        total_plus_hazard = [np.zeros_like(owner.mobile_plus_m2)
                             for owner in state.owners]
        total_minus_hazard = [np.zeros_like(owner.mobile_minus_m2)
                              for owner in state.owners]
        hazard_shares = []
        for exposure, (ia, ib, normal, wa, wb) in zip(
                hazard_exposures, geometry):
            edge_shares = []
            for owner_index, outward, contact in (
                    (ia, normal, wa), (ib, -normal, wb)):
                _, slip, _ = rotated_system_fields(
                    systems, state.owners[owner_index].orientation_rad)
                incidence = np.sum(slip*outward[..., None, :], axis=-1)
                sign_speed = np.sign(stress_directions[owner_index])
                plus = (exposure[..., None]*contact[..., None]
                        *np.maximum(-sign_speed*incidence, 0.0))
                minus = (exposure[..., None]*contact[..., None]
                         *np.maximum(sign_speed*incidence, 0.0))
                total_plus_hazard[owner_index] += plus
                total_minus_hazard[owner_index] += minus
                edge_shares.append((owner_index, plus, minus))
            hazard_shares.append(tuple(edge_shares))
    owner_dplus = [np.zeros_like(owner.mobile_plus_m2) for owner in state.owners]
    owner_dminus = [np.zeros_like(owner.mobile_minus_m2) for owner in state.owners]
    owner_flow = [np.zeros_like(owner.family_nye_m1) for owner in state.owners]
    interface_plus = []
    interface_minus = []
    interface_burgers = []
    interface_activity = []
    b = float(parameters.boundary.burgers_m)

    for edge_index, (component, (ia, ib, normal, wa, wb), extent) in enumerate(
            zip(state.interfaces, geometry, extents)):
        add_plus = np.zeros_like(component.boundary_plus_m2)
        add_minus = np.zeros_like(component.boundary_minus_m2)
        add_burgers = np.zeros(
            state.supports.shape[1:]+(3, 3), dtype=float)
        edge_activity = np.zeros(grid_shape, dtype=float)
        owner_sides = ((ia, normal, wa), (ib, -normal, wb))
        for local_owner_index, (owner_index, outward, contact) in enumerate(
                owner_sides):
            owner = state.owners[owner_index]
            burgers, slip, plane = rotated_system_fields(
                systems, owner.orientation_rad)
            line = np.cross(plane, slip)
            incidence = np.sum(slip*outward[..., None, :], axis=-1)
            if stress_directions is None:
                # Manufactured zero-force fixture: symmetric thermal access.
                crossing_plus = crossing_minus = np.abs(incidence)
                direction_plus = direction_minus = np.sign(incidence)
            else:
                sign_speed = np.sign(stress_directions[owner_index])
                # The codebase Mura convention transports kappa with -v.
                # Positive and negative Burgers populations have opposite v
                # under the same resolved stress.
                crossing_plus = np.maximum(-sign_speed*incidence, 0.0)
                crossing_minus = np.maximum(sign_speed*incidence, 0.0)
                direction_plus = sign_speed
                direction_minus = -sign_speed
            if hazard_shares is None:
                local_plus = extent[..., None]*contact[..., None]*crossing_plus
                local_minus = extent[..., None]*contact[..., None]*crossing_minus
            else:
                _, plus_hazard, minus_hazard = hazard_shares[
                    edge_index][local_owner_index]
                plus_total = total_plus_hazard[owner_index]
                minus_total = total_minus_hazard[owner_index]
                local_plus = np.divide(
                    -np.expm1(-plus_total)*plus_hazard, plus_total,
                    out=np.zeros_like(plus_hazard), where=plus_total > 0.0)
                local_minus = np.divide(
                    -np.expm1(-minus_total)*minus_hazard, minus_total,
                    out=np.zeros_like(minus_hazard), where=minus_total > 0.0)
            cap_plus = local_plus*np.asarray(owner.mobile_plus_m2)
            cap_minus = local_minus*np.asarray(owner.mobile_minus_m2)
            owner_dplus[owner_index] += cap_plus
            owner_dminus[owner_index] += cap_minus
            support = np.asarray(state.supports[owner_index])[..., None]
            edge_activity += np.sum(
                support*(cap_plus+cap_minus), axis=2)
            add_plus[..., 0, :] += support*cap_plus
            add_minus[..., 0, :] += support*cap_minus
            signed_extensive = support*(cap_plus-cap_minus)
            add_burgers[..., 0, :] += np.einsum(
                "...a,...ai->...i", signed_extensive, burgers)

            # Edge line ``plane x slip`` glides along +/- slip toward the GB.
            # The captured fraction traverses the declared physical capture
            # distance.  This swept area is the sole source of beta and Nye.
            displacement_plus = (parameters.capture_distance_m*local_plus
                                 *direction_plus)[..., None]*slip
            displacement_minus = (parameters.capture_distance_m*local_minus
                                  *direction_minus)[..., None]*slip
            swept_plus = np.cross(displacement_plus,
                                  np.asarray(owner.mobile_plus_m2)[..., None]*line)
            swept_minus = np.cross(displacement_minus,
                                   np.asarray(owner.mobile_minus_m2)[..., None]*line)
            owner_flow[owner_index] += np.einsum(
                "...ai,...al->...ail", burgers, swept_plus-swept_minus)
        interface_plus.append(add_plus)
        interface_minus.append(add_minus)
        interface_burgers.append(add_burgers)
        interface_activity.append(edge_activity)

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
    activity_by_interface = np.asarray(interface_activity)
    activity = np.sum(activity_by_interface, axis=0)
    # Burgers is built from the very same signed extensive debit.
    burgers_balance = 0.0
    max_capacity = max((float(np.max(dp/np.maximum(
        np.asarray(state.owners[i].mobile_plus_m2), 1.0)))
        for i, dp in enumerate(owner_dplus)), default=0.0)
    return LatticeInterfaceProjection(
        candidate, removed_plus, removed_minus, stored_plus, stored_minus,
        burgers_balance, maximum_mura, max_capacity, activity,
        activity_by_interface)


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
    before_local = evaluate_complete_multigrain_local_fields(
        state, spacing_m=spacing, wall_parameters=wall_parameters,
        mean_strain=np.asarray(mean_strain),
        topologies=options.get("topologies", ()),
        systems=options.get("systems", systems),
        boundary_line_energy_J_m=options.get("boundary_line_energy_J_m"),
        boundary_junction_energy_J_m=options.get(
            "boundary_junction_energy_J_m"))
    owner_drivings = _owner_drivings_from_common_stress(
        state, CommonWallDriving(mean_strain=np.asarray(mean_strain)),
        systems, wall_parameters)
    stress_directions = tuple(
        resolved_driving_components(
            owner, driving, systems, (), wall_parameters)["speed_m_s"]
        for owner, driving in zip(state.owners, owner_drivings))
    temperature = np.sum(
        np.asarray(state.supports)*np.asarray([
            owner.temperature_K for owner in state.owners]), axis=0)
    edge_fractions = []
    edge_affinities = []
    edge_rates = []
    edge_fraction_fields = []
    edge_affinity_fields = []
    edge_rate_fields = []
    maximum_uncapped_fraction = 0.0
    # Price each interface from the immutable accepted state.  The local
    # affinity is the exact local storage change plus the equilibrium elastic
    # first variation sigma:delta(eigenstrain), divided by the local captured
    # line.  The finite combined event still faces the complete-energy gate
    # below, so this kinetic localization cannot accept an uphill transaction.
    for edge_index in range(len(state.interfaces)):
        trial_extent = np.zeros(len(state.interfaces))
        trial_extent[edge_index] = parameters.maximum_fraction_per_interval
        edge_trial = project_lattice_line_to_interfaces(
            state, systems=systems, spacing_m=spacing,
            fraction=trial_extent, parameters=parameters,
            stress_direction_by_owner=stress_directions)
        edge_local = evaluate_complete_multigrain_local_fields(
            edge_trial.candidate_state, spacing_m=spacing,
            wall_parameters=wall_parameters,
            mean_strain=np.asarray(mean_strain),
            topologies=options.get("topologies", ()),
            systems=options.get("systems", systems),
            boundary_line_energy_J_m=options.get(
                "boundary_line_energy_J_m"),
            boundary_junction_energy_J_m=options.get(
                "boundary_junction_energy_J_m"))
        activity = edge_trial.capture_activity_m2_by_interface[edge_index]
        storage_before = (before_local.defect_storage_J_m3
                          +before_local.signed_junction_storage_J_m3
                          +before_local.boundary_excess_J_m3)
        storage_after = (edge_local.defect_storage_J_m3
                         +edge_local.signed_junction_storage_J_m3
                         +edge_local.boundary_excess_J_m3)
        delta_eigenstrain = edge_local.eigenstrain-before_local.eigenstrain
        elastic_release = np.einsum(
            "...ij,...ij->...", before_local.stress_Pa, delta_eigenstrain)
        release_density = storage_before-storage_after+elastic_release
        affinity_field = np.divide(
            release_density, activity, out=np.zeros_like(release_density),
            where=activity > 0.0)
        accessible = (activity > 0.0) & (affinity_field >= 0.0)
        rate_field = exp_floor_event_rate_Hz(
            np.maximum(affinity_field, 0.0), temperature,
            parameters.boundary)
        uncapped_fraction_field = np.where(
            accessible, -np.expm1(-rate_field*dt), 0.0)
        maximum_uncapped_fraction = max(
            maximum_uncapped_fraction,
            float(np.max(uncapped_fraction_field, initial=0.0)))
        if parameters.complete_physical_clock:
            # ``1-exp(-rate*dt)`` is the exact finite Arrhenius reaction
            # probability.  The small declared extent above remains an
            # affinity probe; it must not silently discard the remainder of
            # the requested physical interval.
            fraction_field = uncapped_fraction_field
        else:
            fraction_field = np.minimum(
                uncapped_fraction_field,
                parameters.maximum_fraction_per_interval)
        activity_sum = float(np.sum(activity, dtype=np.longdouble))
        if activity_sum > 0.0:
            edge_affinity = float(np.sum(
                activity*affinity_field, dtype=np.longdouble)/activity_sum)
            edge_fraction = float(np.sum(
                activity*fraction_field, dtype=np.longdouble)/activity_sum)
            edge_rate = float(np.sum(
                activity*rate_field, dtype=np.longdouble)/activity_sum)
        else:
            edge_affinity = edge_fraction = edge_rate = 0.0
        edge_affinities.append(edge_affinity)
        edge_rates.append(edge_rate)
        edge_fractions.append(edge_fraction)
        edge_affinity_fields.append(affinity_field)
        edge_rate_fields.append(rate_field)
        edge_fraction_fields.append(fraction_field)
    affinity = max(edge_affinities, default=0.0)
    rate = max(edge_rates, default=0.0)
    fraction = max(edge_fractions, default=0.0)
    hazard_exposure = (
        np.asarray(edge_rate_fields)*dt
        if parameters.complete_physical_clock else None)
    projection = project_lattice_line_to_interfaces(
        state, systems=systems, spacing_m=spacing,
        fraction=np.asarray(edge_fraction_fields), parameters=parameters,
        stress_direction_by_owner=stress_directions,
        hazard_exposure_by_interface=hazard_exposure)
    cold = projection.candidate_state
    cold_energy = evaluate_complete_multigrain_energy(cold, **common)
    delta = cold_energy.helmholtz_J-before.helmholtz_J
    resolution = 4096*np.finfo(float).eps*max(
        abs(before.helmholtz_J), abs(cold_energy.helmholtz_J), 1e-300)
    selected_hazard_scale = 1.0
    energy_search_evaluations = 1
    if (parameters.complete_physical_clock and delta > resolution
            and np.any(hazard_exposure > 0.0)):
        # At high temperature/affinity the exact exposure can saturate while
        # finite boundary storage makes complete capture uphill.  This is a
        # stiff constrained relaxation, not permission to discard physical
        # time.  Search the same conservative reaction coordinate and retain
        # its lowest complete Helmholtz state.  A logarithmic search resolves
        # both saturated and near-linear hazards without a fitted scale.
        best = (delta, selected_hazard_scale, projection, cold, cold_energy)
        worse_after_downhill_minimum = 0
        for power in range(1, 41):
            scale = 0.5**power
            trial_projection = project_lattice_line_to_interfaces(
                state, systems=systems, spacing_m=spacing,
                fraction=np.asarray(edge_fraction_fields),
                parameters=parameters,
                stress_direction_by_owner=stress_directions,
                hazard_exposure_by_interface=hazard_exposure*scale)
            trial_cold = trial_projection.candidate_state
            trial_energy = evaluate_complete_multigrain_energy(
                trial_cold, **common)
            trial_delta = trial_energy.helmholtz_J-before.helmholtz_J
            energy_search_evaluations += 1
            if trial_delta < best[0]:
                best = (trial_delta, scale, trial_projection,
                        trial_cold, trial_energy)
                worse_after_downhill_minimum = 0
            elif best[0] < -resolution:
                worse_after_downhill_minimum += 1
                # Along this scalar conservative reaction coordinate the
                # energy returns continuously toward the accepted state as
                # exposure -> 0.  Three resolved rises past a downhill
                # minimum bracket the logarithmic search without spending
                # evaluations on another twenty near-identity trials.
                if worse_after_downhill_minimum >= 3:
                    break
        (delta, selected_hazard_scale, projection,
         cold, cold_energy) = best
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
        fraction, rate, maximum_uncapped_fraction, selected_hazard_scale,
        energy_search_evaluations, affinity, projection,
        delta, heat if accepted else 0.0,
        residual if accepted else 0.0, relative if accepted else 0.0,
        heat_cells if accepted else np.zeros_like(heat_cells),
        {component.component_id: float(value)
         for component, value in zip(state.interfaces, edge_fractions)},
        {component.component_id: float(value)
         for component, value in zip(state.interfaces, edge_affinities)},
        {component.component_id: np.asarray(value)
         for component, value in zip(state.interfaces, edge_fraction_fields)},
        {component.component_id: np.asarray(value)
         for component, value in zip(state.interfaces, edge_affinity_fields)},
        ("LOCAL_COMPLETE_STORAGE_PLUS_ELASTIC_FIRST_VARIATION_"
         +("EXACT_CLOCK_ENERGY_LIMITED" if (
             parameters.complete_physical_clock
             and selected_hazard_scale < 1.0) else "EXACT_CLOCK")
         if parameters.complete_physical_clock else
         "LOCAL_COMPLETE_STORAGE_PLUS_ELASTIC_FIRST_VARIATION_CLIPPED_CLOCK"))
