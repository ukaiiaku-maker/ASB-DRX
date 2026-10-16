"""Whole-state energy and independently dissipative multi-grain transaction.

Every energy is integrated once over the represented three-dimensional patch.
Intrinsic grain-boundary energy resides in the multiphase local/gradient terms;
``boundary_excess_J`` contains only explicit trapped line/junction inventory.
Numerical normalization or compatibility penalties are deliberately absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math

import numpy as np

from .common_tensorial_wall import wall_total_free_energy_density_J_m3
from .complete_front_energy import CompleteFrontEnergy
from .nonlocal_elasticity import elastic_energy_density, solve_periodic_eigenstrain
from .tensorial_nye import spectral_derivatives
from .structural_interface_network import (
    JunctionDisclinationParameters, StructuralBoundaryParameters,
    audit_network_compatibility, interface_lengths_m,
    network_boundary_energy_J_per_m,
    network_junction_disclination_energy_J_per_m,
)
from .multigrain_common_state import (
    LINE_FIELDS, JointTransferResult, MultiGrainCommonState,
    multigrain_state_digest, reconstruct_multigrain_common,
)


@dataclass(frozen=True)
class MultiGrainDissipation:
    """Physical irreversible channels computed independently of closure."""

    glide_J: float = 0.0
    recovery_J: float = 0.0
    boundary_mobility_J: float = 0.0
    neutral_annihilation_J: float = 0.0

    @property
    def generated_heat_J(self):
        return (self.glide_J+self.recovery_J+self.boundary_mobility_J
                +self.neutral_annihilation_J)

    def validate(self):
        values = tuple(float(getattr(self, name)) for name in (
            "glide_J", "recovery_J", "boundary_mobility_J",
            "neutral_annihilation_J"))
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("dissipation channels must be finite and nonnegative")


@dataclass(frozen=True)
class CompleteMultiGrainDecision:
    accepted: bool
    classification: str
    complete_functional: bool
    independent_dissipation: bool
    before_state_digest: str
    candidate_state_digest: str
    configuration_digest: str
    interval_s: float
    before: CompleteFrontEnergy
    candidate: CompleteFrontEnergy
    delta_helmholtz_J: float
    external_work_J: float
    material_sink_export_J: float
    available_change_J: float
    generated_heat_J: float
    thermostat_export_J: float
    first_law_residual_J: float
    dissipation_residual_J: float
    tolerance_J: float
    heat_deposition_mode: str
    heat_source_integral_J: float
    heat_source_closure_J: float
    raw_heat_source_integral_J: float
    heat_source_normalization_factor: float
    raw_heat_source_relative_mismatch: float

    def as_dict(self):
        value = asdict(self)
        value["before"]["helmholtz_J"] = self.before.helmholtz_J
        value["before"]["internal_J"] = self.before.internal_J
        value["candidate"]["helmholtz_J"] = self.candidate.helmholtz_J
        value["candidate"]["internal_J"] = self.candidate.internal_J
        return value


@dataclass(frozen=True)
class CompleteMultiGrainTransaction:
    published_state: MultiGrainCommonState
    candidate_state: MultiGrainCommonState
    decision: CompleteMultiGrainDecision


@dataclass(frozen=True)
class MultiGrainMechanicalEnergyBalance:
    before: CompleteFrontEnergy
    candidate: CompleteFrontEnergy
    external_work_J: float
    thermostat_exchange_J: float
    internal_energy_change_J: float
    first_law_residual_J: float
    first_law_scale_J: float
    relative_first_law_residual: float
    mean_stress_before_Pa: np.ndarray
    mean_stress_candidate_Pa: np.ndarray


@dataclass(frozen=True)
class CompleteMultiGrainLocalFields:
    """Spatial terms conjugate to a stationary lattice/GB transaction.

    Each energy field is in J m^-3.  The split integrates exactly to the
    corresponding entries of :class:`CompleteFrontEnergy`; it is therefore a
    diagnostic/kinetic view of the same functional, not a second energy law.
    """

    defect_storage_J_m3: np.ndarray
    signed_junction_storage_J_m3: np.ndarray
    boundary_excess_J_m3: np.ndarray
    recoverable_elastic_J_m3: np.ndarray
    stress_Pa: np.ndarray
    eigenstrain: np.ndarray

    @property
    def absorption_helmholtz_J_m3(self):
        return (self.defect_storage_J_m3
                +self.signed_junction_storage_J_m3
                +self.boundary_excess_J_m3
                +self.recoverable_elastic_J_m3)


def evaluate_complete_multigrain_local_fields(
        state, *, spacing_m, wall_parameters, mean_strain=None,
        topologies=(), systems=None, boundary_line_energy_J_m=None,
        boundary_junction_energy_J_m=None,
        additional_plastic_distortion=None):
    """Return the local terms changed by lattice/interface absorption.

    The nonlocal elastic solve remains global; its returned density is the
    equilibrium density for the declared common state.  Local kinetics uses
    the exact first variation ``sigma:delta(eigenstrain)`` while the final
    transaction is still accepted only against the complete finite energy.
    """
    state.validate()
    spacing = float(spacing_m)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("positive finite spacing required")
    mixture, _ = reconstruct_multigrain_common(state, spacing)
    grid_shape = state.supports.shape[1:]
    defect = np.zeros(grid_shape, dtype=float)
    junction = np.zeros(grid_shape, dtype=float)
    multiplicity = (np.asarray([t.product_line_multiplicity for t in topologies])
                    if topologies else np.ones(mixture.junction_m2.shape[2]))
    reaction = (np.asarray([t.delta_free_energy_J_m for t in topologies])
                if topologies else np.zeros(mixture.junction_m2.shape[2]))
    for support, owner in zip(state.supports, state.owners):
        weight = np.asarray(support, dtype=float)
        local = wall_total_free_energy_density_J_m3(
            owner, wall_parameters, topologies, systems)
        local_junction = np.sum(
            owner.junction_m2
            *(multiplicity*wall_parameters.junction_energy_J_m+reaction),
            axis=2)
        defect += weight*(local-local_junction)
        junction += weight*local_junction

    line_energy = (wall_parameters.line_energy_J_m
                   if boundary_line_energy_J_m is None
                   else float(boundary_line_energy_J_m))
    junction_energy = (wall_parameters.junction_energy_J_m
                       if boundary_junction_energy_J_m is None
                       else float(boundary_junction_energy_J_m))
    boundary = np.zeros(grid_shape, dtype=float)
    for interface in state.interfaces:
        line_density = np.sum(
            interface.boundary_plus_m2+interface.boundary_minus_m2,
            axis=tuple(range(2, interface.boundary_plus_m2.ndim)))
        junction_density = np.sum(
            interface.boundary_junction_m2,
            axis=tuple(range(2, interface.boundary_junction_m2.ndim)))
        boundary += line_energy*line_density+junction_energy*junction_density

    beta = np.asarray(mixture.beta_p)
    if additional_plastic_distortion is not None:
        extra = np.asarray(additional_plastic_distortion, dtype=float)
        if extra.shape != beta.shape or np.any(~np.isfinite(extra)):
            raise ValueError("additional plastic distortion has invalid shape")
        beta = beta+extra
    eigenstrain = .5*(beta[..., :2, :2]
                      +np.swapaxes(beta[..., :2, :2], -1, -2))
    mean = np.zeros((2, 2)) if mean_strain is None else np.asarray(
        mean_strain, dtype=float)
    stress, strain = solve_periodic_eigenstrain(
        eigenstrain, mean, spacing, wall_parameters.c11_Pa,
        wall_parameters.c12_Pa, wall_parameters.c44_Pa,
        iterations=wall_parameters.elastic_iterations)
    elastic = elastic_energy_density(stress, strain, eigenstrain)
    return CompleteMultiGrainLocalFields(
        defect, junction, boundary, elastic, stress, eigenstrain)


def _mechanical_stress_and_eigenstrain(
        state, spacing_m, wall_parameters, mean_strain):
    mixture, _ = reconstruct_multigrain_common(state, spacing_m)
    beta = np.asarray(mixture.beta_p)
    eigenstrain = .5*(beta[..., :2, :2]
                      +np.swapaxes(beta[..., :2, :2], -1, -2))
    stress, _ = solve_periodic_eigenstrain(
        eigenstrain, np.asarray(mean_strain, dtype=float), float(spacing_m),
        wall_parameters.c11_Pa, wall_parameters.c12_Pa,
        wall_parameters.c44_Pa, iterations=wall_parameters.elastic_iterations)
    return stress, eigenstrain


def _mean_mechanical_stress(state, spacing_m, wall_parameters, mean_strain):
    stress, _ = _mechanical_stress_and_eigenstrain(
        state, spacing_m, wall_parameters, mean_strain)
    return np.mean(stress, axis=(0, 1))


def evaluate_multigrain_mechanical_interval(
        before_state, candidate_state, *, mean_strain_before,
        mean_strain_candidate, spacing_m, represented_thickness_m,
        wall_parameters, energy_kwargs=None, thermostat_exchange_J=0.0):
    """Independently audit strain-controlled work across one owner step."""
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    options.pop("mean_strain", None)
    common = dict(
        spacing_m=spacing_m,
        represented_thickness_m=represented_thickness_m,
        wall_parameters=wall_parameters, **options)
    before = evaluate_complete_multigrain_energy(
        before_state, mean_strain=mean_strain_before, **common)
    candidate = evaluate_complete_multigrain_energy(
        candidate_state, mean_strain=mean_strain_candidate, **common)
    stress_field_before, eigenstrain_before = _mechanical_stress_and_eigenstrain(
        before_state, spacing_m, wall_parameters, mean_strain_before)
    stress_field_candidate, eigenstrain_candidate = (
        _mechanical_stress_and_eigenstrain(
        candidate_state, spacing_m, wall_parameters, mean_strain_candidate)
    )
    stress_before = np.mean(stress_field_before, axis=(0, 1))
    stress_candidate = np.mean(stress_field_candidate, axis=(0, 1))
    delta_strain = (np.asarray(mean_strain_candidate, dtype=float)
                    -np.asarray(mean_strain_before, dtype=float))
    represented_volume = (before_state.supports.shape[1]
                          *before_state.supports.shape[2]
                          *float(spacing_m)**2
                          *float(represented_thickness_m))
    work = float(np.sum(.5*(stress_before+stress_candidate)*delta_strain)
                 *represented_volume)
    # Linear-elastic energy is quadratic.  Its endpoint trapezoidal identity
    # is exact and avoids subtracting O(1e-12 J) totals to audit physical
    # increments as small as O(1e-20 J) after recursive time subdivision.
    # This is an evaluation repair, not a residual correction: both endpoint
    # stresses and the independently evolved eigenstrain enter explicitly.
    average_stress = .5*(stress_field_before+stress_field_candidate)
    elastic_change = float(np.sum(
        average_stress*delta_strain, dtype=np.longdouble)
        *float(spacing_m)**2*float(represented_thickness_m)
        -np.sum(
            average_stress*(eigenstrain_candidate-eigenstrain_before),
            dtype=np.longdouble)
        *float(spacing_m)**2*float(represented_thickness_m))
    nonelastic_components = (
        candidate.defect_storage_J-before.defect_storage_J,
        candidate.signed_junction_storage_J-before.signed_junction_storage_J,
        candidate.boundary_excess_J-before.boundary_excess_J,
        candidate.phase_local_J-before.phase_local_J,
        candidate.phase_gradient_J-before.phase_gradient_J,
        candidate.structural_boundary_J-before.structural_boundary_J,
        candidate.structural_junction_J-before.structural_junction_J,
        candidate.thermal_internal_J-before.thermal_internal_J,
        candidate.numerical_constraint_J-before.numerical_constraint_J,
    )
    nonelastic_change = sum(nonelastic_components)
    delta_internal = elastic_change+nonelastic_change
    # Thermostat exchange is positive into the represented material.  The
    # production thermal operator records cooling as a negative increment, so
    # the open-system first law is Delta U = W + Q_thermostat.
    thermostat = float(thermostat_exchange_J)
    residual = delta_internal-work-thermostat
    # A hold can exchange defect/elastic energy with the thermal reservoir
    # while its exact external work is zero.  The endpoint totals are O(1e-11)
    # J in the production patch, so their floating-point subtraction cannot
    # resolve an O(1e-27) J residual.  Include that independently calculated
    # roundoff floor in the normalization; this does not relax any physically
    # resolvable imbalance and prevents recursive bisection of a zero-load
    # interval down to machine precision.
    roundoff_floor = (4096.0*np.finfo(float).eps
                      *max(abs(before.internal_J),
                           abs(candidate.internal_J), 1e-300))
    # During a fixed-strain hold, recoverable elastic/defect energy is converted
    # to heat while both net internal-energy change and external work approach
    # zero.  Normalizing by either net alone makes every finite-order truncation
    # read as a 100% first-law error.  The gross independently evaluated energy
    # turnover is the physically resolved scale for that cancellation.  It is
    # not a residual correction: the signed residual remains unchanged.
    gross_internal_turnover = abs(elastic_change)+sum(
        abs(value) for value in nonelastic_components)
    scale = max(abs(delta_internal), abs(work), abs(thermostat),
                gross_internal_turnover,
                roundoff_floor, 1e-300)
    return MultiGrainMechanicalEnergyBalance(
        before, candidate, work, thermostat, delta_internal, residual,
        scale, abs(residual)/scale, stress_before, stress_candidate)


def _config_digest(values):
    def convert(value):
        if hasattr(value, "__dataclass_fields__"):
            return {key: convert(item) for key, item in asdict(value).items()}
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (tuple, list)):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value
    encoded = json.dumps(convert(values), sort_keys=True,
                         separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_complete_multigrain_energy(
        state, *, spacing_m, represented_thickness_m, wall_parameters,
        mean_strain=None, topologies=(), systems=None,
        phase_barrier_J_m3=0.0, phase_gradient_J_m=0.0,
        boundary_line_energy_J_m=None, boundary_junction_energy_J_m=None,
        reference_temperature_K=0.0, structural_interfaces=(),
        structural_burgers_m=None, structural_parameters=None,
        junction_disclination_parameters=None,
        additional_plastic_distortion=None):
    """Evaluate the complete shared functional on one multi-owner state."""
    state.validate()
    spacing = float(spacing_m)
    thickness = float(represented_thickness_m)
    if not math.isfinite(spacing) or spacing <= 0.0 or not math.isfinite(
            thickness) or thickness <= 0.0:
        raise ValueError("positive finite spacing and thickness required")
    mixture, _ = reconstruct_multigrain_common(state, spacing)
    volume = spacing*spacing*thickness
    multiplicity = (np.asarray([t.product_line_multiplicity for t in topologies])
                    if topologies else np.ones(mixture.junction_m2.shape[2]))
    reaction = (np.asarray([t.delta_free_energy_J_m for t in topologies])
                if topologies else np.zeros(mixture.junction_m2.shape[2]))
    # Defect storage is an owner-local nonlinear constitutive energy.  Average
    # the owner energies, not the owner densities: g(sum eta_i rho_i) invents
    # cross-grain correlation energy in diffuse interfaces and is not
    # conjugate to the support-weighted owner kinetics.  Elasticity below is
    # deliberately different because beta_p is a shared spatial source.
    total_defect_J = 0.0
    junction_J = 0.0
    for support, owner in zip(state.supports, state.owners):
        weight = np.asarray(support, dtype=float)
        local = wall_total_free_energy_density_J_m3(
            owner, wall_parameters, topologies, systems)
        junction_density = np.sum(
            owner.junction_m2
            *(multiplicity*wall_parameters.junction_energy_J_m+reaction),
            axis=2)
        total_defect_J += float(np.sum(
            weight*local, dtype=np.longdouble)*volume)
        junction_J += float(np.sum(
            weight*junction_density, dtype=np.longdouble)*volume)

    line_energy = (wall_parameters.line_energy_J_m
                   if boundary_line_energy_J_m is None
                   else float(boundary_line_energy_J_m))
    junction_energy = (wall_parameters.junction_energy_J_m
                       if boundary_junction_energy_J_m is None
                       else float(boundary_junction_energy_J_m))
    boundary_J = 0.0
    for interface in state.interfaces:
        line_density = np.sum(
            interface.boundary_plus_m2+interface.boundary_minus_m2,
            axis=tuple(range(2, interface.boundary_plus_m2.ndim)))
        junction_interface = np.sum(
            interface.boundary_junction_m2,
            axis=tuple(range(2, interface.boundary_junction_m2.ndim)))
        boundary_J += float(np.sum(
            line_energy*line_density+junction_energy*junction_interface,
            dtype=np.longdouble)*volume)

    beta = np.asarray(mixture.beta_p)
    if additional_plastic_distortion is not None:
        extra = np.asarray(additional_plastic_distortion, dtype=float)
        if extra.shape != beta.shape or np.any(~np.isfinite(extra)):
            raise ValueError("additional plastic distortion has invalid shape")
        beta = beta+extra
    eigenstrain = .5*(beta[..., :2, :2]
                      +np.swapaxes(beta[..., :2, :2], -1, -2))
    mean = np.zeros((2, 2)) if mean_strain is None else np.asarray(
        mean_strain, dtype=float)
    stress, strain = solve_periodic_eigenstrain(
        eigenstrain, mean, spacing, wall_parameters.c11_Pa,
        wall_parameters.c12_Pa, wall_parameters.c44_Pa,
        iterations=wall_parameters.elastic_iterations)
    elastic_J = float(np.sum(
        elastic_energy_density(stress, strain, eigenstrain),
        dtype=np.longdouble)*volume)

    phases = np.moveaxis(np.asarray(state.supports, dtype=float), 0, -1)
    sum_e2 = np.sum(phases*phases, axis=2)
    sum_e4 = np.sum(phases**4, axis=2)
    phase_local = (.5*float(phase_barrier_J_m3)
                   *np.maximum(sum_e2*sum_e2-sum_e4, 0.0))
    phase_local_J = float(np.sum(phase_local, dtype=np.longdouble)*volume)
    phase_gradient_J = 0.0
    for index in range(phases.shape[2]):
        gx, gy = spectral_derivatives(phases[..., index], spacing)
        phase_gradient_J += float(
            .5*float(phase_gradient_J_m)
            *np.sum(gx*gx+gy*gy, dtype=np.longdouble)*volume)
    thermal_J = float(
        wall_parameters.volumetric_heat_capacity_J_m3_K
        *np.sum(mixture.temperature_K-float(reference_temperature_K),
                dtype=np.longdouble)*volume)
    structural_J = 0.0
    structural_junction_J = 0.0
    if structural_interfaces:
        if structural_burgers_m is None:
            raise ValueError(
                "structural interface energy requires a Burgers magnitude")
        parameters = (StructuralBoundaryParameters()
                      if structural_parameters is None
                      else structural_parameters)
        lengths = interface_lengths_m(
            state.supports, structural_interfaces, state.grain_ids, spacing)
        structural_J = (network_boundary_energy_J_per_m(
            structural_interfaces, lengths, float(structural_burgers_m),
            parameters)*thickness)
        if junction_disclination_parameters is not None:
            junction_parameters = junction_disclination_parameters
            if not isinstance(junction_parameters, JunctionDisclinationParameters):
                raise TypeError("invalid junction disclination parameters")
            structural_junction_J = (
                network_junction_disclination_energy_J_per_m(
                    state.grain_ids, structural_interfaces, lengths,
                    float(structural_burgers_m), junction_parameters)*thickness)
    return CompleteFrontEnergy(
        total_defect_J-junction_J, junction_J, boundary_J, elastic_J,
        phase_local_J, phase_gradient_J, thermal_J, 0.0, structural_J,
        structural_junction_J)


def evaluate_joint_multigrain_transaction(
        before_state: MultiGrainCommonState, capacity: JointTransferResult, *,
        spacing_m, represented_thickness_m, wall_parameters, interval_s,
        dissipation: MultiGrainDissipation, external_work_J=0.0,
        prescribed_temperature=False, energy_kwargs=None,
        heat_source_J_by_cell=None,
        heat_deposition_mode="local_realized_event",
        absolute_tolerance_J=0.0,
        relative_tolerance=8192.0*np.finfo(float).eps,
        precomputed_before_energy: CompleteFrontEnergy | None = None,
        bind_state_identity=True,
        synchronize_structural_owner_orientation=False):
    """Evaluate and atomically publish a complete joint physical event.

    The independent dissipation is deposited as heat (or exported by a
    prescribed-temperature thermostat). No residual is relabelled as heat.
    """
    if not capacity.capacity_feasible:
        raise ValueError("capacity result is infeasible")
    if float(interval_s) <= 0.0 or not math.isfinite(float(interval_s)):
        raise ValueError("positive finite physical interval required")
    dissipation.validate()
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    common = dict(spacing_m=spacing_m,
                  represented_thickness_m=represented_thickness_m,
                  wall_parameters=wall_parameters, **options)
    # A front-force solve prices many candidates from one immutable parent.
    # Its parent functional is therefore exactly state invariant.  Accepting
    # an explicitly supplied value avoids repeated FFT/elastic evaluations;
    # callers remain responsible for computing it with this same ``common``
    # configuration before entering the candidate loop.
    before_energy = (evaluate_complete_multigrain_energy(before_state, **common)
                     if precomputed_before_energy is None
                     else precomputed_before_energy)
    cold_state = capacity.candidate
    if synchronize_structural_owner_orientation:
        interfaces = tuple(options.get("structural_interfaces", ()))
        burgers = options.get("structural_burgers_m")
        if not interfaces or burgers is None:
            raise ValueError(
                "structural orientation synchronization requires graph energy")
        lengths = interface_lengths_m(
            cold_state.supports, interfaces, cold_state.grain_ids, spacing_m)
        reference = float(np.mean(before_state.owners[0].orientation_rad))
        audit = audit_network_compatibility(
            cold_state.grain_ids, interfaces, lengths, float(burgers),
            reference_owner=cold_state.grain_ids[0],
            reference_orientation_rad=reference)
        owners = tuple(
            replace(owner, orientation_rad=(
                np.asarray(owner.orientation_rad)
                +audit.fitted_orientation_rad[index]
                -float(np.mean(owner.orientation_rad))))
            for index, owner in enumerate(cold_state.owners))
        cold_state = replace(cold_state, owners=owners)
        cold_state.validate()
    cold_energy = evaluate_complete_multigrain_energy(cold_state, **common)
    cell_volume = (float(spacing_m)**2*float(represented_thickness_m))
    # Only the explicitly external channel leaves the represented system.
    # Boundary storage and neutral annihilation remain internal and may not be
    # counted again as material export merely because they leave bulk owners.
    material_export = (cell_volume*wall_parameters.line_energy_J_m
                       *capacity.physical_channel_cell_sums_m2.get(
                           "external_sink", 0.0))
    generated_heat = dissipation.generated_heat_J
    thermostat = generated_heat if prescribed_temperature else 0.0
    work = float(external_work_J)
    if heat_deposition_mode not in {"local_realized_event", "uniform_ablation"}:
        raise ValueError("unknown heat deposition mode")
    candidate = cold_state
    heat_source_integral = 0.0
    heat_source_closure = 0.0
    raw_heat_source_integral = 0.0
    heat_source_normalization_factor = 1.0
    raw_heat_source_relative_mismatch = 0.0
    if generated_heat > 0.0 and not prescribed_temperature:
        shape = before_state.supports.shape[1:]
        if heat_deposition_mode == "uniform_ablation":
            source = np.ones(shape, dtype=float)
        elif heat_source_J_by_cell is not None:
            source = np.asarray(heat_source_J_by_cell, dtype=float)
            if source.shape != shape:
                raise ValueError("heat source must match the common-state grid")
            if np.any(~np.isfinite(source)) or np.any(source < 0.0):
                raise ValueError("heat source must be finite and nonnegative")
        else:
            # Generic physical transactions may not have a channel-resolved
            # force field.  Their least-assumptive local support is the
            # realized transfer measure itself, never the requested event.
            source = np.zeros(shape, dtype=float)
            for extent in capacity.accepted_fraction_by_interface.values():
                source += np.asarray(extent, dtype=float)
        source_sum = float(np.sum(source, dtype=np.longdouble))
        if source_sum <= 0.0:
            raise ValueError("positive heat requires a positive realized source")
        raw_heat_source_integral = source_sum
        heat_source_normalization_factor = generated_heat/source_sum
        raw_heat_source_relative_mismatch = abs(source_sum-generated_heat)/max(
            abs(generated_heat), 1e-300)
        # A production local source is already an energy-by-cell ledger made
        # from the same accepted force and extent as the scalar dissipation.
        # Renormalizing a materially inconsistent field would conceal a
        # force/source association bug.  Geometry-only generic sources and the
        # explicitly declared uniform ablation remain normalized by design.
        if (heat_deposition_mode == "local_realized_event"
                and heat_source_J_by_cell is not None
                and raw_heat_source_relative_mismatch
                > max(float(relative_tolerance), 8192.0*np.finfo(float).eps)):
            raise ValueError(
                "local heat-source integral disagrees with generated heat")
        # Normalize the *geometry* independently of its absolute scale.  The
        # imposed integral is exactly the independently computed dissipation;
        # no energy residual is relabelled as heat.
        source_energy = generated_heat*source/source_sum
        heat_source_integral = float(
            np.sum(source_energy, dtype=np.longdouble))
        # Remove the final floating reduction error at one occupied cell.  It
        # is normally sub-ulp but makes the deposited source ledger explicit.
        occupied = np.argwhere(source > 0.0)
        correction = generated_heat-heat_source_integral
        if correction != 0.0:
            source_energy[tuple(occupied[0])] += correction
            heat_source_integral = float(
                np.sum(source_energy, dtype=np.longdouble))
        heat_source_closure = heat_source_integral-generated_heat
        delta_temperature = source_energy/max(
            wall_parameters.volumetric_heat_capacity_J_m3_K*cell_volume,
            1e-300)
        candidate = replace(cold_state, owners=tuple(
            replace(owner, temperature_K=np.asarray(owner.temperature_K)
                    +delta_temperature) for owner in cold_state.owners))
    accepted_extents = {
        key: float(np.sum(value, dtype=np.longdouble))
        for key, value in capacity.accepted_fraction_by_interface.items()}
    total_extent = sum(accepted_extents.values())
    if total_extent > 0.0:
        interfaces = []
        for interface in candidate.interfaces:
            share = accepted_extents.get(interface.component_id, 0.0)/total_extent
            interfaces.append(replace(
                interface,
                cumulative_work_J=(interface.cumulative_work_J+share*work),
                cumulative_heat_J=(interface.cumulative_heat_J
                                   +share*generated_heat)))
        candidate = replace(candidate, interfaces=tuple(interfaces))
    # With no work or heat the physical candidate fields equal the cold
    # fields.  Only zero-valued cumulative interface ledger additions may have
    # been materialized, and those ledgers do not enter the functional.
    candidate_energy = (cold_energy if generated_heat == 0.0 and work == 0.0
                        else evaluate_complete_multigrain_energy(
                            candidate, **common))
    delta_f = cold_energy.helmholtz_J-before_energy.helmholtz_J
    available = delta_f-work+material_export
    scale = max(abs(before_energy.helmholtz_J), abs(cold_energy.helmholtz_J),
                abs(work), abs(material_export), abs(generated_heat), 1e-300)
    tolerance = max(float(absolute_tolerance_J),
                    float(relative_tolerance)*scale)
    dissipation_residual = generated_heat+available
    first_law = (candidate_energy.internal_J-before_energy.internal_J
                 -work+thermostat+material_export)
    finite = all(math.isfinite(value) for value in (
        delta_f, work, material_export, available, generated_heat,
        thermostat, dissipation_residual, first_law, tolerance))
    accepted = (finite and available <= tolerance
                and abs(dissipation_residual) <= tolerance
                and abs(first_law) <= tolerance)
    if not finite:
        classification = "REJECTED_NONFINITE_COMPLETE_PHYSICS"
    elif available > tolerance:
        classification = "REJECTED_UPHILL_COMPLETE_PHYSICAL_ENERGY"
    elif abs(dissipation_residual) > tolerance:
        classification = "REJECTED_INDEPENDENT_DISSIPATION_MISMATCH"
    elif abs(first_law) > tolerance:
        classification = "REJECTED_FIRST_LAW_MISMATCH"
    else:
        classification = "ACCEPTED_COMPLETE_MULTIGRAIN_PHYSICAL_EVENT"
    config = _config_digest({
        "spacing_m": spacing_m,
        "represented_thickness_m": represented_thickness_m,
        "wall_parameters": wall_parameters,
        "interval_s": interval_s,
        "prescribed_temperature": prescribed_temperature,
        "heat_deposition_mode": heat_deposition_mode,
        "energy_kwargs": options,
    })
    if bind_state_identity:
        before_digest = multigrain_state_digest(before_state)
        candidate_digest = multigrain_state_digest(candidate)
    else:
        # Virtual force probes are read-only prices, never publication
        # authorities.  A sentinel keeps accidental publication fail-closed
        # while avoiding repeated full-state serialization and hashing.
        before_digest = "UNBOUND_VIRTUAL_ENERGY_PRICE"
        candidate_digest = "UNBOUND_VIRTUAL_ENERGY_PRICE"
    decision = CompleteMultiGrainDecision(
        accepted, classification, True, True,
        before_digest, candidate_digest, config,
        float(interval_s), before_energy, candidate_energy, delta_f, work,
        material_export, available, generated_heat, thermostat, first_law,
        dissipation_residual, tolerance, heat_deposition_mode,
        heat_source_integral, heat_source_closure,
        raw_heat_source_integral, heat_source_normalization_factor,
        raw_heat_source_relative_mismatch)
    if not accepted:
        return CompleteMultiGrainTransaction(before_state, candidate, decision)
    ledger = replace(
        candidate.ledger,
        energy_accepted_transactions=(
            candidate.ledger.energy_accepted_transactions+1))
    return CompleteMultiGrainTransaction(
        replace(candidate, ledger=ledger), candidate, decision)
