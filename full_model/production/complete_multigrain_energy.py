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
    internal_energy_change_J: float
    first_law_residual_J: float
    relative_first_law_residual: float
    mean_stress_before_Pa: np.ndarray
    mean_stress_candidate_Pa: np.ndarray


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
        wall_parameters, energy_kwargs=None):
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
    nonelastic_change = sum((
        candidate.defect_storage_J-before.defect_storage_J,
        candidate.signed_junction_storage_J-before.signed_junction_storage_J,
        candidate.boundary_excess_J-before.boundary_excess_J,
        candidate.phase_local_J-before.phase_local_J,
        candidate.phase_gradient_J-before.phase_gradient_J,
        candidate.thermal_internal_J-before.thermal_internal_J,
        candidate.numerical_constraint_J-before.numerical_constraint_J,
    ))
    delta_internal = elastic_change+nonelastic_change
    residual = delta_internal-work
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
    scale = max(abs(delta_internal), abs(work), roundoff_floor, 1e-300)
    return MultiGrainMechanicalEnergyBalance(
        before, candidate, work, delta_internal, residual,
        abs(residual)/scale, stress_before, stress_candidate)


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
        reference_temperature_K=0.0):
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
    return CompleteFrontEnergy(
        total_defect_J-junction_J, junction_J, boundary_J, elastic_J,
        phase_local_J, phase_gradient_J, thermal_J, 0.0)


def evaluate_joint_multigrain_transaction(
        before_state: MultiGrainCommonState, capacity: JointTransferResult, *,
        spacing_m, represented_thickness_m, wall_parameters, interval_s,
        dissipation: MultiGrainDissipation, external_work_J=0.0,
        prescribed_temperature=False, energy_kwargs=None,
        absolute_tolerance_J=0.0,
        relative_tolerance=8192.0*np.finfo(float).eps):
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
    before_energy = evaluate_complete_multigrain_energy(before_state, **common)
    cold_state = capacity.candidate
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
    candidate = cold_state
    if generated_heat > 0.0 and not prescribed_temperature:
        represented_volume = before_state.supports.shape[1]*before_state.supports.shape[2]*cell_volume
        delta_temperature = generated_heat/max(
            wall_parameters.volumetric_heat_capacity_J_m3_K
            *represented_volume, 1e-300)
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
    candidate_energy = evaluate_complete_multigrain_energy(candidate, **common)
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
        "energy_kwargs": options,
    })
    decision = CompleteMultiGrainDecision(
        accepted, classification, True, True,
        multigrain_state_digest(before_state),
        multigrain_state_digest(candidate), config,
        float(interval_s), before_energy, candidate_energy, delta_f, work,
        material_export, available, generated_heat, thermostat, first_law,
        dissipation_residual, tolerance)
    if not accepted:
        return CompleteMultiGrainTransaction(before_state, candidate, decision)
    ledger = replace(
        candidate.ledger,
        energy_accepted_transactions=(
            candidate.ledger.energy_accepted_transactions+1))
    return CompleteMultiGrainTransaction(
        replace(candidate, ledger=ledger), candidate, decision)
