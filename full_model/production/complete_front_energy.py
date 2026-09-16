"""Complete physical energy decision for an authoritative common front state.

The routines in this module are deliberately transactional.  A candidate
``CommonFrontState`` is evaluated without mutating the accepted state, and the
caller publishes it only after :func:`decide_complete_front_trial` returns an
accepted decision.  Numerical compatibility penalties are not part of this
functional.

All component values are energies in joules for the represented 3-D patch.
The elastic solve is performed at fixed mean strain.  ``external_work_J`` is
therefore normally zero; it is explicit for load-controlled test protocols.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import numpy as np

try:
    from .common_front_state import commit_front_result, reconstruct_common
    from .common_tensorial_wall import wall_total_free_energy_density_J_m3
    from .nonlocal_elasticity import elastic_energy_density, solve_periodic_eigenstrain
    from .tensorial_nye import nye_from_plastic_distortion, spectral_derivatives
except ImportError:  # pragma: no cover - direct production-script execution
    from common_front_state import commit_front_result, reconstruct_common
    from common_tensorial_wall import wall_total_free_energy_density_J_m3
    from nonlocal_elasticity import elastic_energy_density, solve_periodic_eigenstrain
    from tensorial_nye import nye_from_plastic_distortion, spectral_derivatives


@dataclass(frozen=True)
class CompleteFrontEnergy:
    defect_storage_J: float
    signed_junction_storage_J: float
    boundary_excess_J: float
    recoverable_elastic_J: float
    phase_local_J: float
    phase_gradient_J: float
    thermal_internal_J: float
    numerical_constraint_J: float = 0.0

    @property
    def helmholtz_J(self):
        return (self.defect_storage_J+self.signed_junction_storage_J
                +self.boundary_excess_J+self.recoverable_elastic_J
                +self.phase_local_J+self.phase_gradient_J)

    @property
    def internal_J(self):
        return self.helmholtz_J+self.thermal_internal_J


@dataclass(frozen=True)
class CompleteFrontTrialDecision:
    accepted: bool
    classification: str
    before: CompleteFrontEnergy
    candidate: CompleteFrontEnergy
    delta_helmholtz_J: float
    external_work_J: float
    available_change_J: float
    generated_heat_J: float
    thermostat_export_J: float
    material_sink_export_J: float
    first_law_residual_J: float
    tolerance_J: float

    def as_dict(self):
        value = asdict(self)
        value["before"]["helmholtz_J"] = self.before.helmholtz_J
        value["before"]["internal_J"] = self.before.internal_J
        value["candidate"]["helmholtz_J"] = self.candidate.helmholtz_J
        value["candidate"]["internal_J"] = self.candidate.internal_J
        return value


@dataclass(frozen=True)
class ProductRuleNyeAudit:
    exact_reconstructed_m1: np.ndarray
    independently_assembled_m1: np.ndarray
    owner_bulk_m1: np.ndarray
    support_interface_m1: np.ndarray
    discrete_representation_residual_m1: np.ndarray


@dataclass(frozen=True)
class CommonFrontTransactionResult:
    published_state: object
    published_mixture: object
    candidate_state: object
    candidate_mixture: object
    decision: CompleteFrontTrialDecision


def _curl_from_derivatives(dx, dy):
    alpha = np.zeros_like(dx)
    alpha[..., :, 0] = -dy[..., :, 2]
    alpha[..., :, 1] = dx[..., :, 2]
    alpha[..., :, 2] = dy[..., :, 0]-dx[..., :, 1]
    return alpha


def independently_assemble_product_rule_nye(state, spacing_m):
    """Assemble bulk and support-gradient Nye without a subtract/re-add path.

    The exact discrete curl and the continuum product-rule assembly are both
    returned.  Their difference is explicitly a spectral representation
    residual (aliasing/product truncation), not an unowned physical density.
    """
    weights = state.front.material_support_weights()
    owners = (state.parent, state.child, state.wake)
    bulk = np.zeros(state.interface_nye_m1.shape, dtype=float)
    interface = np.zeros_like(bulk)
    beta_bar = np.zeros_like(owners[0].beta_p, dtype=float)
    for weight, owner in zip(weights, owners):
        beta = np.asarray(owner.beta_p, dtype=float)
        beta_bar += weight[..., None, None]*beta
        bx, by = spectral_derivatives(beta, spacing_m)
        bulk += weight[..., None, None]*_curl_from_derivatives(bx, by)
        wx, wy = spectral_derivatives(weight, spacing_m)
        interface += _curl_from_derivatives(
            wx[..., None, None]*beta, wy[..., None, None]*beta)
    exact = nye_from_plastic_distortion(beta_bar, spacing_m)
    assembled = bulk+interface
    return ProductRuleNyeAudit(
        exact, assembled, bulk, interface, exact-assembled)


def evaluate_complete_front_energy(
        state, eta, *, spacing_m, represented_thickness_m, wall_parameters,
        mean_strain=None, topologies=(), systems=None,
        phase_barrier_J_m3=0.0, phase_gradient_J_m=0.0,
        boundary_line_energy_J_m=None, boundary_junction_energy_J_m=None,
        reference_temperature_K=0.0):
    """Evaluate each physical energy owner once for a common candidate."""
    spacing = float(spacing_m)
    thickness = float(represented_thickness_m)
    if not math.isfinite(spacing) or spacing <= 0.0 or not math.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("positive finite spacing and represented thickness required")
    phases = np.asarray(eta, dtype=float)
    if phases.ndim != 3 or phases.shape[:2] != state.front.chi.shape:
        raise ValueError("eta must have grid x phase layout matching the front")
    if np.any(~np.isfinite(phases)):
        raise ValueError("eta contains nonfinite values")
    mixture, _ = reconstruct_common(state, spacing)
    area = spacing*spacing
    volume = area*thickness
    local = wall_total_free_energy_density_J_m3(
        mixture, wall_parameters, topologies, systems)

    # Junction storage is exposed separately while still using the exact
    # common functional: subtract its declared contribution from the residual
    # defect component, so no term is counted twice.
    multiplicity = (np.asarray([t.product_line_multiplicity for t in topologies])
                    if topologies else np.ones(mixture.junction_m2.shape[2]))
    reaction = (np.asarray([t.delta_free_energy_J_m for t in topologies])
                if topologies else np.zeros(mixture.junction_m2.shape[2]))
    junction_density = np.sum(
        mixture.junction_m2*(multiplicity*wall_parameters.junction_energy_J_m
                             +reaction), axis=2)
    junction_J = float(np.sum(junction_density, dtype=np.longdouble)*volume)
    total_defect_J = float(np.sum(local, dtype=np.longdouble)*volume)

    line_energy = (wall_parameters.line_energy_J_m if boundary_line_energy_J_m is None
                   else float(boundary_line_energy_J_m))
    junction_energy = (wall_parameters.junction_energy_J_m
                       if boundary_junction_energy_J_m is None
                       else float(boundary_junction_energy_J_m))
    boundary_density = line_energy*np.sum(
        state.boundary_plus_m2+state.boundary_minus_m2, axis=(2, 3))
    boundary_density += junction_energy*np.sum(state.boundary_junction_m2, axis=2)
    boundary_J = float(np.sum(boundary_density, dtype=np.longdouble)*volume)

    beta = np.asarray(mixture.beta_p)
    eigenstrain = .5*(beta[..., :2, :2]+np.swapaxes(beta[..., :2, :2], -1, -2))
    mean = np.zeros((2, 2)) if mean_strain is None else np.asarray(mean_strain, dtype=float)
    stress, strain = solve_periodic_eigenstrain(
        eigenstrain, mean, spacing, wall_parameters.c11_Pa,
        wall_parameters.c12_Pa, wall_parameters.c44_Pa,
        iterations=wall_parameters.elastic_iterations)
    elastic_J = float(np.sum(
        elastic_energy_density(stress, strain, eigenstrain),
        dtype=np.longdouble)*volume)

    sum_e2 = np.sum(phases*phases, axis=2)
    sum_e4 = np.sum(phases**4, axis=2)
    phase_local = .5*float(phase_barrier_J_m3)*np.maximum(sum_e2*sum_e2-sum_e4, 0.0)
    phase_local_J = float(np.sum(phase_local, dtype=np.longdouble)*volume)
    phase_gradient_J = 0.0
    for index in range(phases.shape[2]):
        gx, gy = spectral_derivatives(phases[..., index], spacing)
        phase_gradient_J += float(.5*float(phase_gradient_J_m)
                                  *np.sum(gx*gx+gy*gy, dtype=np.longdouble)
                                  *volume)
    thermal_J = float(wall_parameters.volumetric_heat_capacity_J_m3_K
                      *np.sum(mixture.temperature_K-float(reference_temperature_K),
                              dtype=np.longdouble)*volume)
    return CompleteFrontEnergy(
        defect_storage_J=total_defect_J-junction_J,
        signed_junction_storage_J=junction_J,
        boundary_excess_J=boundary_J,
        recoverable_elastic_J=elastic_J,
        phase_local_J=phase_local_J,
        phase_gradient_J=phase_gradient_J,
        thermal_internal_J=thermal_J,
        numerical_constraint_J=0.0)


def decide_complete_front_trial(
        before, candidate, *, external_work_J=0.0, generated_heat_J=0.0,
        thermostat_export_J=0.0, material_sink_export_J=0.0,
        absolute_tolerance_J=0.0,
        relative_tolerance=8192.0*np.finfo(float).eps):
    """Apply fixed-strain free-energy acceptance and audit the first law."""
    work = float(external_work_J)
    heat = float(generated_heat_J)
    export = float(thermostat_export_J)
    sink_export = float(material_sink_export_J)
    delta_f = candidate.helmholtz_J-before.helmholtz_J
    available = delta_f-work
    scale = max(abs(before.helmholtz_J), abs(candidate.helmholtz_J),
                abs(work), abs(heat), abs(export), abs(sink_export), 1e-300)
    tolerance = max(float(absolute_tolerance_J), float(relative_tolerance)*scale)
    accepted = available <= tolerance
    delta_u = candidate.internal_J-before.internal_J
    # Generated heat is an internal conversion and therefore is not another
    # boundary input.  Only declared thermostat/material exports enter the
    # whole-system first-law residual.
    first_law = (delta_u-work+export+sink_export) if accepted else 0.0
    return CompleteFrontTrialDecision(
        accepted,
        "ACCEPTED_COMPLETE_PHYSICAL_ENERGY" if accepted else "REJECTED_UPHILL_COMPLETE_PHYSICAL_ENERGY",
        before, candidate, delta_f, work, available, heat, export, sink_export,
        first_law, tolerance)


def evaluate_common_front_transaction(
        state, accepted_front, eta_before, eta_candidate, *, spacing_m,
        cell_volume_m3, represented_thickness_m, transmission_fraction,
        boundary_storage_fraction, neutral_sink_fraction,
        signed_sink_fraction=0.0, energy_kwargs=None,
        external_work_J=0.0, prescribed_temperature=False):
    """Price an entire common-state transaction before publishing any owner.

    ``accepted_front`` is geometry/history only.  Its legacy material views are
    ignored by :func:`commit_front_result`.  Rejection returns the original
    state and its reconstructed mixture exactly; the fully assembled candidate
    is retained solely for diagnostics.
    """
    options = {} if energy_kwargs is None else dict(energy_kwargs)
    before_energy = evaluate_complete_front_energy(
        state, eta_before, spacing_m=spacing_m,
        represented_thickness_m=represented_thickness_m, **options)
    geometry_identity = (
        np.array_equal(accepted_front.chi, state.front.chi)
        and np.array_equal(accepted_front.processed_max,
                           state.front.processed_max)
        and np.array_equal(accepted_front.cleanup_max,
                           state.front.cleanup_max)
        and accepted_front.parent_label == state.front.parent_label
        and accepted_front.child_label == state.front.child_label)
    if geometry_identity and np.array_equal(eta_before, eta_candidate):
        mixture, _ = reconstruct_common(state, spacing_m)
        decision = decide_complete_front_trial(before_energy, before_energy)
        decision = replace(decision, classification="EXACT_ZERO_EVENT_IDENTITY")
        return CommonFrontTransactionResult(
            state, mixture, state, mixture, decision)
    candidate_state, candidate_mixture = commit_front_result(
        state, accepted_front, spacing_m=spacing_m,
        cell_volume_m3=cell_volume_m3,
        transmission_fraction=transmission_fraction,
        boundary_storage_fraction=boundary_storage_fraction,
        neutral_sink_fraction=neutral_sink_fraction,
        signed_sink_fraction=signed_sink_fraction)
    delta_sink = (candidate_state.ledger.sink_line_m
                  -state.ledger.sink_line_m)
    wall_parameters = options["wall_parameters"]
    sink_export = max(delta_sink, 0.0)*wall_parameters.line_energy_J_m
    cold_energy = evaluate_complete_front_energy(
        candidate_state, eta_candidate, spacing_m=spacing_m,
        represented_thickness_m=represented_thickness_m, **options)
    cold_decision = decide_complete_front_trial(
        before_energy, cold_energy, external_work_J=external_work_J,
        material_sink_export_J=sink_export)
    # Overdamped front kinetics dissipates its *complete* accepted affinity.
    # This is affinity times realized extent, not a residual relabelled after
    # the fact.  Neutral annihilation is one contributor to this same channel
    # and is therefore not deposited a second time using a line-energy proxy.
    heat = (max(float(external_work_J)-sink_export
                -(cold_energy.helmholtz_J-before_energy.helmholtz_J), 0.0)
            if cold_decision.accepted else 0.0)
    # At prescribed temperature generated heat is exported by the thermostat;
    # otherwise it becomes thermal internal energy before publication.
    thermostat = heat if prescribed_temperature else 0.0
    if heat > 0.0 and not prescribed_temperature:
        represented_volume = (state.front.chi.size*float(cell_volume_m3))
        delta_temperature = heat/max(
            wall_parameters.volumetric_heat_capacity_J_m3_K
            *represented_volume, 1e-300)
        owners = {}
        for name in ("parent", "child", "wake"):
            owner = getattr(candidate_state, name)
            owners[name] = replace(
                owner, temperature_K=(np.asarray(owner.temperature_K)
                                      +delta_temperature))
        candidate_state = replace(candidate_state, **owners)
        candidate_mixture, _ = reconstruct_common(candidate_state, spacing_m)
    candidate_energy = evaluate_complete_front_energy(
        candidate_state, eta_candidate, spacing_m=spacing_m,
        represented_thickness_m=represented_thickness_m, **options)
    decision = decide_complete_front_trial(
        before_energy, candidate_energy, external_work_J=external_work_J,
        generated_heat_J=heat, thermostat_export_J=thermostat,
        material_sink_export_J=sink_export)
    if decision.accepted:
        return CommonFrontTransactionResult(
            candidate_state, candidate_mixture, candidate_state,
            candidate_mixture, decision)
    before_mixture, _ = reconstruct_common(state, spacing_m)
    return CommonFrontTransactionResult(
        state, before_mixture, candidate_state, candidate_mixture, decision)
