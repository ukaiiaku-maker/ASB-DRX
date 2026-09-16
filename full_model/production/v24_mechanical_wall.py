"""V24 full-elastic kinematics coupled to ledgered wall line supply.

This seam uses the qualified common nonlocal elastic solve and slip kinematics,
but replaces its legacy unlocalized forest-to-wall conversion with explicit
finite-volume transport/capture.  Ordering receives a zero compatibility
coefficient and no orientation-derived target.  An optional, separately
ledgered junction topology route can operate on captured tangle parents.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .arrhenius_kinetics import (
    ActivatedProcess, activated_rate_array_s, exp_floor_enthalpy_j,
)
from .common_tensorial_wall import (
    CommonWallDriving, CommonWallParameters, CommonWallState,
    resolved_driving_components, wall_free_energy_density_J_m3, wall_residual,
)
from .density_state_map import (
    DensityInventory, checkpoint_arrays, derived_density_fields,
    from_checkpoint_arrays,
)
from .extensive_wall import ExtensiveWallParameters, accepted_ordering_step
from .tensorial_nye import rotated_system_fields
from .mura_kinematics import (
    accept_family_mura_step, family_plastic_flow_from_signed_alignment,
)
from .nonlocal_elasticity import elastic_energy_density, solve_periodic_eigenstrain
from .wall_topology_supply import (
    ReservoirAlignmentState, accepted_junction_topology_step,
    accepted_line_reorientation_step,
    alignment_checkpoint_arrays, alignment_from_checkpoint_arrays,
    apply_signed_ordering_extent, apply_signed_reservoir_exchange,
    accepted_mura_transport_capture_step,
    validate_junction_alignment,
    reservoir_nye_m1,
)


@dataclass(frozen=True)
class V24MechanicalWallState:
    common: CommonWallState
    density: DensityInventory
    reservoir_alignment: ReservoirAlignmentState

    def validate(self, systems, topologies):
        self.common.validate(systems, topologies)
        self.reservoir_alignment.validate(self.density, len(systems))
        validate_junction_alignment(
            self.density, self.reservoir_alignment, topologies)
        pairs = (
            (self.common.mobile_plus_m2, self.density.mobile_plus_m2),
            (self.common.mobile_minus_m2, self.density.mobile_minus_m2),
            (self.common.forest_plus_m2, self.density.forest_plus_m2),
            (self.common.forest_minus_m2, self.density.forest_minus_m2),
            (self.common.wall_plus_m2,
             self.density.wall_tangle_plus_m2+self.density.wall_ordered_plus_m2),
            (self.common.wall_minus_m2,
             self.density.wall_tangle_minus_m2+self.density.wall_ordered_minus_m2),
            (self.common.junction_m2, self.density.junction_m2),
        )
        if any(not np.array_equal(left, right) for left, right in pairs):
            raise ValueError("mechanical and V24 density inventories are not synchronized")


@dataclass(frozen=True)
class V24TopologyKinetics:
    junction_process: ActivatedProcess
    junction_enthalpy_J: float
    critical_stress_Pa: float
    exp_a: float = 2.2
    exp_n: float = 2.5
    exp_floor: float = 0.05
    maximum_junction_fraction_per_step: float = 0.1

    def __post_init__(self):
        if (self.junction_enthalpy_J < 0.0 or self.critical_stress_Pa <= 0.0
                or self.exp_a < 0.0 or self.exp_n < 1.0
                or not 0.0 <= self.exp_floor <= 1.0
                or not 0.0 < self.maximum_junction_fraction_per_step <= 1.0):
            raise ValueError("invalid V24 topology kinetics")


class MuraWorkBudgetError(RuntimeError):
    """Rejected Mura transaction with a scalar, serializable budget audit."""

    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def _elastic_energy_sum_J_m3_cells(common, beta_p, driving, parameters):
    """Recoverable elastic energy at the fixed total-strain substep state."""
    if driving.mean_strain is None:
        return None
    beta2 = np.asarray(beta_p)[..., :2, :2]
    eigenstrain = .5*(beta2+np.swapaxes(beta2, -1, -2))
    if driving.fixed_eigenstrain is not None:
        eigenstrain = eigenstrain+np.asarray(driving.fixed_eigenstrain)
    stress, strain = solve_periodic_eigenstrain(
        eigenstrain, driving.mean_strain, parameters.spacing_m,
        parameters.c11_Pa, parameters.c12_Pa, parameters.c44_Pa,
        iterations=parameters.elastic_iterations)
    return float(np.sum(elastic_energy_density(stress, strain, eigenstrain)))


def _resolved_elastic_energy_sum_J_m3_cells(drive):
    if drive["stress_tensor_Pa"] is None:
        return None
    return float(np.sum(elastic_energy_density(
        drive["stress_tensor_Pa"], drive["compatible_strain"],
        drive["eigenstrain"])))


def _common_with_transport_density(common, density, topologies):
    fields = derived_density_fields(density, topologies)
    return replace(
        common,
        mobile_plus_m2=density.mobile_plus_m2,
        mobile_minus_m2=density.mobile_minus_m2,
        forest_plus_m2=density.forest_plus_m2,
        forest_minus_m2=density.forest_minus_m2,
        wall_plus_m2=density.wall_tangle_plus_m2+density.wall_ordered_plus_m2,
        wall_minus_m2=density.wall_tangle_minus_m2+density.wall_ordered_minus_m2,
        junction_m2=density.junction_m2,
        wall_order=fields["q_wall_diagnostic"])


def _mura_budget_audit(state, transported_density, capture_ledger,
                       family_flow_rate, raw_stress_Pa, schmid_tensors,
                       schmid_norm2, event_scale, dt_s, systems, topologies,
                       parameters, elastic_release_J_m3_cells):
    """Return the complete fixed-strain Mura event budget.

    Plastic work is retained as the conjugate first-order diagnostic.  At
    prescribed total strain the independently recomputed recoverable elastic
    release is the actual mechanical source and is therefore not added to
    plastic work a second time.
    """
    slip_rate = np.divide(
        np.sum(family_flow_rate*schmid_tensors, axis=(-2, -1)), schmid_norm2,
        out=np.zeros_like(schmid_norm2), where=schmid_norm2 > 0.0)
    family_work = float(dt_s)*raw_stress_Pa*slip_rate
    plastic_work = np.sum(family_work, axis=2)
    before = wall_free_energy_density_J_m3(
        state.common, parameters, topologies, systems)
    transported_common = _common_with_transport_density(
        state.common, transported_density, topologies)
    after = wall_free_energy_density_J_m3(
        transported_common, parameters, topologies, systems)
    defect_delta = after-before
    line_creation = parameters.line_energy_J_m*sum(
        np.sum(capture_ledger["sign"][sign]["mura_line_stretching_m2"], axis=2)
        for sign in ("plus", "minus"))
    total_plastic_work = float(np.sum(plastic_work))
    total_defect_delta = float(np.sum(defect_delta))
    mechanical_source = (total_plastic_work if elastic_release_J_m3_cells is None
                         else float(elastic_release_J_m3_cells))
    heat_total = mechanical_source-total_defect_delta
    tolerance = 1e-12*max(abs(mechanical_source), abs(total_defect_delta), 1.0)
    family_storage = parameters.line_energy_J_m*sum(
        capture_ledger["sign"][sign]["mura_line_stretching_m2"]
        for sign in ("plus", "minus"))
    active = np.unravel_index(np.argmax(line_creation), line_creation.shape)
    return {
        "event_scale": float(event_scale),
        "accepted_dt_s": float(dt_s),
        "external_work_J_m3_cells": 0.0 if elastic_release_J_m3_cells is not None else None,
        "plastic_work_J_m3_cells": total_plastic_work,
        "recoverable_elastic_energy_release_J_m3_cells": (
            None if elastic_release_J_m3_cells is None
            else float(elastic_release_J_m3_cells)),
        "existing_defect_energy_release_J_m3_cells": float(
            max(-total_defect_delta, 0.0)),
        "proposed_defect_energy_change_J_m3_cells": total_defect_delta,
        "proposed_line_creation_energy_J_m3_cells": float(np.sum(line_creation)),
        "dissipative_drag_and_heat_J_m3_cells": float(heat_total),
        "admissibility_tolerance_J_m3_cells": float(tolerance),
        "admissible": bool(heat_total >= -tolerance),
        "active_cell_ij": [int(active[0]), int(active[1])],
        "active_cell_line_creation_energy_J_m3": float(line_creation[active]),
        "family_plastic_work_J_m3_cells": [float(value) for value in
                                             np.sum(family_work, axis=(0, 1))],
        "family_line_creation_energy_J_m3_cells": [float(value) for value in
            np.sum(family_storage, axis=(0, 1))],
        "reaction_extents": {
            "mura_line_stretching_m2_cells": float(sum(
                np.sum(capture_ledger["sign"][sign]["mura_line_stretching_m2"])
                for sign in ("plus", "minus"))),
            "captured_line_m2_cells": float(sum(
                np.sum(capture_ledger["sign"][sign]["captured_line_m2"])
                for sign in ("plus", "minus"))),
        },
        "mechanical_source_ownership": (
            "fixed_total_strain_recoverable_elastic_release"
            if elastic_release_J_m3_cells is not None
            else "prescribed_stress_conjugate_plastic_work"),
        "plastic_work_not_double_counted": True,
    }, slip_rate, plastic_work, defect_delta, line_creation


def mechanical_checkpoint_arrays(state):
    """Lossless array payload for the complete V24 mechanical wall state."""
    payload = {"v24_common__"+name: np.asarray(value)
               for name, value in state.common.__dict__.items()}
    payload.update(checkpoint_arrays(state.density, prefix="v24_density__"))
    payload.update(alignment_checkpoint_arrays(state.reservoir_alignment))
    return payload


def mechanical_from_checkpoint_arrays(mapping, systems, topologies):
    common_names = tuple(CommonWallState.__dataclass_fields__)
    missing = [name for name in common_names if "v24_common__"+name not in mapping]
    if missing:
        raise ValueError("incomplete V24 mechanical restart: "+", ".join(missing))
    common = CommonWallState(**{
        name: np.asarray(mapping["v24_common__"+name]).copy()
        for name in common_names})
    density = from_checkpoint_arrays(
        mapping, len(systems), len(topologies), prefix="v24_density__")
    alignment = alignment_from_checkpoint_arrays(
        mapping, density, len(systems))
    result = V24MechanicalWallState(common, density, alignment)
    result.validate(systems, topologies)
    return result


def synchronize_common(state, topologies):
    fields = derived_density_fields(state.density, topologies)
    common = replace(
        state.common,
        mobile_plus_m2=state.density.mobile_plus_m2,
        mobile_minus_m2=state.density.mobile_minus_m2,
        forest_plus_m2=state.density.forest_plus_m2,
        forest_minus_m2=state.density.forest_minus_m2,
        wall_plus_m2=(state.density.wall_tangle_plus_m2
                      +state.density.wall_ordered_plus_m2),
        wall_minus_m2=(state.density.wall_tangle_minus_m2
                       +state.density.wall_ordered_minus_m2),
        junction_m2=state.density.junction_m2,
        wall_order=fields["q_wall_diagnostic"])
    return replace(state, common=common)


def accepted_v24_mechanical_step(
        state, driving, capture_support, systems, topologies,
        common_parameters, extensive_parameters, topology_kinetics, dt_s, *,
        topology_route_enabled=False, maximum_orientation_increment_rad=0.02,
        mura_work_budget_mode="energy_limited",
        maximum_work_budget_backtracks=20):
    """Advance mechanics, line transport/capture, ordering, and topology once."""
    state.validate(systems, topologies)
    _nye_reservoir_initial = reservoir_nye_m1(
        state.reservoir_alignment, systems, state.common.orientation_rad,
        topologies)["total"]
    _nye_beta_initial = np.sum(state.common.family_nye_m1, axis=2)
    if extensive_parameters.nye_match_coefficient_J_m != 0.0:
        raise ValueError("V24 production ordering forbids target-Nye driving")
    # The common residual supplies the authoritative full-elastic stress,
    # physical glide speed, and exactly consistent beta/slip/orientation rates.
    mechanics_parameters = replace(
        common_parameters, wall_order_enabled=False,
        extensive_wall_partition_enabled=True, transport_scheme="upwind",
        mobile_correlation_diffusivity_m2_s=0.0)
    drive = resolved_driving_components(
        state.common, driving, systems, topologies, mechanics_parameters)
    # Reuse the already solved elastic fields in the common residual. Passing
    # the raw RSS makes its Taylor reduction identical, and passing speed
    # avoids a second nonlocal solve without changing the constitutive path.
    resolved = CommonWallDriving(
        glide_speed_m_s=drive["speed_m_s"],
        resolved_stress_Pa=drive["raw_stress_Pa"])
    residual = wall_residual(
        state.common, resolved, systems, topologies, mechanics_parameters)
    _, slip_directions, plane_normals = rotated_system_fields(
        systems, state.common.orientation_rad)
    disordered_line_directions = np.cross(plane_normals, slip_directions)
    velocity_plus_3d = drive["speed_m_s"][..., None]*slip_directions
    velocity_minus_3d = -velocity_plus_3d
    velocity_plus = velocity_plus_3d[..., :2]
    velocity_minus = -velocity_plus
    courant_rate = np.max(
        np.sum(np.abs(velocity_plus), axis=-1)/common_parameters.spacing_m)
    orientation_rate = np.max(np.abs(residual.state_rate.orientation_rad))
    accepted_dt = min(
        float(dt_s), 0.8/max(float(courant_rate), 1e-300),
        maximum_orientation_increment_rad/max(float(orientation_rate), 1e-300))
    # One accepted Mura face event owns scalar population motion, line moments,
    # plastic distortion, and family Nye.  An inadmissible positivity trial is
    # rejected by reducing its extent; no population or alignment is clipped.
    if mura_work_budget_mode not in ("energy_limited", "legacy_reject"):
        raise ValueError("unknown Mura work-budget mode")
    if int(maximum_work_budget_backtracks) < 0:
        raise ValueError("Mura work-budget backtracks cannot be negative")
    proposed_velocity_plus_3d = velocity_plus_3d
    proposed_velocity_minus_3d = velocity_minus_3d
    proposed_family_flow_rate = family_plastic_flow_from_signed_alignment(
        state.reservoir_alignment.mobile_plus_m2,
        state.reservoir_alignment.mobile_minus_m2,
        proposed_velocity_plus_3d, proposed_velocity_minus_3d, systems,
        state.common.orientation_rad)
    schmid_tensors = np.einsum(
        "...ai,...aj->...aij", slip_directions, plane_normals)
    schmid_norm2 = np.sum(schmid_tensors*schmid_tensors, axis=(-2, -1))
    proposed_mura_slip_rate = np.divide(
        np.sum(proposed_family_flow_rate*schmid_tensors, axis=(-2, -1)),
        schmid_norm2, out=np.zeros_like(schmid_norm2), where=schmid_norm2 > 0.0)
    proposed_family_work_rate = np.sum(
        drive["raw_stress_Pa"]*proposed_mura_slip_rate, axis=(0, 1))
    proposed_family_work = accepted_dt*proposed_family_work_rate
    # The exact-off legacy comparator deliberately retains every proposed
    # family.  The production energy-limited path selects families below from
    # their complete finite-event affinity, not from conjugate mechanical work
    # alone.  Keep the full-rate timestep here: the selection is a constrained
    # constitutive rate over this interval, not a hidden timestep subdivision.
    family_event_scales = np.ones(len(systems))
    elastic_before = _resolved_elastic_energy_sum_J_m3_cells(drive)

    def _candidate_model(scales):
        scaled_flow = (proposed_family_flow_rate
                       *scales[None, None, :, None, None])
        scaled_slip = proposed_mura_slip_rate*scales[None, None, :]
        full_beta = state.common.beta_p+accepted_dt*np.sum(scaled_flow, axis=2)
        full_plastic_work = float(np.sum(
            accepted_dt*drive["raw_stress_Pa"]*scaled_slip))
        elastic_quadratic = None
        if elastic_before is not None:
            elastic_after = _elastic_energy_sum_J_m3_cells(
                state.common, full_beta, driving, common_parameters)
            elastic_quadratic = (
                full_plastic_work-(elastic_before-elastic_after))
        return scaled_flow, full_plastic_work, elastic_quadratic

    def _candidate_trial(scales, event_scale, model):
        scaled_flow, full_plastic_work, elastic_quadratic = model
        velocity_plus = (event_scale*proposed_velocity_plus_3d
                         *scales[None, None, :, None])
        velocity_minus = (event_scale*proposed_velocity_minus_3d
                          *scales[None, None, :, None])
        family_flow = event_scale*scaled_flow
        density, alignment, capture = accepted_mura_transport_capture_step(
            state.density, state.reservoir_alignment,
            velocity_plus, velocity_minus, capture_support, systems,
            state.common.orientation_rad, common_parameters.spacing_m,
            accepted_dt, topologies)
        elastic_release = None
        if elastic_before is not None:
            elastic_release = (event_scale*full_plastic_work
                               -event_scale*event_scale*elastic_quadratic)
        audit, slip_rate, plastic_work, defect_delta, line_creation = (
            _mura_budget_audit(
                state, density, capture, family_flow,
                drive["raw_stress_Pa"], schmid_tensors, schmid_norm2,
                event_scale, accepted_dt, systems, topologies,
                common_parameters, elastic_release))
        audit["family_event_scales"] = [float(value) for value in scales]
        audit["proposed_family_work_before_selection_J_m3_cells"] = [
            float(value) for value in proposed_family_work]
        # Retain the V32 key for restart/postprocessor compatibility while
        # making clear in the new ledger that it no longer selects families.
        audit["proposed_family_work_before_stall_J_m3_cells"] = [
            float(value) for value in proposed_family_work]
        return (density, alignment, capture, family_flow, slip_rate,
                plastic_work, defect_delta, line_creation, audit)

    family_candidate_audits = []
    if mura_work_budget_mode == "energy_limited":
        family_event_scales = np.zeros(len(systems))
        for family in range(len(systems)):
            scales = np.zeros(len(systems)); scales[family] = 1.0
            try:
                candidate = _candidate_trial(
                    scales, 1.0, _candidate_model(scales))
                audit = candidate[-1]
                selected = bool(audit["admissible"])
                family_candidate_audits.append({
                    "family": int(family), "result": "EVALUATED",
                    "selected": selected, "audit": audit})
                family_event_scales[family] = float(selected)
            except (ValueError, RuntimeError) as error:
                if ("alignment magnitude exceeds" not in str(error)
                        and "negative mobile density" not in str(error)):
                    raise
                family_candidate_audits.append({
                    "family": int(family),
                    "result": "INADMISSIBLE_KINEMATICS",
                    "selected": False,
                    "error": f"{type(error).__name__}: {error}"})

    selected_model = _candidate_model(family_event_scales)

    budget_trials = []
    accepted_transaction = None
    scales = ([1.0] if mura_work_budget_mode == "legacy_reject" else
              [0.0] if not np.any(family_event_scales > 0.0) else
              [2.0**(-attempt) for attempt in
               range(int(maximum_work_budget_backtracks)+1)]+[0.0])
    for event_scale in scales:
        try:
            (transported_density, transported_alignment, capture_ledger,
             family_flow_rate, mura_slip_rate,
             plastic_work_increment_J_m3, defect_delta, line_creation,
             audit) = _candidate_trial(
                 family_event_scales, event_scale, selected_model)
            budget_trials.append(audit)
            if mura_work_budget_mode == "legacy_reject":
                old_storage = float(np.sum(line_creation))
                old_work = float(np.sum(plastic_work_increment_J_m3))
                if old_storage > old_work+1e-12*max(abs(old_work), 1.0):
                    raise MuraWorkBudgetError(
                        "Mura line storage exceeds available plastic work",
                        {"mode": "legacy_reject", "trials": budget_trials})
            elif not audit["admissible"]:
                continue
            accepted_transaction = (
                event_scale, transported_density, transported_alignment,
                capture_ledger, family_flow_rate, mura_slip_rate,
                plastic_work_increment_J_m3, defect_delta, line_creation, audit)
            break
        except (ValueError, RuntimeError) as error:
            if isinstance(error, MuraWorkBudgetError):
                raise
            if ("alignment magnitude exceeds" not in str(error)
                    and "negative mobile density" not in str(error)):
                raise
            if mura_work_budget_mode == "legacy_reject":
                raise
            continue
    if accepted_transaction is None:
        raise MuraWorkBudgetError(
            "Mura work-budget limiter found no admissible event extent",
            {"mode": mura_work_budget_mode, "trials": budget_trials})
    (event_scale, transported_density, transported_alignment, capture_ledger,
     family_flow_rate, mura_slip_rate, plastic_work_increment_J_m3,
     defect_energy_increment_J_m3, mura_storage_increment_J_m3,
     accepted_budget) = accepted_transaction
    beta_p, family_nye, mura_audit = accept_family_mura_step(
        state.common.beta_p, state.common.family_nye_m1,
        family_flow_rate, accepted_dt, common_parameters.spacing_m)
    alignment_rate = (
        capture_ledger["alignment_rate_plus_m2_s"]
        -capture_ledger["alignment_rate_minus_m2_s"])
    total_flow_rate = np.sum(family_flow_rate, axis=2)
    orientation_rate = common_parameters.orientation_spin_weight*.5*(
        total_flow_rate[..., 1, 0]-total_flow_rate[..., 0, 1])
    mura_plastic_power_W_m3 = np.sum(
        drive["raw_stress_Pa"]*mura_slip_rate, axis=2)
    if mura_work_budget_mode == "legacy_reject":
        accepted_storage_increment_J_m3 = mura_storage_increment_J_m3
        total_work = float(np.sum(plastic_work_increment_J_m3))
        storage_fraction = float(np.sum(mura_storage_increment_J_m3))/max(
            total_work, 1e-300)
        deposited_heat_increment_J_m3 = (
            plastic_work_increment_J_m3*(1.0-storage_fraction))
        ledger_mechanical_source = total_work
    else:
        accepted_storage_increment_J_m3 = defect_energy_increment_J_m3
        total_heat = max(float(accepted_budget[
            "dissipative_drag_and_heat_J_m3_cells"]), 0.0)
        # The Mura event is nonlocal. Deposit its closed global heat on the
        # local nonnegative dissipation support, without converting a
        # compatibility correction into physical heat.
        local_dissipation_support = np.maximum(
            plastic_work_increment_J_m3-defect_energy_increment_J_m3, 0.0)
        support_sum = float(np.sum(local_dissipation_support))
        if support_sum > 0.0:
            deposited_heat_increment_J_m3 = (
                total_heat*local_dissipation_support/support_sum)
        else:
            deposited_heat_increment_J_m3 = np.full_like(
                plastic_work_increment_J_m3,
                total_heat/plastic_work_increment_J_m3.size)
        ledger_mechanical_source = accepted_budget[
            "recoverable_elastic_energy_release_J_m3_cells"]
        if ledger_mechanical_source is None:
            ledger_mechanical_source = accepted_budget["plastic_work_J_m3_cells"]
    # Only slip bookkeeping, the atomic Mura fields, spin, and mechanical heat
    # are accepted from the constitutive residual. Legacy independent beta/Nye
    # and density rates are deliberately excluded.
    common = replace(
        state.common,
        slip=state.common.slip+accepted_dt*mura_slip_rate,
        beta_p=beta_p,
        alignment_m2=state.common.alignment_m2+accepted_dt*alignment_rate,
        family_nye_m1=family_nye,
        orientation_rad=state.common.orientation_rad+accepted_dt*orientation_rate,
        temperature_K=(state.common.temperature_K+deposited_heat_increment_J_m3
                       /common_parameters.volumetric_heat_capacity_J_m3_K))
    _nye_beta_after_kinematics = np.sum(common.family_nye_m1, axis=2)
    _nye_reservoir_after_transport = reservoir_nye_m1(
        transported_alignment, systems, common.orientation_rad,
        topologies)["total"]
    working_density, working_alignment, locking_ledger = (
        apply_signed_reservoir_exchange(
            transported_density, transported_alignment, "mobile", "forest",
            accepted_dt*residual.channel_rates_m2_s["lock_plus"],
            accepted_dt*residual.channel_rates_m2_s["lock_minus"], systems,
            common.orientation_rad, topologies))
    topology_ledger = None
    reorientation_ledger = None
    if topology_route_enabled and topologies:
        family_enthalpy = exp_floor_enthalpy_j(
            np.abs(drive["effective_stress_Pa"]),
            topology_kinetics.junction_enthalpy_J,
            topology_kinetics.critical_stress_Pa, topology_kinetics.exp_a,
            topology_kinetics.exp_n, topology_kinetics.exp_floor)
        family_rate_s = activated_rate_array_s(
            topology_kinetics.junction_process, family_enthalpy,
            common.temperature_K[..., None])
        delta_free_energy_J = (
            (extensive_parameters.ordered_excess_J_m
             -extensive_parameters.disordered_excess_J_m)
            *extensive_parameters.event_length_m)
        thermal_energy_J = 1.380649e-23*common.temperature_K[..., None]
        affinity = delta_free_energy_J/thermal_energy_J
        # Two stable logistic factors retain k_forward/k_reverse=exp(-Delta F/kT)
        # without cancellation when one direction is strongly favored.
        bounded_affinity = np.clip(affinity, -700.0, 700.0)
        forward_rate_s = family_rate_s*2.0/(1.0+np.exp(bounded_affinity))
        reverse_rate_s = family_rate_s*2.0/(1.0+np.exp(-bounded_affinity))
        forward_fraction = np.minimum(
            -np.expm1(-accepted_dt*forward_rate_s),
            topology_kinetics.maximum_junction_fraction_per_step)
        reverse_fraction = np.minimum(
            -np.expm1(-accepted_dt*reverse_rate_s),
            topology_kinetics.maximum_junction_fraction_per_step)
        request_plus = (
            forward_fraction*working_density.wall_tangle_plus_m2
            -reverse_fraction*working_density.wall_ordered_plus_m2)
        request_minus = (
            forward_fraction*working_density.wall_tangle_minus_m2
            -reverse_fraction*working_density.wall_ordered_minus_m2)
        working_density, working_alignment, reorientation_ledger = (
            accepted_line_reorientation_step(
                working_density, working_alignment, request_plus,
                request_minus, (0.0, 0.0, 1.0),
                disordered_line_directions,
                extensive_parameters.event_length_m, systems,
                common.orientation_rad, accepted_dt, topologies))
        reorientation_ledger["thermodynamics"] = {
            "delta_free_energy_per_event_J": delta_free_energy_J,
            "forward_rate_s": forward_rate_s,
            "reverse_rate_s": reverse_rate_s,
            "detailed_balance_ratio": np.divide(
                forward_rate_s, reverse_rate_s,
                out=np.ones_like(forward_rate_s), where=reverse_rate_s > 0.0),
            "expected_ratio": np.exp(-bounded_affinity),
        }
        pair_stress = np.stack([
            np.maximum(np.abs(drive["effective_stress_Pa"][..., item.parent_a]),
                       np.abs(drive["effective_stress_Pa"][..., item.parent_b]))
            for item in topologies], axis=2)
        enthalpy = exp_floor_enthalpy_j(
            pair_stress, topology_kinetics.junction_enthalpy_J,
            topology_kinetics.critical_stress_Pa, topology_kinetics.exp_a,
            topology_kinetics.exp_n, topology_kinetics.exp_floor)
        rate_s = activated_rate_array_s(
            topology_kinetics.junction_process, enthalpy,
            common.temperature_K[..., None])
        requests = []
        for index, item in enumerate(topologies):
            first = getattr(working_density,
                f"wall_tangle_{'plus' if item.sign_a > 0 else 'minus'}_m2")[..., item.parent_a]
            second = getattr(working_density,
                f"wall_tangle_{'plus' if item.sign_b > 0 else 'minus'}_m2")[..., item.parent_b]
            fraction = np.minimum(-np.expm1(-accepted_dt*rate_s[..., index]),
                                  topology_kinetics.maximum_junction_fraction_per_step)
            requests.append(fraction*np.minimum(first, second))
        working_density, working_alignment, topology_ledger = (
            accepted_junction_topology_step(
                working_density, working_alignment, np.stack(requests, axis=2),
                systems, topologies, common.orientation_rad, accepted_dt))
    if topology_route_enabled:
        ordered_density, ordered_alignment = working_density, working_alignment
        ordering_thermo = {"disabled_in_explicit_topology_comparator": True}
        ordering_topology = None
    else:
        # Shared EXP-floor/signed-entropy ordering law. With C_FB=0 and a zero
        # target, only the declared extensive free-energy affinity selects direction.
        zero_target = np.zeros(state.common.orientation_rad.shape+(3, 3))
        ordered_density, ordering_thermo, _ = accepted_ordering_step(
            working_density, systems, topologies, common.orientation_rad,
            zero_target, drive["effective_stress_Pa"], common.temperature_K,
            extensive_parameters, accepted_dt)
        extent_plus = (ordered_density.wall_ordered_plus_m2
                       -working_density.wall_ordered_plus_m2)
        extent_minus = (ordered_density.wall_ordered_minus_m2
                        -working_density.wall_ordered_minus_m2)
        ordered_density, ordered_alignment, ordering_topology = (
            apply_signed_ordering_extent(
                working_density, working_alignment,
                extent_plus, extent_minus, systems, common.orientation_rad,
                topologies))
    result = synchronize_common(V24MechanicalWallState(
        common, ordered_density, ordered_alignment), topologies)
    result.validate(systems, topologies)
    _nye_reservoir_after_reactions = reservoir_nye_m1(
        result.reservoir_alignment, systems, result.common.orientation_rad,
        topologies)["total"]
    _declared_reaction_source = np.zeros_like(_nye_beta_initial)
    if topology_route_enabled:
        _declared_reaction_source = (
            _nye_reservoir_after_reactions-_nye_reservoir_after_transport)
        family_nye_with_source = result.common.family_nye_m1.copy()
        family_nye_with_source[..., 0, :, :] += _declared_reaction_source
        result = replace(result, common=replace(
            result.common, family_nye_m1=family_nye_with_source))
        result.validate(systems, topologies)
    _reference = max(float(np.sqrt(np.mean(_nye_beta_initial**2))), 1.0)
    def _stage(name, reservoir_increment, beta_increment):
        mismatch = reservoir_increment-beta_increment
        return {
            "operator": name,
            "reservoir_increment_rms_m1": float(np.sqrt(np.mean(
                reservoir_increment**2))),
            "plastic_curl_increment_rms_m1": float(np.sqrt(np.mean(
                beta_increment**2))),
            "increment_residual_rms_m1": float(np.sqrt(np.mean(mismatch**2))),
            "increment_residual_relative": float(
                np.sqrt(np.mean(mismatch**2))/_reference),
        }
    _zero = np.zeros_like(_nye_beta_initial)
    _mura_increment = _nye_beta_after_kinematics-_nye_beta_initial
    _mura_stage = _stage("authoritative_mura_face_flux",
                         _mura_increment, _mura_increment)
    _mura_stage["reservoir_first_moment_increment_rms_m1"] = float(
        np.sqrt(np.mean((_nye_reservoir_after_transport
                         -_nye_reservoir_initial)**2)))
    _mura_stage["reservoir_connection_residual_rms_m1"] = float(
        np.sqrt(np.mean((_nye_reservoir_after_transport
                         -_nye_reservoir_initial-_mura_increment)**2)))
    _nye_stages = [
        _mura_stage,
        _stage("ordering_and_topology_reactions",
               _nye_reservoir_after_reactions-_nye_reservoir_after_transport,
               _declared_reaction_source),
    ]
    _violating = next((row["operator"] for row in _nye_stages
                       if row["increment_residual_rms_m1"]
                       > 2e-11*_reference), None)
    _scalar_balance = max(
        abs(float(capture_ledger["sign"][sign][
            "global_scalar_residual_line_per_thickness"]))
        for sign in ("plus", "minus"))
    _alignment_balance = max(
        float(np.max(np.abs(capture_ledger["sign"][sign][
            "global_alignment_residual_line_per_thickness"])))
        for sign in ("plus", "minus"))
    _plastic_work_increment = plastic_work_increment_J_m3
    _deposited_heat_increment = deposited_heat_increment_J_m3
    _mechanical_source = ledger_mechanical_source
    return result, {
        "accepted_dt_s": accepted_dt,
        "mura_event_scale": event_scale,
        "mura_family_event_scales": family_event_scales,
        "mura_work_budget_mode": mura_work_budget_mode,
        "mura_work_budget": {
            "accepted": accepted_budget,
            "trial_count": len(budget_trials),
            "trials": budget_trials,
            "family_selection_rule": (
                "complete_discrete_full_event_affinity_then_joint_backtrack"
                if mura_work_budget_mode == "energy_limited"
                else "legacy_no_family_selection"),
            "family_candidate_audits": family_candidate_audits,
            "joint_selected_family_candidate": budget_trials[0],
            "remainder_fraction_unreacted": float(1.0-event_scale),
            "unreacted_remainder_semantics": (
                "deterministic_constrained_rate_no_debt_recomputed_next_step"
                if mura_work_budget_mode == "energy_limited"
                else "legacy_not_applicable"),
            "family_unreacted_fractions": [float(
                1.0-event_scale*family_event_scales[index])
                for index in range(len(systems))],
            "physical_stall": bool(event_scale == 0.0
                                   or not np.any(family_event_scales > 0.0)),
            "stalled_families": [int(index) for index, scale in
                                 enumerate(family_event_scales) if scale == 0.0],
        },
        "raw_stress_Pa": drive["raw_stress_Pa"],
        "effective_stress_Pa": drive["effective_stress_Pa"],
        "plastic_power_W_m3": mura_plastic_power_W_m3,
        "transport_capture": capture_ledger,
        "locking_unlocking": locking_ledger,
        "ordering_thermodynamics": ordering_thermo,
        "ordering_topology": ordering_topology,
        "line_reorientation_topology": reorientation_ledger,
        "junction_topology": topology_ledger,
        "legacy_common_density_rates_accepted": False,
        "legacy_independent_beta_nye_rates_accepted": False,
        "legacy_independent_slip_rate_accepted": False,
        "orientation_target_used": False,
        "mura_face_event": mura_audit,
        "mura_slip_rate_s": mura_slip_rate,
        "mura_balance_ledger": {
            "maximum_scalar_line_balance_residual_m": _scalar_balance,
            "maximum_alignment_balance_residual_m": _alignment_balance,
            "plastic_work_increment_J_m3": _plastic_work_increment,
            "deposited_heat_increment_J_m3": _deposited_heat_increment,
            "stored_line_energy_increment_J_m3": accepted_storage_increment_J_m3,
            "line_creation_energy_increment_J_m3": mura_storage_increment_J_m3,
            "recoverable_elastic_energy_release_J_m3_cells": float(
                _mechanical_source),
            "global_work_minus_heat_storage_residual_J_m3_cells": float(
                _mechanical_source-np.sum(_deposited_heat_increment)
                -np.sum(accepted_storage_increment_J_m3)),
            "reaction_source_tensor_m1": _declared_reaction_source,
            "declared_topology_source_tensor_m1": _declared_reaction_source,
            "line_stretching_is_declared_mura_geometric_source": True,
        },
        "nye_suboperator_audit": {
            "sign_convention": "alpha=-Curl(beta_p)",
            "stages": _nye_stages,
            "first_violating_suboperator": _violating,
            "accepted_step_hard_invariant_passed": bool(
                _violating is None
                and mura_audit["accepted_step_hard_invariant_passed"]),
            "post_step_projection_used": False,
        },
    }
