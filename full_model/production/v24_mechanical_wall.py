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
from .extensive_wall import (
    ExtensiveWallParameters, accepted_ordering_step,
    extensive_wall_energy_components_J_m3,
)
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


def select_feasible_family_extent(extent_curve):
    """Classify and select a connected finite-event family extent.

    ``extent_curve`` is ordered from the full proposal toward zero.  The
    kinetic proposal fixes that ray; this routine only finds its largest
    thermodynamically admissible sampled extent.  It therefore does not
    maximize energy release or create event debt.
    """
    if not extent_curve or float(extent_curve[0]["extent"]) != 1.0:
        raise ValueError("family extent curve must begin with the full event")
    feasible = [row for row in extent_curve
                if row["result"] == "EVALUATED" and row["admissible"]]
    selected_extent = max(
        (float(row["extent"]) for row in feasible), default=0.0)
    full = extent_curve[0]
    if selected_extent == 1.0:
        classification = "FULL_EVENT_ADMISSIBLE"
    elif selected_extent > 0.0 and full["result"] != "EVALUATED":
        classification = "CAPACITY_LIMITED_EVENT"
    elif selected_extent > 0.0:
        classification = "INITIAL_DIRECTION_DOWNHILL_FULL_EVENT_OVERSHOOTS"
    elif any(row.get("result") == "INADMISSIBLE_KINEMATICS"
             for row in extent_curve):
        classification = "CAPACITY_LIMITED_NO_RESOLVED_FEASIBLE_EXTENT"
    elif any(not row.get("sign_resolved", False) for row in extent_curve):
        classification = "UNRESOLVED_SUBTRACTIVE_CANCELLATION"
    else:
        classification = "GENUINELY_UPHILL_SCREENED_DIRECTION"
    return selected_extent, classification


def adaptive_connected_feasible_extent(evaluate, levels):
    """Find the largest feasible dyadic extent connected to zero.

    The complete affinity along a fixed Mura family ray is screened on the
    same dyadic set used by the exhaustive V35 implementation.  The zero
    event is an exact admissible identity.  Two smallest nonzero extents give
    a one-sided directional-affinity estimate and distinguish a physically
    uphill direction from subtraction at the zero endpoint.  When that
    endpoint is downhill, a binary bracket locates the first admissible
    dyadic extent.  No new extent and no altered constitutive state is used.

    ``evaluate`` must return the same row accepted by
    :func:`select_feasible_family_extent`.  The returned curve contains only
    evaluated points, ordered from full extent toward zero.
    """
    levels = int(levels)
    if levels < 2:
        raise ValueError("feasible-family screen requires at least two levels")
    extents = [2.0**(-level) for level in range(levels)]
    rows = {}

    def at(index):
        if index not in rows:
            rows[index] = evaluate(extents[index])
        return rows[index]

    full = at(0)
    near = at(levels-1)
    near2 = at(levels-2) if levels > 2 else full

    def resolved_admissible(row):
        return (row["result"] == "EVALUATED" and row["admissible"]
                and row.get("sign_resolved", False))

    derivative = None
    derivative_classification = "UNRESOLVED_ROUNDOFF"
    if (near["result"] == "EVALUATED"
            and near2["result"] == "EVALUATED"):
        e1, h1 = float(near["extent"]), float(
            near["complete_affinity_J_m3_cells"])
        e2, h2 = float(near2["extent"]), float(
            near2["complete_affinity_J_m3_cells"])
        # Exact for a quadratic affinity H(e)=a*e+b*e^2 and a more stable
        # zero-endpoint classifier than H(e) itself.
        derivative = (h1*e2*e2-h2*e1*e1)/(e1*e2*(e2-e1))
        derivative_tolerance = max(
            float(near.get("affinity_tolerance_J_m3_cells", 0.0))/e1,
            float(near2.get("affinity_tolerance_J_m3_cells", 0.0))/e2)
        if derivative > derivative_tolerance:
            derivative_classification = "DOWNHILL_FROM_ZERO"
        elif derivative < -derivative_tolerance:
            derivative_classification = "UPHILL_FROM_ZERO"
    else:
        derivative_tolerance = None

    connected = (resolved_admissible(near)
                 and (levels == 2 or resolved_admissible(near2)))
    selected_index = None
    if connected:
        if resolved_admissible(full):
            selected_index = 0
        else:
            # Predicate is false at the full event and true at the near-zero
            # endpoint. Locate the first true member of the preserved dyadic
            # set. Sampled points on the zero side remain admissible, which is
            # the discrete connected-to-zero invariant.
            lo, hi = 0, levels-1
            while hi-lo > 1:
                middle = (lo+hi)//2
                if resolved_admissible(at(middle)):
                    hi = middle
                else:
                    lo = middle
            selected_index = hi
    curve = [rows[index] for index in sorted(rows)]
    selected_extent = (extents[selected_index]
                       if selected_index is not None else 0.0)
    if selected_extent == 1.0:
        classification = "FULL_EVENT_ADMISSIBLE"
    elif selected_extent > 0.0 and full["result"] != "EVALUATED":
        classification = "CAPACITY_LIMITED_EVENT"
    elif selected_extent > 0.0:
        classification = "INITIAL_DIRECTION_DOWNHILL_FULL_EVENT_OVERSHOOTS"
    elif any(row.get("result") == "INADMISSIBLE_KINEMATICS"
             for row in curve):
        classification = "CAPACITY_LIMITED_NO_RESOLVED_FEASIBLE_EXTENT"
    elif derivative_classification == "UPHILL_FROM_ZERO":
        classification = "GENUINELY_UPHILL_SCREENED_DIRECTION"
    else:
        classification = "UNRESOLVED_SUBTRACTIVE_CANCELLATION"
    return selected_extent, classification, curve, {
        "zero_extent_is_exact_identity": True,
        "connected_to_zero_verified_on_sampled_bracket": bool(connected),
        "directional_affinity_J_m3_cells_per_extent": derivative,
        "directional_affinity_tolerance_J_m3_cells_per_extent": (
            derivative_tolerance),
        "directional_classification": derivative_classification,
        "dyadic_levels_available": levels,
        "dyadic_levels_evaluated": len(curve),
    }


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


def accepted_energy_guarded_reservoir_topology_transaction(
        inventory, alignment, systems, topologies, orientation_rad,
        stress_Pa, temperature_K, parameters, dt_s):
    """Accept geometry-neutral ordering only after a complete energy audit.

    The current continuum state has scalar reservoir content and first line
    moments, but no persistent segment endpoints or swept surface from which a
    finite line reorientation can update ``beta_p``.  Consequently the only
    topology operation representable without manufacturing incompatibility is
    a reservoir conversion which carries the existing moment with the line.

    The candidate is built on copied immutable state by the qualified
    extensive ordering integrator.  Publication requires exact scalar-line
    and total reservoir-Nye closure and a non-increasing *discrete* extensive
    free energy, including the ordered-gradient term.  Rejection returns the
    original objects and zero heat, so no state or ledger is partly committed.
    """
    zero_target = np.zeros(np.asarray(orientation_rad).shape+(3, 3))
    before_parts = extensive_wall_energy_components_J_m3(
        inventory, systems, topologies, orientation_rad, zero_target,
        parameters)
    before_nye = reservoir_nye_m1(
        alignment, systems, orientation_rad, topologies)["total"]
    before_line = derived_density_fields(inventory, topologies)["rho_total_m2"]
    candidate_inventory, candidate_alignment, kinetics, _ = (
        accepted_ordering_step(
            inventory, systems, topologies, orientation_rad, zero_target,
            stress_Pa, temperature_K, parameters, dt_s,
            alignment=alignment))
    after_parts = extensive_wall_energy_components_J_m3(
        candidate_inventory, systems, topologies, orientation_rad,
        zero_target, parameters)
    after_nye = reservoir_nye_m1(
        candidate_alignment, systems, orientation_rad, topologies)["total"]
    after_line = derived_density_fields(
        candidate_inventory, topologies)["rho_total_m2"]

    component_changes = {
        name: np.asarray(after_parts[name])-np.asarray(before_parts[name])
        for name in before_parts
    }
    delta_total = float(np.sum(component_changes["total"],
                               dtype=np.longdouble))
    energy_scale = max(
        abs(float(np.sum(before_parts["total"], dtype=np.longdouble))),
        abs(float(np.sum(after_parts["total"], dtype=np.longdouble))), 1.0)
    energy_tolerance = 2e-12*energy_scale
    line_residual = after_line-before_line
    nye_residual = after_nye-before_nye
    line_scale = max(float(np.max(np.abs(before_line))), 1.0)
    nye_scale = max(float(np.sqrt(np.mean(before_nye*before_nye))), 1.0)
    line_closed = bool(np.max(np.abs(line_residual)) <= 2e-12*line_scale)
    nye_closed = bool(np.sqrt(np.mean(nye_residual*nye_residual))
                      <= 2e-12*nye_scale)
    accepted = bool(line_closed and nye_closed
                    and delta_total <= energy_tolerance)

    if not accepted:
        return inventory, alignment, {
            "operator": "energy_guarded_geometry_neutral_topology_transaction",
            "accepted": False,
            "rejection_is_atomic": True,
            "classification": "COMPLETE_TOPOLOGY_CANDIDATE_REJECTED",
            "scalar_line_residual_m2": line_residual,
            "total_nye_residual_m1": nye_residual,
            "component_energy_changes_J_m3": component_changes,
            "complete_energy_change_J_m3_cells": delta_total,
            "energy_tolerance_J_m3_cells": energy_tolerance,
            "irreversible_heat_increment_J_m3": np.zeros_like(
                np.asarray(orientation_rad)),
            "ordering_kinetics": kinetics,
        }

    released = max(-delta_total, 0.0)
    extent_weight = sum(np.abs(np.asarray(
        kinetics["accepted_transfer_m2_s"][sign]))
        for sign in ("plus", "minus"))
    extent_weight = np.sum(extent_weight, axis=2)
    weight_sum = float(np.sum(extent_weight))
    if released == 0.0:
        heat = np.zeros_like(np.asarray(orientation_rad))
    elif weight_sum > 0.0:
        heat = released*extent_weight/weight_sum
    else:
        heat = np.full_like(np.asarray(orientation_rad),
                            released/np.asarray(orientation_rad).size)
    return candidate_inventory, candidate_alignment, {
        "operator": "energy_guarded_geometry_neutral_topology_transaction",
        "accepted": True,
        "rejection_is_atomic": True,
        "classification": "ADMISSIBLE_RESERVOIR_CONVERSION_NO_GEOMETRY_CHANGE",
        "geometry_scope": (
            "existing line and first moment relabeling; no reorientation, "
            "junction creation, swept plastic area, or endpoint creation"),
        "scalar_line_residual_m2": line_residual,
        "total_nye_residual_m1": nye_residual,
        "component_energy_changes_J_m3": component_changes,
        "complete_energy_change_J_m3_cells": delta_total,
        "energy_tolerance_J_m3_cells": energy_tolerance,
        "irreversible_heat_increment_J_m3": heat,
        "ordering_kinetics": kinetics,
    }


def accepted_v24_mechanical_step(
        state, driving, capture_support, systems, topologies,
        common_parameters, extensive_parameters, topology_kinetics, dt_s, *,
        topology_route_enabled=False, maximum_orientation_increment_rad=0.02,
        mura_work_budget_mode="energy_limited",
        maximum_work_budget_backtracks=20,
        feasible_family_extent_levels=12):
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
    if mura_work_budget_mode not in (
            "energy_limited", "energy_limited_feasible_extents",
            "legacy_reject"):
        raise ValueError("unknown Mura work-budget mode")
    if int(maximum_work_budget_backtracks) < 0:
        raise ValueError("Mura work-budget backtracks cannot be negative")
    if int(feasible_family_extent_levels) < 2:
        raise ValueError("feasible-family screen requires at least two levels")
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
    candidate_model_cache = {}

    def _candidate_model(scales):
        key = tuple(float(value) for value in np.asarray(scales))
        if key in candidate_model_cache:
            return candidate_model_cache[key]
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
        model = scaled_flow, full_plastic_work, elastic_quadratic
        candidate_model_cache[key] = model
        return model

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
    if mura_work_budget_mode in (
            "energy_limited", "energy_limited_feasible_extents"):
        family_event_scales = np.zeros(len(systems))
        for family in range(len(systems)):
            unit_scales = np.zeros(len(systems)); unit_scales[family] = 1.0
            unit_model = _candidate_model(unit_scales)

            def _evaluate_extent(extent):
                try:
                    candidate = _candidate_trial(
                        unit_scales, extent, unit_model)
                    audit = candidate[-1]
                    # Preserve the V35 per-family audit semantics: this is the
                    # actual family extent, not the cached unit-ray model.
                    audit["family_event_scales"] = [
                        float(extent*value) for value in unit_scales]
                    heat = float(audit[
                        "dissipative_drag_and_heat_J_m3_cells"])
                    tolerance = float(audit[
                        "admissibility_tolerance_J_m3_cells"])
                    return {
                        "extent": float(extent), "result": "EVALUATED",
                        "admissible": bool(audit["admissible"]),
                        "complete_affinity_J_m3_cells": heat,
                        "sign_resolved": bool(abs(heat) > tolerance),
                        "affinity_tolerance_J_m3_cells": tolerance,
                        "audit": audit}
                except (ValueError, RuntimeError) as error:
                    if ("alignment magnitude exceeds" not in str(error)
                            and "negative mobile density" not in str(error)):
                        raise
                    return {
                        "extent": float(extent),
                        "result": "INADMISSIBLE_KINEMATICS",
                        "admissible": False,
                        "error": f"{type(error).__name__}: {error}"}

            if mura_work_budget_mode == "energy_limited":
                extent_curve = [_evaluate_extent(1.0)]
                selected_extent, classification = (
                    select_feasible_family_extent(extent_curve))
                search_audit = None
            else:
                (selected_extent, classification, extent_curve,
                 search_audit) = adaptive_connected_feasible_extent(
                    _evaluate_extent, feasible_family_extent_levels)
            full = extent_curve[0]
            family_event_scales[family] = selected_extent
            selected_row = next((row for row in extent_curve
                                 if row["extent"] == selected_extent), None)
            family_candidate_audits.append({
                "family": int(family),
                "result": full["result"],
                "selected": bool(selected_extent > 0.0),
                "selected_extent": float(selected_extent),
                "classification": classification,
                "extent_curve": extent_curve,
                **({"adaptive_search": search_audit}
                   if search_audit is not None else {}),
                **({"audit": selected_row["audit"]}
                   if selected_row is not None else {}),
                **({"error": full["error"]}
                   if full["result"] != "EVALUATED" else {}),
            })

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
    if topology_route_enabled:
        # V41 showed that local line reorientation plus a reconstructed Nye
        # source creates incompatibility without a represented swept surface,
        # and omits the dominant ordered-gradient energy.  The production
        # topology flag now selects the strongest event supported by the
        # persistent state: geometry-neutral reservoir conversion with a
        # complete copied-state energy guard.  The legacy reorientation and
        # junction routines remain available as isolated audit fixtures, but
        # cannot publish through this driver until persistent segment/node and
        # swept-surface geometry exists.
        (ordered_density, ordered_alignment,
         topology_ledger) = accepted_energy_guarded_reservoir_topology_transaction(
            working_density, working_alignment, systems, topologies,
            common.orientation_rad, drive["effective_stress_Pa"],
            common.temperature_K, extensive_parameters, accepted_dt)
        topology_heat = topology_ledger["irreversible_heat_increment_J_m3"]
        common = replace(
            common,
            temperature_K=(common.temperature_K+topology_heat
                           /common_parameters.volumetric_heat_capacity_J_m3_K))
        ordering_thermo = topology_ledger["ordering_kinetics"]
        ordering_topology = ordering_thermo.get("topology_ledger")
        reorientation_ledger = {
            "operator": "unrepresentable_geometry_fail_closed",
            "executed": False,
            "reason": (
                "persistent endpoint/swept-surface state is absent; local "
                "moment rotation cannot update beta_p consistently"),
        }
    else:
        # Shared EXP-floor/signed-entropy ordering law. With C_FB=0 and a zero
        # target, only the declared extensive free-energy affinity selects direction.
        zero_target = np.zeros(state.common.orientation_rad.shape+(3, 3))
        (ordered_density, ordered_alignment,
         ordering_thermo, _) = accepted_ordering_step(
            working_density, systems, topologies, common.orientation_rad,
            zero_target, drive["effective_stress_Pa"], common.temperature_K,
            extensive_parameters, accepted_dt, alignment=working_alignment)
        ordering_topology = ordering_thermo.pop("topology_ledger")
    result = synchronize_common(V24MechanicalWallState(
        common, ordered_density, ordered_alignment), topologies)
    result.validate(systems, topologies)
    _nye_reservoir_after_reactions = reservoir_nye_m1(
        result.reservoir_alignment, systems, result.common.orientation_rad,
        topologies)["total"]
    _declared_reaction_source = np.zeros_like(_nye_beta_initial)
    if topology_route_enabled:
        # Geometry-neutral conversion moves each reservoir moment with its
        # line, so its physical source is identically zero.  Never overwrite
        # the plastic-curl Nye field with a measured reservoir residual.
        _declared_reaction_source = np.zeros_like(_nye_beta_initial)
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
                if mura_work_budget_mode == "energy_limited" else
                "complete_affinity_feasible_family_extent_then_joint_backtrack"
                if mura_work_budget_mode == "energy_limited_feasible_extents"
                else "legacy_no_family_selection"),
            "family_candidate_audits": family_candidate_audits,
            "candidate_model_cache_entries": len(candidate_model_cache),
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
        "topology_energy_kinematics": topology_ledger,
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
