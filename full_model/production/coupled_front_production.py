"""Production adapter for an atomic, bidirectional SIBM front event.

The phase solver supplies a *trial* profile.  This module measures its signed
zero-contour motion, constructs both thermodynamic transactions, limits the
motion by their net EXP-floor rate, and commits phase and material state
together.  The legacy phase-first ``advance_front`` operator is deliberately
not called here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math

import numpy as np

try:
    from .arrhenius_kinetics import ActivatedProcess
    from .coupled_front_event import FrontEnergyTerms, propose_bidirectional_front_event
    from .moving_front import (
        DefectState, FrontLedger, SparseFrontState, conservative_front_transfer,
        reconstruct_mixture, supported_front_state_is_exactly_equal,
        total_line_density)
    from .front_topology import (
        TopologySnapshot, diagnostic_ray_crossing_count,
        initialize_front_topology, match_front_topology,
        snapshot_from_dict, snapshot_to_dict)
    from .front_event_measure import physical_front_event_measure
except ImportError:  # pragma: no cover - direct production-script execution
    from arrhenius_kinetics import ActivatedProcess
    from coupled_front_event import FrontEnergyTerms, propose_bidirectional_front_event
    from moving_front import (
        DefectState, FrontLedger, SparseFrontState, conservative_front_transfer,
        reconstruct_mixture, supported_front_state_is_exactly_equal,
        total_line_density)
    from front_topology import (
        TopologySnapshot, diagnostic_ray_crossing_count,
        initialize_front_topology, match_front_topology,
        snapshot_from_dict, snapshot_to_dict)
    from front_event_measure import physical_front_event_measure


SCHEMA = "full-v34-coupled-front-production/v1"


def existing_pair_geometric_envelope(
        eta, *, parent_label: int, child_label: int, normal_axis: int,
        fraction: float = 0.125, direction: int = 1, active_mask=None):
    """Expose a bounded existing-interface translation to the kinetic limiter.

    This is a proposal envelope, not a mobility law.  Only the declared parent
    and child exchange phase fraction; the complete bidirectional rate and
    topology transaction below select the accepted prefix.
    """
    fields = np.asarray(eta, dtype=float)
    parent, child, axis = int(parent_label), int(child_label), int(normal_axis)
    if (fields.ndim != 3 or parent == child
            or min(parent, child) < 0 or max(parent, child) >= fields.shape[2]):
        raise ValueError("invalid existing-pair phase labels")
    if axis not in (0, 1) or direction not in (-1, 1):
        raise ValueError("invalid geometric-envelope direction")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("geometric-envelope fraction must be in (0,1]")
    value = fields[:, :, child]
    if direction > 0:
        bound = np.maximum.reduce(
            (value, np.roll(value, 1, axis=axis),
             np.roll(value, -1, axis=axis)))
    else:
        bound = np.minimum.reduce(
            (value, np.roll(value, 1, axis=axis),
             np.roll(value, -1, axis=axis)))
    proposed_child = np.clip(
        value + float(fraction)*(bound-value), 0.0, 1.0)
    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=bool)
        if mask.shape != value.shape:
            raise ValueError("geometric-envelope active mask is not grid matched")
        proposed_child = np.where(mask, proposed_child, value)
    # On a multiphase junction, the child may neighbor a third label.  The
    # pair envelope can consume only locally available declared parent; it may
    # not borrow phase fraction from an uninvolved grain.
    delta = np.clip(proposed_child-value, -value, fields[:, :, parent])
    proposed_child = value+delta
    proposed_parent = fields[:, :, parent]-delta
    result = fields.copy()
    result[:, :, child] = proposed_child
    result[:, :, parent] = proposed_parent
    if np.max(np.abs(np.sum(result, axis=2)-np.sum(fields, axis=2))) > 1e-12:
        raise RuntimeError("existing-pair envelope failed phase conservation")
    return result


@dataclass(frozen=True)
class CoupledFrontLedger:
    attempts: int = 0
    accepted: int = 0
    rejected_direction: int = 0
    stationary_trials: int = 0
    a_to_b_swept_volume_m3: float = 0.0
    b_to_a_swept_volume_m3: float = 0.0
    revisit_volume_m3: float = 0.0
    a_to_b_line_processed_m: float = 0.0
    b_to_a_line_processed_m: float = 0.0
    transmitted_line_m: float = 0.0
    boundary_line_m: float = 0.0
    annihilated_line_m: float = 0.0
    sink_line_m: float = 0.0
    heat_J: float = 0.0
    maximum_abs_line_closure_m: float = 0.0
    maximum_abs_signed_closure_m2: float = 0.0


@dataclass(frozen=True)
class CoupledFrontRuntime:
    material_a_label: int
    material_b_label: int
    normal_axis: int
    normal_orientation: int
    reference_b_fraction: np.ndarray
    maximum_b_fraction: np.ndarray
    minimum_b_fraction: np.ndarray
    previous_phi: np.ndarray
    topology: TopologySnapshot
    periodic: bool = True
    topology_event_count: int = 0
    ledger: CoupledFrontLedger = CoupledFrontLedger()


@dataclass(frozen=True)
class CoupledFrontDecision:
    accepted: bool
    classification: str
    proposed_signed_volume_m3: float
    accepted_signed_volume_m3: float
    rate_a_to_b_s: float
    rate_b_to_a_s: float
    net_velocity_a_to_b_m_s: float
    detailed_balance_log_residual: float
    maximum_abs_line_closure_m: float
    maximum_abs_signed_closure_m2: float
    heat_increment_J: float
    topology_event: dict | None = None
    component_count: int = 0
    ray_crossing_count_before: int = 0
    ray_crossing_count_after: int = 0
    maximum_component_distance_cells: float = 0.0
    filtered_subcell_components_after: int = 0
    topology_backtrack_fraction: float = 1.0
    positive_swept_volume_m3: float = 0.0
    negative_swept_volume_m3: float = 0.0
    absolute_swept_volume_m3: float = 0.0
    interface_area_m2: float = 0.0
    represented_thickness_m: float = 0.0
    cell_volume_m3: float = 0.0
    maximum_abs_phase_change: float = 0.0
    rms_phase_change: float = 0.0
    component_motion: tuple = ()
    kinetic_free_energy_a_to_b_J: float = 0.0
    kinetic_free_energy_b_to_a_J: float = 0.0
    microscopic_reverse_pair: bool = False
    actual_reverse_edge: bool = False
    detailed_balance_applicable: bool = False
    gross_channel_activity_s: float = 0.0
    channel_a_to_b: dict | None = None
    channel_b_to_a: dict | None = None
    proposal_probe_only: bool = False
    proposal_probe_external_work_J: float = 0.0
    proposal_probe_selected_direction: bool = False
    kinetic_event_volume_m3: float = 0.0
    kinetic_event_length_m: float = 0.0
    front_event_volume_m3: float = 0.0
    front_jump_length_m: float = 0.0
    physical_site_count: float = 0.0
    expected_events_a_to_b: float = 0.0
    expected_events_b_to_a: float = 0.0
    expected_signed_event_count: float = 0.0
    expected_signed_swept_volume_m3: float = 0.0
    expected_normal_velocity_m_s: float = 0.0
    site_count_per_interface_area_m2: float = 0.0
    event_count_per_interface_area: float = 0.0
    deterministic_rate_law: str = "legacy_independent_metropolis"


def initialize_coupled_front_runtime(state: SparseFrontState, phi,
                                     *, normal_axis: int, active_mask=None,
                                     periodic=True):
    value = np.asarray(phi, dtype=float)
    if value.shape != state.chi.shape or not np.all(np.isfinite(value)):
        raise ValueError("front phi is not finite and grid matched")
    axis = int(normal_axis)
    if axis not in (0, 1):
        raise ValueError("normal axis must be 0 or 1")
    ray_count = diagnostic_ray_crossing_count(
        value, normal_axis=axis, active_mask=active_mask, periodic=periodic)
    topology = initialize_front_topology(
        value, active_mask=active_mask, periodic=periodic,
        ray_crossing_count=ray_count)
    return CoupledFrontRuntime(
        state.parent_label, state.child_label, axis, 1,
        state.chi.copy(), state.chi.copy(), state.chi.copy(), value.copy(),
        topology, bool(periodic))


def runtime_metadata_json(runtime: CoupledFrontRuntime):
    return json.dumps({
        "schema": SCHEMA,
        "material_a_label": runtime.material_a_label,
        "material_b_label": runtime.material_b_label,
        "normal_axis": runtime.normal_axis,
        "normal_orientation": runtime.normal_orientation,
        "topology": snapshot_to_dict(runtime.topology),
        "periodic": runtime.periodic,
        "topology_event_count": runtime.topology_event_count,
        "ledger": asdict(runtime.ledger),
    }, sort_keys=True)


def runtime_arrays(runtime: CoupledFrontRuntime):
    return {
        "reference_b_fraction": runtime.reference_b_fraction,
        "maximum_b_fraction": runtime.maximum_b_fraction,
        "minimum_b_fraction": runtime.minimum_b_fraction,
        "previous_phi": runtime.previous_phi,
    }


def runtime_from_checkpoint(metadata_json, arrays, *, fallback_state=None,
                            fallback_phi=None, normal_axis=None,
                            active_mask=None, periodic=True):
    """Load v1 state or migrate a legacy sparse-front checkpoint explicitly."""
    if metadata_json is None:
        if fallback_state is None or fallback_phi is None or normal_axis is None:
            raise ValueError("legacy front migration requires sparse state, phi, and axis")
        return initialize_coupled_front_runtime(
            fallback_state, fallback_phi, normal_axis=normal_axis,
            active_mask=active_mask, periodic=periodic)
    metadata = json.loads(str(metadata_json))
    if metadata.get("schema") != SCHEMA:
        raise ValueError("unsupported coupled-front checkpoint schema")
    required = ("reference_b_fraction", "maximum_b_fraction",
                "minimum_b_fraction", "previous_phi")
    if not all(name in arrays for name in required):
        raise ValueError("partial coupled-front checkpoint")
    values = [np.asarray(arrays[name], dtype=float) for name in required]
    if len({value.shape for value in values}) != 1 or not all(
            np.all(np.isfinite(value)) for value in values):
        raise ValueError("invalid coupled-front checkpoint arrays")
    topology = (snapshot_from_dict(metadata["topology"])
                if "topology" in metadata
                else initialize_front_topology(
                    values[-1], active_mask=active_mask, periodic=periodic,
                    ray_crossing_count=diagnostic_ray_crossing_count(
                        values[-1], normal_axis=int(metadata["normal_axis"]),
                        active_mask=active_mask, periodic=periodic)))
    return CoupledFrontRuntime(
        int(metadata["material_a_label"]), int(metadata["material_b_label"]),
        int(metadata["normal_axis"]), int(metadata["normal_orientation"]),
        *[value.copy() for value in values],
        topology, bool(metadata.get("periodic", periodic)),
        int(metadata.get("topology_event_count", 0)),
        CoupledFrontLedger(**metadata.get("ledger", {})))


def _blend(old, old_weight, incoming, increment):
    w = old_weight[:, :, None] if old.ndim == 3 else old_weight
    q = increment[:, :, None] if old.ndim == 3 else increment
    return np.divide(w*old + q*incoming, w+q, out=np.asarray(old).copy(),
                     where=(w+q) > 0.0)


def _blend_state(old, old_weight, incoming, increment):
    return DefectState(*(
        _blend(a, old_weight, b, increment)
        for a, b in zip((old.rp, old.rm, old.forest, old.wall),
                        (incoming.rp, incoming.rm, incoming.forest, incoming.wall))))


def _non_b_state(state):
    weight = 1.0-state.chi
    virgin = 1.0-state.processed_max
    wake = state.processed_max-state.chi
    return DefectState(*(
        np.divide(
            (virgin[:, :, None] if a.ndim == 3 else virgin)*a
            +(wake[:, :, None] if a.ndim == 3 else wake)*b,
            weight[:, :, None] if a.ndim == 3 else weight,
            out=np.asarray(a).copy(),
            where=(weight[:, :, None] if a.ndim == 3 else weight) > 0.0)
        for a, b in zip((state.parent.rp, state.parent.rm,
                         state.parent.forest, state.parent.wall),
                        (state.recovered_wake.rp, state.recovered_wake.rm,
                         state.recovered_wake.forest, state.recovered_wake.wall))))


def _transaction(state, signed_sweep, *, cell_volume_m3, line_energy_J_m,
                 transmission_fraction, boundary_storage_fraction,
                 neutral_sink_fraction, signed_sink_fraction,
                 boundary_capacity_density_m2):
    """Commit both signs from one accepted field and return exact audit data."""
    signed = np.asarray(signed_sweep, dtype=float)
    if signed.shape != state.chi.shape or not np.all(np.isfinite(signed)):
        raise ValueError("signed sweep is invalid")
    positive = np.minimum(np.maximum(signed, 0.0), 1.0-state.chi)
    negative = np.minimum(np.maximum(-signed, 0.0), state.chi)
    state0 = state
    boundary0 = state.boundary_line_density_m2
    boundary_signed0 = state.boundary_signed_density_m2
    annihilated_density = np.zeros_like(state.chi)
    sink_density = np.zeros_like(state.chi)
    transmitted_density = np.zeros_like(state.chi)
    processed_a_density = np.zeros_like(state.chi)
    processed_b_density = np.zeros_like(state.chi)
    max_signed_closure = 0.0

    if np.any(positive):
        # Draw revisited material from the explicit A-wake first and virgin
        # material from A second.  A phase-average donor would be conservative
        # only on first passage and fails an advance/retreat/re-advance cycle.
        revisit = np.minimum(positive, state.processed_max-state.chi)
        virgin = positive-revisit
        donor = DefectState(*(
            np.divide(
                (revisit[:, :, None] if a.ndim == 3 else revisit)*a
                +(virgin[:, :, None] if a.ndim == 3 else virgin)*b,
                positive[:, :, None] if a.ndim == 3 else positive,
                out=np.asarray(b).copy(),
                where=(positive[:, :, None] if a.ndim == 3 else positive) > 0.0)
            for a, b in zip((state.recovered_wake.rp, state.recovered_wake.rm,
                             state.recovered_wake.forest, state.recovered_wake.wall),
                            (state.parent.rp, state.parent.rm,
                             state.parent.forest, state.parent.wall))))
        transfer = conservative_front_transfer(
            donor, transmission_fraction=transmission_fraction,
            boundary_storage_fraction=boundary_storage_fraction,
            neutral_sink_fraction=neutral_sink_fraction,
            signed_sink_fraction=signed_sink_fraction)
        if boundary_capacity_density_m2 is not None:
            capacity = np.broadcast_to(
                np.asarray(boundary_capacity_density_m2, dtype=float), state.chi.shape)
            available = np.maximum(capacity-state.boundary_line_density_m2, 0.0)
            demand = positive*transfer.boundary_excess_line_density_m2
            positive *= np.clip(np.divide(
                available, demand, out=np.ones_like(demand), where=demand > 0.0),
                0.0, 1.0)
            revisit = np.minimum(positive, state.processed_max-state.chi)
            virgin = positive-revisit
        child = _blend_state(state.child, state.chi, transfer.child, positive)
        processed_a_density += positive*total_line_density(donor)
        transmitted_density += positive*total_line_density(transfer.child)
        annihilated_density += positive*transfer.annihilated_line_density_m2
        sink_density += positive*transfer.sink_line_density_m2
        max_signed_closure = max(max_signed_closure, float(np.max(np.abs(
            positive[:, :, None]*transfer.signed_closure_density_m2))))
        state = replace(
            state, child=child, chi=state.chi+positive,
            processed_max=state.processed_max+virgin,
            cleanup_max=np.maximum(state.cleanup_max, state.processed_max+virgin),
            boundary_line_density_m2=(state.boundary_line_density_m2
                +positive*transfer.boundary_excess_line_density_m2),
            boundary_signed_density_m2=(state.boundary_signed_density_m2
                +positive[:, :, None]*transfer.boundary_excess_signed_density_m2))

    if np.any(negative):
        # Recompute the available child after any simultaneous positive part.
        negative = np.minimum(negative, state.chi)
        donor = state.child
        transfer = conservative_front_transfer(
            donor, transmission_fraction=transmission_fraction,
            boundary_storage_fraction=boundary_storage_fraction,
            neutral_sink_fraction=neutral_sink_fraction,
            signed_sink_fraction=signed_sink_fraction)
        if boundary_capacity_density_m2 is not None:
            capacity = np.broadcast_to(
                np.asarray(boundary_capacity_density_m2, dtype=float), state.chi.shape)
            available = np.maximum(capacity-state.boundary_line_density_m2, 0.0)
            demand = negative*transfer.boundary_excess_line_density_m2
            negative *= np.clip(np.divide(
                available, demand, out=np.ones_like(demand), where=demand > 0.0),
                0.0, 1.0)
        wake_weight = state.processed_max-state.chi
        wake = _blend_state(state.recovered_wake, wake_weight,
                            transfer.child, negative)
        processed_b_density += negative*total_line_density(donor)
        transmitted_density += negative*total_line_density(transfer.child)
        annihilated_density += negative*transfer.annihilated_line_density_m2
        sink_density += negative*transfer.sink_line_density_m2
        max_signed_closure = max(max_signed_closure, float(np.max(np.abs(
            negative[:, :, None]*transfer.signed_closure_density_m2))))
        state = replace(
            state, recovered_wake=wake, chi=state.chi-negative,
            boundary_line_density_m2=(state.boundary_line_density_m2
                +negative*transfer.boundary_excess_line_density_m2),
            boundary_signed_density_m2=(state.boundary_signed_density_m2
                +negative[:, :, None]*transfer.boundary_excess_signed_density_m2))

    before = total_line_density(reconstruct_mixture(state0))
    after = total_line_density(reconstruct_mixture(state))
    boundary_added = state.boundary_line_density_m2-boundary0
    closure_density = before-after-boundary_added-annihilated_density-sink_density
    closure_m = float(np.sum(closure_density, dtype=np.longdouble)*cell_volume_m3)
    scale = max(float(np.sum(np.abs(before), dtype=np.longdouble)*cell_volume_m3), 1e-300)
    if abs(closure_m) > 65536.0*math.ulp(scale):
        raise RuntimeError(f"coupled front line balance failed: {closure_m:.17g} m")
    heat = float(np.sum(annihilated_density, dtype=np.longdouble)
                 *cell_volume_m3*line_energy_J_m)
    audit = dict(
        positive=positive, negative=negative,
        volume_ab=float(np.sum(positive, dtype=np.longdouble)*cell_volume_m3),
        volume_ba=float(np.sum(negative, dtype=np.longdouble)*cell_volume_m3),
        processed_a_m=float(np.sum(processed_a_density, dtype=np.longdouble)*cell_volume_m3),
        processed_b_m=float(np.sum(processed_b_density, dtype=np.longdouble)*cell_volume_m3),
        transmitted_m=float(np.sum(transmitted_density, dtype=np.longdouble)*cell_volume_m3),
        boundary_m=float(np.sum(boundary_added, dtype=np.longdouble)*cell_volume_m3),
        annihilated_m=float(np.sum(annihilated_density, dtype=np.longdouble)*cell_volume_m3),
        sink_m=float(np.sum(sink_density, dtype=np.longdouble)*cell_volume_m3),
        line_closure_m=closure_m, signed_closure_m2=max_signed_closure,
        heat_J=heat)
    old = state.ledger
    state = replace(state, ledger=replace(
        old,
        parent_line_processed_m=old.parent_line_processed_m
            +audit["processed_a_m"]+audit["processed_b_m"],
        child_line_transmitted_m=old.child_line_transmitted_m+audit["transmitted_m"],
        boundary_line_stored_m=old.boundary_line_stored_m+audit["boundary_m"],
        neutral_pair_annihilated_m=old.neutral_pair_annihilated_m+audit["annihilated_m"],
        sink_line_m=old.sink_line_m+audit["sink_m"],
        line_closure_m=old.line_closure_m+closure_m,
        signed_burgers_change_m2=max(old.signed_burgers_change_m2, max_signed_closure),
        line_energy_released_J=old.line_energy_released_J+heat,
        heat_released_J=old.heat_released_J+heat,
        swept_volume_m3=old.swept_volume_m3+audit["volume_ab"]+audit["volume_ba"],
        requested_swept_volume_m3=old.requested_swept_volume_m3
            +float(np.sum(np.abs(signed), dtype=np.longdouble)*cell_volume_m3),
        capacity_limited_volume_m3=old.capacity_limited_volume_m3
            +float(np.sum(np.abs(signed)-positive-negative,
                          dtype=np.longdouble)*cell_volume_m3)))
    return state, audit


def accept_coupled_front_candidate(
        state: SparseFrontState, runtime: CoupledFrontRuntime,
        eta_before, eta_trial, *, spacing_m, represented_thickness_m, dt_s,
        temperature_K, line_energy_J_m, process: ActivatedProcess,
        h0_J, critical_pressure_Pa, exp_a, exp_n, exp_floor,
        driving_pressure_a_to_b_Pa=0.0, applied_pressure_a_to_b_Pa=0.0,
        mobility_enabled=True,
        active_mask=None, periodic=True, transmission_fraction=0.0,
        boundary_storage_fraction=0.0, neutral_sink_fraction=0.0,
        signed_sink_fraction=0.0, boundary_capacity_density_m2=None,
        support_component_reconnection=False,
        topology_backtracking_enabled=False,
        minimum_topology_backtrack_fraction=2.0**-12,
        topology_backtracking_bisections=12,
        kinetic_free_energy_a_to_b_J=None,
        kinetic_free_energy_b_to_a_J=None,
        kinetic_event_volume_m3=None,
        kinetic_event_length_m=None, proposal_probe_only=False,
        actual_reverse_edge=None,
        deterministic_rate_law="legacy_independent_metropolis"):
    """Atomically accept a trial two-phase update and its material transaction."""
    before = np.asarray(eta_before, dtype=float)
    trial = np.asarray(eta_trial, dtype=float)
    if before.shape != trial.shape or before.ndim != 3:
        raise ValueError("phase trial must be matching 3-D arrays")
    a, b = runtime.material_a_label, runtime.material_b_label
    phi0 = before[:, :, b]-before[:, :, a]
    phi1 = trial[:, :, b]-trial[:, :, a]
    # Boundary-condition changes are explicit configuration migrations.  They
    # alter contour connectivity at a periodic seam, so rebuild from the
    # unchanged physical field before attempting any motion.
    if runtime.periodic != bool(periodic):
        rebased = initialize_front_topology(
            phi0, active_mask=active_mask, periodic=periodic,
            ray_crossing_count=diagnostic_ray_crossing_count(
                phi0, normal_axis=runtime.normal_axis,
                active_mask=active_mask, periodic=periodic))
        runtime = replace(runtime, topology=rebased, periodic=bool(periodic))
    ray_before = diagnostic_ray_crossing_count(
        phi0, normal_axis=runtime.normal_axis, active_mask=active_mask,
        periodic=periodic)
    ray_after = diagnostic_ray_crossing_count(
        phi1, normal_axis=runtime.normal_axis, active_mask=active_mask,
        periodic=periodic)
    topology = match_front_topology(
        runtime.topology, phi0, phi1, active_mask=active_mask,
        periodic=periodic, ray_crossing_count=ray_after,
        support_component_reconnection=support_component_reconnection)
    topology_backtrack_fraction = 1.0
    if (topology.event == "PAIR_IDENTITY_LOST"
            and topology_backtracking_enabled):
        # A diffuse phase update can be much larger than the distance for
        # which component identity is well posed.  Find the largest admissible
        # prefix of that same update.  This changes neither direction nor the
        # cut-cell transaction: the accepted prefix is subsequently processed
        # by the ordinary rate and capacity limits.
        unsafe = 1.0
        safe = None
        candidate = 0.5
        minimum = float(minimum_topology_backtrack_fraction)
        while candidate >= minimum:
            candidate_eta = before+candidate*(trial-before)
            candidate_phi = candidate_eta[:, :, b]-candidate_eta[:, :, a]
            candidate_match = match_front_topology(
                runtime.topology, phi0, candidate_phi,
                active_mask=active_mask, periodic=periodic,
                ray_crossing_count=diagnostic_ray_crossing_count(
                    candidate_phi, normal_axis=runtime.normal_axis,
                    active_mask=active_mask, periodic=periodic),
                support_component_reconnection=support_component_reconnection)
            if candidate_match.event is None:
                safe = (candidate, candidate_eta, candidate_phi, candidate_match)
                break
            unsafe = candidate
            candidate *= 0.5
        if safe is not None:
            lower, best_eta, best_phi, best_match = safe
            for _ in range(max(0, int(topology_backtracking_bisections))):
                middle = 0.5*(lower+unsafe)
                candidate_eta = before+middle*(trial-before)
                candidate_phi = candidate_eta[:, :, b]-candidate_eta[:, :, a]
                candidate_match = match_front_topology(
                    runtime.topology, phi0, candidate_phi,
                    active_mask=active_mask, periodic=periodic,
                    ray_crossing_count=diagnostic_ray_crossing_count(
                        candidate_phi, normal_axis=runtime.normal_axis,
                        active_mask=active_mask, periodic=periodic),
                    support_component_reconnection=support_component_reconnection)
                if candidate_match.event is None:
                    lower, best_eta, best_phi, best_match = (
                        middle, candidate_eta, candidate_phi, candidate_match)
                else:
                    unsafe = middle
            topology_backtrack_fraction = lower
            trial, phi1, topology = best_eta, best_phi, best_match
            ray_after = topology.snapshot.ray_crossing_count
    event_volume = float(spacing_m)**2*float(represented_thickness_m)
    front_event_volume = (event_volume if kinetic_event_volume_m3 is None
                          else float(kinetic_event_volume_m3))
    front_jump_length = (float(spacing_m) if kinetic_event_length_m is None
                         else float(kinetic_event_length_m))
    if (not math.isfinite(front_event_volume) or front_event_volume <= 0.0
            or not math.isfinite(front_jump_length)
            or front_jump_length <= 0.0):
        raise ValueError("kinetic event volume and length must be positive")
    pressure = float(driving_pressure_a_to_b_Pa)
    state_a = _non_b_state(state)
    state_b = state.child
    # Algebraic exchange symmetry, not a finite pressure cutoff: an exactly
    # equal supported pair with no external pressure defines identical trials.
    if (float(applied_pressure_a_to_b_Pa) == 0.0
            and supported_front_state_is_exactly_equal(state)):
        pressure = 0.0
        state_b = state_a
    event = propose_bidirectional_front_event(
        state_a, state_b, event_volume_m3=front_event_volume,
        event_length_m=front_jump_length, line_energy_J_m=line_energy_J_m,
        temperature_K=float(np.mean(np.asarray(temperature_K, dtype=float))),
        process=process, h0_J=h0_J, critical_pressure_Pa=critical_pressure_Pa,
        exp_a=exp_a, exp_n=exp_n, exp_floor=exp_floor,
        energy_a_to_b=FrontEnergyTerms(phase_J=-pressure*event_volume),
        energy_b_to_a=FrontEnergyTerms(phase_J=pressure*event_volume),
        mobility_enabled=(mobility_enabled and not proposal_probe_only),
        transmission_fraction=transmission_fraction,
        boundary_storage_fraction=boundary_storage_fraction,
        neutral_sink_fraction=neutral_sink_fraction,
        signed_sink_fraction=signed_sink_fraction,
        kinetic_free_energy_a_to_b_J=kinetic_free_energy_a_to_b_J,
        kinetic_free_energy_b_to_a_J=kinetic_free_energy_b_to_a_J,
        actual_reverse_edge=actual_reverse_edge,
        deterministic_rate_law=deterministic_rate_law)
    proposed = (topology.signed_receiver_area_cells2*float(spacing_m)**2
                *float(represented_thickness_m))
    velocity = event.net_velocity_a_to_b_m_s
    interface_length = (sum(item.interface_length_cells
                            for item in runtime.topology.components)
                        *float(spacing_m))
    _event_measure = physical_front_event_measure(
        interface_length_m=interface_length,
        represented_thickness_m=float(represented_thickness_m),
        burgers_m=front_jump_length,
        rate_a_to_b_per_site_s=event.rate_a_to_b_s,
        rate_b_to_a_per_site_s=event.rate_b_to_a_s,
        dt_s=float(dt_s), event_volume_m3=front_event_volume,
        event_length_m=front_jump_length)
    _site_diagnostics = dict(
        kinetic_free_energy_a_to_b_J=event.kinetic_free_energy_a_to_b_J,
        kinetic_free_energy_b_to_a_J=event.kinetic_free_energy_b_to_a_J,
        microscopic_reverse_pair=event.microscopic_reverse_pair,
        actual_reverse_edge=event.actual_reverse_edge,
        detailed_balance_applicable=event.detailed_balance_applicable,
        gross_channel_activity_s=(event.rate_a_to_b_s
                                  +event.rate_b_to_a_s),
        channel_a_to_b=asdict(event.a_to_b_channel),
        channel_b_to_a=asdict(event.b_to_a_channel),
        proposal_probe_only=bool(proposal_probe_only),
        proposal_probe_external_work_J=0.0,
        proposal_probe_selected_direction=False,
        deterministic_rate_law=event.deterministic_rate_law,
        kinetic_event_volume_m3=_event_measure.event_volume_m3,
        kinetic_event_length_m=_event_measure.event_length_m,
        front_event_volume_m3=_event_measure.event_volume_m3,
        front_jump_length_m=_event_measure.event_length_m,
        physical_site_count=_event_measure.physical_site_count,
        expected_events_a_to_b=_event_measure.expected_events_a_to_b,
        expected_events_b_to_a=_event_measure.expected_events_b_to_a,
        expected_signed_event_count=(
            _event_measure.expected_signed_event_count),
        expected_signed_swept_volume_m3=(
            _event_measure.expected_signed_swept_volume_m3),
        expected_normal_velocity_m_s=(
            _event_measure.expected_normal_velocity_m_s),
        site_count_per_interface_area_m2=(
            _event_measure.site_count_per_interface_area_m2),
        event_count_per_interface_area=(
            _event_measure.event_count_per_interface_area))
    allowed = velocity*float(dt_s)*interface_length*float(represented_thickness_m)
    tolerance = 8192.0*np.finfo(float).eps*max(event_volume, abs(proposed), 1e-300)
    if topology.event is not None:
        ledger = replace(runtime.ledger, attempts=runtime.ledger.attempts+1)
        stopped = replace(
            runtime, topology_event_count=runtime.topology_event_count+1,
            ledger=ledger)
        return state, stopped, before.copy(), CoupledFrontDecision(
            False, topology.event, proposed, 0.0,
            event.rate_a_to_b_s, event.rate_b_to_a_s, velocity,
            event.detailed_balance_log_residual, 0.0, 0.0, 0.0,
            topology_event=topology.event_record,
            component_count=len(runtime.topology.components),
            ray_crossing_count_before=ray_before,
            ray_crossing_count_after=ray_after,
            maximum_component_distance_cells=(
                topology.maximum_component_distance_cells),
            filtered_subcell_components_after=(
                topology.snapshot.filtered_subcell_component_count),
            **_site_diagnostics)
    if abs(proposed) <= tolerance:
        ledger = replace(runtime.ledger, attempts=runtime.ledger.attempts+1,
                         stationary_trials=runtime.ledger.stationary_trials+1)
        return state, replace(runtime, previous_phi=phi1.copy(),
                              topology=topology.snapshot,
                              ledger=ledger), trial.copy(), CoupledFrontDecision(
            False, "STATIONARY_GEOMETRY", proposed, 0.0,
            event.rate_a_to_b_s, event.rate_b_to_a_s, velocity,
            event.detailed_balance_log_residual, 0.0, 0.0, 0.0,
            component_count=len(topology.snapshot.components),
            ray_crossing_count_before=ray_before,
            ray_crossing_count_after=ray_after,
            maximum_component_distance_cells=(
                topology.maximum_component_distance_cells),
            filtered_subcell_components_after=(
                topology.snapshot.filtered_subcell_component_count),
            **_site_diagnostics)
    if (not proposal_probe_only
            and (abs(velocity) <= 0.0 or proposed*velocity <= 0.0)):
        ledger = replace(runtime.ledger, attempts=runtime.ledger.attempts+1,
                         rejected_direction=runtime.ledger.rejected_direction+1)
        return state, replace(runtime, ledger=ledger), before.copy(), CoupledFrontDecision(
            False, "REJECTED_BY_BIDIRECTIONAL_RATE", proposed, 0.0,
            event.rate_a_to_b_s, event.rate_b_to_a_s, velocity,
            event.detailed_balance_log_residual, 0.0, 0.0, 0.0,
            component_count=len(runtime.topology.components),
            ray_crossing_count_before=ray_before,
            ray_crossing_count_after=ray_after,
            maximum_component_distance_cells=(
                topology.maximum_component_distance_cells),
            filtered_subcell_components_after=(
                topology.snapshot.filtered_subcell_component_count),
            **_site_diagnostics)
    # A proposal probe only materializes the already supplied geometric trial
    # so the complete common functional can price both outgoing endpoints. It
    # applies no pressure/work, evaluates no direction preference, and is
    # never a publishable physical acceptance. The later ordinary call uses
    # the independently computed channel kinetics.
    fraction = (1.0 if proposal_probe_only
                else min(1.0, abs(allowed)/abs(proposed)))
    accepted_eta = before+fraction*(trial-before)
    phi_accept = accepted_eta[:, :, b]-accepted_eta[:, :, a]
    accepted_topology = match_front_topology(
        runtime.topology, phi0, phi_accept, active_mask=active_mask,
        periodic=periodic, ray_crossing_count=diagnostic_ray_crossing_count(
            phi_accept, normal_axis=runtime.normal_axis,
            active_mask=active_mask, periodic=periodic),
        support_component_reconnection=support_component_reconnection)
    if accepted_topology.event is not None:
        ledger = replace(runtime.ledger, attempts=runtime.ledger.attempts+1)
        stopped = replace(
            runtime, topology_event_count=runtime.topology_event_count+1,
            ledger=ledger)
        return state, stopped, before.copy(), CoupledFrontDecision(
            False, accepted_topology.event, proposed, 0.0,
            event.rate_a_to_b_s, event.rate_b_to_a_s, velocity,
            event.detailed_balance_log_residual, 0.0, 0.0, 0.0,
            topology_event=accepted_topology.event_record,
            component_count=len(runtime.topology.components),
            ray_crossing_count_before=ray_before,
            ray_crossing_count_after=ray_after,
            maximum_component_distance_cells=(
                accepted_topology.maximum_component_distance_cells),
            filtered_subcell_components_after=(
                accepted_topology.snapshot.filtered_subcell_component_count),
            **_site_diagnostics)
    signed_sweep = (accepted_topology.receiver_fraction_after
                    -accepted_topology.receiver_fraction_before)
    if accepted_topology.signed_receiver_area_cells2 > 0.0:
        signed_sweep = np.maximum(signed_sweep, 0.0)
    else:
        signed_sweep = -np.maximum(-signed_sweep, 0.0)
    signed_sweep = np.clip(signed_sweep, -state.chi, 1.0-state.chi)
    new_state, audit = _transaction(
        state, signed_sweep, cell_volume_m3=event_volume,
        line_energy_J_m=line_energy_J_m,
        transmission_fraction=transmission_fraction,
        boundary_storage_fraction=boundary_storage_fraction,
        neutral_sink_fraction=neutral_sink_fraction,
        signed_sink_fraction=signed_sink_fraction,
        boundary_capacity_density_m2=boundary_capacity_density_m2)
    actual_signed = audit["volume_ab"]-audit["volume_ba"]
    # Capacity limiting is an atomic partial acceptance; scale phase motion by
    # the accepted geometric fraction so state and contour cannot diverge.
    requested_abs = max(
        abs(accepted_topology.signed_receiver_area_cells2*event_volume), 1e-300)
    capacity_scale = min(1.0, abs(actual_signed)/requested_abs)
    if capacity_scale < 1.0:
        accepted_eta = before+capacity_scale*(accepted_eta-before)
        phi_accept = accepted_eta[:, :, b]-accepted_eta[:, :, a]
        accepted_topology = match_front_topology(
            runtime.topology, phi0, phi_accept, active_mask=active_mask,
            periodic=periodic,
            ray_crossing_count=diagnostic_ray_crossing_count(
                phi_accept, normal_axis=runtime.normal_axis,
                active_mask=active_mask, periodic=periodic),
            support_component_reconnection=support_component_reconnection)
    old = runtime.ledger
    revisit = np.minimum(np.maximum(signed_sweep, 0.0),
                         np.maximum(runtime.maximum_b_fraction-state.chi, 0.0))
    revisit += np.minimum(np.maximum(-signed_sweep, 0.0),
                          np.maximum(state.chi-runtime.minimum_b_fraction, 0.0))
    ledger = replace(
        old, attempts=old.attempts+1, accepted=old.accepted+1,
        a_to_b_swept_volume_m3=old.a_to_b_swept_volume_m3+audit["volume_ab"],
        b_to_a_swept_volume_m3=old.b_to_a_swept_volume_m3+audit["volume_ba"],
        revisit_volume_m3=old.revisit_volume_m3
            +float(np.sum(revisit, dtype=np.longdouble)*event_volume),
        a_to_b_line_processed_m=old.a_to_b_line_processed_m+audit["processed_a_m"],
        b_to_a_line_processed_m=old.b_to_a_line_processed_m+audit["processed_b_m"],
        transmitted_line_m=old.transmitted_line_m+audit["transmitted_m"],
        boundary_line_m=old.boundary_line_m+audit["boundary_m"],
        annihilated_line_m=old.annihilated_line_m+audit["annihilated_m"],
        sink_line_m=old.sink_line_m+audit["sink_m"], heat_J=old.heat_J+audit["heat_J"],
        maximum_abs_line_closure_m=max(old.maximum_abs_line_closure_m,
                                       abs(audit["line_closure_m"])),
        maximum_abs_signed_closure_m2=max(old.maximum_abs_signed_closure_m2,
                                          audit["signed_closure_m2"]))
    runtime = replace(
        runtime, maximum_b_fraction=np.maximum(runtime.maximum_b_fraction, new_state.chi),
        minimum_b_fraction=np.minimum(runtime.minimum_b_fraction, new_state.chi),
        previous_phi=(accepted_eta[:, :, b]-accepted_eta[:, :, a]).copy(),
        topology=accepted_topology.snapshot,
        topology_event_count=(runtime.topology_event_count
                              +int(accepted_topology.event_record is not None)),
        ledger=ledger)
    phase_change = accepted_eta-before
    component_motion = tuple({
        "component_id": int(component.component_id),
        "signed_area_m2": float(
            component.signed_receiver_swept_area_cells2*float(spacing_m)**2),
        "interface_length_m": float(
            component.interface_length_cells*float(spacing_m)),
        "normal_displacement_m": float(
            component.signed_receiver_swept_area_cells2*float(spacing_m)
            /max(component.interface_length_cells, 1e-300)),
    } for component in accepted_topology.snapshot.components)
    return new_state, runtime, accepted_eta, CoupledFrontDecision(
        True, ("GEOMETRY_PROBE_ONLY_NOT_PHYSICAL_ACCEPTANCE"
               if proposal_probe_only else "ACCEPTED_ATOMIC_COUPLED_FRONT"),
        proposed, actual_signed,
        (0.0 if proposal_probe_only else event.rate_a_to_b_s),
        (0.0 if proposal_probe_only else event.rate_b_to_a_s),
        (0.0 if proposal_probe_only else velocity),
        event.detailed_balance_log_residual, abs(audit["line_closure_m"]),
        audit["signed_closure_m2"], audit["heat_J"],
        component_count=len(accepted_topology.snapshot.components),
        ray_crossing_count_before=ray_before,
        ray_crossing_count_after=accepted_topology.snapshot.ray_crossing_count,
        maximum_component_distance_cells=(
            accepted_topology.maximum_component_distance_cells),
        filtered_subcell_components_after=(
            accepted_topology.snapshot.filtered_subcell_component_count),
        topology_event=accepted_topology.event_record,
        topology_backtrack_fraction=topology_backtrack_fraction,
        positive_swept_volume_m3=audit["volume_ab"],
        negative_swept_volume_m3=audit["volume_ba"],
        absolute_swept_volume_m3=audit["volume_ab"]+audit["volume_ba"],
        interface_area_m2=(sum(
            item.interface_length_cells
            for item in accepted_topology.snapshot.components)
            *float(spacing_m)*float(represented_thickness_m)),
        represented_thickness_m=float(represented_thickness_m),
        cell_volume_m3=event_volume,
        maximum_abs_phase_change=float(np.max(np.abs(phase_change))),
        rms_phase_change=float(np.sqrt(np.mean(phase_change*phase_change))),
        component_motion=component_motion,
        **_site_diagnostics)
