"""Capacity-limited diagnostic reaction cone for V25 wall qualification.

The cone is deliberately read-only: it answers whether declared events can
span a missing Frank--Bilby circuit inventory.  It never returns a production
source or a target-following rate.  Local reactions that change authoritative
total Nye without swept plastic area or a declared source are inadmissible.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class ReactionEvent:
    name: str
    circuit_increment_per_extent_m: np.ndarray
    line_increment_per_extent: float
    burgers_increment_per_extent_m: np.ndarray
    nye_increment_per_extent_m: np.ndarray
    alignment_increment_per_extent: np.ndarray
    turning_node_increment_per_extent_m1: float
    junction_increment_per_extent: float
    free_energy_increment_per_extent_J_m: float
    capacity_extent_m1: float
    kinetic_exposure_extent_m1: float = 0.0
    swept_area_declared: bool = False
    explicit_source_or_sink: bool = False
    frank_rule_residual_m: float = 0.0
    line_node_residual: float = 0.0

    def validate(self, tolerance=1e-12):
        vector = np.asarray(self.circuit_increment_per_extent_m, dtype=float)
        burgers = np.asarray(self.burgers_increment_per_extent_m, dtype=float)
        nye = np.asarray(self.nye_increment_per_extent_m, dtype=float)
        alignment = np.asarray(self.alignment_increment_per_extent, dtype=float)
        if vector.shape != (3,) or burgers.shape != (3,) or nye.shape != (3, 3):
            raise ValueError("reaction event requires 3-vector circuit/Burgers and 3x3 Nye")
        if alignment.shape != (3,) or any(np.any(~np.isfinite(x)) for x in
                                          (vector, burgers, nye, alignment)):
            raise ValueError("reaction event increments must be finite")
        scalars = (self.line_increment_per_extent,
                   self.turning_node_increment_per_extent_m1,
                   self.junction_increment_per_extent,
                   self.free_energy_increment_per_extent_J_m,
                   self.capacity_extent_m1, self.kinetic_exposure_extent_m1,
                   self.frank_rule_residual_m, self.line_node_residual)
        if not np.all(np.isfinite(scalars)) or self.capacity_extent_m1 < 0.0 \
                or self.kinetic_exposure_extent_m1 < 0.0:
            raise ValueError("reaction event capacities and ledgers must be finite/nonnegative")
        local_nye_change = float(np.linalg.norm(nye))
        scale = max(float(np.linalg.norm(vector)), 1e-300)
        conservative = local_nye_change <= tolerance*scale
        ownership_valid = (conservative or self.swept_area_declared
                           or self.explicit_source_or_sink)
        return {
            "local_total_nye_conservative": bool(conservative),
            "authoritative_nye_ownership_valid": bool(ownership_valid),
            "frank_rule_valid": bool(abs(self.frank_rule_residual_m)
                                      <= tolerance*scale),
            "line_node_valid": bool(abs(self.line_node_residual)
                                     <= tolerance),
        }


def audit_reaction_cone(events, missing_circuit_inventory, *,
                        relative_tolerance=0.05, ownership_tolerance=1e-12):
    """Solve the bounded diagnostic cone and return a machine-readable audit."""
    target = np.asarray(missing_circuit_inventory, dtype=float)
    if target.shape != (3,) or np.any(~np.isfinite(target)):
        raise ValueError("missing Frank--Bilby inventory must be a finite 3-vector")
    validations = [event.validate(ownership_tolerance) for event in events]
    admissible_indices = [i for i, valid in enumerate(validations)
                          if valid["authoritative_nye_ownership_valid"]
                          and valid["frank_rule_valid"]
                          and valid["line_node_valid"]]
    target_norm = float(np.linalg.norm(target))
    if not admissible_indices:
        residual = target_norm
        solution = np.zeros(len(events))
    else:
        matrix = np.stack([
            np.asarray(events[i].circuit_increment_per_extent_m)
            for i in admissible_indices], axis=1)
        upper = np.asarray([events[i].capacity_extent_m1
                            for i in admissible_indices])
        # Optimize a capacity fraction rather than a dimensional extent.  BCC
        # Burgers columns are O(1e-10 m) while integrated extents are O(1e8
        # m^-1); direct dimensional optimization otherwise terminates on a
        # misleadingly small gradient before resolving the physical product.
        scaled_matrix = matrix*upper[None, :]
        answer = lsq_linear(scaled_matrix, target,
                            bounds=(np.zeros_like(upper), np.ones_like(upper)),
                            lsmr_tol="auto", max_iter=1000)
        dimensional = upper*answer.x
        solution = np.zeros(len(events)); solution[admissible_indices] = dimensional
        residual = float(np.linalg.norm(matrix@dimensional-target))
    relative = residual/max(target_norm, 1e-300)
    reachable = bool(relative <= relative_tolerance)
    exposure = float(sum(min(events[i].kinetic_exposure_extent_m1,
                             solution[i]) for i in range(len(events))))
    required = float(np.sum(solution))
    exposed_fraction = exposure/max(required, 1e-300)
    if not reachable:
        classification = "FB_TARGET_OUTSIDE_PHYSICAL_REACTION_CONE"
    elif exposed_fraction < 1.0-relative_tolerance:
        classification = "FB_TARGET_REACHABLE_BUT_KINETICALLY_UNEXPOSED"
    else:
        classification = "FB_TARGET_REACHABLE_WITH_SUFFICIENT_LOCAL_CAPACITY"
    return {
        "classification": classification,
        "target_norm": target_norm,
        "absolute_residual": residual,
        "relative_residual": relative,
        "relative_tolerance": float(relative_tolerance),
        "reachable": reachable,
        "required_total_extent_m1": required,
        "kinetically_exposed_fraction": exposed_fraction,
        "event_names": [event.name for event in events],
        "event_extents_m1": solution.tolist(),
        "event_validation": validations,
        "inadmissible_events": [events[i].name for i in range(len(events))
                                if i not in admissible_indices],
        "optimizer_is_diagnostic_only": True,
    }


def circuit_event_from_line_change(name, burgers_m, line_before, line_after,
                                   circuit_line, capacity_extent_m1, **kwargs):
    """Build a complete local line-reorientation event increment."""
    b = np.asarray(burgers_m, dtype=float)
    before = np.asarray(line_before, dtype=float)
    after = np.asarray(line_after, dtype=float)
    circuit = np.asarray(circuit_line, dtype=float)
    delta_line = after-before
    return ReactionEvent(
        name=name,
        circuit_increment_per_extent_m=b*float(delta_line@circuit),
        line_increment_per_extent=0.0,
        burgers_increment_per_extent_m=np.zeros(3),
        nye_increment_per_extent_m=np.outer(b, delta_line),
        alignment_increment_per_extent=delta_line,
        turning_node_increment_per_extent_m1=float(
            kwargs.pop("turning_node_increment_per_extent_m1", 0.0)),
        junction_increment_per_extent=float(kwargs.pop(
            "junction_increment_per_extent", 0.0)),
        free_energy_increment_per_extent_J_m=float(kwargs.pop(
            "free_energy_increment_per_extent_J_m", 0.0)),
        capacity_extent_m1=float(capacity_extent_m1), **kwargs)
