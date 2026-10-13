"""One-dimensional selective subgrain growth in an orientation gradient.

The state is a row of pre-existing subgrains.  Interior boundaries move under
the difference in current bulk stored energy using the same EXP-floor
Arrhenius front kinetics as the production SIBM operator.  When a cell reaches
zero width, its two boundaries are replaced by the compatible boundary between
its surviving neighbors.  No orientation is created or rotated: a monotonic
orientation gradient naturally concentrates several low-angle steps into the
new boundary.  The topology event is accepted only if the complete bulk plus
Read--Shockley energy does not increase.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .arrhenius_kinetics import (
    activated_rate_array_s,
    exp_floor_enthalpy_j,
)
from .structural_boundary import (
    StructuralBoundaryParameters,
    read_shockley_energy_J_m2,
)


@dataclass(frozen=True)
class SubgrainNetworkState:
    edges_m: np.ndarray
    orientation_rad: np.ndarray
    stored_energy_J_m3: np.ndarray
    physical_time_s: float = 0.0
    cumulative_dissipation_J_m2: float = 0.0
    eliminated_cells: int = 0

    def validate(self):
        edges = np.asarray(self.edges_m, dtype=float)
        orientation = np.asarray(self.orientation_rad, dtype=float)
        stored = np.asarray(self.stored_energy_J_m3, dtype=float)
        if (edges.ndim != 1 or orientation.ndim != 1 or stored.ndim != 1
                or edges.size != orientation.size+1
                or stored.size != orientation.size or orientation.size < 2):
            raise ValueError("network requires n cells, n energies, and n+1 edges")
        if (np.any(~np.isfinite(edges)) or np.any(~np.isfinite(orientation))
                or np.any(~np.isfinite(stored)) or np.any(stored < 0.0)):
            raise ValueError("network state must be finite with nonnegative stored energy")
        if np.any(np.diff(edges) < -64*np.finfo(float).eps*max(abs(edges[-1]), 1.0)):
            raise ValueError("network edges must be ordered")
        if not math.isfinite(self.physical_time_s) or self.physical_time_s < 0.0:
            raise ValueError("physical time must be finite and nonnegative")
        if (not math.isfinite(self.cumulative_dissipation_J_m2)
                or self.cumulative_dissipation_J_m2 < 0.0
                or self.eliminated_cells < 0):
            raise ValueError("network ledger is invalid")


@dataclass(frozen=True)
class SubgrainNetworkParameters:
    structural: StructuralBoundaryParameters
    process: object
    mobility_prefactor_m4_J_s: float
    activation_enthalpy_0_J: float
    critical_pressure_Pa: float
    exp_a: float
    exp_n: float
    exp_floor: float
    maximum_events_per_interval: int = 64

    def __post_init__(self):
        positive = (
            self.mobility_prefactor_m4_J_s, self.activation_enthalpy_0_J,
            self.critical_pressure_Pa, self.exp_n,
        )
        if any(not math.isfinite(float(value)) or value <= 0.0
               for value in positive):
            raise ValueError("network kinetic scales must be positive and finite")
        if (not math.isfinite(self.exp_a) or self.exp_a < 0.0
                or not math.isfinite(self.exp_floor)
                or not 0.0 <= self.exp_floor <= 1.0
                or self.exp_n < 1.0 or self.maximum_events_per_interval < 1):
            raise ValueError("invalid EXP-floor or event control")


@dataclass(frozen=True)
class SubgrainNetworkDecision:
    state: SubgrainNetworkState
    accepted: bool
    classification: str
    energy_before_J_m2: float
    energy_after_J_m2: float
    delta_helmholtz_J_m2: float
    maximum_velocity_m_s: float
    topology_events: tuple[dict, ...]
    orientation_sum_residual_rad: float


def network_energy_J_m2(state, parameters):
    state.validate()
    width = np.diff(state.edges_m)
    bulk = float(np.sum(width*state.stored_energy_J_m3))
    theta = np.diff(state.orientation_rad)
    boundary = float(np.sum(read_shockley_energy_J_m2(
        theta, parameters.structural)))
    return bulk+boundary


def boundary_velocities_m_s(state, temperature_K, parameters):
    """Positive velocity moves an interior edge to the right."""
    state.validate()
    temperature = float(temperature_K)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    pressure = state.stored_energy_J_m3[1:]-state.stored_energy_J_m3[:-1]
    enthalpy = exp_floor_enthalpy_j(
        np.abs(pressure), parameters.activation_enthalpy_0_J,
        parameters.critical_pressure_Pa, parameters.exp_a,
        parameters.exp_n, parameters.exp_floor)
    rate = activated_rate_array_s(parameters.process, enthalpy, temperature)
    kinetic = rate/parameters.process.attempt_frequency_s
    velocity = parameters.mobility_prefactor_m4_J_s*kinetic*pressure
    return velocity


def _eliminate_zero_cell(state, index, parameters, tolerance_J_m2):
    """Remove zero-width interior cell ``index`` and close its two boundaries."""
    n = state.orientation_rad.size
    if not 0 < index < n-1:
        raise ValueError("only an interior subgrain can be eliminated")
    before = network_energy_J_m2(state, parameters)
    left_theta = state.orientation_rad[index]-state.orientation_rad[index-1]
    right_theta = state.orientation_rad[index+1]-state.orientation_rad[index]
    combined_theta = state.orientation_rad[index+1]-state.orientation_rad[index-1]
    content_residual = combined_theta-left_theta-right_theta
    edges = np.delete(state.edges_m, index+1)
    orientation = np.delete(state.orientation_rad, index)
    stored = np.delete(state.stored_energy_J_m3, index)
    candidate = SubgrainNetworkState(
        edges, orientation, stored, state.physical_time_s,
        state.cumulative_dissipation_J_m2, state.eliminated_cells+1)
    after = network_energy_J_m2(candidate, parameters)
    if after > before+float(tolerance_J_m2):
        raise RuntimeError("subgrain elimination would increase complete energy")
    event = {
        "eliminated_index": int(index),
        "eliminated_orientation_rad": float(state.orientation_rad[index]),
        "left_misorientation_rad": float(left_theta),
        "right_misorientation_rad": float(right_theta),
        "combined_misorientation_rad": float(combined_theta),
        "orientation_sum_residual_rad": float(content_residual),
        "topology_energy_change_J_m2": float(after-before),
    }
    return candidate, event, after-before


def advance_subgrain_network(
        state, *, dt_s, temperature_K, parameters,
        energy_tolerance_J_m2=1e-12):
    """Advance motion and exact cell-disappearance events over one interval."""
    state.validate()
    remaining = float(dt_s)
    if not math.isfinite(remaining) or remaining <= 0.0:
        raise ValueError("positive finite interval required")
    before = network_energy_J_m2(state, parameters)
    current = state
    events = []
    maximum_velocity = 0.0
    total_topology_drop = 0.0
    while remaining > 64*np.finfo(float).eps*max(dt_s, 1.0):
        velocity = boundary_velocities_m_s(current, temperature_K, parameters)
        maximum_velocity = max(maximum_velocity,
                               float(np.max(np.abs(velocity), initial=0.0)))
        edge_rate = np.zeros_like(current.edges_m)
        edge_rate[1:-1] = velocity
        width = np.diff(current.edges_m)
        width_rate = np.diff(edge_rate)
        closing = np.where(width_rate < 0.0)[0]
        event_time = math.inf
        event_index = None
        for index in closing:
            candidate_time = -width[index]/width_rate[index]
            if candidate_time < event_time:
                event_time = float(candidate_time); event_index = int(index)
        step = min(remaining, event_time)
        edges = current.edges_m+step*edge_rate
        current = SubgrainNetworkState(
            edges, current.orientation_rad.copy(),
            current.stored_energy_J_m3.copy(),
            current.physical_time_s+step,
            current.cumulative_dissipation_J_m2,
            current.eliminated_cells)
        remaining -= step
        if event_index is not None and event_time <= step+64*np.finfo(float).eps*max(step, 1.0):
            if event_index == 0 or event_index == current.orientation_rad.size-1:
                # Domain-edge cells cannot disappear under fixed boundaries;
                # stop at the exact event rather than silently changing length.
                remaining = 0.0
                break
            current, event, drop = _eliminate_zero_cell(
                current, event_index, parameters, energy_tolerance_J_m2)
            events.append(event)
            total_topology_drop += drop
            if len(events) > parameters.maximum_events_per_interval:
                raise RuntimeError("network exceeded topology-event budget")
            continue
        break
    after = network_energy_J_m2(current, parameters)
    delta = after-before
    if delta > float(energy_tolerance_J_m2):
        raise RuntimeError("subgrain motion increased complete energy")
    current = SubgrainNetworkState(
        current.edges_m, current.orientation_rad,
        current.stored_energy_J_m3, state.physical_time_s+float(dt_s)-remaining,
        state.cumulative_dissipation_J_m2+max(-delta, 0.0),
        current.eliminated_cells)
    residual = max((abs(item["orientation_sum_residual_rad"])
                    for item in events), default=0.0)
    classification = ("TOPOLOGY_COARSENING" if events else
                      "SELECTIVE_MIGRATION" if maximum_velocity > 0.0 else
                      "STATIONARY")
    return SubgrainNetworkDecision(
        current, True, classification, before, after, delta,
        maximum_velocity, tuple(events), residual)
