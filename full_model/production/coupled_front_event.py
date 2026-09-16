"""Bidirectional thermodynamic trial for an atomic phase/front event.

This module contains no parent/child lineage.  Two material states are denoted
``a`` and ``b`` and both directional transactions are assembled before a net
rate is evaluated.  Defect processing is consequently part of the migration
affinity, rather than an operation applied after phase motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .arrhenius_kinetics import (
    ActivatedProcess, KB_J_K, activated_rate_s, exp_floor_enthalpy_j)
from .moving_front import (
    DefectState, conservative_front_transfer, total_line_density)


@dataclass(frozen=True)
class FrontEnergyTerms:
    phase_J: float = 0.0
    interface_J: float = 0.0
    elastic_J: float = 0.0
    gnd_J: float = 0.0
    boundary_intrinsic_J: float = 0.0

    @property
    def nondefect_total_J(self):
        return (self.phase_J + self.interface_J + self.elastic_J
                + self.gnd_J + self.boundary_intrinsic_J)


@dataclass(frozen=True)
class DirectionalFrontTrial:
    donor_name: str
    receiver_name: str
    event_volume_m3: float
    donor_line_m: float
    transmitted_line_m: float
    boundary_line_m: float
    annihilated_line_m: float
    sink_line_m: float
    line_closure_m: float
    signed_burgers_closure_m2: float
    defect_free_energy_change_J: float
    full_free_energy_change_J: float
    heat_J: float
    sink_export_energy_J: float
    energy_closure_J: float


@dataclass(frozen=True)
class BidirectionalFrontEvent:
    a_to_b: DirectionalFrontTrial
    b_to_a: DirectionalFrontTrial
    rate_a_to_b_s: float
    rate_b_to_a_s: float
    net_velocity_a_to_b_m_s: float
    detailed_balance_log_residual: float


def _mean_density(field):
    return float(np.mean(np.asarray(field, dtype=float), dtype=np.longdouble))


def directional_front_trial(donor: DefectState, *, donor_name: str,
                            receiver_name: str, event_volume_m3: float,
                            line_energy_J_m: float,
                            energy_terms=FrontEnergyTerms(),
                            transmission_fraction=0.0,
                            boundary_storage_fraction=0.0,
                            neutral_sink_fraction=0.0,
                            signed_sink_fraction=0.0):
    """Assemble a conservative donor-to-receiver transaction before motion."""
    volume = float(event_volume_m3)
    line_energy = float(line_energy_J_m)
    if (not math.isfinite(volume) or volume <= 0.0 or not math.isfinite(line_energy)
            or line_energy < 0.0):
        raise ValueError("event volume and line energy are invalid")
    transfer = conservative_front_transfer(
        donor, transmission_fraction=transmission_fraction,
        boundary_storage_fraction=boundary_storage_fraction,
        neutral_sink_fraction=neutral_sink_fraction,
        signed_sink_fraction=signed_sink_fraction)
    donor_line = _mean_density(total_line_density(donor)) * volume
    transmitted = _mean_density(total_line_density(transfer.child)) * volume
    boundary = _mean_density(transfer.boundary_excess_line_density_m2) * volume
    annihilated = _mean_density(transfer.annihilated_line_density_m2) * volume
    sink = _mean_density(transfer.sink_line_density_m2) * volume
    line_closure = _mean_density(transfer.line_closure_density_m2) * volume
    signed_closure = float(np.max(np.abs(transfer.signed_closure_density_m2))) * volume
    defect_delta = line_energy * (transmitted + boundary - donor_line)
    heat = line_energy * annihilated
    sink_export = line_energy * sink
    full_delta = float(energy_terms.nondefect_total_J) + defect_delta
    # The nondefect terms are recoverable free-energy increments and are not
    # silently converted into heat.  This closure audits defect processing.
    closure = defect_delta + heat + sink_export
    return DirectionalFrontTrial(
        str(donor_name), str(receiver_name), volume, donor_line, transmitted,
        boundary, annihilated, sink, line_closure, signed_closure,
        defect_delta, full_delta, heat, sink_export, closure)


def _metropolis_pair(base_rate_s, delta_delta_f_J, temperature_K):
    """Bounded rate pair with an exact local detailed-balance ratio."""
    thermal = KB_J_K * float(temperature_K)
    exponent = float(delta_delta_f_J) / thermal
    if exponent >= 0.0:
        return base_rate_s * math.exp(-min(exponent, 700.0)), base_rate_s
    return base_rate_s, base_rate_s * math.exp(-min(-exponent, 700.0))


def propose_bidirectional_front_event(
        state_a: DefectState, state_b: DefectState, *, event_volume_m3,
        event_length_m, line_energy_J_m, temperature_K,
        process: ActivatedProcess, h0_J, critical_pressure_Pa, exp_a, exp_n,
        exp_floor, energy_a_to_b=FrontEnergyTerms(),
        energy_b_to_a=FrontEnergyTerms(), mobility_enabled=True,
        transmission_fraction=0.0, boundary_storage_fraction=0.0,
        neutral_sink_fraction=0.0, signed_sink_fraction=0.0):
    """Construct both transactions and then evaluate their detailed-balance rate."""
    common = dict(
        event_volume_m3=event_volume_m3, line_energy_J_m=line_energy_J_m,
        transmission_fraction=transmission_fraction,
        boundary_storage_fraction=boundary_storage_fraction,
        neutral_sink_fraction=neutral_sink_fraction,
        signed_sink_fraction=signed_sink_fraction)
    ab = directional_front_trial(
        state_a, donor_name="a", receiver_name="b",
        energy_terms=energy_a_to_b, **common)
    ba = directional_front_trial(
        state_b, donor_name="b", receiver_name="a",
        energy_terms=energy_b_to_a, **common)
    length = float(event_length_m)
    temperature = float(temperature_K)
    if (not math.isfinite(length) or length < 0.0 or not math.isfinite(temperature)
            or temperature <= 0.0):
        raise ValueError("event length or temperature is invalid")
    pressure = abs(ab.full_free_energy_change_J-ba.full_free_energy_change_J) / float(event_volume_m3)
    enthalpy = exp_floor_enthalpy_j(
        pressure, h0_J, critical_pressure_Pa, exp_a, exp_n, exp_floor)
    base = activated_rate_s(process, enthalpy, temperature) if mobility_enabled else 0.0
    delta = ab.full_free_energy_change_J-ba.full_free_energy_change_J
    rate_ab, rate_ba = _metropolis_pair(base, delta, temperature)
    expected_log_ratio = -delta/(KB_J_K*temperature)
    if rate_ab == 0.0 and rate_ba == 0.0:
        residual = 0.0
    else:
        residual = math.log(rate_ab/rate_ba)-expected_log_ratio
    return BidirectionalFrontEvent(
        ab, ba, rate_ab, rate_ba, length*(rate_ab-rate_ba), residual)
