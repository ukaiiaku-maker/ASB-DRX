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

try:
    from .arrhenius_kinetics import (
        ActivatedProcess, KB_J_K, activated_rate_s, exp_floor_enthalpy_j)
    from .moving_front import (
        DefectState, conservative_front_transfer, total_line_density)
except ImportError:  # pragma: no cover - direct production-script execution
    from arrhenius_kinetics import (
        ActivatedProcess, KB_J_K, activated_rate_s, exp_floor_enthalpy_j)
    from moving_front import (
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
    kinetic_free_energy_a_to_b_J: float = 0.0
    kinetic_free_energy_b_to_a_J: float = 0.0
    microscopic_reverse_pair: bool = False


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


def _directional_metropolis_pair(base_rate_s, delta_f_ab_J, delta_f_ba_J,
                                 temperature_K):
    """Evaluate two actual directional trials without a factor-two bias.

    When the second trial is the microscopic reverse of the first,
    ``delta_f_ba = -delta_f_ab`` and this construction gives the required
    ratio ``exp(-delta_f_ab/kT)``.  Subtracting the two directional energies
    would instead double the exponent.  Irreversible line removal can make the
    two trials non-reverses; their rates then remain separately meaningful but
    are not presented as a microscopic reverse pair.
    """
    thermal = KB_J_K * float(temperature_K)
    ab_penalty = max(float(delta_f_ab_J)/thermal, 0.0)
    ba_penalty = max(float(delta_f_ba_J)/thermal, 0.0)
    return (base_rate_s*math.exp(-min(ab_penalty, 700.0)),
            base_rate_s*math.exp(-min(ba_penalty, 700.0)))


def propose_bidirectional_front_event(
        state_a: DefectState, state_b: DefectState, *, event_volume_m3,
        event_length_m, line_energy_J_m, temperature_K,
        process: ActivatedProcess, h0_J, critical_pressure_Pa, exp_a, exp_n,
        exp_floor, energy_a_to_b=FrontEnergyTerms(),
        energy_b_to_a=FrontEnergyTerms(), mobility_enabled=True,
        transmission_fraction=0.0, boundary_storage_fraction=0.0,
        neutral_sink_fraction=0.0, signed_sink_fraction=0.0,
        kinetic_free_energy_a_to_b_J=None,
        kinetic_free_energy_b_to_a_J=None):
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
    kinetic_ab = (ab.full_free_energy_change_J
                  if kinetic_free_energy_a_to_b_J is None
                  else float(kinetic_free_energy_a_to_b_J))
    kinetic_ba = (ba.full_free_energy_change_J
                  if kinetic_free_energy_b_to_a_J is None
                  else float(kinetic_free_energy_b_to_a_J))
    pressure = max(abs(kinetic_ab), abs(kinetic_ba))/float(event_volume_m3)
    enthalpy = exp_floor_enthalpy_j(
        pressure, h0_J, critical_pressure_Pa, exp_a, exp_n, exp_floor)
    base = activated_rate_s(process, enthalpy, temperature) if mobility_enabled else 0.0
    rate_ab, rate_ba = _directional_metropolis_pair(
        base, kinetic_ab, kinetic_ba, temperature)
    expected_log_ratio = float(np.clip(
        (-max(kinetic_ab, 0.0)+max(kinetic_ba, 0.0))
        /(KB_J_K*temperature), -700.0, 700.0))
    if rate_ab == 0.0 and rate_ba == 0.0:
        residual = 0.0
    else:
        residual = math.log(rate_ab/rate_ba)-expected_log_ratio
    reverse_scale = max(abs(kinetic_ab), abs(kinetic_ba), 1e-300)
    microscopic_reverse = bool(abs(kinetic_ab+kinetic_ba) <= (
        4096.0*np.finfo(float).eps*reverse_scale))
    return BidirectionalFrontEvent(
        ab, ba, rate_ab, rate_ba, length*(rate_ab-rate_ba), residual,
        kinetic_ab, kinetic_ba, microscopic_reverse)
