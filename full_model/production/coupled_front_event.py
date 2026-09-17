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
        ActivatedProcess, KB_J_K, activated_rate_s, exp_floor_enthalpy_j,
        free_barrier_j)
    from .moving_front import (
        DefectState, conservative_front_transfer, total_line_density)
except ImportError:  # pragma: no cover - direct production-script execution
    from arrhenius_kinetics import (
        ActivatedProcess, KB_J_K, activated_rate_s, exp_floor_enthalpy_j,
        free_barrier_j)
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
class DirectionalRateChannel:
    """One physical outgoing channel from the currently accepted state.

    ``acceptance_probability`` is the thermodynamic uphill acceptance.  It is
    distinct from ``availability_factor`` (whether the channel exists) and
    from the EXP-floor transition-state rate.  Keeping these factors separate
    prevents two different downhill endpoints from inheriting one shared rate
    merely because both acceptances saturate at unity.
    """

    endpoint_free_energy_change_J: float
    driving_pressure_magnitude_Pa: float
    activation_enthalpy_J: float
    activation_entropy_over_kB: float
    activation_free_barrier_J: float
    attempt_frequency_s: float
    identifiable_prefactor_s: float
    transition_state_rate_s: float
    acceptance_probability: float
    availability_factor: float
    gross_activity_s: float


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
    actual_reverse_edge: bool = False
    detailed_balance_applicable: bool = False
    a_to_b_channel: DirectionalRateChannel | None = None
    b_to_a_channel: DirectionalRateChannel | None = None


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


def _directional_rate_channel(process, *, delta_f_J, event_volume_m3,
                              temperature_K, h0_J, critical_pressure_Pa,
                              exp_a, exp_n, exp_floor, availability_factor):
    """Evaluate one outgoing channel without reference to its competitor."""
    delta_f = float(delta_f_J)
    temperature = float(temperature_K)
    availability = float(availability_factor)
    if (not math.isfinite(delta_f) or not math.isfinite(availability)
            or availability < 0.0 or availability > 1.0):
        raise ValueError("channel energy and availability must be finite")
    pressure = abs(delta_f)/float(event_volume_m3)
    enthalpy = exp_floor_enthalpy_j(
        pressure, h0_J, critical_pressure_Pa, exp_a, exp_n, exp_floor)
    barrier = free_barrier_j(
        enthalpy, temperature, process.entropy_over_kB)
    transition_rate = activated_rate_s(process, enthalpy, temperature)
    acceptance = math.exp(-min(max(
        delta_f/(KB_J_K*temperature), 0.0), 700.0))
    gross = availability*transition_rate*acceptance
    return DirectionalRateChannel(
        delta_f, pressure, enthalpy, process.entropy_over_kB, barrier,
        process.attempt_frequency_s, process.identifiable_prefactor_s,
        transition_rate, acceptance, availability, gross)


def _directional_metropolis_pair(channel_ab, channel_ba):
    """Return gross activities for two independently priced channels.

    When the second trial is the microscopic reverse of the first,
    ``delta_f_ba = -delta_f_ab`` and this construction gives the required
    ratio ``exp(-delta_f_ab/kT)``.  Subtracting the two directional energies
    would instead double the exponent.  Irreversible line removal can make the
    two trials distinct outgoing channels; their EXP-floor transition states
    must then be evaluated separately rather than assigned one shared base.
    """
    return channel_ab.gross_activity_s, channel_ba.gross_activity_s


def propose_bidirectional_front_event(
        state_a: DefectState, state_b: DefectState, *, event_volume_m3,
        event_length_m, line_energy_J_m, temperature_K,
        process: ActivatedProcess, h0_J, critical_pressure_Pa, exp_a, exp_n,
        exp_floor, energy_a_to_b=FrontEnergyTerms(),
        energy_b_to_a=FrontEnergyTerms(), mobility_enabled=True,
        transmission_fraction=0.0, boundary_storage_fraction=0.0,
        neutral_sink_fraction=0.0, signed_sink_fraction=0.0,
        kinetic_free_energy_a_to_b_J=None,
        kinetic_free_energy_b_to_a_J=None,
        availability_a_to_b=1.0, availability_b_to_a=1.0,
        actual_reverse_edge=None):
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
    mobility = 1.0 if mobility_enabled else 0.0
    channel_ab = _directional_rate_channel(
        process, delta_f_J=kinetic_ab, event_volume_m3=event_volume_m3,
        temperature_K=temperature, h0_J=h0_J,
        critical_pressure_Pa=critical_pressure_Pa, exp_a=exp_a,
        exp_n=exp_n, exp_floor=exp_floor,
        availability_factor=mobility*float(availability_a_to_b))
    channel_ba = _directional_rate_channel(
        process, delta_f_J=kinetic_ba, event_volume_m3=event_volume_m3,
        temperature_K=temperature, h0_J=h0_J,
        critical_pressure_Pa=critical_pressure_Pa, exp_a=exp_a,
        exp_n=exp_n, exp_floor=exp_floor,
        availability_factor=mobility*float(availability_b_to_a))
    rate_ab, rate_ba = _directional_metropolis_pair(channel_ab, channel_ba)
    expected_log_ratio = (
        math.log(channel_ab.transition_state_rate_s
                 /channel_ba.transition_state_rate_s)
        +math.log(max(channel_ab.availability_factor, 1e-300)
                  /max(channel_ba.availability_factor, 1e-300))
        +math.log(channel_ab.acceptance_probability
                  /channel_ba.acceptance_probability))
    if rate_ab <= 0.0 or rate_ba <= 0.0:
        residual = 0.0
    else:
        residual = math.log(rate_ab/rate_ba)-expected_log_ratio
    reverse_scale = max(abs(kinetic_ab), abs(kinetic_ba), 1e-300)
    energy_reverse = bool(abs(kinetic_ab+kinetic_ba) <= (
        4096.0*np.finfo(float).eps*reverse_scale))
    transaction_reverse = bool(
        float(transmission_fraction) == 1.0
        and float(boundary_storage_fraction) == 0.0
        and float(neutral_sink_fraction) == 0.0
        and float(signed_sink_fraction) == 0.0)
    reverse_edge = (energy_reverse and transaction_reverse
                    if actual_reverse_edge is None
                    else bool(actual_reverse_edge))
    detailed_balance_applicable = bool(
        reverse_edge
        and channel_ab.availability_factor > 0.0
        and channel_ba.availability_factor > 0.0
        and channel_ab.availability_factor == channel_ba.availability_factor)
    return BidirectionalFrontEvent(
        ab, ba, rate_ab, rate_ba, length*(rate_ab-rate_ba), residual,
        kinetic_ab, kinetic_ba, energy_reverse, reverse_edge,
        detailed_balance_applicable, channel_ab, channel_ba)
