"""Stateful full-model DRX embryos used by the v34 recovery trunk.

This is a selective back-port of the record/ledger idea from the retired reduced
campaign.  It deliberately has no dependency on :mod:`asb_drx` and cannot
allocate a grain label.  The full driver owns phase-field support and promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from typing import Literal

from .arrhenius_kinetics import ActivatedProcess, activated_rate_s, exp_floor_enthalpy_j


EmbryoStatus = Literal["active", "promotable", "promoted", "retired", "rejected"]


@dataclass(frozen=True)
class EmbryoEvent:
    step: int
    time_s: float
    kind: str
    hazard: float
    stochastic_threshold: float
    barrier_J: float
    rate_s: float


@dataclass(frozen=True)
class EmbryoSample:
    step: int
    time_s: float
    radius_m: float
    stored_relief_J_m3: float
    orientation_penalty_J_m3: float
    compatibility_penalty_J_m3: float
    interface_energy_J: float
    bulk_energy_J: float
    total_excess_energy_J: float
    radial_velocity_m_s: float
    phase_support_area_m2: float
    phase_purity: float
    gb_contact_fraction: float
    wall_contact_fraction: float
    gnd_contact_fraction: float


@dataclass(frozen=True)
class EmbryoRecord:
    embryo_id: int
    parent_grain: int
    parent_lineage: str
    position_m: tuple[float, float]
    orientation_rad: float
    parent_orientation_rad: float
    radius_m: float
    birth_step: int
    birth_time_s: float
    birth_strain: float
    rng_stream: str
    rng_state_json: str
    cumulative_hazard: float
    status: EmbryoStatus = "active"
    age_s: float = 0.0
    support_time_s: float = 0.0
    maximum_radius_m: float = 0.0
    events: tuple[EmbryoEvent, ...] = ()
    history: tuple[EmbryoSample, ...] = ()

    def __post_init__(self):
        if self.embryo_id < 0 or self.parent_grain < 0 or self.birth_step < 0:
            raise ValueError("embryo, parent, and birth identifiers must be nonnegative")
        if not self.parent_lineage or not self.rng_stream:
            raise ValueError("lineage and RNG stream must be declared")
        if len(self.position_m) != 2 or not all(math.isfinite(v) for v in self.position_m):
            raise ValueError("position must contain two finite coordinates")
        for name in ("radius_m", "birth_time_s", "birth_strain", "age_s",
                     "support_time_s", "maximum_radius_m", "cumulative_hazard"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.radius_m <= 0.0:
            raise ValueError("radius must be positive")
        if self.status not in ("active", "promotable", "promoted", "retired", "rejected"):
            raise ValueError("invalid embryo status")


@dataclass(frozen=True)
class EmbryoParameters:
    represented_thickness_m: float
    interface_energy_J_m2: float
    mobility_prefactor_m4_J_s: float
    mobility_process: ActivatedProcess
    mobility_enthalpy_0_J: float
    mobility_critical_pressure_Pa: float
    mobility_exp_a: float
    mobility_exp_n: float
    mobility_enthalpy_floor: float
    minimum_resolved_radius_m: float
    minimum_survival_time_s: float
    minimum_support_time_s: float
    minimum_phase_purity: float
    minimum_misorientation_rad: float
    orientation_symmetry_order: int

    def __post_init__(self):
        positive = (
            "represented_thickness_m", "interface_energy_J_m2",
            "mobility_prefactor_m4_J_s", "mobility_critical_pressure_Pa",
            "minimum_resolved_radius_m",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.mobility_enthalpy_0_J < 0.0 or self.mobility_exp_a < 0.0:
            raise ValueError("mobility enthalpy and EXP coefficient must be nonnegative")
        if self.mobility_exp_n < 1.0:
            raise ValueError("production mobility EXP exponent must be >= 1")
        if not 0.0 <= self.mobility_enthalpy_floor <= 1.0:
            raise ValueError("mobility enthalpy floor must be in [0,1]")
        if self.minimum_survival_time_s < 0.0 or self.minimum_support_time_s < 0.0:
            raise ValueError("minimum times must be nonnegative")
        if not 0.0 < self.minimum_phase_purity <= 1.0:
            raise ValueError("phase purity threshold must be in (0,1]")
        if self.orientation_symmetry_order < 1:
            raise ValueError("orientation symmetry order must be positive")


@dataclass(frozen=True)
class EmbryoEnvironment:
    temperature_K: float
    stored_relief_J_m3: float
    orientation_penalty_J_m3: float
    compatibility_penalty_J_m3: float
    phase_support_area_m2: float
    phase_purity: float
    gb_contact_fraction: float
    wall_contact_fraction: float
    gnd_contact_fraction: float

    def __post_init__(self):
        if not math.isfinite(self.temperature_K) or self.temperature_K <= 0.0:
            raise ValueError("temperature must be positive")
        for name in ("stored_relief_J_m3", "orientation_penalty_J_m3",
                     "compatibility_penalty_J_m3", "phase_support_area_m2"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("phase_purity", "gb_contact_fraction", "wall_contact_fraction",
                     "gnd_contact_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")


@dataclass(frozen=True)
class EmbryoLedger:
    old_excess_energy_J: float
    new_excess_energy_J: float
    free_energy_change_J: float
    dissipated_energy_J: float
    closure_error_J: float


@dataclass(frozen=True)
class EmbryoStep:
    record: EmbryoRecord
    ledger: EmbryoLedger
    accepted_dt_s: float
    halvings: int


@dataclass(frozen=True)
class EmbryoPopulation:
    next_id: int
    records: tuple[EmbryoRecord, ...] = ()

    def __post_init__(self):
        identifiers = [record.embryo_id for record in self.records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("embryo IDs must be unique")
        if self.next_id < 0 or (identifiers and self.next_id <= max(identifiers)):
            raise ValueError("next embryo ID must exceed every allocated ID")


def _energies(radius_m, environment, parameters):
    interface = (parameters.represented_thickness_m * 2.0 * math.pi * radius_m
                 * parameters.interface_energy_J_m2)
    net_bulk = (environment.orientation_penalty_J_m3
                + environment.compatibility_penalty_J_m3
                - environment.stored_relief_J_m3)
    bulk = (parameters.represented_thickness_m * math.pi * radius_m**2 * net_bulk)
    return interface, bulk, interface + bulk


def _misorientation(record, symmetry_order):
    period = 2.0 * math.pi / symmetry_order
    delta = (record.orientation_rad - record.parent_orientation_rad + 0.5 * period) % period
    return abs(delta - 0.5 * period)


def evolve_embryo(record, step, time_s, proposed_dt_s, environment, parameters,
                   maximum_halvings=40):
    """Advance one full-model embryo without allocating a phase/grain label."""
    if record.status not in ("active", "promotable"):
        _, _, energy = _energies(record.radius_m, environment, parameters)
        ledger = EmbryoLedger(energy, energy, 0.0, 0.0, 0.0)
        return EmbryoStep(record, ledger, 0.0, 0)
    if proposed_dt_s <= 0.0 or not math.isfinite(proposed_dt_s):
        raise ValueError("timestep must be finite and positive")

    resistance = (environment.orientation_penalty_J_m3
                  + environment.compatibility_penalty_J_m3
                  + parameters.interface_energy_J_m2 / record.radius_m)
    force = environment.stored_relief_J_m3 - resistance
    enthalpy = exp_floor_enthalpy_j(
        abs(force), parameters.mobility_enthalpy_0_J,
        parameters.mobility_critical_pressure_Pa, parameters.mobility_exp_a,
        parameters.mobility_exp_n, parameters.mobility_enthalpy_floor)
    rate = activated_rate_s(parameters.mobility_process, enthalpy, environment.temperature_K)
    mobility = (parameters.mobility_prefactor_m4_J_s * rate
                / max(parameters.mobility_process.attempt_frequency_s, 1e-300))
    radial_velocity = mobility * force
    old_interface, old_bulk, old_energy = _energies(record.radius_m, environment, parameters)

    dt_s = proposed_dt_s
    for halvings in range(maximum_halvings + 1):
        new_radius = record.radius_m + radial_velocity * dt_s
        if new_radius <= 0.0:
            dt_s *= 0.5
            continue
        new_interface, new_bulk, new_energy = _energies(new_radius, environment, parameters)
        tolerance = 128.0 * math.ulp(max(abs(old_energy), abs(new_energy), 1e-300))
        if new_energy <= old_energy + tolerance:
            break
        dt_s *= 0.5
    else:
        raise RuntimeError("no energy-dissipating embryo step exists")

    age_s = record.age_s + dt_s
    supported = (environment.phase_purity >= parameters.minimum_phase_purity
                 and environment.phase_support_area_m2 >= math.pi * new_radius**2)
    support_time_s = record.support_time_s + dt_s if supported else 0.0
    net_bulk_drive = (environment.stored_relief_J_m3
                      - environment.orientation_penalty_J_m3
                      - environment.compatibility_penalty_J_m3)
    critical_radius = (parameters.interface_energy_J_m2 / net_bulk_drive
                       if net_bulk_drive > 0.0 else math.inf)
    promotable = (
        new_radius > critical_radius
        and radial_velocity > 0.0
        and age_s >= parameters.minimum_survival_time_s
        and support_time_s >= parameters.minimum_support_time_s
        and supported
        and _misorientation(record, parameters.orientation_symmetry_order)
            >= parameters.minimum_misorientation_rad
        and environment.stored_relief_J_m3 > 0.0
    )
    status = "promotable" if promotable else "active"
    if new_radius < parameters.minimum_resolved_radius_m:
        status = "retired"
    sample = EmbryoSample(
        int(step), float(time_s + dt_s), new_radius,
        environment.stored_relief_J_m3, environment.orientation_penalty_J_m3,
        environment.compatibility_penalty_J_m3, new_interface, new_bulk, new_energy,
        radial_velocity, environment.phase_support_area_m2, environment.phase_purity,
        environment.gb_contact_fraction, environment.wall_contact_fraction,
        environment.gnd_contact_fraction,
    )
    updated = replace(
        record, radius_m=new_radius, status=status, age_s=age_s,
        support_time_s=support_time_s,
        maximum_radius_m=max(record.maximum_radius_m, new_radius),
        history=record.history + (sample,),
    )
    change = new_energy - old_energy
    dissipated = max(-change, 0.0)
    ledger = EmbryoLedger(old_energy, new_energy, change, dissipated, -change-dissipated)
    return EmbryoStep(updated, ledger, dt_s, halvings)


def population_to_json(population):
    payload = {
        "schema": "full-v34-stateful-embryos/v1",
        "next_id": population.next_id,
        "records": [asdict(record) for record in population.records],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def population_from_json(text):
    payload = json.loads(text)
    if payload.get("schema") != "full-v34-stateful-embryos/v1":
        raise ValueError("unsupported embryo schema")
    records = []
    for raw in payload["records"]:
        raw["position_m"] = tuple(raw["position_m"])
        raw["events"] = tuple(EmbryoEvent(**event) for event in raw["events"])
        raw["history"] = tuple(EmbryoSample(**sample) for sample in raw["history"])
        records.append(EmbryoRecord(**raw))
    return EmbryoPopulation(int(payload["next_id"]), tuple(records))
