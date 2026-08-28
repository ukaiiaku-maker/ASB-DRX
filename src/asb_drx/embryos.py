"""Stateful finite-amplitude DRX embryos and physical promotion gate.

Embryos are physical candidate objects, not grid-cell counters or phase labels.
This module advances their isolated circular-limit dynamics and audits promotion;
it does not allocate an order-parameter field or reset dislocation density.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Literal

from .grains import GrainRecord, crystallographic_misorientation_rad


EmbryoStatus = Literal["active", "promoted", "retired", "rejected"]


@dataclass(frozen=True)
class EmbryoAttempt:
    time_s: float
    applied_shear: float
    temperature_K: float
    barrier_J: float
    event_probability: float
    uniform_draw: float
    accepted: bool

    def __post_init__(self) -> None:
        for name in ("time_s", "applied_shear", "temperature_K", "barrier_J"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("event_probability", "uniform_draw"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0,1]")


@dataclass(frozen=True)
class EmbryoHistorySample:
    time_s: float
    radius_m: float
    local_driving_energy_J_m3: float
    excess_energy_J: float
    radial_velocity_m_s: float
    phase_support_area_m2: float
    phase_purity: float


@dataclass(frozen=True)
class EmbryoRecord:
    embryo_id: str
    position_m: tuple[float, float]
    radius_m: float
    orientation_rad: float
    parent_label: int
    parent_orientation_rad: float
    lineage_id: str
    birth_time_s: float
    birth_applied_shear: float
    rng_lineage: str
    attempts: tuple[EmbryoAttempt, ...]
    status: EmbryoStatus = "active"
    age_s: float = 0.0
    integrated_positive_driving_J_s_m3: float = 0.0
    support_steps: int = 0
    maximum_radius_m: float = 0.0
    promoted_time_s: float | None = None
    retired_time_s: float | None = None
    history: tuple[EmbryoHistorySample, ...] = ()

    def __post_init__(self) -> None:
        if not self.embryo_id or not self.lineage_id or not self.rng_lineage:
            raise ValueError("embryo, lineage, and RNG identifiers must be nonempty")
        if len(self.position_m) != 2 or not all(math.isfinite(x) for x in self.position_m):
            raise ValueError("position_m must contain two finite coordinates")
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("radius_m must be finite and positive")
        if not math.isfinite(self.orientation_rad) or not math.isfinite(self.parent_orientation_rad):
            raise ValueError("orientations must be finite")
        if self.parent_label < 0:
            raise ValueError("parent_label must be nonnegative")
        for name in ("birth_time_s", "birth_applied_shear", "age_s",
                     "integrated_positive_driving_J_s_m3", "maximum_radius_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.status not in ("active", "promoted", "retired", "rejected"):
            raise ValueError("invalid embryo status")
        if self.support_steps < 0:
            raise ValueError("support_steps must be nonnegative")


@dataclass(frozen=True)
class EmbryoEvolutionParameters:
    boundary_energy_J_m2: float
    represented_thickness_m: float
    radial_mobility_m4_J_s: float
    minimum_resolved_radius_m: float
    minimum_survival_time_s: float
    minimum_support_steps: int
    minimum_phase_purity: float
    minimum_misorientation_rad: float
    symmetry_order: int

    def __post_init__(self) -> None:
        for name in (
            "boundary_energy_J_m2", "represented_thickness_m",
            "radial_mobility_m4_J_s", "minimum_resolved_radius_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.minimum_survival_time_s) or self.minimum_survival_time_s < 0.0:
            raise ValueError("minimum_survival_time_s must be finite and nonnegative")
        if self.minimum_support_steps < 1:
            raise ValueError("minimum_support_steps must be positive")
        if not math.isfinite(self.minimum_phase_purity) or not 0.0 < self.minimum_phase_purity <= 1.0:
            raise ValueError("minimum_phase_purity must be in (0,1]")
        if not math.isfinite(self.minimum_misorientation_rad) or self.minimum_misorientation_rad < 0.0:
            raise ValueError("minimum_misorientation_rad must be finite and nonnegative")
        if self.symmetry_order < 1:
            raise ValueError("symmetry_order must be positive")


@dataclass(frozen=True)
class EmbryoStepLedger:
    old_excess_energy_J: float
    new_excess_energy_J: float
    free_energy_change_J: float
    released_heat_J: float
    closure_error_J: float


@dataclass(frozen=True)
class EmbryoStep:
    record: EmbryoRecord
    ledger: EmbryoStepLedger
    accepted_dt_s: float
    halvings: int


@dataclass(frozen=True)
class EmbryoPopulation:
    records: tuple[EmbryoRecord, ...]

    def __post_init__(self) -> None:
        identifiers = tuple(item.embryo_id for item in self.records)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("embryo identifiers must be unique")


def circular_excess_energy_J(
    radius_m: float,
    local_driving_energy_J_m3: float,
    parameters: EmbryoEvolutionParameters,
) -> float:
    if not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be finite and positive")
    if not math.isfinite(local_driving_energy_J_m3) or local_driving_energy_J_m3 <= 0.0:
        raise ValueError("local driving energy must be finite and positive")
    return parameters.represented_thickness_m * (
        2.0 * math.pi * radius_m * parameters.boundary_energy_J_m2
        - math.pi * radius_m**2 * local_driving_energy_J_m3
    )


def create_embryo(
    *,
    embryo_id: str,
    position_m: tuple[float, float],
    radius_m: float,
    orientation_rad: float,
    parent_label: int,
    parent_orientation_rad: float,
    parent_lineage_id: str,
    birth_time_s: float,
    birth_applied_shear: float,
    rng_lineage: str,
    attempt: EmbryoAttempt,
    parameters: EmbryoEvolutionParameters,
) -> EmbryoRecord:
    misorientation = crystallographic_misorientation_rad(
        orientation_rad, parent_orientation_rad, parameters.symmetry_order
    )
    accepted = attempt.accepted and misorientation >= parameters.minimum_misorientation_rad
    return EmbryoRecord(
        embryo_id,
        position_m,
        radius_m,
        orientation_rad,
        parent_label,
        parent_orientation_rad,
        f"{parent_lineage_id}/{embryo_id}",
        birth_time_s,
        birth_applied_shear,
        rng_lineage,
        (attempt,),
        status="active" if accepted else "rejected",
        maximum_radius_m=radius_m,
    )


def evolve_embryo(
    record: EmbryoRecord,
    proposed_dt_s: float,
    local_driving_energy_J_m3: float,
    phase_support_area_m2: float,
    phase_purity: float,
    parameters: EmbryoEvolutionParameters,
    *,
    maximum_halvings: int = 40,
) -> EmbryoStep:
    if not math.isfinite(proposed_dt_s) or proposed_dt_s <= 0.0:
        raise ValueError("proposed_dt_s must be finite and positive")
    if not math.isfinite(local_driving_energy_J_m3) or local_driving_energy_J_m3 <= 0.0:
        raise ValueError("local driving energy must be finite and positive")
    if not math.isfinite(phase_support_area_m2) or phase_support_area_m2 < 0.0:
        raise ValueError("phase support area must be finite and nonnegative")
    if not math.isfinite(phase_purity) or not 0.0 <= phase_purity <= 1.0:
        raise ValueError("phase_purity must be in [0,1]")
    if record.status != "active":
        energy = circular_excess_energy_J(record.radius_m, local_driving_energy_J_m3, parameters)
        return EmbryoStep(record, EmbryoStepLedger(energy, energy, 0.0, 0.0, 0.0), 0.0, 0)

    critical_radius = parameters.boundary_energy_J_m2 / local_driving_energy_J_m3
    escape_radius = 2.0 * critical_radius
    old_energy = circular_excess_energy_J(
        record.radius_m, local_driving_energy_J_m3, parameters
    )
    velocity = parameters.radial_mobility_m4_J_s * (
        local_driving_energy_J_m3
        - parameters.boundary_energy_J_m2 / record.radius_m
    )
    dt_s = proposed_dt_s
    for halvings in range(maximum_halvings + 1):
        candidate_radius = record.radius_m + dt_s * velocity
        if candidate_radius <= 0.0:
            dt_s *= 0.5
            continue
        new_energy = circular_excess_energy_J(
            candidate_radius, local_driving_energy_J_m3, parameters
        )
        tolerance = 64.0 * math.ulp(max(abs(old_energy), abs(new_energy), 1.0e-300))
        if new_energy > old_energy + tolerance:
            dt_s *= 0.5
            continue
        break
    else:
        raise RuntimeError("no energy-decreasing embryo step found")

    age = record.age_s + dt_s
    support_required_area = math.pi * parameters.minimum_resolved_radius_m**2
    supported = (
        phase_support_area_m2 >= support_required_area
        and phase_purity >= parameters.minimum_phase_purity
    )
    support_steps = record.support_steps + 1 if supported else 0
    positive_force = max(
        local_driving_energy_J_m3
        - parameters.boundary_energy_J_m2 / record.radius_m,
        0.0,
    )
    integrated = record.integrated_positive_driving_J_s_m3 + positive_force * dt_s
    new_time = record.birth_time_s + age
    status: EmbryoStatus = "active"
    promoted_time = None
    retired_time = None
    if candidate_radius < parameters.minimum_resolved_radius_m:
        status = "retired"
        retired_time = new_time
    elif (
        candidate_radius >= escape_radius
        and age >= parameters.minimum_survival_time_s
        and support_steps >= parameters.minimum_support_steps
        and integrated > 0.0
    ):
        status = "promoted"
        promoted_time = new_time
    sample = EmbryoHistorySample(
        new_time,
        candidate_radius,
        local_driving_energy_J_m3,
        new_energy,
        velocity,
        phase_support_area_m2,
        phase_purity,
    )
    updated = replace(
        record,
        radius_m=candidate_radius,
        status=status,
        age_s=age,
        integrated_positive_driving_J_s_m3=integrated,
        support_steps=support_steps,
        maximum_radius_m=max(record.maximum_radius_m, candidate_radius),
        promoted_time_s=promoted_time,
        retired_time_s=retired_time,
        history=record.history + (sample,),
    )
    free_change = new_energy - old_energy
    heat = max(-free_change, 0.0)
    return EmbryoStep(
        updated,
        EmbryoStepLedger(old_energy, new_energy, free_change, heat, -free_change - heat),
        dt_s,
        halvings,
    )


def promoted_embryo_grain_record(record: EmbryoRecord, label: int) -> GrainRecord:
    """Create grain provenance only after the physical embryo gate passes."""
    if record.status != "promoted" or record.promoted_time_s is None:
        raise ValueError("only a promoted embryo can seed a grain record")
    return GrainRecord(
        label=label,
        orientation_rad=record.orientation_rad,
        parent_label=record.parent_label,
        lineage_id=record.lineage_id,
        birth_time_s=record.birth_time_s,
        source_embryo_id=record.embryo_id,
        embryo_gate_passed=True,
    )


def save_embryo_population(path: Path, population: EmbryoPopulation) -> None:
    payload = {
        "schema": "asb-drx-embryo-population/v1",
        "records": [asdict(item) for item in population.records],
    }
    Path(path).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def load_embryo_population(path: Path) -> EmbryoPopulation:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "asb-drx-embryo-population/v1":
        raise ValueError("unsupported embryo checkpoint schema")
    records = []
    for raw in payload["records"]:
        raw["position_m"] = tuple(raw["position_m"])
        raw["attempts"] = tuple(EmbryoAttempt(**item) for item in raw["attempts"])
        raw["history"] = tuple(EmbryoHistorySample(**item) for item in raw["history"])
        records.append(EmbryoRecord(**raw))
    return EmbryoPopulation(tuple(records))
