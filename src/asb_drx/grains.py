"""Auditable physical-grain classification for multi-order-parameter fields.

This module measures resolved grain support.  It does not nucleate grains or
advance phase fields, and its generic verification fixtures are not material
parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np


GrainStatus = Literal["allocated", "active", "promoted", "retired", "rejected"]


@dataclass(frozen=True)
class GrainCriteria:
    purity_threshold: float
    minimum_area_m2: float
    minimum_persistence_steps: int
    retirement_grace_steps: int
    minimum_misorientation_rad: float
    symmetry_order: int
    growth_relative_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if not math.isfinite(self.purity_threshold) or not 0.0 < self.purity_threshold <= 1.0:
            raise ValueError("purity_threshold must be finite and in (0, 1]")
        if not math.isfinite(self.minimum_area_m2) or self.minimum_area_m2 <= 0.0:
            raise ValueError("minimum_area_m2 must be finite and positive")
        if self.minimum_persistence_steps < 1:
            raise ValueError("minimum_persistence_steps must be positive")
        if self.retirement_grace_steps < 1:
            raise ValueError("retirement_grace_steps must be positive")
        if (
            not math.isfinite(self.minimum_misorientation_rad)
            or self.minimum_misorientation_rad < 0.0
        ):
            raise ValueError("minimum_misorientation_rad must be finite and nonnegative")
        if self.symmetry_order < 1:
            raise ValueError("symmetry_order must be positive")
        if (
            not math.isfinite(self.growth_relative_tolerance)
            or self.growth_relative_tolerance < 0.0
        ):
            raise ValueError("growth_relative_tolerance must be finite and nonnegative")


@dataclass(frozen=True)
class GrainRecord:
    label: int
    orientation_rad: float
    parent_label: int | None
    lineage_id: str
    birth_time_s: float
    status: GrainStatus = "allocated"
    consecutive_support_steps: int = 0
    inactive_steps: int = 0
    current_area_m2: float = 0.0
    maximum_area_m2: float = 0.0
    ever_grew: bool = False
    promoted_time_s: float | None = None

    def __post_init__(self) -> None:
        if self.label < 0:
            raise ValueError("label must be nonnegative")
        if not math.isfinite(self.orientation_rad):
            raise ValueError("orientation_rad must be finite")
        if self.parent_label is not None and self.parent_label < 0:
            raise ValueError("parent_label must be nonnegative when present")
        if not self.lineage_id:
            raise ValueError("lineage_id must be nonempty")
        if not math.isfinite(self.birth_time_s) or self.birth_time_s < 0.0:
            raise ValueError("birth_time_s must be finite and nonnegative")
        if self.status not in ("allocated", "active", "promoted", "retired", "rejected"):
            raise ValueError("invalid grain status")
        if self.consecutive_support_steps < 0 or self.inactive_steps < 0:
            raise ValueError("support counters must be nonnegative")
        for name in ("current_area_m2", "maximum_area_m2"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.promoted_time_s is not None and (
            not math.isfinite(self.promoted_time_s) or self.promoted_time_s < self.birth_time_s
        ):
            raise ValueError("promoted_time_s must be finite and no earlier than birth")


@dataclass(frozen=True)
class GrainTrackerState:
    records: tuple[GrainRecord, ...]
    updates: int = 0
    time_s: float = 0.0

    def __post_init__(self) -> None:
        if self.updates < 0:
            raise ValueError("updates must be nonnegative")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("time_s must be finite and nonnegative")
        labels = tuple(record.label for record in self.records)
        if labels != tuple(range(len(self.records))):
            raise ValueError("records must be ordered with contiguous labels from zero")
        for record in self.records:
            if record.parent_label is not None and record.parent_label >= len(self.records):
                raise ValueError("parent_label must reference an allocated record")


@dataclass(frozen=True)
class GrainMetrics:
    allocated_labels: int
    topology_components: int
    resolved_labels: int
    physical_grains: int
    recrystallized_grains: int
    recrystallized_area_fraction: float
    rejected_labels: int
    retired_labels: int


def crystallographic_misorientation_rad(
    orientation_a_rad: float, orientation_b_rad: float, symmetry_order: int
) -> float:
    """Smallest scalar-orientation separation modulo an n-fold symmetry."""

    if not math.isfinite(orientation_a_rad) or not math.isfinite(orientation_b_rad):
        raise ValueError("orientations must be finite")
    if symmetry_order < 1:
        raise ValueError("symmetry_order must be positive")
    period = 2.0 * math.pi / symmetry_order
    difference = (orientation_a_rad - orientation_b_rad) % period
    return min(difference, period - difference)


def periodic_component_count(mask: np.ndarray) -> int:
    """Count four-connected components on a doubly periodic grid."""

    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype.kind != "b":
        raise ValueError("mask must be a two-dimensional Boolean array")
    rows, columns = mask.shape
    if rows < 1 or columns < 1:
        raise ValueError("mask dimensions must be nonempty")
    visited = np.zeros_like(mask, dtype=bool)
    components = 0
    for row, column in zip(*np.nonzero(mask & ~visited)):
        if visited[row, column]:
            continue
        components += 1
        stack = [(int(row), int(column))]
        visited[row, column] = True
        while stack:
            current_row, current_column = stack.pop()
            for next_row, next_column in (
                ((current_row - 1) % rows, current_column),
                ((current_row + 1) % rows, current_column),
                (current_row, (current_column - 1) % columns),
                (current_row, (current_column + 1) % columns),
            ):
                if mask[next_row, next_column] and not visited[next_row, next_column]:
                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))
    return components


def update_grain_tracker(
    eta_fields: np.ndarray,
    state: GrainTrackerState,
    time_s: float,
    dx_m: float,
    criteria: GrainCriteria,
) -> tuple[GrainTrackerState, GrainMetrics]:
    """Update immutable records from dominant, pure, connected field support."""

    fields = np.asarray(eta_fields, dtype=float)
    if fields.ndim != 3 or fields.shape[1] < 1 or fields.shape[2] < 1:
        raise ValueError("eta_fields must have shape (labels, rows, columns)")
    if fields.shape[0] != len(state.records):
        raise ValueError("eta_fields label count must match tracker records")
    if not np.all(np.isfinite(fields)) or np.any(fields < 0.0) or np.any(fields > 1.0):
        raise ValueError("eta_fields must be finite and in [0, 1]")
    if not math.isfinite(time_s) or time_s < state.time_s:
        raise ValueError("time_s must be finite and monotone")
    if not math.isfinite(dx_m) or dx_m <= 0.0:
        raise ValueError("dx_m must be finite and positive")

    dominant = np.argmax(fields, axis=0)
    masks = tuple(
        (dominant == label) & (fields[label] >= criteria.purity_threshold)
        for label in range(fields.shape[0])
    )
    components = tuple(periodic_component_count(mask) for mask in masks)
    areas = tuple(float(np.count_nonzero(mask) * dx_m**2) for mask in masks)
    records: list[GrainRecord] = []
    for old, area_m2, component_count in zip(state.records, areas, components):
        if old.status in ("retired", "rejected"):
            records.append(replace(old, current_area_m2=area_m2))
            continue
        resolved = area_m2 >= criteria.minimum_area_m2 and component_count == 1
        consecutive = old.consecutive_support_steps + 1 if resolved else 0
        inactive = 0 if resolved else old.inactive_steps + 1
        grew = old.ever_grew or (
            old.maximum_area_m2 > 0.0
            and area_m2 > old.maximum_area_m2 * (1.0 + criteria.growth_relative_tolerance)
        )
        maximum = max(old.maximum_area_m2, area_m2)
        status = old.status
        promoted_time = old.promoted_time_s
        persistent = consecutive >= criteria.minimum_persistence_steps
        if persistent and status == "allocated":
            if old.parent_label is None:
                status = "active"
            else:
                parent = state.records[old.parent_label]
                misorientation = crystallographic_misorientation_rad(
                    old.orientation_rad, parent.orientation_rad, criteria.symmetry_order
                )
                valid_lineage = old.lineage_id.startswith(parent.lineage_id + "/")
                if valid_lineage and misorientation >= criteria.minimum_misorientation_rad:
                    status = "promoted"
                    promoted_time = time_s
                else:
                    status = "rejected"
        if status in ("active", "promoted") and inactive >= criteria.retirement_grace_steps:
            status = "retired"
        records.append(
            replace(
                old,
                status=status,
                consecutive_support_steps=consecutive,
                inactive_steps=inactive,
                current_area_m2=area_m2,
                maximum_area_m2=maximum,
                ever_grew=grew,
                promoted_time_s=promoted_time,
            )
        )

    new_state = GrainTrackerState(tuple(records), state.updates + 1, time_s)
    currently_resolved = tuple(
        area >= criteria.minimum_area_m2 and count == 1
        for area, count in zip(areas, components)
    )
    physical = sum(
        resolved and record.status in ("active", "promoted")
        for resolved, record in zip(currently_resolved, records)
    )
    recrystallized = sum(
        resolved and record.status == "promoted"
        for resolved, record in zip(currently_resolved, records)
    )
    recrystallized_area = sum(
        area
        for area, resolved, record in zip(areas, currently_resolved, records)
        if resolved and record.status == "promoted"
    )
    domain_area_m2 = fields.shape[1] * fields.shape[2] * dx_m**2
    metrics = GrainMetrics(
        allocated_labels=len(records),
        topology_components=sum(components),
        resolved_labels=sum(currently_resolved),
        physical_grains=physical,
        recrystallized_grains=recrystallized,
        recrystallized_area_fraction=recrystallized_area / domain_area_m2,
        rejected_labels=sum(record.status == "rejected" for record in records),
        retired_labels=sum(record.status == "retired" for record in records),
    )
    return new_state, metrics


def save_grain_tracker(path: Path, state: GrainTrackerState) -> None:
    payload = {
        "schema": "asb-drx-grain-tracker/v1",
        "updates": state.updates,
        "time_s": state.time_s,
        "records": [asdict(record) for record in state.records],
    }
    Path(path).write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def load_grain_tracker(path: Path) -> GrainTrackerState:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "asb-drx-grain-tracker/v1":
        raise ValueError("unsupported grain-tracker checkpoint schema")
    records = tuple(GrainRecord(**record) for record in payload["records"])
    return GrainTrackerState(records, int(payload["updates"]), float(payload["time_s"]))
