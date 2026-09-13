"""Physical-grain recognition for the full two-dimensional PF recovery trunk.

The tracker is diagnostic: it cannot allocate labels or modify phase fields.
Initial matrix grains and recrystallized descendants are reported separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math

import numpy as np


@dataclass(frozen=True)
class GrainCriteria:
    interface_width_m: float
    minimum_area_width_factor: float
    purity_threshold: float
    minimum_persistence_s: float
    stable_support_s: float
    retirement_grace_s: float
    minimum_misorientation_rad: float
    orientation_symmetry_order: int
    minimum_stored_energy_drop_J_m3: float
    growth_relative_tolerance: float = 1e-6

    def __post_init__(self):
        for name in ("interface_width_m", "minimum_area_width_factor"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("minimum_persistence_s", "stable_support_s", "retirement_grace_s",
                     "minimum_misorientation_rad", "minimum_stored_energy_drop_J_m3",
                     "growth_relative_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0.0 < self.purity_threshold <= 1.0:
            raise ValueError("purity threshold must be in (0,1]")
        if self.orientation_symmetry_order < 1:
            raise ValueError("orientation symmetry order must be positive")

    @property
    def minimum_area_m2(self):
        return self.minimum_area_width_factor * self.interface_width_m**2


@dataclass(frozen=True)
class GrainRecord:
    label: int
    orientation_rad: float
    parent_label: int | None
    lineage: str
    birth_time_s: float
    source_embryo_id: int | None
    embryo_promoted: bool
    status: str = "allocated"
    support_time_s: float = 0.0
    absent_time_s: float = 0.0
    current_area_m2: float = 0.0
    maximum_area_m2: float = 0.0
    ever_grew: bool = False
    lower_energy_time_s: float = 0.0
    recognition_time_s: float | None = None

    def __post_init__(self):
        if self.label < 0 or not self.lineage:
            raise ValueError("label and lineage must be declared")
        if self.parent_label is not None and self.parent_label < 0:
            raise ValueError("parent label must be nonnegative")
        if self.source_embryo_id is not None and self.source_embryo_id < 0:
            raise ValueError("source embryo ID must be nonnegative")
        if self.status not in ("allocated", "initial", "recrystallized", "retired", "rejected"):
            raise ValueError("invalid grain status")
        for name in ("birth_time_s", "support_time_s", "absent_time_s",
                     "current_area_m2", "maximum_area_m2", "lower_energy_time_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class GrainTracker:
    records: tuple[GrainRecord, ...]
    time_s: float = 0.0

    def __post_init__(self):
        if tuple(record.label for record in self.records) != tuple(range(len(self.records))):
            raise ValueError("grain records must be contiguous and label ordered")
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("tracker time must be finite and nonnegative")


@dataclass(frozen=True)
class GrainMetrics:
    allocated_labels: int
    topology_components: int
    resolved_labels: int
    physical_matrix_grains: int
    physical_drx_grains: int
    recrystallized_area_fraction: float
    rejected_labels: int
    retired_labels: int


def crystallographic_misorientation_rad(a, b, symmetry_order):
    period = 2.0 * math.pi / symmetry_order
    delta = (float(a) - float(b)) % period
    return min(delta, period - delta)


def periodic_component_count(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("component mask must be two-dimensional")
    visited = np.zeros_like(mask)
    count = 0
    for start in zip(*np.nonzero(mask)):
        if visited[start]:
            continue
        count += 1
        visited[start] = True
        stack = [start]
        while stack:
            i, j = stack.pop()
            for neighbor in (((i-1) % mask.shape[0], j), ((i+1) % mask.shape[0], j),
                             (i, (j-1) % mask.shape[1]), (i, (j+1) % mask.shape[1])):
                if mask[neighbor] and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
    return count


def update_tracker(eta, stored_energy_J_m3, tracker, time_s, dx_m, dy_m, criteria):
    """Measure full PF fields and update physical-time recognition state."""
    fields = np.asarray(eta, dtype=float)
    stored = np.asarray(stored_energy_J_m3, dtype=float)
    if fields.ndim != 3 or fields.shape[:2] != stored.shape:
        raise ValueError("eta must be (Nx,Ny,Ng) and match stored-energy shape")
    if fields.shape[2] != len(tracker.records):
        raise ValueError("eta label count must match tracker")
    if not np.all(np.isfinite(fields)) or np.any(fields < 0.0) or np.any(fields > 1.0):
        raise ValueError("eta must be finite and in [0,1]")
    if not np.all(np.isfinite(stored)) or np.any(stored < 0.0):
        raise ValueError("stored energy must be finite and nonnegative")
    if time_s < tracker.time_s or dx_m <= 0.0 or dy_m <= 0.0:
        raise ValueError("time must be monotone and cell dimensions positive")
    dt_s = time_s - tracker.time_s
    dominant = np.argmax(fields, axis=2)
    masks = tuple((dominant == g) & (fields[:, :, g] >= criteria.purity_threshold)
                  for g in range(fields.shape[2]))
    areas = tuple(float(np.count_nonzero(mask) * dx_m * dy_m) for mask in masks)
    components = tuple(periodic_component_count(mask) for mask in masks)
    mean_energy = tuple(float(np.mean(stored[mask])) if np.any(mask) else math.inf
                        for mask in masks)

    updated = []
    for old, area, component_count, own_energy in zip(
            tracker.records, areas, components, mean_energy):
        if old.status in ("retired", "rejected"):
            updated.append(replace(old, current_area_m2=area))
            continue
        resolved = area >= criteria.minimum_area_m2 and component_count == 1
        support = old.support_time_s + dt_s if resolved else 0.0
        absent = 0.0 if resolved else old.absent_time_s + dt_s
        grew = old.ever_grew or (
            old.maximum_area_m2 > 0.0
            and area > old.maximum_area_m2 * (1.0 + criteria.growth_relative_tolerance))
        lower = False
        valid_lineage = True
        distinct = True
        if old.parent_label is not None:
            parent = tracker.records[old.parent_label]
            lower = own_energy + criteria.minimum_stored_energy_drop_J_m3 < mean_energy[old.parent_label]
            valid_lineage = old.lineage.startswith(parent.lineage + "/")
            distinct = crystallographic_misorientation_rad(
                old.orientation_rad, parent.orientation_rad,
                criteria.orientation_symmetry_order) >= criteria.minimum_misorientation_rad
        lower_time = old.lower_energy_time_s + dt_s if lower else 0.0
        status = old.status
        recognition = old.recognition_time_s
        if old.parent_label is None and resolved and support >= criteria.minimum_persistence_s:
            status = "initial"
            recognition = recognition if recognition is not None else time_s
        elif (old.parent_label is not None and resolved
              and support >= criteria.minimum_persistence_s
              and lower_time >= criteria.minimum_persistence_s
              and valid_lineage and distinct
              and old.source_embryo_id is not None and old.embryo_promoted
              and (grew or support >= criteria.stable_support_s)):
            status = "recrystallized"
            recognition = recognition if recognition is not None else time_s
        elif (old.parent_label is not None and support >= criteria.minimum_persistence_s
              and (not valid_lineage or not distinct or old.source_embryo_id is None
                   or not old.embryo_promoted)):
            status = "rejected"
        if status in ("initial", "recrystallized") and absent > criteria.retirement_grace_s:
            status = "retired"
        updated.append(replace(
            old, status=status, support_time_s=support, absent_time_s=absent,
            current_area_m2=area, maximum_area_m2=max(old.maximum_area_m2, area),
            ever_grew=grew, lower_energy_time_s=lower_time,
            recognition_time_s=recognition))

    new_tracker = GrainTracker(tuple(updated), float(time_s))
    resolved_flags = tuple(area >= criteria.minimum_area_m2 and count == 1
                           for area, count in zip(areas, components))
    drx_area = sum(area for area, resolved, record in zip(areas, resolved_flags, updated)
                   if resolved and record.status == "recrystallized")
    metrics = GrainMetrics(
        allocated_labels=len(updated), topology_components=sum(components),
        resolved_labels=sum(resolved_flags),
        physical_matrix_grains=sum(r and x.status == "initial" for r, x in zip(resolved_flags, updated)),
        physical_drx_grains=sum(r and x.status == "recrystallized" for r, x in zip(resolved_flags, updated)),
        recrystallized_area_fraction=drx_area/(stored.size*dx_m*dy_m),
        rejected_labels=sum(x.status == "rejected" for x in updated),
        retired_labels=sum(x.status == "retired" for x in updated))
    return new_tracker, metrics


def tracker_to_json(tracker):
    return json.dumps({"schema": "full-v34-physical-grains/v1", "time_s": tracker.time_s,
                       "records": [asdict(record) for record in tracker.records]},
                      sort_keys=True, separators=(",", ":"))


def tracker_from_json(text):
    payload = json.loads(text)
    if payload.get("schema") != "full-v34-physical-grains/v1":
        raise ValueError("unsupported physical-grain schema")
    return GrainTracker(tuple(GrainRecord(**raw) for raw in payload["records"]),
                        float(payload["time_s"]))
