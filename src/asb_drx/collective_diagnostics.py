"""Structural diagnostics for persistent-contact collective event histories.

The quantities in this module are falsification diagnostics. They are not a
parameter-fitting route for the production constitutive model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Iterable


def coefficient_of_variation(values: Iterable[float]) -> float | None:
    values = list(values)
    if len(values) < 2:
        return None
    mean = fmean(values)
    if mean == 0.0:
        return None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / mean


def depin_count_diagnostics(path: str | Path, cluster_windows: tuple[int, ...] = (0, 1, 10, 100)) -> dict:
    """Summarize count clustering without interpreting it as a physical closure."""

    rows: list[tuple[int, int]] = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append((int(row["step"]), int(row["event_count"])))
    if not rows:
        raise ValueError(f"no depinning events in {path}")
    rows.sort()
    total_events = sum(count for _, count in rows)
    multi_events = sum(count for _, count in rows if count > 1)
    waits = [right[0] - left[0] for left, right in zip(rows, rows[1:])]

    clusters: dict[str, dict] = {}
    for window in cluster_windows:
        sizes: list[int] = []
        current = rows[0][1]
        previous_step = rows[0][0]
        for step, count in rows[1:]:
            if step - previous_step <= window:
                current += count
            else:
                sizes.append(current)
                current = count
            previous_step = step
        sizes.append(current)
        mean_size = fmean(sizes)
        clusters[str(window)] = {
            "cluster_count": len(sizes),
            "mean_events_per_cluster": mean_size,
            "max_events_per_cluster": max(sizes),
            "galton_watson_R_proxy": 1.0 - 1.0 / mean_size,
        }

    return {
        "event_steps": len(rows),
        "total_events": total_events,
        "max_same_step_events": max(count for _, count in rows),
        "multi_hit_step_fraction": sum(count > 1 for _, count in rows) / len(rows),
        "multi_hit_event_fraction": multi_events / total_events,
        "inter_event_step_cv": coefficient_of_variation(waits),
        "clusters_by_window_steps": clusters,
        "interpretation": "count-process diagnostic only; not a production branching parameter",
    }


def _power_spectral_radius(columns: dict[int, dict[int, float]], iterations: int = 500) -> float:
    node_ids = sorted(set(columns) | {i for children in columns.values() for i in children})
    if not node_ids:
        return 0.0
    x = {node: 1.0 / len(node_ids) for node in node_ids}
    eigenvalue = 0.0
    for _ in range(iterations):
        y = {node: 0.0 for node in node_ids}
        for parent, children in columns.items():
            parent_weight = x.get(parent, 0.0)
            for child, value in children.items():
                y[child] += value * parent_weight
        scale = sum(y.values())
        if scale == 0.0:
            return 0.0
        y = {node: value / scale for node, value in y.items()}
        if max(abs(y[node] - x[node]) for node in node_ids) < 1.0e-13:
            eigenvalue = scale
            break
        x = y
        eigenvalue = scale
    return eigenvalue


def native_audit_branching_diagnostics(path: str | Path) -> dict:
    """Construct a one-step contact-to-contact branching proxy from a native audit.

    For an accepted release at step s, contacts surviving to the next audited
    step receive an excess probability weight Rdt_i(s) * [rate_i(s+)/rate_i(s)-1].
    A centered version removes the median log-rate change shared by all surviving
    contacts, which suppresses the common applied-loading increment.
    """

    groups: dict[int, list[dict]] = defaultdict(list)
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                groups[int(row["step"])].append(row)
    steps = sorted(groups)
    if not steps:
        raise ValueError(f"empty native audit {path}")

    parent_counts: Counter[int] = Counter()
    raw_columns: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    centered_columns: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    raw_event_totals: list[float] = []
    centered_event_totals: list[float] = []
    redistribution_samples = 0
    centered_positive = 0
    accepted_rows = 0
    threshold_valid = 0
    accepted_steps: list[int] = []

    for step, next_step in zip(steps, steps[1:]):
        current = {int(row["contact_id"]): row for row in groups[step]}
        following = {int(row["contact_id"]): row for row in groups[next_step]}
        parents = [row for row in groups[step] if bool(row.get("accepted"))]
        for parent in parents:
            accepted_rows += 1
            accepted_steps.append(step)
            before = float(parent["accumulated_hazard_before"])
            after = float(parent["accumulated_hazard_after"])
            threshold = float(parent["threshold"])
            if before < threshold <= after:
                threshold_valid += 1
            parent_id = int(parent["contact_id"])
            parent_counts[parent_id] += 1
            shared = sorted((set(current) & set(following)) - {parent_id})
            changes: list[tuple[int, float, float]] = []
            for child_id in shared:
                rate_before = float(current[child_id]["rate_s"])
                rate_after = float(following[child_id]["rate_s"])
                rdt_before = float(current[child_id]["Rdt"])
                if rate_before > 0.0 and rate_after > 0.0 and rdt_before >= 0.0:
                    changes.append(
                        (child_id, math.log(rate_after) - math.log(rate_before), rdt_before)
                    )
            if not changes:
                raw_event_totals.append(0.0)
                centered_event_totals.append(0.0)
                continue
            common_log_increment = median(change for _, change, _ in changes)
            raw_total = 0.0
            centered_total = 0.0
            for child_id, log_change, rdt_before in changes:
                redistribution_samples += 1
                raw_weight = rdt_before * max(math.expm1(min(log_change, 700.0)), 0.0)
                residual = log_change - common_log_increment
                centered_weight = rdt_before * max(math.expm1(min(residual, 700.0)), 0.0)
                if centered_weight > 0.0:
                    centered_positive += 1
                raw_columns[parent_id][child_id] += raw_weight
                centered_columns[parent_id][child_id] += centered_weight
                raw_total += raw_weight
                centered_total += centered_weight
            raw_event_totals.append(raw_total)
            centered_event_totals.append(centered_total)

    for parent_id, count in parent_counts.items():
        for child_id in list(raw_columns[parent_id]):
            raw_columns[parent_id][child_id] /= count
        for child_id in list(centered_columns[parent_id]):
            centered_columns[parent_id][child_id] /= count

    simultaneous = Counter(accepted_steps)
    waits = [right - left for left, right in zip(sorted(accepted_steps), sorted(accepted_steps)[1:])]
    return {
        "accepted_release_rows": accepted_rows,
        "unique_parent_contacts": len(parent_counts),
        "threshold_crossing_valid_fraction": threshold_valid / accepted_rows if accepted_rows else None,
        "simultaneous_release_step_fraction": (
            sum(count > 1 for count in simultaneous.values()) / len(simultaneous) if simultaneous else None
        ),
        "accepted_release_interval_step_cv": coefficient_of_variation(waits),
        "redistribution_samples": redistribution_samples,
        "centered_positive_response_fraction": (
            centered_positive / redistribution_samples if redistribution_samples else None
        ),
        "mean_raw_excess_probability_per_release": (
            fmean(raw_event_totals) if raw_event_totals else None
        ),
        "mean_centered_excess_probability_per_release": (
            fmean(centered_event_totals) if centered_event_totals else None
        ),
        "raw_one_step_branching_spectral_radius_proxy": _power_spectral_radius(raw_columns),
        "centered_one_step_branching_spectral_radius_proxy": _power_spectral_radius(centered_columns),
        "operator_definition": "B_ij=<Rdt_i max(exp(delta log rate_i)-1,0)> over releases of parent j",
        "centering": "subtract median delta log rate across surviving contacts for each release",
        "limitations": [
            "one audited step only",
            "contact-to-contact response, not continuum elastic Green function",
            "common-mode centering is a diagnostic convention",
            "spectral radii are structural proxies and never production parameters",
        ],
    }
