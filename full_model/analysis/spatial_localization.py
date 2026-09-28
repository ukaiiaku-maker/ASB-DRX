"""Outcome-neutral spatial concentration and periodic connectivity metrics."""

from __future__ import annotations

import numpy as np


def _periodic_components(mask):
    nx, ny = mask.shape
    unseen = set(map(tuple, np.argwhere(mask)))
    components = []
    while unseen:
        start = unseen.pop()
        queue = [start]
        unwrapped = {start: np.asarray(start, dtype=int)}
        winding = np.zeros(2, dtype=bool)
        while queue:
            cell = queue.pop()
            base = unwrapped[cell]
            for shift in ((1, 0), (-1, 0), (0, 1), (0, -1),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)):
                raw = base+shift
                neighbor = (int(raw[0] % nx), int(raw[1] % ny))
                if not mask[neighbor]:
                    continue
                if neighbor not in unwrapped:
                    unwrapped[neighbor] = raw
                    unseen.discard(neighbor)
                    queue.append(neighbor)
                else:
                    mismatch = raw-unwrapped[neighbor]
                    winding |= np.abs(mismatch) >= np.asarray((nx, ny))
        coordinates = np.stack(tuple(unwrapped.values()))
        span = np.ptp(coordinates, axis=0)+1
        components.append((len(unwrapped), span, winding))
    return components


def spatial_localization_metrics(field, threshold_multiple=2.0):
    """Describe signed signal, concentration, and connected hot structure."""
    value = np.asarray(field, dtype=float)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("localization field must be a finite 2-D array")
    positive = np.maximum(value, 0.0)
    mean = float(np.mean(positive))
    threshold = float(threshold_multiple)*mean
    mask = (positive >= threshold if threshold > 0.0 else
            np.zeros(positive.shape, dtype=bool))
    components = _periodic_components(mask)
    if components:
        count, span, winding = max(components, key=lambda item: item[0])
    else:
        count = 0; span = np.zeros(2); winding = np.zeros(2, dtype=bool)
    return {
        "signed_min_W_m3": float(np.min(value)),
        "positive_mean_W_m3": mean,
        "positive_max_W_m3": float(np.max(positive)),
        "maximum_to_mean": float(np.max(positive)/max(mean, 1e-300)),
        "above_threshold_fraction": float(np.mean(mask)),
        "threshold_multiple": float(threshold_multiple),
        "largest_component_cells": int(count),
        "largest_component_fraction": float(count/value.size),
        "component_span_fraction": [
            float(span[0]/value.shape[0]), float(span[1]/value.shape[1])],
        "periodic_winding": [bool(winding[0]), bool(winding[1])],
        "band_like": bool(
            count >= min(value.shape)
            and (np.max(span/np.asarray(value.shape)) >= .5
                 or np.any(winding))),
    }
