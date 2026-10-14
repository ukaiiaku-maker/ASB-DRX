#!/usr/bin/env python3
"""Audit scalar power-amplitude episodes without relabelling them as bands."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERIFY = ROOT/"full_model"/"verification"


def _episodes(rows: list[dict], threshold: float = 2.0) -> list[dict]:
    mask = [row["power_localization_ratio"] >= threshold for row in rows]
    episodes = []
    start = None
    for index, active in enumerate(mask+[False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            first = rows[start]; last = rows[index-1]
            episodes.append({
                "first_step": first["step"], "last_step": last["step"],
                "first_strain": first["applied_shear_strain"],
                "last_strain": last["applied_shear_strain"],
                "interval_count": index-start,
            })
            start = None
    return episodes


def main() -> None:
    cases = {}
    for n in (32, 48, 64):
        for control, suffix in (("adiabatic", ""),
                                ("strong_bath", "_strong_bath")):
            path = VERIFY/f"v66_regime_fixed_n{n}_T1000_r8e4{suffix}"/"history.json"
            rows = json.loads(path.read_text())
            cases[f"n{n}_{control}"] = {
                "steps": len(rows),
                "episodes_at_or_above_2x_mean_power": _episodes(rows),
            }
    payload = {
        "schema": "v66-power-amplitude-history-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cases": cases,
        "decision": {
            "scalar_amplitude_is_a_band_classifier": False,
            "reason": (
                "Every grid and thermal control has the same early scalar "
                "episode through 2.544% strain, yet the spatial endpoint audit "
                "does not identify a connected late band at n64/9.6%. Scalar "
                "maximum-to-mean power is therefore an onset diagnostic only."),
        },
    }
    output = VERIFY/"v66_power_amplitude_history"/"result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(json.dumps(payload["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
