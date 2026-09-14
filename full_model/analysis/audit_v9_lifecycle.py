#!/usr/bin/env python3
"""Build the v9 lifecycle ledger from deterministic v8 checkpoint series."""

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def scalar(z, name, default=0.0):
    return float(z[name]) if name in z.files else float(default)


def checkpoint_record(path):
    with np.load(path, allow_pickle=True) as z:
        params = json.loads(str(z["P_json"].item()))
        area = (float(params["L_phys"])/int(params["Nx"]))**2
        thickness = float(params.get("nuc_barrier_thickness_b", 2.0))*float(params["b"])
        factor = area*thickness
        rp, rm = np.asarray(z["rp"]), np.asarray(z["rm"])
        forest = np.asarray(z["rho_forest"])
        wall = np.asarray(z["rho_wall"])
        gb = np.asarray(z["rho_GB"])
        eta = np.asarray(z["eta"])
        ng = int(z["Ng"])
        child = np.zeros(eta.shape[:2])
        if ng > 12:
            h = eta**2*(3.0-2.0*eta)
            child = h[:, :, -1]/np.maximum(np.sum(h, axis=2), 1e-300)
        event_records = json.loads(str(z["atomic_promotion_events_json"].item()))
        event = event_records[-1] if event_records else {}
        return {
            "step": int(z["step"]), "time_s": scalar(z, "sim_time"),
            "stress_Pa": scalar(z, "sigma_bar"),
            "temperature_mean_K": float(np.mean(z["T"])),
            "mobile_line_m": float(np.sum(rp+rm, dtype=np.longdouble)*factor),
            "forest_line_m": float(np.sum(forest, dtype=np.longdouble)*factor),
            "wall_line_m": float(np.sum(wall, dtype=np.longdouble)*factor),
            "boundary_line_m": float(np.sum(gb, dtype=np.longdouble)*factor),
            "signed_abs_line_m": float(np.sum(np.abs(rp-rm), dtype=np.longdouble)*factor),
            "child_fraction_integral_m2": float(np.sum(child)*area),
            "child_pure_core_area_m2": float(np.count_nonzero(child >= 0.8)*area),
            "atomic_commit_total": int(z["atomic_promotion_commit_total"]),
            "atomic_event": event,
        }


def audit_variant(directory):
    paths = sorted(glob.glob(str(Path(directory)/"drx_v25_restart_*.npz")))
    records = [checkpoint_record(path) for path in paths]
    commits = [r for r in records if r["atomic_commit_total"]]
    pre = next((r for r in reversed(records)
                if not r["atomic_commit_total"]), records[0])
    first = commits[0] if commits else None
    final = records[-1]
    peak = max(records, key=lambda r: r["child_pure_core_area_m2"])
    event = first["atomic_event"] if first else {}
    physical = peak["child_pure_core_area_m2"] > 1e-11
    collapsed = bool(first and final["child_pure_core_area_m2"] == 0.0)
    irreversible = float(event.get("line_content_pair_annihilation_m", 0.0)) > 0.0
    if physical:
        classification = "PHYSICAL_GRAIN_FORMATION"
    elif collapsed and irreversible:
        classification = "IRREVERSIBLE_CLEANUP_WITH_FAILED_CHILD"
    elif first and irreversible:
        classification = "RECOVERY_EVENT_WITH_TRANSIENT_CHILD_SUPPORT"
    else:
        classification = "REVERSIBLE_PHASE_EXCURSION_WITH_NO_NET_CLEANUP"
    return {
        "classification": classification,
        "pre_promotion": pre, "first_post_promotion": first,
        "peak_child_support": peak, "final": final,
        "checkpoint_series": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("v8_root")
    parser.add_argument("output")
    args = parser.parse_args()
    result = {
        "schema": "full-v34-v9-lifecycle-audit/v1",
        "represented_thickness": "nuc_barrier_thickness_b*b",
        "variants": {
            name: audit_variant(Path(args.v8_root)/name)
            for name in ("lineage_scoped", "common_variational",
                         "disabled", "sign_reversed")},
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
