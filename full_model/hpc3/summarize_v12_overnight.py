#!/usr/bin/env python3
"""Aggregate all independently finalized v12 trajectories."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

case_root, status_root, manifest_path, csv_path, json_path = map(Path, sys.argv[1:6])
records = []
for final in sorted(status_root.glob("*/final.json")):
    records.append(json.loads(final.read_text()))
fields = ["case_id", "tier", "classification", "hard_invariants_passed", "grid",
          "steps_requested", "rate_s-1", "temperature_K", "radius_um", "window_um",
          "mobility_multiplier", "drag_pressure_Pa", "wallclock_seconds",
          "final_checkpoint_sha256"]
with open(csv_path, "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for record in records:
        writer.writerow({key: record.get(key) for key in fields})
by_id = {r["case_id"]: r for r in records}
def tip(case):
    r = by_id.get(case)
    if not r or not r.get("last_contour"): return None
    return float(r["last_contour"]["bulge_tip_displacement_m"])-float(r["first_contour"]["bulge_tip_displacement_m"])
a1, a2 = tip("A1_baseline_bulge"), tip("A2_no_bulge")
normal_relative = None if a1 is None or a2 is None else a1-a2
criticality = all(by_id.get(x, {}).get("classification") == "SIBM_CRITICALITY_CONTROL_PASSED"
                  for x in ("E1_subcritical_fixture", "E2_supercritical_fixture"))
decision = {
    "schema": "full-v34-v12-sibm-comparison/v1", "cases": records,
    "case_count": len(records), "all_hard_invariants_passed": bool(records) and
        all(r["hard_invariants_passed"] for r in records),
    "baseline_tip_relative_to_control_m": normal_relative,
    "criticality_fixture_passed": criticality,
    "additional_pair_cases": 0,
    "additional_pair_reason": "source has no second pair with >=250 pure-core cells on both sides",
}
matched = {
    "A1_baseline_bulge": "A2_no_bulge", "A2_no_bulge": "A1_baseline_bulge",
    "D1_grid192_bulge": "D2_grid192_control", "D2_grid192_control": "D1_grid192_bulge",
}
for record in records:
    flags = record["scientific_flags"]
    flags["matched_control_available"] = record["case_id"] in matched and matched[record["case_id"]] in by_id
    if record["case_id"] == "A1_baseline_bulge" and normal_relative is not None:
        flags["normal_growth_relative_to_control"] = normal_relative > 0.0
    flags["full_model_sibm_mechanism_supported"] = False
decision["cases"] = records
json_path.write_text(json.dumps(decision, indent=2, sort_keys=True)+"\n")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for record in records:
    path = case_root/record["case_id"]/"sibm_contour_diagnostics.csv"
    if not path.exists(): continue
    rows = list(csv.DictReader(open(path)))
    if not rows: continue
    time = np.array([float(r["time_s"]) for r in rows])*1e3
    axes[0].plot(time, np.array([float(r["bulge_tip_displacement_m"]) for r in rows])*1e6,
                 label=record["case_id"], lw=.8)
    axes[1].plot(time, np.array([float(r["excess_bulge_area_m2"]) for r in rows])*1e12,
                 label=record["case_id"], lw=.8)
axes[0].set(xlabel="additional time (ms)", ylabel="tip displacement (um)")
axes[1].set(xlabel="additional time (ms)", ylabel="excess area (um2)")
axes[1].legend(fontsize=5, ncol=2)
fig.tight_layout(); fig.savefig(json_path.parent/"v12_contour_tip_area.png", dpi=160); plt.close(fig)

a1_record = by_id.get("A1_baseline_bulge")
if a1_record and a1_record.get("final_checkpoint"):
    checkpoint = case_root/a1_record["case_id"]/Path(a1_record["final_checkpoint"]).name
    with np.load(checkpoint, allow_pickle=True) as z:
        boundary = a1_record.get("boundary", {}); p=boundary.get("parent_label",3); c=boundary.get("child_label",5)
        panels = [(z["eta"][:,:,c]-z["eta"][:,:,p], "eta_child-eta_parent"),
                  (np.log10(np.maximum(z["rho"],1)), "log10 total density"),
                  (z["kappa_tot"], "signed/GND content"), (z["T"], "temperature (K)")]
    fig, axes = plt.subplots(2,2,figsize=(9,8))
    for ax,(field,title) in zip(axes.flat,panels):
        image=ax.imshow(field.T,origin="lower",cmap="coolwarm"); ax.set_title(title); fig.colorbar(image,ax=ax)
    fig.tight_layout(); fig.savefig(json_path.parent/"v12_representative_fields.png",dpi=160); plt.close(fig)
(json_path.parent/"v12_decision_note.md").write_text(
    "# Directive v12 decision\n\n"
    f"Finalized {len(records)} cases. Baseline tip displacement relative to its matched control: "
    f"{normal_relative!r} m. Criticality fixture passed: {criticality}. "
    "The full SIBM support decision requires all declared comparisons and remains false unless the comparison JSON demonstrates them.\n")
manifest = {
    "schema": "full-v34-v12-overnight-manifest/v1",
    "scientific_source_sha": __import__('os').environ.get("SCIENTIFIC_SOURCE_SHA", "unknown"),
    "case_count_finalized": len(records), "case_final_sha256": {},
    "remaining_cases": (status_root/"remaining_cases.txt").read_text().splitlines(),
}
import hashlib
for final in sorted(status_root.glob("*/final.json")):
    manifest["case_final_sha256"][final.parent.name] = hashlib.sha256(final.read_bytes()).hexdigest()
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
