from __future__ import annotations

import argparse
import json
from pathlib import Path

from asb_drx.fixtures import SingleGliderDDDParameterization

from run_local_boundary_matrix import run_condition


OBSERVABLES = (
    "final_mean_stress_Pa",
    "peak_mean_stress_Pa",
    "maximum_temperature_K",
    "maximum_matched_temperature_excess_K",
    "mean_density_change_m2",
    "final_child_order_fraction",
)


def _relative_change(coarse: dict, fine: dict, name: str) -> float:
    denominator = max(abs(float(fine[name])), abs(float(coarse[name])), 1.0e-30)
    return abs(float(fine[name]) - float(coarse[name])) / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=950.0)
    parser.add_argument("--rate", type=float, default=4500.0)
    parser.add_argument("--density-ratio", type=float, default=2.0)
    parser.add_argument("--target-shear", type=float, default=0.3)
    parser.add_argument("--retained-samples", type=int, default=75)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--execution-site", default="local")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = SingleGliderDDDParameterization()
    specifications = (
        ("time_coarse", 16, 75),
        ("time_medium_grid_coarse", 16, 150),
        ("reference", 16, 300),
        ("grid_medium", 24, 300),
        ("grid_fine", 32, 300),
    )
    records = {}
    for label, points, steps in specifications:
        print(f"starting {label}: points={points} steps={steps}", flush=True)
        records[label] = run_condition(
            args.temperature,
            args.rate,
            args.density_ratio,
            points,
            steps,
            args.target_shear,
            20000,
            fixture,
            args.retained_samples,
        )
        if records[label]["status"] != "complete":
            raise RuntimeError(f"refinement member {label} did not complete")
    comparisons = {}
    pairs = {
        "timestep_final": ("time_medium_grid_coarse", "reference"),
        "grid_final": ("grid_medium", "grid_fine"),
    }
    for label, (coarse_name, fine_name) in pairs.items():
        coarse = records[coarse_name]
        fine = records[fine_name]
        changes = {
            name: _relative_change(coarse, fine, name) for name in OBSERVABLES
        }
        comparisons[label] = {
            "coarse": coarse_name,
            "fine": fine_name,
            "relative_changes": changes,
            "maximum_relative_change": max(changes.values()),
            "below_five_percent": max(changes.values()) < 0.05,
            "same_localization_class": coarse["localized"] == fine["localized"],
        }
    report = {
        "schema": "asb-drx-local-antiplane-refinement/v1",
        "source_commit": args.source_commit,
        "execution_site": args.execution_site,
        "condition": {
            "temperature_K": args.temperature,
            "shear_rate_s_inv": args.rate,
            "density_ratio": args.density_ratio,
            "target_shear": args.target_shear,
            "retained_samples": args.retained_samples,
        },
        "recovery_design": {
            "reference_temperature_K": fixture.recovery_law().reference_temperature_K,
            "relaxation_time_ref_s": fixture.recovery_law().relaxation_time_ref_s,
            "activation_energy_J": fixture.recovery_law().activation_energy_J,
        },
        "records": records,
        "comparisons": comparisons,
        "passes_provisional_five_percent_gate": all(
            item["below_five_percent"] and item["same_localization_class"]
            for item in comparisons.values()
        ),
        "limitations": [
            "single deterministic perturbation and one generic boundary condition",
            "child-order fraction is not a physical DRX fraction",
            "absence of localization cannot establish convergence of onset or band width",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
