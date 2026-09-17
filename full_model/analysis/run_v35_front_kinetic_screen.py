#!/usr/bin/env python3
"""Bounded V35 physical-site front-kinetics hypothesis screen.

This is not a calibration.  It evaluates the declared b^3,b microscopic event
law for 15 temperature/stored-energy hypotheses at exactly zero applied
pressure and reports the corresponding physical site/event measure.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.coupled_front_event import (
    FrontEnergyTerms, propose_bidirectional_front_event)
from full_model.production.front_event_measure import (
    combine_front_event_measures, physical_front_event_measure)
from full_model.production.moving_front import DefectState


TEMPERATURES_K = (900.0, 1100.0, 1300.0)
STORED_PRESSURES_PA = (-600e6, -200e6, 0.0, 200e6, 600e6)


def _equal_state():
    density = np.full((1, 1, 1), 2.5e17)
    return DefectState(
        density.copy(), density.copy(), density.copy(),
        np.full((1, 1), 1.0e17))


def run_screen(*, burgers_m=2.48e-10, interface_length_m=10e-6,
               represented_thickness_m=None, dt_s=1e-7):
    b = float(burgers_m)
    thickness = 2.0*b if represented_thickness_m is None else float(
        represented_thickness_m)
    event_volume = b**3
    process = ActivatedProcess(
        "v35-b3-b-front-screen", 1.0e8, drag_rate_s=1.0e8,
        negative_barrier_mode="drag")
    state = _equal_state()
    records = []
    for temperature in TEMPERATURES_K:
        for stored_pressure in STORED_PRESSURES_PA:
            # Zero external work.  The signed event free energies come only
            # from the stored-energy difference of the two physical states.
            delta_ab = -float(stored_pressure)*event_volume
            delta_ba = float(stored_pressure)*event_volume
            event = propose_bidirectional_front_event(
                state, state, event_volume_m3=event_volume,
                event_length_m=b, line_energy_J_m=0.0,
                temperature_K=temperature, process=process,
                h0_J=.35*EV_J, critical_pressure_Pa=1.0e9,
                exp_a=2.0, exp_n=1.5, exp_floor=.10,
                energy_a_to_b=FrontEnergyTerms(),
                energy_b_to_a=FrontEnergyTerms(),
                transmission_fraction=1.0,
                kinetic_free_energy_a_to_b_J=delta_ab,
                kinetic_free_energy_b_to_a_J=delta_ba)
            measure = physical_front_event_measure(
                interface_length_m=interface_length_m,
                represented_thickness_m=thickness, burgers_m=b,
                rate_a_to_b_per_site_s=event.rate_a_to_b_s,
                rate_b_to_a_per_site_s=event.rate_b_to_a_s,
                dt_s=dt_s, event_volume_m3=event_volume,
                event_length_m=b)
            records.append({
                "temperature_K": temperature,
                "stored_energy_pressure_Pa": stored_pressure,
                "applied_pressure_Pa": 0.0,
                "kinetic_free_energy_a_to_b_J": delta_ab,
                "kinetic_free_energy_b_to_a_J": delta_ba,
                "rate_a_to_b_per_site_s": event.rate_a_to_b_s,
                "rate_b_to_a_per_site_s": event.rate_b_to_a_s,
                "net_velocity_a_to_b_m_s": event.net_velocity_a_to_b_m_s,
                "detailed_balance_log_residual": (
                    event.detailed_balance_log_residual),
                "microscopic_reverse_pair": event.microscopic_reverse_pair,
                "physical_event_measure": measure.to_dict(),
            })

    baseline = physical_front_event_measure(
        interface_length_m=interface_length_m,
        represented_thickness_m=thickness, burgers_m=b,
        rate_a_to_b_per_site_s=3.0e6,
        rate_b_to_a_per_site_s=1.0e6, dt_s=dt_s)
    patch = combine_front_event_measures([
        physical_front_event_measure(
            interface_length_m=fraction*interface_length_m,
            represented_thickness_m=thickness, burgers_m=b,
            rate_a_to_b_per_site_s=3.0e6,
            rate_b_to_a_per_site_s=1.0e6, dt_s=dt_s)
        for fraction in (.2, .3, .5)])
    thick = physical_front_event_measure(
        interface_length_m=interface_length_m,
        represented_thickness_m=4.0*thickness, burgers_m=b,
        rate_a_to_b_per_site_s=3.0e6,
        rate_b_to_a_per_site_s=1.0e6, dt_s=dt_s)
    invariance = {
        "patch_site_count_relative_residual": (
            patch.physical_site_count/baseline.physical_site_count-1.0),
        "patch_swept_volume_relative_residual": (
            patch.expected_signed_swept_volume_m3
            /baseline.expected_signed_swept_volume_m3-1.0),
        "thickness_site_count_ratio": (
            thick.physical_site_count/baseline.physical_site_count),
        "thickness_velocity_relative_residual": (
            thick.expected_normal_velocity_m_s
            /baseline.expected_normal_velocity_m_s-1.0),
        "thickness_areal_site_density_relative_residual": (
            thick.site_count_per_interface_area_m2
            /baseline.site_count_per_interface_area_m2-1.0),
        "event_volume_is_b3": baseline.event_volume_m3 == b**3,
        "event_length_is_b": baseline.event_length_m == b,
        "site_area_is_b2": baseline.site_area_m2 == b**2,
    }
    return {
        "schema": "asb-drx/v35-front-kinetic-screen/v1",
        "created_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"),
        "classification": "BOUNDED_HYPOTHESIS_SCREEN_NOT_CALIBRATION",
        "event_baseline": {
            "burgers_m": b, "event_volume_m3": event_volume,
            "event_length_m": b, "site_area_m2": b**2,
            "attempt_frequency_per_site_s": process.attempt_frequency_s,
            "activation_h0_eV": .35, "critical_pressure_Pa": 1.0e9,
            "exp_a": 2.0, "exp_n": 1.5, "exp_floor": .10,
        },
        "hypotheses": [
            "zero stored-energy difference gives equal directional rates",
            "stored-energy sign reverses the preferred direction",
            "temperature changes rate magnitude without changing direction",
            "b3,b site measure is mesh and patch independent",
        ],
        "case_count": len(records), "records": records,
        "invariance_controls": invariance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_screen()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({
        "case_count": result["case_count"],
        "classification": result["classification"],
        "invariance_controls": result["invariance_controls"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
