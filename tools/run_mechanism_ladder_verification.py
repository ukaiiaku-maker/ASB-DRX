from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

from asb_drx.analytical import ExpFloorLaw
from asb_drx.localization import LocalizationCriteria, localization_history
from asb_drx.mechanism_ladder import (
    classify_mechanism_trace, matched_isothermal_case, run_mechanism_trace,
    standard_mechanism_ladder,
)
from asb_drx.multi_order import BinaryCircularLimit, diffuse_binary_circle
from asb_drx.spatial_coupled import SpatialCoupledParameters, SpatialCoupledState

EV_J = 1.602176634e-19


def main() -> None:
    law = ExpFloorLaw(1.5*EV_J, 1.2e9, 1000.0, .2, 2.0, 2.5, 1e12, 4.0, 2.5e-10)
    p = SpatialCoupledParameters(8e10, 3.5e6, 5.0, 5e-9, 1e14, 2e6, 1e-6, 5e-7)
    points = 16; dx_m = 1.6e-5/points; steps = 20; proposed_dt_s = 1e-5
    interface = 2*math.sqrt(p.gradient_coefficient_J_m/p.pair_penalty_J_m3)
    boundary = math.sqrt(p.gradient_coefficient_J_m*p.pair_penalty_J_m3)/3
    limit = BinaryCircularLimit(boundary,p.stored_line_energy_J_m*4e13,1.0)
    eta = diffuse_binary_circle(points,dx_m,1.35*limit.critical_radius_m,interface)
    rho=np.empty_like(eta); rho[0]=5e13; rho[1]=1e13
    coordinate=np.linspace(0,2*math.pi,points,endpoint=False)
    temperature=1000.+.25*np.sin(coordinate)[None,:]
    temperature=np.broadcast_to(temperature,(points,points)).copy()
    initial=SpatialCoupledState(1e8,0,np.zeros((points,points)),temperature,rho,eta)
    criteria=LocalizationCriteria(.4,20.,.1,3.,3,.05)
    records=[]
    for case in standard_mechanism_ladder(10.,1000.):
        trace=run_mechanism_trace(initial,case,dx_m,proposed_dt_s,steps,law,p)
        control_case=matched_isothermal_case(case)
        if not case.controls.evolve_temperature:
            control=trace
        else:
            control=run_mechanism_trace(initial,control_case,dx_m,proposed_dt_s,steps,law,p)
        decision=classify_mechanism_trace(trace,control,dx_m,interface,criteria)
        history=localization_history(trace.plastic_rate_s_inv,trace.temperature_K,control.temperature_K,trace.stress_Pa,dx_m)
        final=trace.states[-1]
        records.append({
            "name":case.name,
            "applied_shear_rate_s_inv":case.applied_shear_rate_s_inv,
            "evolve_temperature":case.controls.evolve_temperature,
            "evolve_phase":case.controls.evolve_phase,
            "unload_initial_stress":case.unload_initial_stress,
            "matched_control":control.case.name,
            "localized":decision.localized,
            "failed_criteria":list(decision.failed_criteria),
            "minimum_active_fraction":min(item.active_fraction for item in history),
            "maximum_temperature_excess_K":max(item.temperature_excess_K for item in history),
            "maximum_softening_fraction":max(item.softening_fraction for item in history),
            "minimum_effective_width_m":min(item.effective_width_m for item in history),
            "final_stress_Pa":final.stress_Pa,
            "maximum_temperature_K":float(np.max(trace.temperature_K)),
            "phase_change_l2":float(np.linalg.norm(final.eta_fields-initial.eta_fields)),
            "cumulative_external_work_J_m3":sum(item.external_work_J_m3 for item in trace.ledgers),
            "cumulative_bath_heat_J_m3":sum(item.bath_heat_J_m3 for item in trace.ledgers),
            "cumulative_global_closure_error_J_m3":sum(item.global_closure_error_J_m3 for item in trace.ledgers),
        })
    report={
        "schema":"asb-drx-mechanism-ladder-verification/v1",
        "scientific_disposition":"generic matched common-equation controls; no material calibration or ASB/DRX validation",
        "points":points,"dx_m":dx_m,"steps":steps,"proposed_dt_s":proposed_dt_s,
        "interface_width_m":interface,
        "localization_criteria":criteria.__dict__,
        "cases":records,
    }
    output=Path("output/mechanism_ladder_verification.json"); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")


if __name__ == "__main__": main()
