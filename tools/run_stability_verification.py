from __future__ import annotations

import json
import math
from pathlib import Path

from asb_drx.analytical import ExpFloorLaw
from asb_drx.stability import StabilityParameters, common_stress_rate_tangents, thermal_storage_mode

EV_J=1.602176634e-19


def main() -> None:
    law=ExpFloorLaw(1.5*EV_J,1.2e9,1000.,.2,2.,2.5,1e12,4.,2.5e-10,.3,.1)
    p=StabilityParameters(3.5e6,5.,5e-9,1e14)
    stress=3e8; rho=5e13; temperature=1000.; domain_m=1.6e-5
    tangent=common_stress_rate_tangents(law,stress,rho,temperature)
    modes=[]
    for index in (1,2,4,8):
        wavenumber=2*math.pi*index/domain_m
        mode=thermal_storage_mode(law,stress,rho,temperature,wavenumber,p)
        modes.append({
            "mode_index":index,"wavenumber_m_inv":wavenumber,
            "wavelength_m":domain_m/index,
            "jacobian_s_inv":mode.jacobian_s_inv.tolist(),
            "eigenvalues_s_inv":[{"real":float(x.real),"imag":float(x.imag)} for x in mode.eigenvalues_s_inv],
            "maximum_growth_rate_s_inv":mode.maximum_growth_rate_s_inv,
            "unstable":mode.maximum_growth_rate_s_inv>0,
        })
    report={
        "schema":"asb-drx-thermal-storage-stability/v1",
        "scientific_disposition":"generic frozen-common-stress finite-wavenumber tangent; not material calibration or nonlinear ASB prediction",
        "state":{"macroscopic_stress_Pa":stress,"density_m2":rho,"temperature_K":temperature,"domain_m":domain_m},
        "rate_tangents":tangent.__dict__,"modes":modes,
    }
    output=Path("output/stability_verification.json"); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")


if __name__=="__main__": main()
