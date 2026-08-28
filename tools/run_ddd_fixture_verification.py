from __future__ import annotations

import json
from pathlib import Path

from asb_drx.fixtures import SingleGliderDDDParameterization


def main() -> None:
    fixture=SingleGliderDDDParameterization(); law=fixture.law()
    peaks=[]
    for temperature in (850.,900.,950.,1000.,1050.):
        peak=law.peak(temperature,fixture.source_strain_rate_s_inv)
        peaks.append({
            "temperature_K":temperature,
            "density_m2":peak.density_m2,
            "local_activation_stress_Pa":peak.local_activation_stress_Pa,
            "macroscopic_strength_Pa":peak.macroscopic_strength_Pa,
            "barrier_scale_J":law.barrier_scale_J(temperature),
        })
    report={
        "schema":"asb-drx-single-glider-ddd-fixture/v1",
        "scientific_disposition":"user-authorized generic parameterization and arbitrary-boundary source; not material calibration",
        "provenance":{"path":fixture.source_path,"sha256":fixture.source_sha256,"jobs_sha256":fixture.campaign_jobs_sha256},
        "parameters":fixture.__dict__,"analytical_peaks_at_ddd_rate":peaks,
        "spatial_parameters":fixture.spatial_parameters().__dict__,
    }
    output=Path("output/ddd_fixture_verification.json"); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")


if __name__=="__main__": main()
