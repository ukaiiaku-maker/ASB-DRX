from full_model.analysis.analyze_v65_stress_curvature import analyze


def test_consistent_quadratic_projection_is_not_called_observed_peak():
    peak_s = 1.15e-6
    rows = []
    for index in range(600):
        time_s = 1.0e-6+index*2.0e-10
        delta_us = (time_s-peak_s)*1e6
        stress_GPa = 4.0-1.2*delta_us**2
        rows.append({"physical_time_s": time_s,
                     "adaptive_interval": index,
                     "shear_stress_Pa": stress_GPa*1e9})

    result = analyze(rows, [], (64, 128, 256, 512), 1.25e-6)

    assert result["decision"]["classification"] == (
        "CONSISTENT_QUADRATIC_PEAK_PROJECTION_NOT_OBSERVED"
    )
    assert not result["decision"]["stress_peak_observed"]
    assert not result["decision"]["strict_asb_claim"]
    assert all(abs(fit["projected_peak_time_s"]-peak_s) < 1e-15
               for fit in result["fits"])
