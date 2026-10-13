from full_model.analysis.monitor_v65_peak_forecast import evaluate


def forecast():
    return {"preterminal_stress_peak_projection": {
        "projected_peak_times_s": [1.1465e-6, 1.1468e-6],
    }}


def test_forecast_validation_waits_for_predeclared_post_peak_lag():
    rows = [
        {"physical_time_s": 1.140e-6, "adaptive_interval": 1,
         "shear_stress_Pa": 3.99e9},
        {"physical_time_s": 1.147e-6, "adaptive_interval": 2,
         "shear_stress_Pa": 4.00e9},
        {"physical_time_s": 1.150e-6, "adaptive_interval": 3,
         "shear_stress_Pa": 3.999e9},
    ]
    result = evaluate(rows, forecast(), confirmation_lag_s=5e-9,
                      forecast_tolerance_s=2e-9)
    assert result["classification"] == "PEAK_FORECAST_VALIDATION_PENDING"


def test_forecast_validation_confirms_only_local_scalar_peak():
    rows = [
        {"physical_time_s": 1.140e-6, "adaptive_interval": 1,
         "shear_stress_Pa": 3.99e9},
        {"physical_time_s": 1.147e-6, "adaptive_interval": 2,
         "shear_stress_Pa": 4.00e9},
        {"physical_time_s": 1.153e-6, "adaptive_interval": 3,
         "shear_stress_Pa": 3.998e9},
    ]
    result = evaluate(rows, forecast(), confirmation_lag_s=5e-9,
                      forecast_tolerance_s=2e-9)
    assert result["classification"] == "FORECAST_VALIDATED_LOCAL_SCALAR_PEAK"
    assert result["local_scalar_peak_confirmed"]
    assert not result["strict_asb_claim"]
