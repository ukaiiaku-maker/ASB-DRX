from full_model.analysis.postprocess_v59_rate_exposure import compare


def _record(strain, time):
    row = {
        "additional_engineering_shear": strain, "physical_time_s": time,
        "post_front_equilibrated_stress_Pa": 1.0, "temperature_mean_K": 2.0,
        "temperature_max_minus_mean_K": 3.0,
        "temperature_max_minus_min_K": 4.0, "power_participation": .2,
        "power_width_minor_m": 1e-6, "power_aspect_ratio": 4.0,
        "fresh_sweep_volume_m3": 1e-21,
    }
    return row


def test_common_strain_and_time_are_distinct_contracts():
    slow = _record(.01, 5e-6); fast = _record(.01, .5e-6)
    assert compare(slow, fast, mode="common_strain")["exposure_matched"]
    assert not compare(slow, fast, mode="common_time")["exposure_matched"]
