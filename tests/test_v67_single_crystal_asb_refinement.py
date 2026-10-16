from copy import deepcopy

from full_model.analysis.analyze_v67_single_crystal_asb_refinement import classify


def _case(n, dt, steps, control, *, scale=1.0):
    ad = control == "adiabatic"
    return {
        "source_tree_clean": True,
        "configuration": {
            "n": n, "initial_shear_strain": .012,
            "shear_rate_s": 8e4, "dt_s": dt, "target_steps": steps},
        "initial_state_digest": f"n{n}",
        "result": {
            "maximum_mechanical_relative_first_law_residual": 5e-4,
            "final_temperature_contrast_K": 132.0*scale if ad else 1e-9,
            "post_peak_stress_drop_fraction": .057*scale if ad else .005,
        },
        "final_power_localization": {
            "band_like": ad, "maximum_to_mean": 2.24*scale if ad else 1.0},
        "band_onset_strain": .119*scale if ad else None,
    }


def test_converged_heat_triggered_pair_qualifies():
    result = classify(
        _case(32, 2e-9, 700, "adiabatic"),
        _case(32, 2e-9, 700, "bath"),
        _case(48, 2e-9, 700, "adiabatic", scale=1.001),
        _case(48, 2e-9, 700, "bath"),
        _case(32, 1e-9, 1400, "adiabatic", scale=.999),
        _case(32, 1e-9, 1400, "bath"))
    assert result["decision"]["strict_prepared_asb_qualified"]


def test_bath_band_blocks_qualification():
    bath = _case(48, 2e-9, 700, "bath")
    bath["final_power_localization"]["band_like"] = True
    result = classify(
        _case(32, 2e-9, 700, "adiabatic"),
        _case(32, 2e-9, 700, "bath"),
        _case(48, 2e-9, 700, "adiabatic"), bath,
        _case(32, 1e-9, 1400, "adiabatic"),
        _case(32, 1e-9, 1400, "bath"))
    assert not result["decision"]["strict_prepared_asb_qualified"]
