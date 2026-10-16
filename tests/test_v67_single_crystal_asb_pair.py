from copy import deepcopy

from full_model.analysis.analyze_v67_single_crystal_asb_pair import classify


def _case(control):
    return {
        "configuration": {
            "n": 32, "thermal_control": control,
            "thermal_bath_rate_s": 0.0 if control == "adiabatic" else 5e9,
        },
        "source_tree_clean": True,
        "initial_state_digest": "same",
        "initial_temperature": {"contrast_K": 29.0},
        "initial_power_localization": {
            "maximum_to_mean": 1.5, "band_like": False},
        "final_power_localization": {"band_like": control == "adiabatic"},
        "result": {
            "maximum_mechanical_relative_first_law_residual": 5e-4,
            "grain_ids_unchanged": True,
            "terminal_band_persistence_intervals": (
                25 if control == "adiabatic" else 0),
            "final_temperature_contrast_K": (
                130.0 if control == "adiabatic" else 1e-10),
            "post_peak_stress_drop_fraction": (
                .06 if control == "adiabatic" else .005),
        },
    }


def test_causal_pair_qualifies_candidate():
    result = classify(_case("adiabatic"), _case("strong_bath"))
    assert result["decision"]["heat_triggered_single_crystal_asb_candidate"]
    assert not result["decision"]["strict_asb_qualified"]


def test_bath_band_fails_causal_classification():
    bath = deepcopy(_case("strong_bath"))
    bath["final_power_localization"]["band_like"] = True
    result = classify(_case("adiabatic"), bath)
    assert not result["decision"]["heat_triggered_single_crystal_asb_candidate"]
