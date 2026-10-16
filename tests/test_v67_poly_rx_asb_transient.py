from full_model.analysis.analyze_v67_poly_rx_asb_transient import classify


def _case(control, stage):
    ad = control == "adiabatic"
    onset = stage == "onset"
    return {
        "checkpoint_sha256": f"{control}-{stage}",
        "parent_checkpoint_sha256": (
            "root" if onset else f"{control}-onset"),
        "source_tree_clean": True,
        "step": 183 if onset else 233,
        "load": .02664 if onset else .03464,
        "thermal_control": control,
        "thermal_bath_rate_s": 0.0 if ad else 5e9,
        "front_transmission_fraction": .25,
        "active_physical_band_intervals": (
            3 if ad and onset else 32 if ad else 0 if onset else 19),
        "decision": {
            "all_hard_absorption_balances_passed": True,
            "mura_hard_balance_passed": True,
            "first_law_hard_balance_passed": True,
            "grain_ids_unchanged": True,
            "final_connected_band": ad and onset,
            "growing_owner_intersects_active_band": not (not ad and onset),
        },
    }


def test_transient_heat_advanced_coexistence_classifies_without_persistent_claim():
    result = classify(
        _case("adiabatic", "onset"), _case("strong_bath", "onset"),
        _case("adiabatic", "later"), _case("strong_bath", "later"))
    assert result["decision"]["transient_heat_advanced_rx_band_coexistence"]
    assert not result["decision"]["persistent_polycrystal_asb_qualified"]


def test_persistent_terminal_band_rejects_transient_dissolution_classification():
    later = _case("adiabatic", "later")
    later["decision"]["final_connected_band"] = True
    result = classify(
        _case("adiabatic", "onset"), _case("strong_bath", "onset"),
        later, _case("strong_bath", "later"))
    assert not result["decision"]["transient_heat_advanced_rx_band_coexistence"]
