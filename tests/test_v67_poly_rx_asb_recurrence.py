from full_model.analysis.analyze_v67_poly_rx_asb_recurrence import classify


def _case(control, stage):
    adiabatic = control == "adiabatic"
    final_band = stage == 0 and adiabatic or stage == 2
    return {
        "checkpoint_sha256": f"{control}-{stage}",
        "parent_checkpoint_sha256": (
            "root" if stage == 0 else f"{control}-{stage - 1}"),
        "source_tree_clean": True,
        "step": (183, 233, 333)[stage],
        "load": (.02664, .03464, .05064)[stage],
        "thermal_control": control,
        "thermal_bath_rate_s": 0.0 if adiabatic else 5e9,
        "front_transmission_fraction": .25,
        "active_physical_band_intervals": (
            (3, 32, 8) if adiabatic else (0, 19, 6))[stage],
        "decision": {
            "all_hard_absorption_balances_passed": True,
            "mura_hard_balance_passed": True,
            "first_law_hard_balance_passed": True,
            "grain_ids_unchanged": True,
            "final_connected_band": final_band,
            "growing_owner_intersects_active_band": stage > 0 or adiabatic,
        },
    }


def test_recurrent_pair_separates_mechanical_recurrence_from_strict_asb():
    result = classify(
        [_case("adiabatic", stage) for stage in range(3)],
        [_case("strong_bath", stage) for stage in range(3)])
    assert result["decision"]["recurrent_mechanical_rx_band_coexistence"]
    assert not result["decision"]["strict_heat_triggered_terminal_polycrystal_asb"]
    assert not result["decision"]["persistent_polycrystal_asb_qualified"]


def test_broken_checkpoint_chain_rejects_recurrence_qualification():
    bath = [_case("strong_bath", stage) for stage in range(3)]
    bath[-1]["parent_checkpoint_sha256"] = "wrong-parent"
    result = classify(
        [_case("adiabatic", stage) for stage in range(3)], bath)
    assert not result["decision"]["recurrent_mechanical_rx_band_coexistence"]


def test_requires_three_matched_stages():
    try:
        classify([_case("adiabatic", 0)], [_case("strong_bath", 0)])
    except ValueError:
        pass
    else:
        raise AssertionError("short trajectory must be rejected")
