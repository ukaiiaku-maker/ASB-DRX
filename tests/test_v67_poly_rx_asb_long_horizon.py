from full_model.analysis.analyze_v67_poly_rx_asb_long_horizon import classify


def _case(control, stage, *, soften=True, persistent=None):
    adiabatic = control == "adiabatic"
    if persistent is None:
        persistent = stage == 2 and adiabatic
    peak = (1.0, 2.0, 1.8 if soften and adiabatic else 2.0)[stage]
    return {
        "checkpoint_sha256": f"{control}-{stage}",
        "parent_checkpoint_sha256": (
            "root" if stage == 0 else f"{control}-{stage - 1}"),
        "source_tree_clean": True,
        "step": (100, 200, 300)[stage],
        "load": (.02, .04, .06)[stage],
        "thermal_control": control,
        "thermal_bath_rate_s": 0.0 if adiabatic else 5e9,
        "target_steps": 100,
        "mean_shear_stress_Pa": peak,
        "temperature_range_K": 50.0 if adiabatic else 1.0,
        "decision": {
            "all_hard_absorption_balances_passed": True,
            "mura_hard_balance_passed": True,
            "first_law_hard_balance_passed": True,
            "grain_ids_unchanged": True,
            "final_connected_band": persistent,
            "band_persisted_every_interval": persistent,
            "growing_owner_intersects_active_band": persistent,
        },
    }


def _series(control, **kwargs):
    return [_case(control, stage, **kwargs) for stage in range(3)]


def test_heat_dependent_persistent_asb_with_rx_qualifies():
    result = classify(_series("adiabatic"), _series("strong_bath"))
    assert result["decision"]["heat_dependent_polycrystal_asb_with_rx"]
    assert result["decision"]["strict_heat_triggered_terminal_band"]


def test_geometric_band_without_softening_does_not_qualify():
    result = classify(
        _series("adiabatic", soften=False), _series("strong_bath"))
    assert result["decision"][
        "persistent_physical_band_with_existing_boundary_rx"]
    assert not result["decision"]["heat_dependent_polycrystal_asb_with_rx"]
    assert result["decision"]["classification"] == (
        "PERSISTENT_RX_BAND_WITHOUT_ASB_SOFTENING")


def test_bath_band_separates_strict_trigger_from_thermal_asb():
    bath = _series("strong_bath")
    bath[-1]["decision"]["final_connected_band"] = True
    bath[-1]["decision"]["band_persisted_every_interval"] = True
    result = classify(_series("adiabatic"), bath)
    assert result["decision"]["heat_dependent_polycrystal_asb_with_rx"]
    assert not result["decision"]["strict_heat_triggered_terminal_band"]
