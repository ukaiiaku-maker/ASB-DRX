from full_model.analysis.analyze_v67_asb_seed_bracket import classify


def _row(seed, ratio, persistence):
    return {
        "seed_peak_K": seed,
        "final_power_maximum_to_mean": ratio,
        "band_intervals": persistence,
        "terminal_band_persistence_intervals": persistence,
        "initial_power_maximum_to_mean": 1.0+.02*seed,
        "source_tree_clean": True,
        "configuration": {"n": 32, "temperature_seed_peak_K": seed},
    }


def test_amplitude_basin_is_bracketed():
    result = classify([
        _row(15.0, 1.8, 0), _row(17.5, 2.001, 1),
        _row(18.75, 2.11, 16), _row(19.375, 2.18, 23)])
    assert result["decision"]["finite_amplitude_basin_bracketed"]
    assert result["brackets"]["any_band_onset_seed_peak_K"] == {
        "last_negative": 15.0, "first_positive": 17.5}
    assert result["brackets"]["twenty_interval_persistence_seed_peak_K"] == {
        "last_negative": 18.75, "first_positive": 19.375}


def test_nonmonotone_response_is_rejected():
    result = classify([_row(10.0, 1.9, 0), _row(20.0, 1.8, 25)])
    assert not result["decision"]["finite_amplitude_basin_bracketed"]
