import numpy as np

from full_model.analysis.postprocess_v59_physical_asb import (
    PhysicalASBCriteria, _verify_refinement_certificate, _wall_parameters,
    classify_physical_episode,
    component_morphology, periodic_identity,
    component_temperature_excess,
    source_association,
    temperature_intervention_certificate,
)


def test_refinement_requires_source_bound_fixed_scale_comparison_records():
    digest = "a"*64
    valid = {
        "passed": True, "fixed_physical_scales_verified": True,
        "comparison_records": [{
            "coarse_checkpoint_sha256": digest,
            "fine_checkpoint_sha256": digest,
            "common_physical_time_s": 2.5e-6,
        }],
    }
    assert _verify_refinement_certificate(valid)["passed"]
    invalid = dict(valid, comparison_records=[])
    assert not _verify_refinement_certificate(invalid)["passed"]
    assert not _verify_refinement_certificate(None)["passed"]


def _row(step, time_s, stress=100.0, contrast=100.0, matched=100.0,
         participation=.1, width=3.0, aspect=4.0, colocation=.8):
    return {
        "step": step, "physical_time_s": time_s,
        "post_front_equilibrated_stress_Pa": stress,
        "spacing_m": 1.0, "interface_width_m": 1.0,
        "domain_length_m": 20.0,
        "temperature_max_minus_mean_K": contrast,
        "matched_temperature_excess_K": matched,
        "plastic_power": {"inverse_participation_fraction": participation},
        "power_width_minor_m": width, "power_aspect_ratio": aspect,
        "heat_power_component_overlap": colocation,
    }


def _decision(rows, refinement=True):
    mask = np.zeros((8, 8), dtype=bool); mask[:, 3] = True
    return classify_physical_episode(
        rows, {row["step"]: mask for row in rows}, PhysicalASBCriteria(),
        matched_control_available=True, refinement_passed=refinement)


def test_two_picosecond_episode_cannot_pass_physical_persistence():
    rows = [_row(0, 0.0, 100.0), _row(1, 1e-12, 70.0),
            _row(2, 2e-12, 70.0)]
    assert not _decision(rows)["inherited_strict_asb"]


def test_temperature_after_band_does_not_complete_prior_episode():
    rows = [_row(0, 0.0, 100.0, contrast=1.0),
            _row(1, 1e-6, 70.0, contrast=1.0),
            _row(2, 2e-6, 70.0, participation=.9)]
    assert not _decision(rows)["inherited_strict_asb"]


def test_monotonic_hardening_fails_post_peak_softening():
    rows = [_row(i, i*.5e-6, stress=100.0+i) for i in range(4)]
    assert not _decision(rows)["inherited_strict_asb"]


def test_same_200ns_episode_is_sampling_count_invariant():
    two = [_row(0, 0.0, 100.0), _row(1, .2e-6, 70.0)]
    three = [_row(0, 0.0, 100.0), _row(1, .1e-6, 70.0),
             _row(2, .2e-6, 70.0)]
    assert not _decision(two)["inherited_strict_asb"]
    assert not _decision(three)["inherited_strict_asb"]


def test_gap_and_component_switch_break_episode():
    rows = [_row(0, 0.0, 100.0), _row(1, .5e-6, 70.0),
            _row(2, 2e-6, 70.0)]
    assert not _decision(rows)["inherited_strict_asb"]


def test_isotropic_patch_fails_registered_elongation():
    rows = [_row(0, 0.0, 100.0, aspect=1.0),
            _row(1, 1e-6, 70.0, aspect=1.0)]
    assert not _decision(rows)["inherited_strict_asb"]


def test_periodic_component_identity_recovers_wrapped_translation():
    left = np.zeros((8, 8), dtype=bool); left[:, 0] = True
    right = np.roll(left, 3, axis=1)
    result = periodic_identity(left, right, 2.0)
    assert result["overlap"] == 1.0
    assert abs(result["displacement_m"][1]) == 6.0


def test_component_identity_rejects_an_unphysical_translation_but_retains_diagnostic():
    left = np.zeros((16, 16), dtype=bool); left[:, 1] = True
    right = np.roll(left, 5, axis=1)
    result = periodic_identity(
        left, right, 1.0, maximum_displacement_m=1.0)
    assert result["overlap"] == 0.0
    assert result["unrestricted_alignment_diagnostic"]["overlap"] == 1.0


def test_component_morphology_resolves_band_without_background_bias():
    field = np.ones((32, 32)); field[:, 14:18] = 20.0
    component = field > field.mean()+field.std()
    morphology = component_morphology(
        field, component, field.mean()+field.std(), 1.0)
    assert morphology["status"] == "DEFINED_THRESHOLD_COMPONENT_EXCESS"
    assert morphology["aspect_ratio"] > 3.0
    # The transverse width follows the four-cell component, not the box-sized
    # positive background.
    assert morphology["second_moment_widths"]["minor_gaussian_fwhm_m"] < 4.0


def test_component_morphology_keeps_square_patch_isotropic():
    field = np.ones((32, 32)); field[8:16, 8:16] = 20.0
    component = field > field.mean()+field.std()
    morphology = component_morphology(
        field, component, field.mean()+field.std(), 1.0)
    assert np.isclose(morphology["aspect_ratio"], 1.0)


def test_diagonal_winding_band_width_is_translation_invariant():
    n = 48
    row, column = np.indices((n, n))
    distance = (column-row+n/2) % n-n/2
    field = 1.0+20.0*np.exp(-.5*(distance/1.5)**2)
    component = np.abs(distance) <= 4.0
    threshold = 1.0
    reference = component_morphology(field, component, threshold, 2.0)
    translated = component_morphology(
        np.roll(field, (9, -7), axis=(0, 1)),
        np.roll(component, (9, -7), axis=(0, 1)), threshold, 2.0)
    assert reference["second_moment_widths"]["topology"] == "rank_one_winding"
    assert reference["second_moment_widths"]["winding_vectors"] == [[1, 1]]
    assert np.isclose(
        reference["second_moment_widths"]["minor_gaussian_fwhm_m"],
        translated["second_moment_widths"]["minor_gaussian_fwhm_m"],
        rtol=2e-14)
    assert np.isclose(reference["aspect_ratio"], translated["aspect_ratio"],
                      rtol=2e-14)


def test_curved_periodic_band_retains_winding_under_translation():
    n = 64
    row, column = np.indices((n, n))
    center = row+4.0*np.sin(2.0*np.pi*row/n)
    distance = (column-center+n/2) % n-n/2
    field = 1.0+15.0*np.exp(-.5*(distance/1.75)**2)
    component = np.abs(distance) <= 4.0
    first = component_morphology(field, component, 1.0, 1.0)
    second = component_morphology(
        np.roll(field, (11, 13), axis=(0, 1)),
        np.roll(component, (11, 13), axis=(0, 1)), 1.0, 1.0)
    assert first["second_moment_widths"]["topology"] == "rank_one_winding"
    assert np.isclose(first["power_aspect_ratio"] if "power_aspect_ratio" in first
                      else first["aspect_ratio"], second["aspect_ratio"], rtol=1e-13)


def test_matched_temperature_excess_must_be_on_power_component():
    baseline = np.zeros((8, 8)); control = np.zeros((8, 8))
    component = np.zeros((8, 8), dtype=bool); component[:, 3] = True
    baseline[0, 0] = 100.0
    baseline[:, 3] = 20.0
    result = component_temperature_excess(baseline, control, component)
    assert result["component_maximum_K"] == 20.0
    assert result["whole_field_maximum_diagnostic_K"] == 100.0


def test_postprocess_reconstructs_independent_temperature_controls():
    configuration = {
        "maximum_fraction_per_step": .1, "thermal_diffusivity_m2_s": 1e-8,
        "temperature_K": 900.0, "flow_temperature_mode": "frozen",
        "recovery_temperature_mode": "frozen",
    }
    parameters = _wall_parameters(configuration, 1e-7)
    assert parameters.flow_temperature_override_K == 900.0
    assert parameters.recovery_temperature_override_K == 900.0
    configuration["recovery_temperature_mode"] = "physical"
    parameters = _wall_parameters(configuration, 1e-7)
    assert parameters.flow_temperature_override_K == 900.0
    assert parameters.recovery_temperature_override_K is None


def test_temperature_intervention_requires_every_arrhenius_channel():
    base = {
        "n": 32, "shear_rate_s": 4e4,
        "flow_temperature_mode": "physical",
        "recovery_temperature_mode": "physical",
        "front_temperature_mode": "physical",
    }
    control = dict(base)
    control.update(flow_temperature_mode="frozen",
                   recovery_temperature_mode="frozen",
                   front_temperature_mode="frozen")
    kwargs = dict(
        baseline_provenance={"source_commit": "abc"},
        control_provenance={"source_commit": "abc"},
        baseline_step=10, control_step=10,
        baseline_time_s=1e-6, control_time_s=1e-6,
        baseline_gamma=.04, control_gamma=.04,
        baseline_initial_volume=np.ones(3),
        control_initial_volume=np.ones(3),
        baseline_grain_ids=(10, 20, 30),
        control_grain_ids=(10, 20, 30))
    assert temperature_intervention_certificate(
        base, control, **kwargs)["passed"]
    control["front_temperature_mode"] = "physical"
    decision = temperature_intervention_certificate(base, control, **kwargs)
    assert not decision["passed"]
    assert not decision["checks"][
        "control_matches_declared_temperature_intervention"]
    assert temperature_intervention_certificate(
        base, control, intervention_scope="flow_recovery", **kwargs)["passed"]


def test_full_one_microsecond_conjunction_passes_only_with_refinement():
    rows = [_row(0, 0.0, 100.0), _row(1, .5e-6, 70.0),
            _row(2, 1e-6, 70.0), _row(3, 1.5e-6, 70.0)]
    assert _decision(rows, True)["inherited_strict_asb"]
    assert not _decision(rows, False)["inherited_strict_asb"]


def test_preceding_peak_can_predate_attributable_control_window():
    rows = [
        _row(0, 0.0, stress=100.0, matched=None),
        _row(1, 1.1e-6, stress=75.0),
        _row(2, 2.2e-6, stress=75.0),
    ]
    mask = np.zeros((8, 8), dtype=bool); mask[:, 3] = True
    decision = classify_physical_episode(
        rows, {row["step"]: mask for row in rows}, PhysicalASBCriteria(),
        matched_control_available=True, refinement_passed=True)
    assert decision["inherited_strict_asb"]
    assert rows[1]["preceding_peak_time_s"] == 0.0


def test_declared_segment_cadence_is_not_reduced_by_one_short_interval():
    rows = [_row(index, time, stress=100.0 if index == 0 else 70.0)
            for index, time in enumerate((0.0, 25e-9, 75e-9, 125e-9))]
    for row in rows:
        row.update(sampling_segment_id="segment-a",
                   declared_sampling_cadence_s=50e-9)
    _decision(rows)
    assert all(row["sampling_continuity"]["passed"] for row in rows)
    assert rows[-1]["sampling_continuity"]["gap_limit_s"] == 75e-9


def test_declared_cadence_rejects_real_missing_interval_and_deduplicates():
    rows = [_row(0, 0.0), _row(1, 50e-9, stress=70.0),
            _row(1, 50e-9, stress=70.0), _row(3, 150e-9, stress=70.0)]
    for row in rows:
        row.update(sampling_segment_id="segment-a",
                   declared_sampling_cadence_s=50e-9)
    decision = _decision(rows)
    assert decision["duplicate_record_count"] == 1
    assert not rows[-1]["sampling_continuity"]["passed"]


def test_operator_transition_retains_whole_peak_but_separates_segment_drop():
    rows = [_row(0, 0.0, stress=100.0),
            _row(1, .5e-6, stress=80.0),
            _row(2, 1.0e-6, stress=70.0)]
    rows[0].update(operator_id="uniform", sampling_segment_id="old",
                   declared_sampling_cadence_s=.5e-6)
    for row in rows[1:]:
        row.update(operator_id="local", sampling_segment_id="new",
                   declared_sampling_cadence_s=.5e-6,
                   source_seam_verified=row is rows[1])
    mask = np.zeros((8, 8), dtype=bool); mask[:, 3] = True
    decision = classify_physical_episode(
        rows, {row["step"]: mask for row in rows}, PhysicalASBCriteria(
            maximum_sampling_gap_s=1e-6), matched_control_available=True,
        refinement_passed=True, qualifying_operator_id="local")
    assert decision["whole_history_peak"]["stress_Pa"] == 100.0
    assert decision["qualifying_operator_peak"]["stress_Pa"] == 80.0
    assert np.isclose(rows[-1]["whole_history_softening_fraction"], .3)
    assert np.isclose(rows[-1]["operator_segment_softening_fraction"], .125)
    assert np.isclose(decision["softening_summary"][
        "maximum_whole_history_fraction"], .3)
    assert np.isclose(decision["softening_summary"][
        "maximum_qualifying_operator_segment_fraction"], .125)
    assert not rows[0]["qualifying_operator_eligible"]


def test_source_association_finds_smaller_colocated_heat_component():
    candidate = np.zeros((16, 16), dtype=bool); candidate[2:4, 2:4] = True
    source = np.zeros((16, 16), dtype=float)
    source[2:4, 2:4] = 10.0
    source[9:13, 9:13] = 20.0
    association = source_association(source, candidate)
    assert association["independently_largest_component_overlap"] == 0.0
    assert association["maximum_overlap_with_any_source_component"] == 1.0
    assert association["positive_source_fraction_on_candidate"] > 0.0
