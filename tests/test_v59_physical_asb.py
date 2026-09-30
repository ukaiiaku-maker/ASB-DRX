import numpy as np

from full_model.analysis.postprocess_v59_physical_asb import (
    PhysicalASBCriteria, _wall_parameters, classify_physical_episode,
    component_morphology, periodic_identity,
    component_temperature_excess,
    temperature_intervention_certificate,
)


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
