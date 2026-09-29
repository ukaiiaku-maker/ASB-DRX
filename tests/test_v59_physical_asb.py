import numpy as np

from full_model.analysis.postprocess_v59_physical_asb import (
    PhysicalASBCriteria, _wall_parameters, classify_physical_episode,
    component_morphology, periodic_identity,
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


def test_full_one_microsecond_conjunction_passes_only_with_refinement():
    rows = [_row(0, 0.0, 100.0), _row(1, .5e-6, 70.0),
            _row(2, 1e-6, 70.0), _row(3, 1.5e-6, 70.0)]
    assert _decision(rows, True)["inherited_strict_asb"]
    assert not _decision(rows, False)["inherited_strict_asb"]
