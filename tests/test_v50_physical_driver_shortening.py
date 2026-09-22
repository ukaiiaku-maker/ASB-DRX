from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from full_model.analysis.run_v49_physical_continuation import (
    advance_consistent_midpoint_segment,
)


@dataclass(frozen=True)
class ToyState:
    eta: np.ndarray
    plastic_shear: float = 0.0
    heat: float = 0.0


def toy_energy(context, state, drive):
    gamma = float(drive.mean_strain[0, 1])
    elastic = .5*context["G"]*(gamma-state.plastic_shear)**2
    return SimpleNamespace(
        recoverable_elastic_J=elastic,
        internal_J=elastic+state.heat)


def scripted_runner(accepted_durations, midpoint_log, state_log):
    calls = {"count": 0}
    def run(context, state, eta, drive, controls):
        index = calls["count"]; calls["count"] += 1
        accepted = accepted_durations[index]
        gamma = float(drive.mean_strain[0, 1])
        dp = context["plastic_rate"]*accepted
        before = .5*context["G"]*(gamma-state.plastic_shear)**2
        after = .5*context["G"]*(gamma-state.plastic_shear-dp)**2
        candidate = ToyState(
            state.eta.copy(), state.plastic_shear+dp,
            state.heat+before-after)
        midpoint_log.append(gamma); state_log.append(state)
        return candidate, {"mura": {
            "accepted_dt_s": accepted,
            "ordering_linear_iterations": 1,
            "ordering_internal_substeps": 1,
        }}
    return run


def test_shortened_ramp_candidates_are_discarded_and_rerun_at_own_midpoint():
    H = 1e-6
    state = ToyState(np.zeros((1, 1, 2)), plastic_shear=2e-4)
    context = {"G": 89.1e9, "plastic_rate": 10.0}
    midpoint_log = []; state_log = []
    result = advance_consistent_midpoint_segment(
        context, state, grid=1, initial_tensor_shear=.01,
        protocol="continued_deformation", rate=100.0,
        physical_time=0.0, load_origin=0.0, requested_duration=H,
        cycle_runner=scripted_runner(
            (H/2, H/4, H/4), midpoint_log, state_log),
        energy_evaluator=toy_energy)
    assert result["accepted_duration_s"] == H/4
    assert result["discarded_shortened_candidates"] == 2
    assert [x["candidate_published"] for x in
            result["duration_selection_attempts"]] == [False, False, True]
    # Tensor shear rate is 100/s; each midpoint follows its attempted clock.
    np.testing.assert_allclose(
        midpoint_log, [.01+100*H/2, .01+100*H/4, .01+100*H/8])
    assert all(item is state for item in state_log)
    assert result["first_law_passed"]


def test_shortening_matches_explicit_small_step_and_restart_state():
    H = 1e-6; h = H/2
    initial = ToyState(np.zeros((1, 1, 2)))
    context = {"G": 89.1e9, "plastic_rate": 10.0}
    shortened = advance_consistent_midpoint_segment(
        context, initial, grid=1, initial_tensor_shear=.01,
        protocol="continued_deformation", rate=100.0,
        physical_time=0.0, load_origin=0.0, requested_duration=H,
        cycle_runner=scripted_runner((h, h), [], []),
        energy_evaluator=toy_energy)
    explicit = advance_consistent_midpoint_segment(
        context, initial, grid=1, initial_tensor_shear=.01,
        protocol="continued_deformation", rate=100.0,
        physical_time=0.0, load_origin=0.0, requested_duration=h,
        cycle_runner=scripted_runner((h,), [], []),
        energy_evaluator=toy_energy)
    np.testing.assert_array_equal(shortened["candidate"].eta,
                                  explicit["candidate"].eta)
    assert shortened["candidate"].plastic_shear == explicit[
        "candidate"].plastic_shear
    assert shortened["candidate"].heat == explicit["candidate"].heat
    np.testing.assert_allclose(shortened["external_work_J"],
                               explicit["external_work_J"], rtol=0, atol=1e-20)
    # A serialized/reconstructed immutable state gives the same next segment.
    checkpoint = ToyState(
        shortened["candidate"].eta.copy(),
        shortened["candidate"].plastic_shear,
        shortened["candidate"].heat)
    next_a = advance_consistent_midpoint_segment(
        context, shortened["candidate"], grid=1,
        initial_tensor_shear=.01, protocol="continued_deformation", rate=100.,
        physical_time=h, load_origin=0., requested_duration=h,
        cycle_runner=scripted_runner((h,), [], []),
        energy_evaluator=toy_energy)
    next_b = advance_consistent_midpoint_segment(
        context, checkpoint, grid=1, initial_tensor_shear=.01,
        protocol="continued_deformation", rate=100., physical_time=h,
        load_origin=0., requested_duration=h,
        cycle_runner=scripted_runner((h,), [], []),
        energy_evaluator=toy_energy)
    np.testing.assert_array_equal(next_a["candidate"].eta,
                                  next_b["candidate"].eta)
    assert next_a["candidate"].plastic_shear == next_b[
        "candidate"].plastic_shear
    assert next_a["candidate"].heat == next_b["candidate"].heat


def test_repricing_old_candidate_has_manufactured_nonzero_residual():
    G = 89.1e9; engineering_rate = 200.0
    H = 1e-6; h = .5e-6; delta_p = 1e-5
    residual_J_m3 = .5*G*engineering_rate*(H-h)*delta_p
    np.testing.assert_allclose(residual_J_m3, 44.55, rtol=1e-15)
