import numpy as np

from full_model.production.groma_patterning_oracle import (
    GromaPatterningParameters, conservative_rhs, fastest_linear_mode,
    linear_dispersion, rk4_step,
)


def test_groma_flux_is_conservative_without_projection():
    n = 64
    x = np.arange(n)*.5
    rho = 1.0+1e-3*np.cos(2*np.pi*x/(n*.5))
    kappa = 2e-4*np.sin(2*np.pi*x/(n*.5))
    p = GromaPatterningParameters()
    rho_rate, kappa_rate, _, _ = conservative_rhs(rho, kappa, .5, p)
    assert abs(np.sum(rho_rate)) < 1e-13
    assert abs(np.sum(kappa_rate)) < 1e-13
    after_rho, after_kappa = rk4_step(rho, kappa, .01, .5, p)
    assert np.isclose(np.sum(after_rho), np.sum(rho), rtol=0, atol=2e-14)
    assert np.isclose(np.sum(after_kappa), np.sum(kappa), rtol=0, atol=2e-14)


def test_linear_dispersion_has_finite_fastest_unstable_mode():
    p = GromaPatterningParameters()
    prediction = fastest_linear_mode(p)
    assert prediction["unstable"]
    k = np.linspace(.001, 1.0, 10000)
    growth, _ = linear_dispersion(k, p)
    selected = k[np.argmax(growth)]
    assert np.isclose(selected, prediction["wavenumber"], rtol=5e-4)
    assert np.isclose(np.max(growth), prediction["growth_rate"], rtol=1e-6)
    assert linear_dispersion(np.array([2.0]), p)[0][0] < 0.0


def test_above_instability_window_is_damped():
    p = GromaPatterningParameters(applied_stress=1.75)
    assert not fastest_linear_mode(p)["unstable"]
    growth, _ = linear_dispersion(np.linspace(.01, 2.0, 100), p)
    assert np.all(growth < 0.0)


def test_exp_floor_mobility_is_odd_bounded_and_thermally_activated():
    from full_model.production.groma_patterning_oracle import glide_speed
    p = GromaPatterningParameters(mobility_law="exp_floor")
    stress = np.array([-2.0, -.5, 0.0, .5, 2.0])
    speed = glide_speed(stress, p)
    assert np.allclose(speed, -speed[::-1])
    assert speed[2] == 0.0
    assert np.all(np.abs(speed) <= p.arrhenius_speed_scale)
    hotter = GromaPatterningParameters(
        mobility_law="exp_floor", arrhenius_inverse_temperature=5.0)
    assert glide_speed(.5, hotter) > glide_speed(.5, p)
