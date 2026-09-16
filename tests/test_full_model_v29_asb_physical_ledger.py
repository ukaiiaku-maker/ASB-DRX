import numpy as np
import pytest

from full_model.production.asb_physical_ledger import (
    DissipationRates, build_asb_energy_ledger, compatibility_dimensional_audit)
from full_model.production.compatibility_energy import decompose_compatibility_energy


def _energy(kappa, rho_gb, psi, dx=2e-9):
    gx = (np.roll(psi, -1, 0)-np.roll(psi, 1, 0))/(2*dx)
    gy = (np.roll(psi, -1, 1)-np.roll(psi, 1, 1))/(2*dx)
    return decompose_compatibility_energy(
        signed_gnd_density_m2=kappa, boundary_density_m2=rho_gb,
        orientation_gradient_m1=np.hypot(gx, gy), line_tension_J_m=1e-9,
        cell_area_m2=dx*dx, represented_thickness_m=5e-10,
        burgers_m=2.5e-10, alpha_penalty_coefficient=5e-9,
        gb_penalty_coefficient=2e-8)


def test_zero_field_has_zero_physical_and_constraint_energy():
    z = np.zeros((16, 16))
    e = _energy(z, z, z)
    assert e.physical_total_J == e.numerical_total_J == 0.0


def test_uniform_and_rigidly_shifted_rotation_are_energy_gauge_invariant():
    z = np.zeros((16, 16))
    a = _energy(z, z, np.full_like(z, 0.2))
    b = _energy(z, z, np.full_like(z, 1.7))
    assert a == b


def test_compatible_simple_shear_like_orientation_gradient_has_zero_residual():
    n, dx, b = 20, 2e-9, 2.5e-10
    # Periodic sinusoid avoids a boundary jump; set both line fields to grad/b.
    x = np.arange(n)[:, None]*2*np.pi/n
    psi = np.broadcast_to(0.01*np.sin(x), (n, n)).copy()
    gx = (np.roll(psi, -1, 0)-np.roll(psi, 1, 0))/(2*dx)
    target = np.abs(gx)/b
    e = _energy(target, target, psi, dx)
    assert e.alpha_residual_rms_m2 < 2.0
    assert e.gb_residual_rms_m2 < 2.0


def test_manufactured_gnd_and_gb_residual_scale_with_volume_and_A_r_squared():
    audit = compatibility_dimensional_audit(5e-9, 2e14, 3e-24)
    assert audit["energy_density_J_m3"] == pytest.approx(1e20)
    assert audit["represented_energy_J"] == pytest.approx(3e-4)
    assert not audit["eligible_as_physical_heat"]


def test_numerical_penalty_cannot_cancel_physical_first_law():
    ledger = build_asb_energy_ledger(
        physical_free_energy_change_J_m3=2.0,
        recoverable_elastic_change_J_m3=3.0,
        thermal_energy_change_J_m3=4.0, external_work_J_m3=10.0,
        independent_dissipation_J_m3=1.0,
        compatibility_alpha_penalty_change_J_m3=1e23,
        compatibility_gb_penalty_change_J_m3=-1e23)
    assert ledger.physical_first_law_residual_J_m3 == 0.0
    assert ledger.numerical_constraint_change_J_m3 == 0.0


def test_all_independent_dissipation_channels_are_nonnegative():
    rates = DissipationRates(1, 2, 3, 4, 5, 6, 7)
    assert rates.total_W_m3 == 28
    with pytest.raises(ValueError):
        DissipationRates(recovery_W_m3=-1e-9)
