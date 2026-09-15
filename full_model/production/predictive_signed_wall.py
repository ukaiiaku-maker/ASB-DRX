"""Predictive signed mobile/forest/wall reactions and stability operators.

This V19 module is intentionally independent of orientation recognition.  Wall
line can only be produced by declared reservoir transfers.  Densities have
units m^-2; the normalized stability coordinates divide them by ``rho_scale``.
The local reaction operator is an Onsager form driven by the same free-energy
chemical potentials used by the projected Hessian audit::

    dz/dt = - B M B.T grad(Psi) + neutral annihilation.

``B`` contains only mobile<->forest and forest<->wall transfers and therefore
conserves total line and signed Burgers content family by family.  Neutral
annihilation is a separately ledgered sink which removes equal positive and
negative content.  No term contains an orientation gradient or a recognition
mask as a wall target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .arrhenius_kinetics import ActivatedProcess, activated_rate_s, exp_floor_enthalpy_j
from .dislocation_free_energy import DislocationFreeEnergyParameters, free_energy_J_m3
from .wall_ordering_energy import WallOrderingParameters, local_wall_energy_J_m3


@dataclass(frozen=True)
class SignedWallState:
    mobile_plus: np.ndarray
    mobile_minus: np.ndarray
    forest_plus: np.ndarray
    forest_minus: np.ndarray
    wall_plus: np.ndarray
    wall_minus: np.ndarray
    order: np.ndarray

    def arrays(self):
        return (self.mobile_plus, self.mobile_minus, self.forest_plus,
                self.forest_minus, self.wall_plus, self.wall_minus)

    def validate(self):
        arrays = tuple(np.asarray(a, dtype=float) for a in self.arrays())
        if arrays[0].ndim < 1 or any(a.shape != arrays[0].shape for a in arrays):
            raise ValueError("all signed reservoirs must share a family axis")
        if any(np.any(~np.isfinite(a)) or np.any(a < 0.0) for a in arrays):
            raise ValueError("signed-population magnitudes must be finite and nonnegative")
        q = np.asarray(self.order, dtype=float)
        if q.shape != arrays[0].shape[:-1] or np.any(~np.isfinite(q)) \
                or np.any(q < 0.0) or np.any(q > 1.0):
            raise ValueError("wall order must match the spatial grid and lie in [0,1]")
        return arrays, q


@dataclass(frozen=True)
class SignedWallParameters:
    rho_scale_m2: float
    storage: DislocationFreeEnergyParameters
    ordering: WallOrderingParameters
    lock_process: ActivatedProcess
    trap_process: ActivatedProcess
    release_process: ActivatedProcess
    annihilation_process: ActivatedProcess
    order_process: ActivatedProcess
    lock_enthalpy_J: float
    trap_enthalpy_J: float
    release_enthalpy_J: float
    annihilation_enthalpy_J: float
    order_enthalpy_J: float
    critical_stress_Pa: float
    exp_a: float = 2.2
    exp_n: float = 2.5
    exp_floor: float = 0.0
    neutral_capture_area_m2: float = 1.0e-16
    order_energy_scale_J_m3: float = 1.0e6
    density_gradient_J_m3_m2: float = 0.0
    order_gradient_J_m: float = 0.0
    compatibility_J_m: float = 0.0
    burgers_m: float = 2.48e-10
    diffusivity_m2_s: float = 0.0
    orientation_mobility_m3_J_s: float = 0.0
    plastic_spin_rate_s: float = 0.0

    def __post_init__(self):
        positive = (self.rho_scale_m2, self.critical_stress_Pa,
                    self.order_energy_scale_J_m3, self.burgers_m)
        nonnegative = (self.lock_enthalpy_J, self.trap_enthalpy_J,
                       self.release_enthalpy_J, self.annihilation_enthalpy_J,
                       self.order_enthalpy_J, self.exp_a, self.exp_floor,
                       self.neutral_capture_area_m2, self.density_gradient_J_m3_m2,
                       self.order_gradient_J_m, self.compatibility_J_m,
                       self.diffusivity_m2_s, self.orientation_mobility_m3_J_s,
                       self.plastic_spin_rate_s)
        if any(not math.isfinite(float(x)) or x <= 0.0 for x in positive):
            raise ValueError("positive V19 wall parameters are invalid")
        if any(not math.isfinite(float(x)) or x < 0.0 for x in nonnegative):
            raise ValueError("nonnegative V19 wall parameters are invalid")
        if self.exp_n < 1.0 or self.exp_floor > 1.0:
            raise ValueError("invalid EXP-floor shape")


@dataclass(frozen=True)
class SignedWallLedger:
    locked_m2: float
    trapped_m2: float
    junctioned_m2: float
    released_m2: float
    neutral_annihilated_m2: float
    total_line_change_m2: float
    signed_change_by_family_m2: np.ndarray


def state_vector(state: SignedWallState):
    arrays, q = state.validate()
    if q.ndim != 0:
        raise ValueError("state_vector requires a homogeneous scalar state")
    return np.concatenate([a.reshape(-1) for a in arrays] + [[float(q)]])


def state_from_vector(vector, families):
    x = np.asarray(vector, dtype=float)
    if x.shape != (6*families+1,):
        raise ValueError("state-vector length does not match family count")
    arrays = [x[i*families:(i+1)*families].copy() for i in range(6)]
    return SignedWallState(*arrays, np.asarray(x[-1]))


def conservation_vectors(families):
    """Return total-line and family-resolved signed-Burgers row vectors."""
    n = 6*families+1
    total = np.zeros(n)
    total[:6*families] = 1.0
    signed = np.zeros((families, n))
    for family in range(families):
        for reservoir in (0, 2, 4):
            signed[family, reservoir*families+family] = 1.0
        for reservoir in (1, 3, 5):
            signed[family, reservoir*families+family] = -1.0
    return total, signed


def conservative_stoichiometric_matrix(families, include_order=True):
    """Return transfer plus explicit cross-family junction columns."""
    n = 6*families+1
    columns = []
    for family in range(families):
        for source, target in ((0, 2), (1, 3), (2, 4), (3, 5)):
            column = np.zeros(n)
            column[source*families+family] = -1.0
            column[target*families+family] = 1.0
            columns.append(column)
    # Opposite signs on distinct families lock as a junction pair and transfer
    # into the corresponding signed wall reservoirs. These columns are
    # stoichiometrically dependent on two trapping columns but are retained so
    # the physical realizing channel and its Arrhenius mobility are explicit.
    for first in range(families):
        for second in range(first+1, families):
            for sign_first, sign_second in ((0, 1), (1, 0)):
                column = np.zeros(n)
                column[(2+sign_first)*families+first] = -1.0
                column[(2+sign_second)*families+second] = -1.0
                column[(4+sign_first)*families+first] = 1.0
                column[(4+sign_second)*families+second] = 1.0
                columns.append(column)
    if include_order:
        column = np.zeros(n); column[-1] = 1.0
        columns.append(column)
    return np.column_stack(columns)


def admissible_basis(families):
    """Orthonormal basis for the actual conservative reaction tangent space."""
    B = conservative_stoichiometric_matrix(families)
    U, singular, _ = np.linalg.svd(B, full_matrices=False)
    tolerance = max(B.shape)*np.finfo(float).eps*max(float(singular[0]), 1.0)
    rank = int(np.sum(singular > tolerance))
    return U[:, :rank]


def homogeneous_free_energy_J_m3(normalized_state, parameters):
    x = np.asarray(normalized_state, dtype=float)
    families = (x.size-1)//6
    if x.shape != (6*families+1,) or np.any(x[:-1] < 0.0) \
            or not 0.0 <= x[-1] <= 1.0:
        raise ValueError("inadmissible normalized homogeneous state")
    density = parameters.rho_scale_m2*float(np.sum(x[:-1]))
    wall = parameters.rho_scale_m2*float(np.sum(
        x[4*families:6*families]))
    return float(free_energy_J_m3(density, parameters.storage)
                 + local_wall_energy_J_m3(density, wall, x[-1],
                                          parameters.ordering))


def numerical_hessian(function, point, relative_step=1.0e-4):
    x = np.asarray(point, dtype=float)
    n = x.size
    H = np.empty((n, n))
    steps = relative_step*np.maximum(np.abs(x), 1.0)
    f0 = function(x)
    for i in range(n):
        ei = np.zeros(n); ei[i] = steps[i]
        H[i, i] = (function(x+ei)-2.0*f0+function(x-ei))/steps[i]**2
        for j in range(i):
            ej = np.zeros(n); ej[j] = steps[j]
            H[i, j] = H[j, i] = (
                function(x+ei+ej)-function(x+ei-ej)
                -function(x-ei+ej)+function(x-ei-ej))/(4.0*steps[i]*steps[j])
    return 0.5*(H+H.T)


def projected_hessian(normalized_state, parameters):
    """Return raw and accessible Hessians in normalized coordinates."""
    x = np.asarray(normalized_state, dtype=float)
    families = (x.size-1)//6
    H = numerical_hessian(lambda y: homogeneous_free_energy_J_m3(y, parameters), x)
    Q = admissible_basis(families)
    Hp = 0.5*(Q.T@H@Q+(Q.T@H@Q).T)
    values, vectors = np.linalg.eigh(Hp)
    physical_vectors = Q@vectors
    total, signed = conservation_vectors(families)
    modes = []
    for value, vector in zip(values, physical_vectors.T):
        # At a strictly positive base state both signs of every infinitesimal
        # tangent are feasible.  Boundary states require an inward direction.
        eps = 1.0e-7
        plus_ok = np.all(x[:-1]+eps*vector[:-1] >= 0.0) and 0 <= x[-1]+eps*vector[-1] <= 1
        minus_ok = np.all(x[:-1]-eps*vector[:-1] >= 0.0) and 0 <= x[-1]-eps*vector[-1] <= 1
        modes.append({
            "eigenvalue_J_m3_per_normalized_state2": float(value),
            "eigenvector": vector.tolist(),
            "total_line_change_normalized": float(total@vector),
            "signed_change_by_family_normalized": (signed@vector).tolist(),
            "wall_order_change": float(vector[-1]),
            "nonnegativity_admissible": bool(plus_ok or minus_ok),
            "realizing_channels": "mobile<->forest, forest<->wall, and wall-order relaxation",
        })
    return H, Hp, modes


def _rate(process, enthalpy, stress_pa, temperature_K, p):
    barrier = exp_floor_enthalpy_j(abs(float(stress_pa)), enthalpy,
                                  p.critical_stress_Pa, p.exp_a, p.exp_n,
                                  p.exp_floor)
    return activated_rate_s(process, barrier, temperature_K)


def reaction_mobilities_s(normalized_state, stress_pa, temperature_K, parameters):
    """Diagonal mobilities for the columns of the stoichiometric matrix."""
    families = (len(normalized_state)-1)//6
    lock = _rate(parameters.lock_process, parameters.lock_enthalpy_J,
                 stress_pa, temperature_K, parameters)
    trap = _rate(parameters.trap_process, parameters.trap_enthalpy_J,
                 stress_pa, temperature_K, parameters)
    order = _rate(parameters.order_process, parameters.order_enthalpy_J,
                  stress_pa, temperature_K, parameters)
    energy_scale = max(parameters.order_energy_scale_J_m3, 1.0)
    junction_count = families*(families-1)
    return np.asarray(([lock/energy_scale, lock/energy_scale,
                        trap/energy_scale, trap/energy_scale]*families)
                      +[trap/energy_scale]*junction_count
                      +[order/energy_scale])


def linearized_operator(normalized_state, wavevector_m_inv, family_velocities_m_s,
                        stress_pa, temperature_K, parameters, *, local_hessian=None):
    """Full isothermal/fixed-macrostress Fourier-space kinetic operator.

    The local reaction block uses the actual Arrhenius mobilities above.
    Signed mobile populations advect in opposite directions.  Density and
    order gradient energies enter the chemical-potential Hessian.  Orientation
    is appended as the final state and is driven by signed mobile plastic spin;
    compatibility couples it to signed wall content without targeting either.
    """
    x = np.asarray(normalized_state, dtype=float)
    families = (x.size-1)//6
    k = np.asarray(wavevector_m_inv, dtype=float)
    velocities = np.asarray(family_velocities_m_s, dtype=float)
    if k.shape != (2,) or velocities.shape != (families, 2):
        raise ValueError("wavevector or family velocities have the wrong shape")
    H = (numerical_hessian(
        lambda y: homogeneous_free_energy_J_m3(y, parameters), x)
         if local_hessian is None else np.asarray(local_hessian, dtype=float).copy())
    if H.shape != (x.size, x.size) or np.any(~np.isfinite(H)):
        raise ValueError("local Hessian has the wrong shape or nonfinite entries")
    k2 = float(k@k)
    H = H.astype(complex)
    H[:-1, :-1] += np.eye(H.shape[0]-1)*parameters.density_gradient_J_m3_m2*k2
    H[-1, -1] += parameters.order_gradient_J_m*k2
    B = conservative_stoichiometric_matrix(families)
    mobility = reaction_mobilities_s(x, stress_pa, temperature_K, parameters)
    A = (B*mobility[None, :])@B.T
    Ldens = -A@H
    if parameters.diffusivity_m2_s:
        Ldens[:-1, :-1] -= parameters.diffusivity_m2_s*k2*np.eye(6*families)
    for family in range(families):
        omega = float(k@velocities[family])
        Ldens[family, family] -= 1j*omega
        Ldens[families+family, families+family] += 1j*omega

    # Add continuous orientation as a nonconserved state. Compatibility is an
    # energetic comparison between orientation gradient and evolved signed wall.
    L = np.zeros((x.size+1, x.size+1), dtype=complex)
    L[:-1, :-1] = Ldens
    theta = x.size
    signed_wall_coeff = np.zeros(x.size, dtype=complex)
    signed_wall_coeff[4*families:5*families] = -parameters.burgers_m*parameters.rho_scale_m2
    signed_wall_coeff[5*families:6*families] = +parameters.burgers_m*parameters.rho_scale_m2
    ctheta = 1j*math.sqrt(k2)
    if parameters.compatibility_J_m and parameters.orientation_mobility_m3_J_s:
        coeff = np.append(signed_wall_coeff, ctheta)
        Hcomp = parameters.compatibility_J_m*np.outer(np.conj(coeff), coeff)
        orient_mob = parameters.orientation_mobility_m3_J_s
        L[theta, :] -= orient_mob*Hcomp[theta, :]
        # Wall populations respond through their existing conservative reaction
        # mobility, never through a direct orientation-derived source.
        L[:-1, :] -= np.column_stack((A, np.zeros(x.size)))@Hcomp
    spin_weights = np.ones(families)/max(families, 1)
    L[theta, :families] += parameters.plastic_spin_rate_s*spin_weights
    L[theta, families:2*families] -= parameters.plastic_spin_rate_s*spin_weights
    return L


def advance_local_reactions(state, chemical_potential_J_m, stress_pa,
                            temperature_K, dt_s, parameters):
    """Advance one local state with conservative limited transfers and a sink.

    ``chemical_potential_J_m`` has the same six-reservoir layout as the state.
    It is supplied by the common thermodynamic functional.  Transfers are
    donor-limited, so nonnegativity is preserved without a density floor source.
    """
    arrays, q = state.validate()
    if q.ndim != 0 or arrays[0].ndim != 1:
        raise ValueError("local reaction update expects one homogeneous cell")
    families = arrays[0].size
    mu = np.asarray(chemical_potential_J_m, dtype=float)
    if mu.shape != (6, families) or np.any(~np.isfinite(mu)):
        raise ValueError("chemical potentials must have shape (6,families)")
    if dt_s < 0.0 or not math.isfinite(dt_s):
        raise ValueError("time increment must be finite and nonnegative")
    values = [a.copy() for a in arrays]
    before_total = sum(float(np.sum(a)) for a in values)
    before_signed = sum(values[i] if i % 2 == 0 else -values[i]
                        for i in range(6))
    rates = {
        "lock": _rate(parameters.lock_process, parameters.lock_enthalpy_J,
                      stress_pa, temperature_K, parameters),
        "trap": _rate(parameters.trap_process, parameters.trap_enthalpy_J,
                      stress_pa, temperature_K, parameters),
        "release": _rate(parameters.release_process, parameters.release_enthalpy_J,
                         stress_pa, temperature_K, parameters),
        "ann": _rate(parameters.annihilation_process, parameters.annihilation_enthalpy_J,
                     stress_pa, temperature_K, parameters),
        "order": _rate(parameters.order_process, parameters.order_enthalpy_J,
                       stress_pa, temperature_K, parameters),
    }

    totals = dict(locked=0.0, trapped=0.0, junctioned=0.0, released=0.0)
    # Thermodynamic direction; tanh bounds the trial fraction. Reverse transfer
    # uses the same column and is recorded as a negative forward amount.
    for name, source, target in (("locked", 0, 2), ("locked", 1, 3),
                                 ("trapped", 2, 4), ("trapped", 3, 5)):
        rate = rates["lock" if name == "locked" else "trap"]
        affinity = mu[target]-mu[source]
        fraction = np.tanh(np.abs(affinity)/max(parameters.order_energy_scale_J_m3/
                                                parameters.rho_scale_m2, 1e-300))
        forward = affinity < 0.0
        donor = np.where(forward, values[source], values[target])
        amount = np.minimum(dt_s*rate*fraction*donor, donor)
        values[source] += np.where(forward, -amount, amount)
        values[target] += np.where(forward, amount, -amount)
        totals[name] += float(np.sum(np.where(forward, amount, -amount)))

    # Cross-family junction locking. Only the forward forest->wall direction is
    # used in the nonlinear update; climb/cross-slip release is declared below.
    for first in range(families):
        for second in range(first+1, families):
            for sign_first, sign_second in ((0, 1), (1, 0)):
                sources = ((2+sign_first, first), (2+sign_second, second))
                targets = ((4+sign_first, first), (4+sign_second, second))
                affinity = sum(mu[t_res, t_family]-mu[s_res, s_family]
                               for (s_res, s_family), (t_res, t_family)
                               in zip(sources, targets))
                drive = math.tanh(max(float(-affinity), 0.0)/max(
                    parameters.order_energy_scale_J_m3/parameters.rho_scale_m2,
                    1e-300))
                donor = min(values[s][f] for s, f in sources)
                amount = min(dt_s*rates["trap"]*drive*donor, donor)
                for source, target in zip(sources, targets):
                    values[source[0]][source[1]] -= amount
                    values[target[0]][target[1]] += amount
                totals["junctioned"] += 2.0*amount

    # Thermally activated wall release is a distinct wall->mobile channel.
    release_fraction = min(dt_s*rates["release"], 1.0)
    for wall_i, mobile_i in ((4, 0), (5, 1)):
        amount = release_fraction*values[wall_i]
        values[wall_i] -= amount; values[mobile_i] += amount
        totals["released"] += float(np.sum(amount))

    # Equal-sign-pair removal conserves signed Burgers content exactly.
    ann_fraction = min(dt_s*rates["ann"], 1.0)
    pair = ann_fraction*np.minimum(values[0], values[1])
    pair = np.minimum(pair, parameters.neutral_capture_area_m2*
                      values[0]*values[1])
    values[0] -= pair; values[1] -= pair
    annihilated = 2.0*float(np.sum(pair))

    wall_fraction = float(np.sum(values[4]+values[5]))/max(
        sum(float(np.sum(a)) for a in values), 1e-300)
    qeq = np.clip(wall_fraction, 0.0, 1.0)
    qnew = float(q)+(qeq-float(q))*min(dt_s*rates["order"], 1.0)
    result = SignedWallState(*values, np.asarray(np.clip(qnew, 0.0, 1.0)))
    result.validate()
    after_total = sum(float(np.sum(a)) for a in values)
    after_signed = sum(values[i] if i % 2 == 0 else -values[i]
                         for i in range(6))
    return result, SignedWallLedger(
        totals["locked"], totals["trapped"], totals["junctioned"],
        totals["released"], annihilated,
        after_total-before_total, after_signed-before_signed)
