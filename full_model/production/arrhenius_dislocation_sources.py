"""Spatial Hall--Petch dislocation-source hypotheses for the common state.

The Arrhenius hazard is shared by boundary/TJ and random-bulk controls.  The
controls differ only in the normalized spatial site field.  The post-event
transition kernel creates sign-balanced mobile pairs, so it cannot inject net
Burgers content.  This is a declared coarse-grained source hypothesis: the
Azizi--Dillon fixed-state Hall--Petch law supplies the event rate but does not
itself determine this state transition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np


@dataclass(frozen=True)
class ArrheniusSourceParameters:
    burgers_m: float = 2.48e-10
    grain_size_m: float = 5.0e-6
    activation_enthalpy_eV: float = 1.90
    activation_entropy_kB: float = 0.0
    intrinsic_activation_volume_m3: float = (2.48e-10)**3
    barrier_floor_eV: float = 0.05
    attempt_frequency_Hz: float = 1.0e12
    pair_increment_m2: float = 1.0e10
    tj_site_weight: float = 1.0

    def __post_init__(self):
        positive = (
            "burgers_m", "grain_size_m", "activation_enthalpy_eV",
            "intrinsic_activation_volume_m3", "barrier_floor_eV",
            "attempt_frequency_Hz", "pair_increment_m2",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.activation_entropy_kB):
            raise ValueError("activation entropy must be finite")
        if not math.isfinite(self.tj_site_weight) or self.tj_site_weight < 0.0:
            raise ValueError("TJ site weight must be finite and nonnegative")
        if self.barrier_floor_eV > self.activation_enthalpy_eV:
            raise ValueError("barrier floor cannot exceed activation enthalpy")


@dataclass(frozen=True)
class PairSourceLedger:
    created_pair_density_m2: float
    annihilated_pair_density_m2: float
    signed_content_change_m2: float
    line_content_change_m2: float
    line_balance_residual_m2: float
    source_mode: str


def _normalize_site_field(field):
    value = np.asarray(field, dtype=float)
    if value.ndim != 2 or np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError("site field must be a finite nonnegative 2-D array")
    mean = float(np.mean(value))
    if mean <= 0.0:
        raise ValueError("site field must contain positive measure")
    return value/mean


def boundary_tj_site_field(supports, *, tj_site_weight=1.0):
    """Return a normalized diffuse GB/TJ source field from owner supports."""
    eta = np.asarray(supports, dtype=float)
    if (eta.ndim != 3 or np.any(~np.isfinite(eta)) or np.any(eta < 0.0)
            or np.any(eta > 1.0)):
        raise ValueError("supports must be finite (owner,nx,ny) fractions")
    total = np.sum(eta, axis=0)
    if np.max(np.abs(total-1.0)) > 2e-6:
        raise ValueError("owner supports must sum to one")
    boundary = np.maximum(1.0-np.sum(eta*eta, axis=0), 0.0)
    # Third-largest support is a bounded differentiable proxy for triple-line
    # participation in the 2-D section; it vanishes for a two-owner boundary.
    if eta.shape[0] >= 3:
        third = np.sort(eta, axis=0)[-3]
        tj = 3.0*third
    else:
        tj = np.zeros_like(boundary)
    return _normalize_site_field(boundary+float(tj_site_weight)*tj)


def random_bulk_site_field(shape, seed, *, spectral_cutoff_cells=2.0):
    """Seeded positive random field with unit mean and a declared cutoff."""
    if len(shape) != 2 or any(int(n) < 4 for n in shape):
        raise ValueError("a two-dimensional shape of at least 4x4 is required")
    cutoff = float(spectral_cutoff_cells)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("spectral cutoff must be finite and positive")
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(size=tuple(map(int, shape)))
    kx = np.fft.fftfreq(shape[0])[:, None]
    ky = np.fft.fftfreq(shape[1])[None, :]
    kernel = np.exp(-2.0*math.pi**2*cutoff**2*(kx*kx+ky*ky))
    smooth = np.fft.ifft2(np.fft.fft2(noise)*kernel).real
    smooth -= np.mean(smooth)
    standard = float(np.std(smooth))
    if standard <= 0.0:
        raise ValueError("random site field has zero variance")
    # Lognormal weights preserve positivity and form a fixed quenched site map.
    return _normalize_site_field(np.exp(smooth/standard))


def randomized_matched_site_field(reference_field, seed):
    """Randomly relocate an existing site-strength distribution exactly.

    This is the strict source-location control: integrated measure, histogram,
    extrema, and every site weight are identical to the reference, while their
    locations are independently permuted.
    """
    reference = _normalize_site_field(reference_field)
    rng = np.random.default_rng(int(seed))
    return rng.permutation(reference.ravel()).reshape(reference.shape)


def hall_petch_nucleation_hazard_Hz(
        resolved_stress_Pa, temperature_K, parameters):
    """Azizi--Dillon Hall--Petch nucleation hazard before spatial weighting.

    ``N=b^2/d^2`` and ``phi=sqrt(d/b)``.  The EXP floor is retained exactly;
    no independent Taylor-peak or scalar-density softening is introduced.
    """
    stress = np.asarray(resolved_stress_Pa, dtype=float)
    temperature = np.asarray(temperature_K, dtype=float)
    stress, temperature = np.broadcast_arrays(stress, temperature)
    if (np.any(~np.isfinite(stress)) or np.any(~np.isfinite(temperature))
            or np.any(temperature <= 0.0)):
        raise ValueError("finite stress and positive temperature required")
    eV_J = 1.602176634e-19
    kB = 1.380649e-23
    multiplicity = parameters.burgers_m**2/parameters.grain_size_m**2
    concentration = math.sqrt(parameters.grain_size_m/parameters.burgers_m)
    work_eV = (concentration*np.abs(stress)
               *parameters.intrinsic_activation_volume_m3/eV_J)
    free_barrier_eV = (parameters.activation_enthalpy_eV
                       -temperature*kB*parameters.activation_entropy_kB/eV_J
                       -work_eV)
    barrier_eV = np.maximum(parameters.barrier_floor_eV, free_barrier_eV)
    return (parameters.attempt_frequency_Hz*multiplicity
            *np.exp(-barrier_eV*eV_J/(kB*temperature)))


def pair_source_rate_m2_s(
        site_field, resolved_stress_Pa, temperature_K, parameters):
    site = _normalize_site_field(site_field)
    hazard = hall_petch_nucleation_hazard_Hz(
        resolved_stress_Pa, temperature_K, parameters)
    try:
        hazard = np.broadcast_to(hazard, site.shape)
    except ValueError as exc:
        raise ValueError("stress and temperature must broadcast to the site field") from exc
    return parameters.pair_increment_m2*site*hazard


def apply_sign_balanced_pair_source(
        state, pair_rate_m2_s, dt_s, *, family_weights=None,
        sink_field=None, neutral_sink_rate_s=0.0, source_mode="unspecified"):
    """Apply a sign-balanced source and optional neutral boundary sink.

    Source and sink operate on the mobile populations only.  Each process is
    pointwise sign balanced, hence it preserves every family's signed content.
    The ledger closes total created/removed line independently.
    """
    rate = np.asarray(pair_rate_m2_s, dtype=float)
    dt = float(dt_s)
    shape = state.mobile_plus_m2.shape
    if rate.shape != shape[:2] or np.any(~np.isfinite(rate)) or np.any(rate < 0.0):
        raise ValueError("pair rate must be a finite nonnegative grid field")
    if not math.isfinite(dt) or dt < 0.0:
        raise ValueError("time increment must be finite and nonnegative")
    families = shape[2]
    if family_weights is None:
        weights = np.full(families, 1.0/families)
    else:
        weights = np.asarray(family_weights, dtype=float)
        if (weights.shape != (families,) or np.any(~np.isfinite(weights))
                or np.any(weights < 0.0) or np.sum(weights) <= 0.0):
            raise ValueError("family weights must be finite and nonnegative")
        weights = weights/np.sum(weights)
    created = dt*rate[..., None]*weights
    plus = np.asarray(state.mobile_plus_m2)+created
    minus = np.asarray(state.mobile_minus_m2)+created

    if sink_field is None:
        sink_weight = np.zeros(shape[:2])
    else:
        sink_weight = np.asarray(sink_field, dtype=float)
        if (sink_weight.shape != shape[:2] or np.any(~np.isfinite(sink_weight))
                or np.any(sink_weight < 0.0)):
            raise ValueError("sink field must be finite and nonnegative")
    sink_rate = float(neutral_sink_rate_s)
    if not math.isfinite(sink_rate) or sink_rate < 0.0:
        raise ValueError("neutral sink rate must be finite and nonnegative")
    fraction = 1.0-np.exp(-sink_rate*dt*sink_weight[..., None])
    neutral = np.minimum(plus, minus)*fraction
    plus -= neutral
    minus -= neutral
    candidate = replace(state, mobile_plus_m2=plus, mobile_minus_m2=minus)

    signed_before = np.sum(state.mobile_plus_m2-state.mobile_minus_m2)
    signed_after = np.sum(plus-minus)
    created_pairs = float(np.sum(created))
    annihilated_pairs = float(np.sum(neutral))
    line_change = float(np.sum(plus+minus-state.mobile_plus_m2-state.mobile_minus_m2))
    balance = line_change-2.0*(created_pairs-annihilated_pairs)
    return candidate, PairSourceLedger(
        created_pairs, annihilated_pairs, float(signed_after-signed_before),
        line_change, balance, str(source_mode))
