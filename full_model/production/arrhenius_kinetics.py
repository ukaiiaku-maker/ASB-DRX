"""Dimensionally explicit Arrhenius helpers for the full v34 recovery model."""

from dataclasses import dataclass
import math
import numpy as np


KB_J_K = 1.380649e-23
EV_J = 1.602176634e-19


@dataclass(frozen=True)
class ActivatedProcess:
    name: str
    attempt_frequency_s: float
    entropy_over_kB: float = 0.0
    drag_rate_s: float = math.inf
    negative_barrier_mode: str = "drag"

    def __post_init__(self):
        if not self.name:
            raise ValueError("activated process name must be nonempty")
        if not math.isfinite(self.attempt_frequency_s) or self.attempt_frequency_s <= 0.0:
            raise ValueError("attempt frequency must be finite and positive")
        if not math.isfinite(self.entropy_over_kB):
            raise ValueError("activation entropy must be finite")
        if self.drag_rate_s < 0.0 or math.isnan(self.drag_rate_s):
            raise ValueError("drag rate must be nonnegative or infinite")
        if self.negative_barrier_mode not in ("drag", "reject"):
            raise ValueError("negative barrier mode must be drag or reject")

    @property
    def identifiable_prefactor_s(self):
        """Constant entropy and attempt frequency are identifiable as a product."""
        return self.attempt_frequency_s * math.exp(self.entropy_over_kB)


def exp_floor_enthalpy_j(stress_pa, h0_j, critical_stress_pa, a, n, floor):
    """EXP-floor activation enthalpy [J] for scalar or array stress input."""
    stress = np.asarray(stress_pa, dtype=float)
    values = (h0_j, critical_stress_pa, a, n, floor)
    if not np.all(np.isfinite(stress)) or not all(math.isfinite(float(value)) for value in values):
        raise ValueError("EXP-floor inputs must be finite")
    if n < 1.0:
        raise ValueError("production EXP-floor exponent n must be >= 1")
    if critical_stress_pa <= 0.0 or h0_j < 0.0 or a < 0.0:
        raise ValueError("invalid EXP-floor parameter")
    ff = min(max(float(floor), 0.0), 1.0)
    if stress.ndim == 0:
        ratio = max(float(stress), 0.0) / float(critical_stress_pa)
        return float(h0_j) * (ff + (1.0 - ff) * math.exp(-float(a) * ratio**float(n)))
    ratio = np.maximum(stress, 0.0) / float(critical_stress_pa)
    return float(h0_j) * (ff + (1.0 - ff) * np.exp(-float(a) * ratio**float(n)))


def free_barrier_j(enthalpy_j, temperature_k, entropy_over_kB):
    """Return ΔG*=ΔH*-TΔS* [J], with ΔS*/kB dimensionless."""
    values = (enthalpy_j, temperature_k, entropy_over_kB)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("free-barrier inputs must be finite")
    if float(enthalpy_j) < 0.0 or float(temperature_k) <= 0.0:
        raise ValueError("enthalpy must be nonnegative and temperature positive")
    return float(enthalpy_j) - KB_J_K * float(temperature_k) * float(entropy_over_kB)


def activated_rate_s(process, enthalpy_j, temperature_k):
    """Event frequency [1/s] with an explicit negative-barrier drag branch."""
    temperature_k = float(temperature_k)
    if temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    barrier = free_barrier_j(enthalpy_j, temperature_k, process.entropy_over_kB)
    if barrier < 0.0:
        if process.negative_barrier_mode != "drag":
            raise ValueError(f"{process.name}: negative free barrier outside validity envelope")
        return min(float(process.attempt_frequency_s), float(process.drag_rate_s))
    return float(process.attempt_frequency_s) * math.exp(-barrier / (KB_J_K * temperature_k))


def event_velocity_m_s(event_rate_s, glide_distance_m):
    """Convert event frequency [1/s] to physical velocity [m/s]."""
    if glide_distance_m < 0.0:
        raise ValueError("glide distance must be nonnegative")
    return float(event_rate_s) * float(glide_distance_m)
