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


def exp_floor_activation_volume_m3(
        stress_pa, h0_j, critical_stress_pa, a, n, floor):
    """Return ``-d DeltaG*/d stress`` for the declared EXP-floor input.

    Activation entropy is stress independent here, so this is also
    ``-d DeltaH*/d stress``.  The result is an activation volume [m^3], not
    the geometric event measure and not the signed complete event affinity.
    """
    stress = np.asarray(stress_pa, dtype=float)
    values = (h0_j, critical_stress_pa, a, n, floor)
    if (not np.all(np.isfinite(stress))
            or not all(math.isfinite(float(value)) for value in values)):
        raise ValueError("EXP-floor activation-volume inputs must be finite")
    if n < 1.0 or critical_stress_pa <= 0.0 or h0_j < 0.0 or a < 0.0:
        raise ValueError("invalid EXP-floor activation-volume parameter")
    ff = min(max(float(floor), 0.0), 1.0)
    positive = np.maximum(stress, 0.0)
    ratio = positive/float(critical_stress_pa)
    result = (float(h0_j)*(1.0-ff)*float(a)*float(n)
              *np.exp(-float(a)*ratio**float(n))
              *ratio**(float(n)-1.0)/float(critical_stress_pa))
    result = np.where(stress > 0.0, result, 0.0)
    return float(result) if result.ndim == 0 else result


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


def activated_rate_array_s(process, enthalpy_j, temperature_k):
    """Vectorized event frequency [1/s] using the same free barrier once.

    This is the field-valued counterpart of :func:`activated_rate_s`.  The
    activation entropy is subtracted exactly once and negative free barriers
    follow the process' declared ``drag`` or ``reject`` validity policy.
    """
    enthalpy = np.asarray(enthalpy_j, dtype=float)
    temperature = np.asarray(temperature_k, dtype=float)
    try:
        enthalpy, temperature = np.broadcast_arrays(enthalpy, temperature)
    except ValueError as error:
        raise ValueError("enthalpy and temperature are not broadcast-compatible") from error
    if (np.any(~np.isfinite(enthalpy)) or np.any(~np.isfinite(temperature))
            or np.any(enthalpy < 0.0) or np.any(temperature <= 0.0)):
        raise ValueError("enthalpy must be finite/nonnegative and temperature finite/positive")
    barrier = enthalpy-KB_J_K*temperature*process.entropy_over_kB
    negative = barrier < 0.0
    if np.any(negative) and process.negative_barrier_mode == "reject":
        raise ValueError(f"{process.name}: negative free barrier outside validity envelope")
    thermal = process.attempt_frequency_s*np.exp(np.clip(
        -np.maximum(barrier, 0.0)/(KB_J_K*temperature), -700.0, 0.0))
    if np.any(negative):
        thermal = np.where(negative, min(process.attempt_frequency_s,
                                         process.drag_rate_s), thermal)
    return thermal


def event_velocity_m_s(event_rate_s, glide_distance_m):
    """Convert event frequency [1/s] to physical velocity [m/s]."""
    if glide_distance_m < 0.0:
        raise ValueError("glide distance must be nonnegative")
    return float(event_rate_s) * float(glide_distance_m)
