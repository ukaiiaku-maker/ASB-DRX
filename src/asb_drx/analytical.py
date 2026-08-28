"""Analytical EXP-floor activation law and rate-temperature strength peak."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import lambertw
from scipy.optimize import brentq


KB_J_PER_K = 1.380649e-23


@dataclass(frozen=True)
class PeakSolution:
    """Closed-form interior maximum of the independent-node strength curve."""

    temperature_K: float
    shear_rate_s_inv: float
    D: float
    normalized_barrier_y: float
    required_barrier_fraction: float
    taylor_ratio_q: float
    density_m2: float
    local_activation_stress_Pa: float
    macroscopic_strength_Pa: float


@dataclass(frozen=True)
class NetPeakSolution:
    """Interior strength maximum for forward-minus-reverse continuum flow."""

    temperature_K: float
    shear_rate_s_inv: float
    taylor_ratio_q: float
    density_m2: float
    local_activation_stress_Pa: float
    macroscopic_strength_Pa: float
    forward_rate_s_inv: float
    reverse_rate_s_inv: float


@dataclass(frozen=True)
class ExpFloorLaw:
    """EXP-floor barrier coupled to an independent Taylor-node rate law.

    The model is intentionally material-agnostic. Its parameters must be supplied
    from an explicitly versioned calibration rather than inherited legacy values.
    """

    barrier_ref_J: float
    stress_ref_Pa: float
    reference_temperature_K: float
    floor_fraction: float
    shape_a: float
    shape_n: float
    rate_prefactor_s_inv: float
    density_exponent_p: float
    burgers_m: float
    barrier_temperature_coefficient: float = 0.0
    stress_temperature_coefficient: float = 0.0
    barrier_entropy_kB: float = 0.0
    taylor_geometry_factor: float = math.sqrt(2.0)

    def __post_init__(self) -> None:
        positive = {
            "barrier_ref_J": self.barrier_ref_J,
            "stress_ref_Pa": self.stress_ref_Pa,
            "reference_temperature_K": self.reference_temperature_K,
            "shape_a": self.shape_a,
            "shape_n": self.shape_n,
            "rate_prefactor_s_inv": self.rate_prefactor_s_inv,
            "density_exponent_p": self.density_exponent_p,
            "burgers_m": self.burgers_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.barrier_entropy_kB):
            raise ValueError("barrier_entropy_kB must be finite")
        if not math.isfinite(self.taylor_geometry_factor) or self.taylor_geometry_factor <= 0.0:
            raise ValueError("taylor_geometry_factor must be finite and positive")
        if not math.isfinite(self.floor_fraction) or not 0.0 <= self.floor_fraction < 1.0:
            raise ValueError("floor_fraction must satisfy 0 <= f < 1")

    def barrier_scale_J(self, temperature_K: float) -> float:
        self._check_temperature(temperature_K)
        reduced = (temperature_K - self.reference_temperature_K) / self.reference_temperature_K
        scale = self.barrier_ref_J * math.exp(-self.barrier_temperature_coefficient * reduced)
        scale -= KB_J_PER_K * self.barrier_entropy_kB * (
            temperature_K - self.reference_temperature_K
        )
        if scale <= 0.0:
            raise ValueError("barrier scale is nonpositive at requested temperature")
        return scale

    def barrier_scale_temperature_derivative_J_K(self, temperature_K: float) -> float:
        self._check_temperature(temperature_K)
        reduced = (temperature_K - self.reference_temperature_K) / self.reference_temperature_K
        exponential = self.barrier_ref_J * math.exp(-self.barrier_temperature_coefficient * reduced)
        return (
            -self.barrier_temperature_coefficient * exponential / self.reference_temperature_K
            - KB_J_PER_K * self.barrier_entropy_kB
        )

    def stress_scale_Pa(self, temperature_K: float) -> float:
        self._check_temperature(temperature_K)
        reduced = (temperature_K - self.reference_temperature_K) / self.reference_temperature_K
        return self.stress_ref_Pa * math.exp(-self.stress_temperature_coefficient * reduced)

    def barrier_J(self, local_stress_Pa: float, temperature_K: float) -> float:
        if not math.isfinite(local_stress_Pa) or local_stress_Pa < 0.0:
            raise ValueError("local_stress_Pa must be finite and nonnegative")
        G0 = self.barrier_scale_J(temperature_K)
        ratio = local_stress_Pa / self.stress_scale_Pa(temperature_K)
        variable = math.exp(-self.shape_a * ratio**self.shape_n)
        return G0 * (self.floor_fraction + (1.0 - self.floor_fraction) * variable)

    def activation_volume_m3(self, local_stress_Pa: float, temperature_K: float) -> float:
        if not math.isfinite(local_stress_Pa) or local_stress_Pa < 0.0:
            raise ValueError("local_stress_Pa must be finite and nonnegative")
        G0 = self.barrier_scale_J(temperature_K)
        tau_c = self.stress_scale_Pa(temperature_K)
        ratio = local_stress_Pa / tau_c
        if ratio == 0.0 and self.shape_n < 1.0:
            return math.inf
        return (
            G0
            * (1.0 - self.floor_fraction)
            * self.shape_a
            * self.shape_n
            * ratio ** (self.shape_n - 1.0)
            * math.exp(-self.shape_a * ratio**self.shape_n)
            / tau_c
        )

    def shear_rate_s_inv(
        self, local_stress_Pa: float, density_m2: float, temperature_K: float
    ) -> float:
        q = self.taylor_ratio(density_m2)
        exponent = -self.barrier_J(local_stress_Pa, temperature_K) / (KB_J_PER_K * temperature_K)
        return self.rate_prefactor_s_inv * q**self.density_exponent_p * math.exp(exponent)

    def net_shear_rate_s_inv(
        self, signed_local_stress_Pa: float, density_m2: float, temperature_K: float
    ) -> float:
        """Odd forward-minus-reverse rate used by continuum plastic flow.

        ``shear_rate_s_inv`` remains the positive one-way event rate represented
        by the source single-glider equations.  Subtracting the unloaded reverse
        event rate supplies detailed-balance symmetry without a fitted stress
        regularization and makes the continuum rate exactly zero at zero stress.
        """
        if not math.isfinite(signed_local_stress_Pa):
            raise ValueError("signed_local_stress_Pa must be finite")
        if signed_local_stress_Pa == 0.0:
            return 0.0
        forward = self.shear_rate_s_inv(
            abs(signed_local_stress_Pa), density_m2, temperature_K
        )
        reverse = self.shear_rate_s_inv(0.0, density_m2, temperature_K)
        return math.copysign(max(forward - reverse, 0.0), signed_local_stress_Pa)

    def local_stress_Pa(
        self, density_m2: float, temperature_K: float, shear_rate_s_inv: float
    ) -> float:
        q = self.taylor_ratio(density_m2)
        self._check_rate(shear_rate_s_inv)
        G0 = self.barrier_scale_J(temperature_K)
        h = (KB_J_PER_K * temperature_K / G0) * (
            math.log(self.rate_prefactor_s_inv / shear_rate_s_inv)
            + self.density_exponent_p * math.log(q)
        )
        if not self.floor_fraction < h < 1.0:
            raise ValueError(
                "requested state is outside the interior EXP-floor inverse "
                f"(required barrier fraction h={h:.16g})"
            )
        y = (h - self.floor_fraction) / (1.0 - self.floor_fraction)
        return self.stress_scale_Pa(temperature_K) * (
            -math.log(y) / self.shape_a
        ) ** (1.0 / self.shape_n)

    def macroscopic_strength_Pa(
        self, density_m2: float, temperature_K: float, shear_rate_s_inv: float
    ) -> float:
        q = self.taylor_ratio(density_m2)
        return q * self.local_stress_Pa(density_m2, temperature_K, shear_rate_s_inv)

    def net_local_stress_Pa(
        self, density_m2: float, temperature_K: float, shear_rate_s_inv: float
    ) -> float:
        """Invert the positive branch of the forward-minus-reverse net rate."""
        q = self.taylor_ratio(density_m2)
        self._check_rate(shear_rate_s_inv)
        reverse = self.shear_rate_s_inv(0.0, density_m2, temperature_K)
        forward = shear_rate_s_inv + reverse
        G0 = self.barrier_scale_J(temperature_K)
        h = -(KB_J_PER_K * temperature_K / G0) * math.log(
            forward / (self.rate_prefactor_s_inv * q**self.density_exponent_p)
        )
        if not self.floor_fraction < h < 1.0:
            raise ValueError(
                "requested state is outside the interior net EXP-floor inverse "
                f"(required barrier fraction h={h:.16g})"
            )
        y = (h - self.floor_fraction) / (1.0 - self.floor_fraction)
        return self.stress_scale_Pa(temperature_K) * (
            -math.log(y) / self.shape_a
        ) ** (1.0 / self.shape_n)

    def net_macroscopic_strength_Pa(
        self, density_m2: float, temperature_K: float, shear_rate_s_inv: float
    ) -> float:
        return self.taylor_ratio(density_m2) * self.net_local_stress_Pa(
            density_m2, temperature_K, shear_rate_s_inv
        )

    def peak(self, temperature_K: float, shear_rate_s_inv: float) -> PeakSolution:
        self._check_temperature(temperature_K)
        self._check_rate(shear_rate_s_inv)
        G0 = self.barrier_scale_J(temperature_K)
        D = (
            self.density_exponent_p
            * KB_J_PER_K
            * temperature_K
            / (self.shape_n * G0 * (1.0 - self.floor_fraction))
        )
        if not 0.0 < D <= 1.0 / math.e:
            raise ValueError(
                "no interior strength maximum: peak existence requires "
                f"0 < D <= 1/e, obtained D={D:.16g}"
            )
        W0 = float(lambertw(-D, k=0).real)
        y = math.exp(W0)
        h = self.floor_fraction + (1.0 - self.floor_fraction) * y
        log_q = (
            math.log(shear_rate_s_inv / self.rate_prefactor_s_inv)
            + G0 * h / (KB_J_PER_K * temperature_K)
        ) / self.density_exponent_p
        q = math.exp(log_q)
        density = (q / (self.taylor_geometry_factor * self.burgers_m)) ** 2
        local_stress = self.stress_scale_Pa(temperature_K) * (
            -math.log(y) / self.shape_a
        ) ** (1.0 / self.shape_n)
        return PeakSolution(
            temperature_K=temperature_K,
            shear_rate_s_inv=shear_rate_s_inv,
            D=D,
            normalized_barrier_y=y,
            required_barrier_fraction=h,
            taylor_ratio_q=q,
            density_m2=density,
            local_activation_stress_Pa=local_stress,
            macroscopic_strength_Pa=q * local_stress,
        )

    def net_peak(
        self, temperature_K: float, shear_rate_s_inv: float
    ) -> NetPeakSolution:
        """Predict the continuum strength maximum for the net signed rate."""
        self._check_temperature(temperature_K)
        self._check_rate(shear_rate_s_inv)
        G0 = self.barrier_scale_J(temperature_K)
        thermal = KB_J_PER_K * temperature_K
        unloaded_factor = math.exp(-G0 / thermal)
        floor_factor = math.exp(-G0 * self.floor_fraction / thermal)
        if floor_factor <= unloaded_factor:
            raise ValueError("no admissible net EXP-floor rate interval")
        c = G0 / thermal

        def state(barrier_fraction: float) -> tuple[float, float, float]:
            difference = unloaded_factor * math.expm1(c * (1.0 - barrier_fraction))
            q = (
                shear_rate_s_inv / (self.rate_prefactor_s_inv * difference)
            ) ** (1.0 / self.density_exponent_p)
            reverse = self.rate_prefactor_s_inv * q**self.density_exponent_p * unloaded_factor
            y = (barrier_fraction - self.floor_fraction) / (1.0 - self.floor_fraction)
            stress = self.stress_scale_Pa(temperature_K) * (
                -math.log(y) / self.shape_a
            ) ** (1.0 / self.shape_n)
            return q, stress, reverse

        def log_strength_derivative(barrier_fraction: float) -> float:
            y = (barrier_fraction - self.floor_fraction) / (1.0 - self.floor_fraction)
            z_over_difference = 1.0 / (
                1.0 - math.exp(-c * (1.0 - barrier_fraction))
            )
            return (
                c * z_over_difference / self.density_exponent_p
                - 1.0
                / (
                    self.shape_n
                    * (barrier_fraction - self.floor_fraction)
                    * (-math.log(y))
                )
            )

        y_values = np.unique(
            np.concatenate(
                (
                    np.geomspace(1.0e-12, 1.0e-2, 160),
                    np.linspace(1.0e-2, 1.0 - 1.0e-2, 400),
                    1.0 - np.geomspace(1.0e-12, 1.0e-2, 160),
                )
            )
        )
        h_values = self.floor_fraction + (1.0 - self.floor_fraction) * y_values
        roots = []
        previous_h = float(h_values[0])
        previous_value = log_strength_derivative(previous_h)
        for candidate_h in h_values[1:]:
            candidate_h = float(candidate_h)
            candidate_value = log_strength_derivative(candidate_h)
            if previous_value > 0.0 and candidate_value < 0.0:
                roots.append(
                    brentq(log_strength_derivative, previous_h, candidate_h, xtol=1.0e-14)
                )
            previous_h = candidate_h
            previous_value = candidate_value
        if not roots:
            raise ValueError("no interior maximum of net EXP-floor strength")
        q, local_stress, reverse = state(roots[-1])
        density = (q / (self.taylor_geometry_factor * self.burgers_m)) ** 2
        forward = shear_rate_s_inv + reverse
        return NetPeakSolution(
            temperature_K,
            shear_rate_s_inv,
            q,
            density,
            local_stress,
            q * local_stress,
            forward,
            reverse,
        )

    def taylor_ratio(self, density_m2: float) -> float:
        if not math.isfinite(density_m2) or density_m2 <= 0.0:
            raise ValueError("density_m2 must be finite and positive")
        return self.taylor_geometry_factor * self.burgers_m * math.sqrt(density_m2)

    @staticmethod
    def _check_temperature(temperature_K: float) -> None:
        if not math.isfinite(temperature_K) or temperature_K <= 0.0:
            raise ValueError("temperature_K must be finite and positive")

    @staticmethod
    def _check_rate(shear_rate_s_inv: float) -> None:
        if not math.isfinite(shear_rate_s_inv) or shear_rate_s_inv <= 0.0:
            raise ValueError("shear_rate_s_inv must be finite and positive")
