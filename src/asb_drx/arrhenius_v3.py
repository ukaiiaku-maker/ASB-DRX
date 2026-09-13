"""Versioned EXP-floor enthalpy plus independent activation entropy.

This module is the Mission-v3 kinetic backbone.  It is deliberately separate
from :mod:`asb_drx.analytical`, whose historical temperature parameterization
remains a regression fixture.  Energies are per event in joules and entropy is
reported in units of Boltzmann's constant per event.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


KB_J_PER_K = 1.380649e-23


@dataclass(frozen=True)
class ExpFloorEnthalpy:
    """Positive EXP-floor activation enthalpy with smooth temperature scales."""

    reference_enthalpy_J: float
    reference_stress_Pa: float
    reference_temperature_K: float
    floor_fraction: float
    shape_a: float
    shape_n: float
    enthalpy_temperature_coefficient: float = 0.0
    stress_temperature_coefficient: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "reference_enthalpy_J", "reference_stress_Pa",
            "reference_temperature_K", "shape_a",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.shape_n) or self.shape_n < 1.0:
            raise ValueError(
                "production shape_n must be finite and at least one to avoid "
                "a zero-stress activation-volume singularity"
            )
        if not math.isfinite(self.floor_fraction) or not 0.0 < self.floor_fraction <= 1.0:
            raise ValueError("floor_fraction must satisfy 0 < f <= 1")
        for name in (
            "enthalpy_temperature_coefficient",
            "stress_temperature_coefficient",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")

    def enthalpy_scale_J(self, temperature_K: float) -> float:
        _check_temperature(temperature_K)
        reduced = (
            temperature_K - self.reference_temperature_K
        ) / self.reference_temperature_K
        return self.reference_enthalpy_J * math.exp(
            -self.enthalpy_temperature_coefficient * reduced
        )

    def stress_scale_Pa(self, temperature_K: float) -> float:
        _check_temperature(temperature_K)
        reduced = (
            temperature_K - self.reference_temperature_K
        ) / self.reference_temperature_K
        return self.reference_stress_Pa * math.exp(
            -self.stress_temperature_coefficient * reduced
        )

    def enthalpy_J(self, driving_stress_Pa: float, temperature_K: float) -> float:
        if not math.isfinite(driving_stress_Pa) or driving_stress_Pa < 0.0:
            raise ValueError("driving_stress_Pa must be finite and nonnegative")
        ratio = driving_stress_Pa / self.stress_scale_Pa(temperature_K)
        shape = math.exp(-self.shape_a * ratio**self.shape_n)
        return self.enthalpy_scale_J(temperature_K) * (
            self.floor_fraction + (1.0 - self.floor_fraction) * shape
        )

    def activation_volume_m3(
        self, driving_stress_Pa: float, temperature_K: float
    ) -> float:
        """Return ``-d(Delta H*)/d(tau)``, which must be nonnegative."""

        if not math.isfinite(driving_stress_Pa) or driving_stress_Pa < 0.0:
            raise ValueError("driving_stress_Pa must be finite and nonnegative")
        scale = self.enthalpy_scale_J(temperature_K)
        stress_scale = self.stress_scale_Pa(temperature_K)
        ratio = driving_stress_Pa / stress_scale
        if ratio == 0.0:
            if self.shape_n < 1.0:
                return math.inf
            if self.shape_n > 1.0:
                return 0.0
        return (
            scale * (1.0 - self.floor_fraction) * self.shape_a * self.shape_n
            * ratio ** (self.shape_n - 1.0)
            * math.exp(-self.shape_a * ratio**self.shape_n) / stress_scale
        )


@dataclass(frozen=True)
class BoundedActivationEntropy:
    """Small, signed, bounded entropy family in units of ``k_B`` per event."""

    reference_kB: float = 0.0
    temperature_amplitude_kB: float = 0.0
    temperature_width_K: float = 1.0
    reference_temperature_K: float = 300.0

    def __post_init__(self) -> None:
        for name in (
            "reference_kB", "temperature_amplitude_kB",
            "temperature_width_K", "reference_temperature_K",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.temperature_width_K <= 0.0 or self.reference_temperature_K <= 0.0:
            raise ValueError("entropy temperature scales must be positive")

    def entropy_kB(self, temperature_K: float) -> float:
        _check_temperature(temperature_K)
        return self.reference_kB + self.temperature_amplitude_kB * math.tanh(
            (temperature_K - self.reference_temperature_K)
            / self.temperature_width_K
        )


@dataclass(frozen=True)
class ArrheniusMechanism:
    """Forward-minus-unloaded-reverse kinetics for one mechanism.

    A constant entropy and the attempt frequency enter only through their
    identifiable effective prefactor.  Directional symmetry uses the same
    mechanism entropy in both event directions and therefore gives exactly
    zero net rate at zero stress.
    """

    enthalpy: ExpFloorEnthalpy
    entropy: BoundedActivationEntropy
    attempt_frequency_s_inv: float
    event_increment: float = 1.0
    density_exponent: float = 0.0
    density_reference_m2: float = 1.0
    validity_temperature_K: tuple[float, float] = (1.0, math.inf)
    validity_stress_Pa: tuple[float, float] = (0.0, math.inf)

    def __post_init__(self) -> None:
        for name in (
            "attempt_frequency_s_inv", "event_increment",
            "density_reference_m2",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.density_exponent):
            raise ValueError("density_exponent must be finite")
        t0, t1 = self.validity_temperature_K
        s0, s1 = self.validity_stress_Pa
        if not 0.0 < t0 < t1 or not 0.0 <= s0 < s1:
            raise ValueError("invalid Arrhenius validity envelope")

    def effective_prefactor_s_inv(self, temperature_K: float) -> float:
        return self.attempt_frequency_s_inv * math.exp(
            self.entropy.entropy_kB(temperature_K)
        )

    def activation_free_energy_J(
        self, driving_stress_Pa: float, temperature_K: float
    ) -> float:
        self._check_envelope(driving_stress_Pa, temperature_K)
        free = self.enthalpy.enthalpy_J(
            driving_stress_Pa, temperature_K
        ) - KB_J_PER_K * temperature_K * self.entropy.entropy_kB(temperature_K)
        if free < 0.0:
            raise ValueError(
                "activation free energy is negative; an explicit barrierless/drag "
                "transition is required"
            )
        return free

    def directional_event_rates_s_inv(
        self, signed_stress_Pa: float, temperature_K: float,
        density_m2: float | None = None,
    ) -> tuple[float, float]:
        if not math.isfinite(signed_stress_Pa):
            raise ValueError("signed_stress_Pa must be finite")
        magnitude = abs(signed_stress_Pa)
        self._check_envelope(magnitude, temperature_K)
        density_factor = self._density_factor(density_m2)
        # Entropy enters exactly once through Delta G = Delta H - T Delta S.
        # Equivalently this is nu0*exp(Delta S/kB)*exp(-Delta H/kBT).
        prefactor = self.attempt_frequency_s_inv * density_factor
        loaded = prefactor * math.exp(
            -self.activation_free_energy_J(magnitude, temperature_K)
            / (KB_J_PER_K * temperature_K)
        )
        unloaded = prefactor * math.exp(
            -self.activation_free_energy_J(0.0, temperature_K)
            / (KB_J_PER_K * temperature_K)
        )
        if signed_stress_Pa >= 0.0:
            return loaded, unloaded
        return unloaded, loaded

    def one_way_rate_s_inv(
        self, driving_stress_Pa: float, temperature_K: float,
        density_m2: float | None = None,
    ) -> float:
        """Rate of one declared dissipative event channel.

        Recovery, capture, and ordering are not signed glide pairs. Their
        reverse processes require separately declared mechanisms and must not
        be created by subtracting the unloaded event rate.
        """

        self._check_envelope(driving_stress_Pa, temperature_K)
        return (
            self.attempt_frequency_s_inv * self._density_factor(density_m2)
            * math.exp(
                -self.activation_free_energy_J(driving_stress_Pa, temperature_K)
                / (KB_J_PER_K * temperature_K)
            )
        )

    def net_rate_s_inv(
        self, signed_stress_Pa: float, temperature_K: float,
        density_m2: float | None = None,
    ) -> float:
        """Return a generalized dimensionless event increment per second.

        This legacy-compatible constitutive rate is appropriate for slip or
        order parameters whose ``event_increment`` is dimensionless. Spatial
        CDD must instead call :meth:`glide_velocity_m_s` with an explicitly
        dimensioned event distance.
        """

        return self.event_increment * self.net_event_frequency_s_inv(
            signed_stress_Pa, temperature_K, density_m2
        )

    def net_event_frequency_s_inv(
        self, signed_stress_Pa: float, temperature_K: float,
        density_m2: float | None = None,
    ) -> float:
        """Return forward-minus-reverse event frequency in ``s^-1``."""

        forward, reverse = self.directional_event_rates_s_inv(
            signed_stress_Pa, temperature_K, density_m2
        )
        return forward - reverse

    def glide_velocity_m_s(
        self, signed_stress_Pa: float, temperature_K: float,
        event_length_m: float, density_m2: float | None = None,
    ) -> float:
        """Convert signed event frequency to a physical glide velocity."""

        if not math.isfinite(event_length_m) or event_length_m <= 0.0:
            raise ValueError("event_length_m must be finite and positive")
        return event_length_m * self.net_event_frequency_s_inv(
            signed_stress_Pa, temperature_K, density_m2
        )

    def _density_factor(self, density_m2: float | None) -> float:
        if density_m2 is None:
            if self.density_exponent != 0.0:
                raise ValueError("density_m2 is required for nonzero density exponent")
            return 1.0
        if not math.isfinite(density_m2) or density_m2 <= 0.0:
            raise ValueError("density_m2 must be finite and positive")
        return (density_m2 / self.density_reference_m2) ** self.density_exponent

    def _check_envelope(self, stress_Pa: float, temperature_K: float) -> None:
        _check_temperature(temperature_K)
        t0, t1 = self.validity_temperature_K
        s0, s1 = self.validity_stress_Pa
        if not t0 <= temperature_K <= t1 or not s0 <= stress_Pa <= s1:
            raise ValueError("state is outside the declared Arrhenius validity envelope")


def _check_temperature(temperature_K: float) -> None:
    if not math.isfinite(temperature_K) or temperature_K <= 0.0:
        raise ValueError("temperature_K must be finite and positive")


PARAMETER_CLASSIFICATION = {
    "barrier_form": "immutable_framework",
    "reference_enthalpy_J": "generic_development_parameter",
    "reference_stress_Pa": "generic_development_parameter",
    "floor_fraction": "generic_development_parameter",
    "shape_a": "generic_development_parameter",
    "shape_n": "generic_development_parameter",
    "activation_entropy_kB": "generic_development_parameter",
    "attempt_frequency_s_inv": "physical_literature_bound",
    "validity_envelope": "immutable_framework",
}
