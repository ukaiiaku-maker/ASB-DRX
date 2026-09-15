"""Common-functional HAGB migration and circular-bulge fixtures.

The graph state represents an existing boundary only.  It never allocates an
orientation or grain label.  Positive normal motion advances the already
declared left grain into the right grain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

import numpy as np

try:
    from .arrhenius_kinetics import ActivatedProcess
    from .moving_front import activated_front_fraction
except ImportError:  # pragma: no cover
    from arrhenius_kinetics import ActivatedProcess
    from moving_front import activated_front_fraction


@dataclass(frozen=True)
class SIBMParameters:
    boundary_energy_J_m2: float
    mobility_prefactor_m4_J_s: float
    process: ActivatedProcess
    activation_enthalpy_0_J: float
    critical_pressure_Pa: float
    exp_a: float
    exp_n: float
    exp_floor: float
    represented_thickness_m: float
    drag_pressure_Pa: float = 0.0


@dataclass(frozen=True)
class BoundaryLedger:
    free_energy_before_J: float = 0.0
    free_energy_after_J: float = 0.0
    free_energy_change_J: float = 0.0
    dissipated_energy_J: float = 0.0
    heat_released_J: float = 0.0
    energy_closure_J: float = 0.0
    signed_area_change_m2: float = 0.0
    maximum_abs_pressure_Pa: float = 0.0


@dataclass(frozen=True)
class BoundaryGraphState:
    height_m: np.ndarray
    left_label: int
    right_label: int
    left_orientation_rad: float
    right_orientation_rad: float
    ledger: BoundaryLedger = BoundaryLedger()


@dataclass(frozen=True)
class ResolvedHAGB:
    parent_label: int
    child_label: int
    parent_mean_density_m2: float
    child_mean_density_m2: float
    parent_pure_core_cells: int
    child_pure_core_cells: int
    misorientation_rad: float
    boundary_cells: int
    boundary_band: np.ndarray
    centre_index: tuple[int, int]
    advance_direction_index: tuple[int, int]


def select_resolved_hagb(eta, orientations_rad, density_m2, *,
                         purity_threshold=0.8, min_pure_core_cells=16,
                         min_misorientation_deg=15.0,
                         parent_label_override=None,
                         child_label_override=None):
    """Select a resolved HAGB without using unstable argmax-label edges."""
    eta = np.asarray(eta, dtype=float)
    orientations = np.asarray(orientations_rad, dtype=float)
    density = np.asarray(density_m2, dtype=float)
    if eta.ndim != 3 or density.shape != eta.shape[:2]:
        raise ValueError("eta and density grids are inconsistent")
    if orientations.shape != (eta.shape[2],):
        raise ValueError("one orientation is required per phase")
    pure = [eta[:, :, g] >= purity_threshold for g in range(eta.shape[2])]
    resolved = [g for g, mask in enumerate(pure)
                if int(np.sum(mask)) >= int(min_pure_core_cells)]
    means = {g: float(np.mean(density[pure[g]])) for g in resolved}
    period = 0.5*np.pi
    viable = []
    for ia, a in enumerate(resolved):
        for b in resolved[ia+1:]:
            band = ((eta[:, :, a] > 0.15) & (eta[:, :, b] > 0.15)
                    & (eta[:, :, a]+eta[:, :, b] > 0.70))
            count = int(np.sum(band))
            mis = abs((orientations[a]-orientations[b]+0.5*period)
                      % period-0.5*period)
            if count >= 8 and np.rad2deg(mis) >= min_misorientation_deg:
                viable.append((abs(means[a]-means[b]), count, (a, b), mis, band))
    if not viable:
        raise ValueError("no resolved existing HAGB satisfies the SIBM criterion")
    if ((parent_label_override is None) != (child_label_override is None)):
        raise ValueError("parent and child label overrides must be supplied together")
    if parent_label_override is not None:
        declared = (int(parent_label_override), int(child_label_override))
        if declared[0] == declared[1]:
            raise ValueError("parent and child labels must be distinct")
        candidates = [row for row in viable if set(row[2]) == set(declared)]
        if not candidates:
            raise ValueError("declared parent/child labels do not form a resolved HAGB")
        _, count, pair, mis, band = max(
            candidates, key=lambda row: (row[0], row[1], row[2]))
        parent, child = declared
    else:
        _, count, pair, mis, band = max(
            viable, key=lambda row: (row[0], row[1], row[2]))
        # Choose the lower-energy child deterministically, then take the other
        # member as parent.  Computing both extrema independently can select the
        # same label when the phase means tie exactly, which invalidates the
        # equal-stored-energy control.
        child = min(pair, key=lambda g: (means[g], g))
        parent = pair[1] if child == pair[0] else pair[0]
    difference = eta[:, :, child]-eta[:, :, parent]
    gx = 0.5*(np.roll(difference, -1, axis=0)-np.roll(difference, 1, axis=0))
    gy = 0.5*(np.roll(difference, -1, axis=1)-np.roll(difference, 1, axis=1))
    magnitude = np.hypot(gx, gy)
    centre = np.unravel_index(
        int(np.argmax(np.where(band, magnitude, -np.inf))), band.shape)
    if abs(gx[centre]) >= abs(gy[centre]):
        advance = (-int(np.sign(gx[centre]) or 1), 0)
    else:
        advance = (0, -int(np.sign(gy[centre]) or 1))
    return ResolvedHAGB(
        parent, child, means[parent], means[child], int(np.sum(pure[parent])),
        int(np.sum(pure[child])), float(mis), count, band,
        (int(centre[0]), int(centre[1])), advance)


def _derivatives_periodic(values, spacing_m):
    f = np.asarray(values, dtype=float)
    first = (np.roll(f, -1)-np.roll(f, 1))/(2.0*spacing_m)
    second = (np.roll(f, -1)-2.0*f+np.roll(f, 1))/(spacing_m**2)
    return first, second


def curvature(state, spacing_m):
    """Signed curvature: a positive upward bulge has positive curvature."""
    slope, second = _derivatives_periodic(state.height_m, spacing_m)
    return -second/np.power(1.0+slope*slope, 1.5)


def common_free_energy_J(state, spacing_m, parameters, stored_difference_Pa,
                         frank_bilby_pressure_Pa=0.0):
    """Declared graph functional whose derivative drives both directions."""
    slope, _ = _derivatives_periodic(state.height_m, spacing_m)
    length = float(np.sum(np.sqrt(1.0+slope*slope))*spacing_m)
    drive = np.asarray(stored_difference_Pa, dtype=float) \
        - np.asarray(frank_bilby_pressure_Pa, dtype=float)
    bulk = float(np.sum(drive*np.asarray(state.height_m))*spacing_m)
    t = parameters.represented_thickness_m
    return t*(parameters.boundary_energy_J_m2*length-bulk)


def common_pressure_Pa(state, spacing_m, parameters, stored_difference_Pa,
                       frank_bilby_pressure_Pa=0.0):
    """Negative variational derivative per swept volume."""
    drive = np.asarray(stored_difference_Pa, dtype=float) \
        - np.asarray(frank_bilby_pressure_Pa, dtype=float)
    pressure = drive-parameters.boundary_energy_J_m2*curvature(state, spacing_m)
    if parameters.drag_pressure_Pa:
        pressure = pressure-parameters.drag_pressure_Pa*np.sign(pressure)
        pressure = np.where(np.sign(pressure) == np.sign(drive), pressure, 0.0)
    return np.asarray(pressure, dtype=float)


def advance_boundary(state, *, spacing_m, dt_s, temperature_K, parameters,
                     stored_difference_Pa, frank_bilby_pressure_Pa=0.0):
    """Take one energy-monotone, forward/reverse-symmetric HAGB step."""
    h0 = np.asarray(state.height_m, dtype=float)
    if h0.ndim != 1 or h0.size < 5 or not np.all(np.isfinite(h0)):
        raise ValueError("boundary height must be a finite 1-D periodic graph")
    if spacing_m <= 0.0 or dt_s <= 0.0 or temperature_K <= 0.0:
        raise ValueError("spacing, timestep, and temperature must be positive")
    before = common_free_energy_J(
        state, spacing_m, parameters, stored_difference_Pa,
        frank_bilby_pressure_Pa)
    pressure = common_pressure_Pa(
        state, spacing_m, parameters, stored_difference_Pa,
        frank_bilby_pressure_Pa)
    fraction = activated_front_fraction(
        parameters.process, np.abs(pressure), temperature_K, dt_s,
        h0_J=parameters.activation_enthalpy_0_J,
        critical_stress_pa=parameters.critical_pressure_Pa,
        exp_a=parameters.exp_a, exp_n=parameters.exp_n,
        exp_floor=parameters.exp_floor)
    kinetic_factor = fraction/max(parameters.process.attempt_frequency_s*dt_s, 1e-300)
    velocity = parameters.mobility_prefactor_m4_J_s*kinetic_factor*pressure
    # Backtrack only for numerical energy monotonicity; this changes temporal
    # resolution, not the critical condition or mobility parameter.
    scale = 1.0
    for _ in range(30):
        trial = BoundaryGraphState(
            h0+scale*dt_s*velocity, state.left_label, state.right_label,
            state.left_orientation_rad, state.right_orientation_rad,
            state.ledger)
        after = common_free_energy_J(
            trial, spacing_m, parameters, stored_difference_Pa,
            frank_bilby_pressure_Pa)
        if after <= before+256.0*math.ulp(max(abs(before), 1e-300)):
            break
        scale *= 0.5
    else:
        raise RuntimeError("no energy-monotone SIBM timestep exists")
    change = after-before
    dissipated = max(-change, 0.0)
    old = state.ledger
    ledger = BoundaryLedger(
        before, after, change, dissipated, 0.0, -change-dissipated,
        old.signed_area_change_m2
        +float(np.sum(trial.height_m-h0)*spacing_m),
        max(old.maximum_abs_pressure_Pa, float(np.max(np.abs(pressure)))))
    return BoundaryGraphState(
        trial.height_m, state.left_label, state.right_label,
        state.left_orientation_rad, state.right_orientation_rad, ledger), velocity*scale


def circular_bulge_pressure_Pa(radius_m, stored_difference_Pa,
                               frank_bilby_pressure_Pa, parameters):
    """Exact local pressure for a circular 2-D bulge, Rc=gamma/drive."""
    if radius_m <= 0.0:
        raise ValueError("bulge radius must be positive")
    return (stored_difference_Pa-frank_bilby_pressure_Pa
            -parameters.boundary_energy_J_m2/radius_m)


def state_json(state):
    return json.dumps({
        "schema": "full-v34-sibm-boundary/v1",
        "left_label": state.left_label, "right_label": state.right_label,
        "left_orientation_rad": state.left_orientation_rad,
        "right_orientation_rad": state.right_orientation_rad,
        "ledger": asdict(state.ledger)}, sort_keys=True, separators=(",", ":"))


def state_from_json(metadata, height_m):
    raw = json.loads(str(metadata))
    if raw.get("schema") != "full-v34-sibm-boundary/v1":
        raise ValueError("unsupported SIBM boundary schema")
    return BoundaryGraphState(
        np.asarray(height_m, dtype=float).copy(), int(raw["left_label"]),
        int(raw["right_label"]), float(raw["left_orientation_rad"]),
        float(raw["right_orientation_rad"]), BoundaryLedger(**raw["ledger"]))
