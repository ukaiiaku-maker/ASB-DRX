"""Centralized geometric measures for prepared-grain response histories."""

from __future__ import annotations

import numpy as np

from full_model.production.tensorial_nye import spectral_derivatives


CONTOUR_DISPLACEMENT_CONVENTION = "endpoint_average_spectral_perimeter"


def spectral_perimeter_m(support, spacing_m: float) -> float:
    """Diffuse contour measure, integral |grad eta| dA, in metres."""
    gx, gy = spectral_derivatives(np.asarray(support, dtype=float), spacing_m)
    return float(np.sum(
        np.sqrt(gx*gx+gy*gy), dtype=np.longdouble)*spacing_m**2)


def grain_response(initial_supports, final_supports, *, spacing_m: float,
                   represented_thickness_m: float) -> dict:
    """Return volume growth and the registered endpoint-average displacement.

    The historical V58 criterion divides each volume change by the represented
    thickness times the arithmetic mean of the initial and final spectral
    perimeters.  Keeping that convention here prevents classifiers from
    silently switching to the initial-perimeter-only approximation.
    """
    initial = np.asarray(initial_supports, dtype=float)
    final = np.asarray(final_supports, dtype=float)
    if initial.shape != final.shape or initial.ndim != 3:
        raise ValueError("grain supports must have matching (grain, x, y) shape")
    cell_volume = float(spacing_m)**2*float(represented_thickness_m)
    initial_volume = np.sum(initial, axis=(1, 2))*cell_volume
    final_volume = np.sum(final, axis=(1, 2))*cell_volume
    initial_perimeter = np.asarray([
        spectral_perimeter_m(field, spacing_m) for field in initial])
    final_perimeter = np.asarray([
        spectral_perimeter_m(field, spacing_m) for field in final])
    delta = final_volume-initial_volume
    average_denominator = (float(represented_thickness_m)
                           *.5*(initial_perimeter+final_perimeter))
    initial_denominator = (float(represented_thickness_m)*initial_perimeter)
    displacement = np.divide(
        delta, average_denominator, out=np.zeros_like(delta),
        where=average_denominator > 0.0)
    initial_only = np.divide(
        delta, initial_denominator, out=np.zeros_like(delta),
        where=initial_denominator > 0.0)
    growth = np.divide(
        delta, initial_volume, out=np.zeros_like(delta),
        where=initial_volume > 0.0)
    return {
        "convention": CONTOUR_DISPLACEMENT_CONVENTION,
        "initial_volume_m3": initial_volume,
        "final_volume_m3": final_volume,
        "volume_growth_fraction": growth,
        "initial_perimeter_m": initial_perimeter,
        "final_perimeter_m": final_perimeter,
        "equivalent_contour_displacement_m": displacement,
        "initial_perimeter_only_displacement_m": initial_only,
    }
