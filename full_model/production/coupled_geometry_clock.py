"""Common-clock composition of bulk evolution and prepared line geometry."""

from __future__ import annotations

import numpy as np

from .state_dependent_subcell import accepted_state_dependent_subcell_x_faces
from .v24_mechanical_wall import accepted_v24_mechanical_step


def _bulk_exposure(
        state, duration_s, driving, capture_support, systems, topologies,
        common_parameters, extensive_parameters, topology_kinetics, *,
        maximum_substeps=4096):
    current = state; remaining = float(duration_s); rows = []
    tolerance = 64*np.finfo(float).eps*max(float(duration_s), 1e-300)
    for index in range(int(maximum_substeps)):
        if remaining <= tolerance:
            return current, {
                "accepted": True, "requested_exposure_s": float(duration_s),
                "accepted_exposure_s": float(duration_s), "substeps": rows,
            }
        candidate, audit = accepted_v24_mechanical_step(
            current, driving, capture_support, systems, topologies,
            common_parameters, extensive_parameters, topology_kinetics,
            remaining, topology_route_enabled=False,
            mura_transport_operator="compatible_dealiased")
        accepted = float(audit.get("accepted_dt_s", 0.0))
        if accepted <= 0.0 or accepted > remaining+tolerance:
            return state, {
                "accepted": False,
                "classification": "BULK_EXPOSURE_REJECTED_COMPLETE_MACRO_ROLLBACK",
                "requested_exposure_s": float(duration_s),
                "accepted_exposure_s": 0.0,
                "unpublished_attempted_prefix_s": float(duration_s-remaining),
                "substeps": rows, "rejected_audit": audit,
            }
        rows.append({"substep": index+1, "accepted_duration_s": accepted,
                     "audit": audit})
        current = candidate; remaining -= accepted
    raise RuntimeError("bulk common-clock exposure exceeded substep limit")


def _geometry_exposure(
        state, duration_s, directions, systems, topologies, driving,
        common_parameters, extensive_parameters, kinetics, *,
        quadrature_order, maximum_retries=64):
    current = state; remaining = float(duration_s); rows = []
    tolerance = 64*np.finfo(float).eps*max(float(duration_s), 1e-300)
    for index in range(int(maximum_retries)):
        if remaining <= tolerance:
            return current, {
                "accepted": True, "requested_exposure_s": float(duration_s),
                "accepted_exposure_s": float(duration_s), "substeps": rows,
            }
        candidate, ledger = accepted_state_dependent_subcell_x_faces(
            current, directions, systems, topologies, driving,
            common_parameters, extensive_parameters, kinetics, remaining,
            quadrature_order=quadrature_order)
        consumed = float(ledger.get("consumed_duration_s", 0.0))
        if not ledger.get("accepted", False) or consumed <= 0.0:
            return state, {
                "accepted": False,
                "classification": (
                    "GEOMETRY_EXPOSURE_REJECTED_COMPLETE_MACRO_ROLLBACK"),
                "requested_exposure_s": float(duration_s),
                "accepted_exposure_s": 0.0,
                "unpublished_attempted_prefix_s": float(duration_s-remaining),
                "substeps": rows, "rejected_ledger": ledger,
            }
        if consumed > remaining+tolerance:
            raise RuntimeError("geometry operator consumed more than requested")
        rows.append({"retry": index+1, "accepted_duration_s": consumed,
                     "ledger": ledger})
        current = candidate; remaining -= consumed
        if ledger.get("macro_complete", consumed >= remaining-tolerance):
            # A stalled identity may consume the complete remaining exposure.
            remaining = max(remaining, 0.0)
    raise RuntimeError("geometry common-clock exposure exceeded retry limit")


def accepted_common_clock_subcell_macro(
        state, duration_s, directions, systems, topologies, driving,
        capture_support, common_parameters, extensive_parameters,
        topology_kinetics, geometry_kinetics, *, quadrature_order=16):
    """Advance B(H/2) -> G(H) -> B(H/2) over physical elapsed time H.

    Bulk and geometry operator exposures are ledgered separately.  Any
    unfillable exposure rolls the complete macro back to ``state``.
    """
    duration = float(duration_s)
    if duration <= 0.0 or not np.isfinite(duration):
        raise ValueError("common-clock duration must be finite and positive")
    first, bulk_a = _bulk_exposure(
        state, .5*duration, driving, capture_support, systems, topologies,
        common_parameters, extensive_parameters, topology_kinetics)
    if not bulk_a["accepted"]:
        return state, {**bulk_a, "operator": "B_half_G_B_half",
                       "physical_elapsed_time_s": 0.0,
                       "complete_macro_rollback": True}
    geometric, geometry = _geometry_exposure(
        first, duration, directions, systems, topologies, driving,
        common_parameters, extensive_parameters, geometry_kinetics,
        quadrature_order=quadrature_order)
    if not geometry["accepted"]:
        return state, {**geometry, "operator": "B_half_G_B_half",
                       "first_bulk_attempt": bulk_a,
                       "physical_elapsed_time_s": 0.0,
                       "complete_macro_rollback": True}
    final, bulk_b = _bulk_exposure(
        geometric, .5*duration, driving, capture_support, systems, topologies,
        common_parameters, extensive_parameters, topology_kinetics)
    if not bulk_b["accepted"]:
        return state, {**bulk_b, "operator": "B_half_G_B_half",
                       "first_bulk_attempt": bulk_a,
                       "geometry_attempt": geometry,
                       "physical_elapsed_time_s": 0.0,
                       "complete_macro_rollback": True}
    final.validate(systems, topologies)
    return final, {
        "operator": "B_half_G_B_half",
        "accepted": True,
        "classification": "COMMON_CLOCK_MACRO_ACCEPTED",
        "physical_elapsed_time_s": duration,
        "bulk_operator_exposure_s": duration,
        "geometry_operator_exposure_s": duration,
        "composition": "B(H/2)->G(H)->B(H/2)",
        "elapsed_clock_is_H_not_2H": True,
        "complete_macro_rollback": False,
        "first_bulk": bulk_a,
        "geometry": geometry,
        "second_bulk": bulk_b,
    }
