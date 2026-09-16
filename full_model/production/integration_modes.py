"""Configuration contracts for explicitly integrated production modes."""

from __future__ import annotations


MODE = "v32_existing_boundary_common_state"


def common_front_integration_requested(parameters):
    """Whether one run requests both common Mura and the sparse front."""
    common = bool(parameters.get("v31_asb_common_mura_ledger", False)
                  or parameters.get("v21_common_tensorial_wall_enabled", False)
                  or parameters.get("v22_common_tensorial_wall_enabled", False))
    front = bool(parameters.get("use_sparse_common_front_state", False)
                 and parameters.get("use_sibm_existing_boundary", False))
    return common and front


def validate_common_front_integration(parameters, *, adapter_active=False):
    """Fail closed until one transaction owns every shared signed state.

    The existing sparse-front transaction owns ``rp/rm`` plus scalar forest
    and wall reservoirs.  The common Mura operator additionally owns signed
    forest/wall arrays, beta/Nye/alignment, and junction state.  Running both
    without an adapter silently creates two owners and a next-step mismatch.

    ``adapter_active`` is an internal source capability, deliberately not a
    user parameter.  It can become true only when that transaction exists.
    """
    if not common_front_integration_requested(parameters):
        return "INACTIVE_EXACT_OFF"
    if not bool(parameters.get(MODE, False)):
        raise ValueError(
            "UNSUPPORTED_COMMON_FRONT_DUAL_OWNER: simultaneous common Mura "
            "and existing-boundary sparse-front evolution requires explicit "
            f"{MODE}=true and a unified signed common-state transaction")
    if not adapter_active:
        raise ValueError(
            "V32_COMMON_FRONT_ADAPTER_UNAVAILABLE: the requested existing-"
            "boundary mode cannot update signed forest/wall, beta/Nye/"
            "alignment, junction, and sparse-front reservoirs atomically")
    forbidden = {
        "disable_nucleation": True,
        "use_hazard_nucleation": False,
        "use_stateful_embryos": False,
        "use_component_relabel": False,
    }
    wrong = {key: parameters.get(key) for key, expected in forbidden.items()
             if parameters.get(key) is not expected}
    if wrong:
        raise ValueError(
            "V32_COMMON_FRONT_LABEL_CREATION_FORBIDDEN: "+repr(wrong))
    return "ACTIVE_UNIFIED_EXISTING_BOUNDARY"
