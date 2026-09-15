"""V24 signed line-alignment state and conservative wall-supply operators.

The scalar inventory counts line length [m^-2].  Every signed reservoir also
carries its first line-direction moment ``kappa`` [m^-2], constrained by
``|kappa| <= rho``.  The Nye tensor reconstructed from the state is

    alpha = sum_a b_a tensor (kappa_a^+ - kappa_a^-)  [m^-1].

Capture and topology ordering are separate operators with separate ledgers.
Both move existing line and its alignment; neither accepts an orientation- or
Frank--Bilby-derived target.  Frank--Bilby closure is therefore an independent
postprocessing test, not a source term.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

try:
    from .density_state_map import DensityInventory, SIGNED_RESERVOIRS
    from .tensorial_nye import rotated_system_fields
except ImportError:
    from density_state_map import DensityInventory, SIGNED_RESERVOIRS
    from tensorial_nye import rotated_system_fields


@dataclass(frozen=True)
class ReservoirAlignmentState:
    """First line-direction moments for every signed scalar reservoir."""

    mobile_plus_m2: np.ndarray
    mobile_minus_m2: np.ndarray
    forest_plus_m2: np.ndarray
    forest_minus_m2: np.ndarray
    wall_tangle_plus_m2: np.ndarray
    wall_tangle_minus_m2: np.ndarray
    wall_ordered_plus_m2: np.ndarray
    wall_ordered_minus_m2: np.ndarray

    def validate(self, inventory: DensityInventory, family_count: int,
                 tolerance=2e-13):
        scalar_shape = inventory.validate(family_count,
                                           inventory.junction_m2.shape[-1])
        for name in SIGNED_RESERVOIRS:
            moment = np.asarray(getattr(self, name), dtype=float)
            density = np.asarray(getattr(inventory, name), dtype=float)
            if moment.shape != scalar_shape+(3,) or np.any(~np.isfinite(moment)):
                raise ValueError(f"inadmissible alignment reservoir: {name}")
            excess = np.linalg.norm(moment, axis=-1)-density
            scale = np.maximum(density, 1.0)
            if np.any(excess > float(tolerance)*scale):
                raise ValueError(f"alignment magnitude exceeds line density: {name}")
        return scalar_shape


def zero_alignment_state(inventory, family_count):
    shape = inventory.validate(family_count, inventory.junction_m2.shape[-1])
    zero = np.zeros(shape+(3,))
    return ReservoirAlignmentState(**{
        name: zero.copy() for name in SIGNED_RESERVOIRS})


def alignment_checkpoint_arrays(alignments, prefix="v24_alignment__"):
    return {prefix+name: np.asarray(getattr(alignments, name))
            for name in SIGNED_RESERVOIRS}


def alignment_from_checkpoint_arrays(mapping, inventory, family_count,
                                     prefix="v24_alignment__"):
    missing = [name for name in SIGNED_RESERVOIRS if prefix+name not in mapping]
    if missing:
        raise ValueError("incomplete V24 alignment restart: "+", ".join(missing))
    result = ReservoirAlignmentState(**{
        name: np.asarray(mapping[prefix+name]).copy()
        for name in SIGNED_RESERVOIRS})
    result.validate(inventory, family_count)
    return result


def aligned_state_from_directions(inventory, directions):
    """Manufactured helper: assign declared unit directions to all line.

    This helper is for initial conditions and verification.  It performs no
    reconstruction from orientation and cannot be called by an evolution
    residual without an explicit user-supplied direction field.
    """
    direction = np.asarray(directions, dtype=float)
    scalar_shape = np.asarray(inventory.mobile_plus_m2).shape
    if direction.shape != scalar_shape+(3,):
        raise ValueError("directions require grid x family x vector layout")
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    if np.any(~np.isfinite(direction)) or np.any(norm <= 0.0):
        raise ValueError("line directions must be finite and nonzero")
    unit = direction/norm
    result = ReservoirAlignmentState(**{
        name: np.asarray(getattr(inventory, name))[..., None]*unit
        for name in SIGNED_RESERVOIRS})
    result.validate(inventory, scalar_shape[-1])
    return result


def reservoir_nye_m1(alignments, systems, orientation_rad):
    """Return reservoir-resolved and total Nye tensors from evolved moments."""
    burgers, _, _ = rotated_system_fields(systems, orientation_rad)
    parts = {}
    for stem in ("mobile", "forest", "wall_tangle", "wall_ordered"):
        signed = (np.asarray(getattr(alignments, f"{stem}_plus_m2"))
                  -np.asarray(getattr(alignments, f"{stem}_minus_m2")))
        parts[stem] = np.einsum("...ai,...aj->...ij", burgers, signed)
    parts["total"] = sum(parts.values())
    return parts


def _accepted_extent(requested_m2, donor_m2):
    request = np.maximum(np.asarray(requested_m2, dtype=float), 0.0)
    donor = np.asarray(donor_m2, dtype=float)
    if request.shape != donor.shape or np.any(~np.isfinite(request)):
        raise ValueError("requested transfer must match donor and be finite/nonnegative")
    return np.minimum(request, donor)


def _transfer_signed_reservoir(inventory, alignments, source_stem,
                               destination_stem, requested, family_count):
    density_updates = {}; alignment_updates = {}; ledgers = {}
    for sign in ("plus", "minus"):
        name0 = f"{source_stem}_{sign}_m2"
        name1 = f"{destination_stem}_{sign}_m2"
        donor = np.asarray(getattr(inventory, name0), dtype=float)
        extent = _accepted_extent(requested[sign], donor)
        donor_alignment = np.asarray(getattr(alignments, name0), dtype=float)
        fraction = np.divide(extent, donor, out=np.zeros_like(extent), where=donor > 0)
        alignment_extent = fraction[..., None]*donor_alignment
        density_updates[name0] = donor-extent
        density_updates[name1] = np.asarray(getattr(inventory, name1))+extent
        alignment_updates[name0] = donor_alignment-alignment_extent
        alignment_updates[name1] = (np.asarray(getattr(alignments, name1))
                                    +alignment_extent)
        ledgers[sign] = {
            "accepted_line_m2": extent,
            "accepted_alignment_m2": alignment_extent,
            "scalar_residual_m2": (density_updates[name0]+density_updates[name1]
                                    -donor-np.asarray(getattr(inventory, name1))),
            "alignment_residual_m2": (
                alignment_updates[name0]+alignment_updates[name1]
                -donor_alignment-np.asarray(getattr(alignments, name1))),
        }
    new_inventory = replace(inventory, **density_updates)
    new_alignments = replace(alignments, **alignment_updates)
    new_alignments.validate(new_inventory, family_count)
    return new_inventory, new_alignments, ledgers


def accepted_transport_capture(inventory, alignments, captured_plus_m2,
                               captured_minus_m2, systems, orientation_rad):
    """Capture mobile line crossing a physical trap into the tangle pool.

    The caller must calculate ``captured_*`` from a transport flux and a
    declared physical capture region.  This operator only accepts and limits
    that crossing; it has no orientation or target-Nye argument.
    """
    before = reservoir_nye_m1(alignments, systems, orientation_rad)
    updated, aligned, ledger = _transfer_signed_reservoir(
        inventory, alignments, "mobile", "wall_tangle",
        {"plus": captured_plus_m2, "minus": captured_minus_m2}, len(systems))
    after = reservoir_nye_m1(aligned, systems, orientation_rad)
    return updated, aligned, {
        "operator": "transport_capture",
        "source": "accepted crossing of declared physical capture support",
        "sign": ledger,
        "total_nye_residual_m1": after["total"]-before["total"],
        "reservoir_nye_change_m1": {
            key: after[key]-before[key] for key in before if key != "total"},
    }


def conservative_transport_capture_step(
        inventory, alignments, velocity_plus_m_s, velocity_minus_m_s,
        capture_support, systems, orientation_rad, spacing_m, dt_s):
    """Periodic donor-cell transport with capture only on entry to a trap.

    Mobile line crossing from outside to inside ``capture_support`` is diverted
    to the destination cell's tangle reservoir.  All other face transfers stay
    mobile.  Scalar line and its alignment moment traverse the identical face
    event.  The declared CFL restriction makes every donor update nonnegative.

    The capture support is an externally declared physical trap geometry; it
    must not be inferred from orientation, target Nye, or a future wall label.
    """
    if spacing_m <= 0.0 or dt_s < 0.0:
        raise ValueError("positive spacing and nonnegative step required")
    shape = inventory.validate(len(systems), inventory.junction_m2.shape[-1])
    alignments.validate(inventory, len(systems))
    support = np.asarray(capture_support, dtype=bool)
    if support.shape != shape[:2]:
        raise ValueError("capture support must match spatial grid")
    velocities = {
        "plus": np.asarray(velocity_plus_m_s, dtype=float),
        "minus": np.asarray(velocity_minus_m_s, dtype=float),
    }
    for velocity in velocities.values():
        if velocity.shape != shape+(2,) or np.any(~np.isfinite(velocity)):
            raise ValueError("velocity requires grid x family x in-plane layout")
        courant = dt_s*np.sum(np.abs(velocity), axis=-1)/spacing_m
        if np.any(courant > 1.0+5e-15):
            raise ValueError("donor-cell transport violates multidimensional CFL <= 1")

    density_updates = {}; alignment_updates = {}; sign_ledgers = {}
    for sign in ("plus", "minus"):
        mobile_name = f"mobile_{sign}_m2"
        tangle_name = f"wall_tangle_{sign}_m2"
        mobile0 = np.asarray(getattr(inventory, mobile_name), dtype=float)
        tangle0 = np.asarray(getattr(inventory, tangle_name), dtype=float)
        amobile0 = np.asarray(getattr(alignments, mobile_name), dtype=float)
        atangle0 = np.asarray(getattr(alignments, tangle_name), dtype=float)
        mobile = mobile0.copy(); tangle = tangle0.copy()
        amobile = amobile0.copy(); atangle = atangle0.copy()
        captured = np.zeros_like(mobile)
        captured_alignment = np.zeros_like(amobile)
        velocity = velocities[sign]
        nx, ny, families = shape
        # Each donor's outgoing amounts are based on the beginning-of-step
        # state. This makes the event independent of iteration ordering.
        for i in range(nx):
            for j in range(ny):
                for family in range(families):
                    rho0 = mobile0[i, j, family]
                    if rho0 == 0.0:
                        continue
                    for axis, size in ((0, nx), (1, ny)):
                        speed = velocity[i, j, family, axis]
                        if speed == 0.0:
                            continue
                        destination = [i, j]
                        destination[axis] = ((i if axis == 0 else j)
                                             +(1 if speed > 0 else -1)) % size
                        di, dj = destination
                        fraction = dt_s*abs(speed)/spacing_m
                        amount = fraction*rho0
                        alignment_amount = fraction*amobile0[i, j, family]
                        mobile[i, j, family] -= amount
                        amobile[i, j, family] -= alignment_amount
                        if support[di, dj] and not support[i, j]:
                            tangle[di, dj, family] += amount
                            atangle[di, dj, family] += alignment_amount
                            captured[di, dj, family] += amount
                            captured_alignment[di, dj, family] += alignment_amount
                        else:
                            mobile[di, dj, family] += amount
                            amobile[di, dj, family] += alignment_amount
        # Roundoff at an exhausted donor may be slightly negative only at the
        # scale of machine precision; anything larger is a failed CFL ledger.
        scale = max(float(np.max(mobile0)), 1.0)
        if np.min(mobile) < -32*np.finfo(float).eps*scale:
            raise RuntimeError("transport produced negative mobile density")
        mobile = np.maximum(mobile, 0.0)
        density_updates[mobile_name] = mobile
        density_updates[tangle_name] = tangle
        alignment_updates[mobile_name] = amobile
        alignment_updates[tangle_name] = atangle
        sign_ledgers[sign] = {
            "captured_line_m2": captured,
            "captured_alignment_m2": captured_alignment,
            "global_scalar_residual_line_per_thickness": float(
                np.sum(mobile+tangle-mobile0-tangle0)*spacing_m**2),
            "global_alignment_residual_line_per_thickness": np.sum(
                amobile+atangle-amobile0-atangle0, axis=(0, 1, 2))*spacing_m**2,
        }
    updated = replace(inventory, **density_updates)
    aligned = replace(alignments, **alignment_updates)
    aligned.validate(updated, len(systems))
    before = reservoir_nye_m1(alignments, systems, orientation_rad)
    after = reservoir_nye_m1(aligned, systems, orientation_rad)
    return updated, aligned, {
        "operator": "finite_volume_transport_and_entry_capture",
        "capture_geometry_source": "caller-declared physical support",
        "sign": sign_ledgers,
        "global_nye_integral_residual_m": np.sum(
            after["total"]-before["total"], axis=(0, 1))*spacing_m**2,
        "local_nye_change_m1": after["total"]-before["total"],
    }


def accepted_topology_ordering(inventory, alignments, ordered_plus_m2,
                               ordered_minus_m2, systems, orientation_rad):
    """Convert captured tangle line to ordered line without creating alignment.

    This is an explicit topology/capture handoff, not a direction generator.
    Unpolarized tangle remains unpolarized.  A future reorientation mechanism
    must provide a separately ledgered spatial/topological Nye source.
    """
    before = reservoir_nye_m1(alignments, systems, orientation_rad)
    updated, aligned, ledger = _transfer_signed_reservoir(
        inventory, alignments, "wall_tangle", "wall_ordered",
        {"plus": ordered_plus_m2, "minus": ordered_minus_m2}, len(systems))
    after = reservoir_nye_m1(aligned, systems, orientation_rad)
    return updated, aligned, {
        "operator": "topology_ordering",
        "source": "existing captured tangle line and its evolved alignment",
        "sign": ledger,
        "total_nye_residual_m1": after["total"]-before["total"],
        "reservoir_nye_change_m1": {
            key: after[key]-before[key] for key in before if key != "total"},
    }


def maximum_ledger_residual(ledger):
    scalar = max(float(np.max(np.abs(item["scalar_residual_m2"])))
                 for item in ledger["sign"].values())
    alignment = max(float(np.max(np.abs(item["alignment_residual_m2"])))
                    for item in ledger["sign"].values())
    nye = float(np.max(np.abs(ledger["total_nye_residual_m1"])))
    return {"scalar_line_m2": scalar, "alignment_m2": alignment,
            "total_nye_m1": nye}
