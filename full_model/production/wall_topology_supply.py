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
    from .density_state_map import (
        DensityInventory, SIGNED_RESERVOIRS, derived_density_fields,
    )
    from .tensorial_nye import (
        junction_closure_metrics, rotated_system_fields, rotation_z,
    )
except ImportError:
    from density_state_map import (
        DensityInventory, SIGNED_RESERVOIRS, derived_density_fields,
    )
    from tensorial_nye import (
        junction_closure_metrics, rotated_system_fields, rotation_z,
    )


TOPOLOGY_MOMENT_FIELDS = (
    "wall_turning_nodes_plus_m3", "wall_turning_nodes_minus_m3",
    "wall_curvature_plus_m3", "wall_curvature_minus_m3",
)


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
    junction_alignment_m2: np.ndarray
    wall_turning_nodes_plus_m3: np.ndarray
    wall_turning_nodes_minus_m3: np.ndarray
    wall_curvature_plus_m3: np.ndarray
    wall_curvature_minus_m3: np.ndarray

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
        junction = np.asarray(self.junction_alignment_m2, dtype=float)
        if junction.shape != inventory.junction_m2.shape+(3,) or np.any(
                ~np.isfinite(junction)):
            raise ValueError("inadmissible junction alignment reservoir")
        for name in TOPOLOGY_MOMENT_FIELDS:
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != scalar_shape or np.any(~np.isfinite(value)) \
                    or np.any(value < 0.0):
                raise ValueError(f"inadmissible topology moment: {name}")
        return scalar_shape


def zero_alignment_state(inventory, family_count):
    shape = inventory.validate(family_count, inventory.junction_m2.shape[-1])
    zero = np.zeros(shape+(3,))
    return ReservoirAlignmentState(**{
        **{name: zero.copy() for name in SIGNED_RESERVOIRS},
        "junction_alignment_m2": np.zeros(inventory.junction_m2.shape+(3,)),
        **{name: np.zeros(shape) for name in TOPOLOGY_MOMENT_FIELDS},
    })


def alignment_checkpoint_arrays(alignments, prefix="v24_alignment__"):
    names = SIGNED_RESERVOIRS+("junction_alignment_m2",)+TOPOLOGY_MOMENT_FIELDS
    return {prefix+name: np.asarray(getattr(alignments, name)) for name in names}


def alignment_from_checkpoint_arrays(mapping, inventory, family_count,
                                     prefix="v24_alignment__"):
    names = SIGNED_RESERVOIRS+("junction_alignment_m2",)+TOPOLOGY_MOMENT_FIELDS
    missing = [name for name in names if prefix+name not in mapping]
    if missing:
        raise ValueError("incomplete V24 alignment restart: "+", ".join(missing))
    result = ReservoirAlignmentState(**{
        name: np.asarray(mapping[prefix+name]).copy()
        for name in names})
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
        **{name: np.asarray(getattr(inventory, name))[..., None]*unit
           for name in SIGNED_RESERVOIRS},
        "junction_alignment_m2": np.zeros(inventory.junction_m2.shape+(3,)),
        **{name: np.zeros(scalar_shape) for name in TOPOLOGY_MOMENT_FIELDS},
    })
    result.validate(inventory, scalar_shape[-1])
    return result


def reservoir_nye_m1(alignments, systems, orientation_rad, topologies=()):
    """Return reservoir-resolved and total Nye tensors from evolved moments."""
    burgers, _, _ = rotated_system_fields(systems, orientation_rad)
    parts = {}
    for stem in ("mobile", "forest", "wall_tangle", "wall_ordered"):
        signed = (np.asarray(getattr(alignments, f"{stem}_plus_m2"))
                  -np.asarray(getattr(alignments, f"{stem}_minus_m2")))
        parts[stem] = np.einsum("...ai,...aj->...ij", burgers, signed)
    if len(topologies) != alignments.junction_alignment_m2.shape[-2]:
        raise ValueError("topology list does not match junction alignment state")
    if topologies:
        product_burgers0 = np.stack(
            [item.product_burgers_m for item in topologies])
        product_burgers = np.einsum(
            "...ij,aj->...ai", rotation_z(orientation_rad), product_burgers0)
        parts["junction"] = np.einsum(
            "...ai,...aj->...ij", product_burgers,
            np.asarray(alignments.junction_alignment_m2))
    else:
        parts["junction"] = np.zeros(orientation_rad.shape+(3, 3))
    parts["total"] = sum(parts.values())
    return parts


def validate_junction_alignment(inventory, alignments, topologies,
                                tolerance=2e-13):
    """Enforce product-line first moment <= declared junction line content."""
    if len(topologies) != inventory.junction_m2.shape[-1]:
        raise ValueError("topology list does not match junction inventory")
    multiplicity = np.asarray([
        item.product_line_multiplicity for item in topologies])
    bound = np.asarray(inventory.junction_m2)*multiplicity
    norm = np.linalg.norm(alignments.junction_alignment_m2, axis=-1)
    if np.any(norm-bound > float(tolerance)*np.maximum(bound, 1.0)):
        raise ValueError("junction alignment exceeds declared product line")
    return True


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
                               captured_minus_m2, systems, orientation_rad,
                               topologies=()):
    """Capture mobile line crossing a physical trap into the tangle pool.

    The caller must calculate ``captured_*`` from a transport flux and a
    declared physical capture region.  This operator only accepts and limits
    that crossing; it has no orientation or target-Nye argument.
    """
    before = reservoir_nye_m1(alignments, systems, orientation_rad, topologies)
    updated, aligned, ledger = _transfer_signed_reservoir(
        inventory, alignments, "mobile", "wall_tangle",
        {"plus": captured_plus_m2, "minus": captured_minus_m2}, len(systems))
    after = reservoir_nye_m1(aligned, systems, orientation_rad, topologies)
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
        capture_support, systems, orientation_rad, spacing_m, dt_s,
        topologies=()):
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
        # Each donor's outgoing amounts are based on the beginning-of-step
        # state. Vectorized rolls perform the same periodic face scatter as a
        # cell loop while making long exposure calculations practical.
        outside = ~support
        for axis in (0, 1):
            component = velocity[..., axis]
            for step, positive_speed in ((1, np.maximum(component, 0.0)),
                                         (-1, np.maximum(-component, 0.0))):
                fraction = dt_s*positive_speed/spacing_m
                amount = fraction*mobile0
                alignment_amount = fraction[..., None]*amobile0
                destination_support = np.roll(support, -step, axis=axis)
                capture_from_source = (outside & destination_support)[..., None]
                captured_amount = amount*capture_from_source
                captured_alignment_amount = (
                    alignment_amount*capture_from_source[..., None])
                transmitted_amount = amount-captured_amount
                transmitted_alignment = alignment_amount-captured_alignment_amount
                mobile -= amount
                amobile -= alignment_amount
                mobile += np.roll(transmitted_amount, step, axis=axis)
                amobile += np.roll(transmitted_alignment, step, axis=axis)
                tangle += np.roll(captured_amount, step, axis=axis)
                atangle += np.roll(captured_alignment_amount, step, axis=axis)
                captured += np.roll(captured_amount, step, axis=axis)
                captured_alignment += np.roll(
                    captured_alignment_amount, step, axis=axis)
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
    before = reservoir_nye_m1(alignments, systems, orientation_rad, topologies)
    after = reservoir_nye_m1(aligned, systems, orientation_rad, topologies)
    return updated, aligned, {
        "operator": "finite_volume_transport_and_entry_capture",
        "capture_geometry_source": "caller-declared physical support",
        "sign": sign_ledgers,
        "global_nye_integral_residual_m": np.sum(
            after["total"]-before["total"], axis=(0, 1))*spacing_m**2,
        "local_nye_change_m1": after["total"]-before["total"],
    }


def accepted_topology_ordering(inventory, alignments, ordered_plus_m2,
                               ordered_minus_m2, systems, orientation_rad,
                               topologies=()):
    """Convert captured tangle line to ordered line without creating alignment.

    This is an explicit topology/capture handoff, not a direction generator.
    Unpolarized tangle remains unpolarized.  A future reorientation mechanism
    must provide a separately ledgered spatial/topological Nye source.
    """
    before = reservoir_nye_m1(alignments, systems, orientation_rad, topologies)
    updated, aligned, ledger = _transfer_signed_reservoir(
        inventory, alignments, "wall_tangle", "wall_ordered",
        {"plus": ordered_plus_m2, "minus": ordered_minus_m2}, len(systems))
    after = reservoir_nye_m1(aligned, systems, orientation_rad, topologies)
    return updated, aligned, {
        "operator": "topology_ordering",
        "source": "existing captured tangle line and its evolved alignment",
        "sign": ledger,
        "total_nye_residual_m1": after["total"]-before["total"],
        "reservoir_nye_change_m1": {
            key: after[key]-before[key] for key in before if key != "total"},
    }


def apply_signed_ordering_extent(inventory, alignments, extent_plus_m2,
                                 extent_minus_m2, systems, orientation_rad,
                                 topologies=()):
    """Apply a signed tangle-to-ordered extent while moving alignment exactly.

    Positive extent orders tangle; negative extent disorders ordered line.
    This map is used after a thermodynamic/Arrhenius law has selected the
    direction and trial magnitude. Local donor availability is enforced here.
    """
    before = reservoir_nye_m1(alignments, systems, orientation_rad, topologies)
    density_updates = {}; alignment_updates = {}; sign_ledger = {}
    for sign, raw in (("plus", extent_plus_m2), ("minus", extent_minus_m2)):
        tname = f"wall_tangle_{sign}_m2"
        oname = f"wall_ordered_{sign}_m2"
        tangle = np.asarray(getattr(inventory, tname), dtype=float)
        ordered = np.asarray(getattr(inventory, oname), dtype=float)
        atangle = np.asarray(getattr(alignments, tname), dtype=float)
        aordered = np.asarray(getattr(alignments, oname), dtype=float)
        trial = np.asarray(raw, dtype=float)
        if trial.shape != tangle.shape or np.any(~np.isfinite(trial)):
            raise ValueError("ordering extent must match signed family layout")
        accepted = np.where(trial >= 0.0, np.minimum(trial, tangle),
                            -np.minimum(-trial, ordered))
        donor = np.where(accepted >= 0.0, tangle, ordered)
        fraction = np.divide(np.abs(accepted), donor,
                             out=np.zeros_like(accepted), where=donor > 0.0)
        moved = np.where((accepted >= 0.0)[..., None],
                         fraction[..., None]*atangle,
                         -fraction[..., None]*aordered)
        density_updates[tname] = tangle-accepted
        density_updates[oname] = ordered+accepted
        alignment_updates[tname] = atangle-moved
        alignment_updates[oname] = aordered+moved
        sign_ledger[sign] = {
            "accepted_tangle_to_ordered_m2": accepted,
            "accepted_alignment_m2": moved,
        }
    updated = replace(inventory, **density_updates)
    aligned = replace(alignments, **alignment_updates)
    aligned.validate(updated, len(systems))
    after = reservoir_nye_m1(aligned, systems, orientation_rad, topologies)
    return updated, aligned, {
        "operator": "signed_topology_ordering_extent",
        "sign": sign_ledger,
        "total_nye_residual_m1": after["total"]-before["total"],
    }


def accepted_line_reorientation_step(
        inventory, alignments, requested_plus_m2, requested_minus_m2,
        ordered_line_direction, event_length_m, systems, orientation_rad,
        dt_s, topologies=()):
    """Form kink-pair-bounded ordered segments with an explicit Nye source.

    A finite segment of existing tangle line is reoriented between two turning
    nodes. The scalar line and signed Burgers family are unchanged. The change
    in line direction is recorded as ``R_topology``; paired node and integrated
    curvature inventories are incremented from the declared event length.
    No orientation-gradient or Frank--Bilby target is an input.
    """
    if event_length_m <= 0.0 or dt_s <= 0.0:
        raise ValueError("reorientation requires positive event length and time")
    direction = np.asarray(ordered_line_direction, dtype=float)
    shape = inventory.mobile_plus_m2.shape
    if direction.shape == (3,):
        direction = np.broadcast_to(direction, shape+(3,))
    if direction.shape != shape+(3,) or np.any(~np.isfinite(direction)):
        raise ValueError("ordered line direction must be vector or grid x family x vector")
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("ordered line direction cannot vanish")
    unit = direction/norm
    before = reservoir_nye_m1(alignments, systems, orientation_rad, topologies)
    density_updates = {}; alignment_updates = {}; topology_updates = {}
    sign_ledger = {}
    for sign, request in (("plus", requested_plus_m2),
                          ("minus", requested_minus_m2)):
        tname = f"wall_tangle_{sign}_m2"
        oname = f"wall_ordered_{sign}_m2"
        nname = f"wall_turning_nodes_{sign}_m3"
        cname = f"wall_curvature_{sign}_m3"
        tangle = np.asarray(getattr(inventory, tname), dtype=float)
        ordered = np.asarray(getattr(inventory, oname), dtype=float)
        atangle = np.asarray(getattr(alignments, tname), dtype=float)
        aordered = np.asarray(getattr(alignments, oname), dtype=float)
        extent = _accepted_extent(request, tangle)
        fraction = np.divide(extent, tangle, out=np.zeros_like(extent),
                             where=tangle > 0.0)
        removed_alignment = fraction[..., None]*atangle
        created_alignment = extent[..., None]*unit
        source_unit = np.divide(
            atangle, np.linalg.norm(atangle, axis=-1, keepdims=True),
            out=np.zeros_like(atangle),
            where=np.linalg.norm(atangle, axis=-1, keepdims=True) > 0.0)
        turning_angle = np.arccos(np.clip(
            np.sum(source_unit*unit, axis=-1), -1.0, 1.0))
        density_updates[tname] = tangle-extent
        density_updates[oname] = ordered+extent
        alignment_updates[tname] = atangle-removed_alignment
        alignment_updates[oname] = aordered+created_alignment
        node_increment = 2.0*extent/event_length_m
        curvature_increment = turning_angle*extent/event_length_m
        topology_updates[nname] = getattr(alignments, nname)+node_increment
        topology_updates[cname] = getattr(alignments, cname)+curvature_increment
        sign_ledger[sign] = {
            "accepted_reoriented_line_m2": extent,
            "removed_alignment_m2": removed_alignment,
            "created_alignment_m2": created_alignment,
            "turning_node_increment_m3": node_increment,
            "curvature_increment_m3": curvature_increment,
            "paired_node_closure_residual_m3": node_increment
                                                  -2.0*extent/event_length_m,
        }
    updated = replace(inventory, **density_updates)
    aligned = replace(alignments, **alignment_updates, **topology_updates)
    aligned.validate(updated, len(systems))
    after = reservoir_nye_m1(aligned, systems, orientation_rad, topologies)
    return updated, aligned, {
        "operator": "finite_segment_kink_pair_reorientation",
        "event_length_m": float(event_length_m),
        "sign": sign_ledger,
        "scalar_line_balance_residual_m2": (
            derived_density_fields(updated, topologies)["rho_total_m2"]
            -derived_density_fields(inventory, topologies)["rho_total_m2"]),
        "R_topology_m1_s": (after["total"]-before["total"])/dt_s,
    }


def accepted_junction_topology_step(
        inventory, alignments, requested_extent_m2, systems, topologies,
        orientation_rad, dt_s):
    """Create explicit junction products from two signed tangle parents.

    Each accepted event consumes equal extent from its two declared parents
    and adds one junction extent.  Frank's rule closes vector Burgers content.
    The product line direction and multiplicity come from the declared node
    geometry.  Any tensorial Nye change caused by the topology event is not
    hidden: it is returned as ``R_topology`` in the balance ledger.
    """
    if not topologies:
        raise ValueError("at least one declared junction topology is required")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("topology step requires a finite positive duration")
    shape = inventory.validate(len(systems), len(topologies))
    alignments.validate(inventory, len(systems))
    request = np.asarray(requested_extent_m2, dtype=float)
    if request.shape != shape[:2]+(len(topologies),) or np.any(
            ~np.isfinite(request)) or np.any(request < 0.0):
        raise ValueError("junction request requires grid x topology nonnegative extent")
    before_nye = reservoir_nye_m1(
        alignments, systems, orientation_rad, topologies)
    before_total = derived_density_fields(inventory, topologies)["rho_total_m2"]
    density = {name: np.asarray(getattr(inventory, name)).copy()
               for name in SIGNED_RESERVOIRS}
    alignment = {name: np.asarray(getattr(alignments, name)).copy()
                 for name in SIGNED_RESERVOIRS}
    junction = np.asarray(inventory.junction_m2).copy()
    junction_alignment = np.asarray(alignments.junction_alignment_m2).copy()
    rotation = rotation_z(orientation_rad)
    accepted = np.zeros_like(request)
    scalar_topology_source = np.zeros(shape[:2])
    vector_burgers_residual = np.zeros(shape[:2]+(3,))
    energy_change = np.zeros(shape[:2])
    closure = []
    for index, topology in enumerate(topologies):
        sign_a = "plus" if topology.sign_a > 0 else "minus"
        sign_b = "plus" if topology.sign_b > 0 else "minus"
        name_a = f"wall_tangle_{sign_a}_m2"
        name_b = f"wall_tangle_{sign_b}_m2"
        donor_a = density[name_a][..., topology.parent_a]
        donor_b = density[name_b][..., topology.parent_b]
        extent = np.minimum(request[..., index], np.minimum(donor_a, donor_b))
        accepted[..., index] = extent
        for name, family in ((name_a, topology.parent_a),
                             (name_b, topology.parent_b)):
            donor = density[name][..., family].copy()
            fraction = np.divide(extent, donor, out=np.zeros_like(extent),
                                 where=donor > 0.0)
            density[name][..., family] -= extent
            alignment[name][..., family, :] *= (1.0-fraction[..., None])
        junction[..., index] += extent
        product_line = np.einsum(
            "...ij,j->...i", rotation, topology.product_line_direction)
        product_alignment = (topology.product_line_multiplicity
                             *extent[..., None]*product_line)
        junction_alignment[..., index, :] += product_alignment
        scalar_topology_source += (topology.product_line_multiplicity-2.0)*extent
        parent_b = (topology.sign_a*systems[topology.parent_a].burgers_vector_m
                    +topology.sign_b*systems[topology.parent_b].burgers_vector_m)
        vector_burgers_residual += extent[..., None]*(
            topology.product_burgers_m-parent_b)
        energy_change += extent*topology.delta_free_energy_J_m
        closure.append({"topology": index,
                        **junction_closure_metrics(topology, systems)})
    updated = replace(inventory, junction_m2=junction, **density)
    aligned = replace(
        alignments, junction_alignment_m2=junction_alignment, **alignment)
    aligned.validate(updated, len(systems))
    validate_junction_alignment(updated, aligned, topologies)
    after_nye = reservoir_nye_m1(
        aligned, systems, orientation_rad, topologies)
    after_total = derived_density_fields(updated, topologies)["rho_total_m2"]
    scalar_residual = after_total-before_total-scalar_topology_source
    return updated, aligned, {
        "operator": "explicit_junction_topology",
        "accepted_junction_extent_m2": accepted,
        "declared_scalar_line_source_m2": scalar_topology_source,
        "scalar_line_balance_residual_m2": scalar_residual,
        "vector_burgers_balance_residual_m1": vector_burgers_residual,
        "R_topology_m1_s": (after_nye["total"]-before_nye["total"])/dt_s,
        "free_energy_change_J_m3": energy_change,
        "irreversible_heat_rate_W_m3": np.maximum(-energy_change/dt_s, 0.0),
        "frank_and_node_closure": closure,
    }


def maximum_ledger_residual(ledger):
    scalar = max(float(np.max(np.abs(item["scalar_residual_m2"])))
                 for item in ledger["sign"].values())
    alignment = max(float(np.max(np.abs(item["alignment_residual_m2"])))
                    for item in ledger["sign"].values())
    nye = float(np.max(np.abs(ledger["total_nye_residual_m1"])))
    return {"scalar_line_m2": scalar, "alignment_m2": alignment,
            "total_nye_m1": nye}
