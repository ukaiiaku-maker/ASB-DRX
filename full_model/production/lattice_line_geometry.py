"""Persistent plaquette/edge geometry for conservative line-network events.

The representation is a small discrete chain complex on the same periodic
two-dimensional material grid used by the production wall model.  A signed
plaquette sweep is a physical surface.  Its oriented boundary is stored as
line links and the Burgers-weighted swept surface supplies ``beta_p``.  Thus
line geometry, plastic distortion, and Nye are not independently assigned.

The two sign slots denote positive and negative Burgers populations.  Link
orientation remains signed inside either slot; this is required for a closed
loop.  Scalar density uses link magnitude, while the first moment uses its
orientation.  The out-of-plane plaquette normal makes this a generic
climb/cross-slip surface for the present 2-D section, not a claim of glide on
an arbitrary BCC plane.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .tensorial_nye import nye_from_plastic_distortion, rotated_system_fields


GEOMETRY_PREFIX = "v43_geometry__"


@dataclass(frozen=True)
class LatticeLineGeometry:
    """Persistent oriented links and their generating swept surfaces.

    ``swept_quanta`` has layout ``(nx, ny, family, burgers_sign)`` and is a
    signed number of cell plaquettes.  ``swept_burgers_area_m3`` retains the
    actual Burgers vector at event time, so later lattice rotation does not
    rewrite geometric history.  Edge arrays are deliberately retained rather
    than reconstructed on restart; validation proves that they are the exact
    boundary of the stored surfaces.
    """

    swept_quanta: np.ndarray
    swept_burgers_area_m3: np.ndarray
    edge_x_quanta: np.ndarray
    edge_y_quanta: np.ndarray
    edge_x_burgers_m: np.ndarray
    edge_y_burgers_m: np.ndarray
    accepted_event_count: np.ndarray
    spacing_m: np.ndarray
    section_thickness_m: np.ndarray

    def validate(self, family_count: int, tolerance=2e-12):
        q = np.asarray(self.swept_quanta, dtype=float)
        if q.ndim != 4 or q.shape[2:] != (int(family_count), 2):
            raise ValueError("geometry requires grid x family x Burgers-sign plaquettes")
        vector_shape = q.shape+(3,)
        arrays = (self.swept_burgers_area_m3, self.edge_x_burgers_m,
                  self.edge_y_burgers_m)
        if any(np.asarray(value).shape != vector_shape for value in arrays):
            raise ValueError("Burgers-weighted geometry has an invalid layout")
        if (np.asarray(self.edge_x_quanta).shape != q.shape
                or np.asarray(self.edge_y_quanta).shape != q.shape):
            raise ValueError("oriented edge geometry has an invalid layout")
        if any(np.any(~np.isfinite(np.asarray(value))) for value in
               (q, self.edge_x_quanta, self.edge_y_quanta)+arrays):
            raise ValueError("geometry state must be finite")
        dx = float(np.asarray(self.spacing_m))
        thickness = float(np.asarray(self.section_thickness_m))
        count = float(np.asarray(self.accepted_event_count))
        if dx <= 0.0 or thickness <= 0.0 or count < 0.0 or not count.is_integer():
            raise ValueError("geometry metric and event count are invalid")
        expected_x, expected_y = boundary_of_plaquettes(q)
        scale = max(float(np.max(np.abs(q))), 1.0)
        if (np.max(np.abs(np.asarray(self.edge_x_quanta)-expected_x))
                > tolerance*scale or
                np.max(np.abs(np.asarray(self.edge_y_quanta)-expected_y))
                > tolerance*scale):
            raise ValueError("line links are not the boundary of the swept surface")
        bx, by = boundary_of_plaquettes(
            np.asarray(self.swept_burgers_area_m3)/dx**2)
        bscale = max(float(np.max(np.abs(self.swept_burgers_area_m3)))/dx**2,
                     1e-300)
        if (np.max(np.abs(np.asarray(self.edge_x_burgers_m)-bx))
                > tolerance*bscale or
                np.max(np.abs(np.asarray(self.edge_y_burgers_m)-by))
                > tolerance*bscale):
            raise ValueError("Burgers links are not the boundary of swept Burgers area")
        node = link_node_balance(np.asarray(self.edge_x_quanta),
                                 np.asarray(self.edge_y_quanta))
        if np.max(np.abs(node)) > tolerance*scale:
            raise ValueError("periodic line network has an open endpoint")
        return q.shape[:2]


def boundary_of_plaquettes(plaquettes):
    """Return oriented x/y links for the boundary of periodic plaquettes."""
    value = np.asarray(plaquettes, dtype=float)
    if value.ndim < 2:
        raise ValueError("plaquette field requires two spatial axes")
    # x link (i,j)->(i+1,j): lower cell contributes +, upper cell -.
    edge_x = value-np.roll(value, 1, axis=1)
    # y link (i,j)->(i,j+1): right/left orientation of a CCW boundary.
    edge_y = np.roll(value, 1, axis=0)-value
    return edge_x, edge_y


def link_node_balance(edge_x, edge_y):
    """Discrete boundary-of-boundary residual at periodic nodes."""
    ex = np.asarray(edge_x, dtype=float); ey = np.asarray(edge_y, dtype=float)
    return ex-np.roll(ex, 1, axis=0)+ey-np.roll(ey, 1, axis=1)


def empty_lattice_geometry(shape, family_count, spacing_m,
                           section_thickness_m=None):
    dx = float(spacing_m)
    thickness = dx if section_thickness_m is None else float(section_thickness_m)
    q = np.zeros(tuple(shape)+(int(family_count), 2))
    qb = np.zeros(q.shape+(3,))
    result = LatticeLineGeometry(
        q, qb, q.copy(), q.copy(), qb.copy(), qb.copy(), np.asarray(0),
        np.asarray(dx), np.asarray(thickness))
    result.validate(family_count)
    return result


def geometry_checkpoint_arrays(state, prefix=GEOMETRY_PREFIX):
    return {prefix+name: np.asarray(getattr(state, name))
            for name in state.__dataclass_fields__}


def geometry_from_checkpoint_arrays(mapping, family_count,
                                    prefix=GEOMETRY_PREFIX):
    names = tuple(LatticeLineGeometry.__dataclass_fields__)
    present = [prefix+name in mapping for name in names]
    if not any(present):
        return None
    if not all(present):
        missing = [name for name, ok in zip(names, present) if not ok]
        raise ValueError("incomplete V43 geometry restart: "+", ".join(missing))
    state = LatticeLineGeometry(**{
        name: np.asarray(mapping[prefix+name]).copy() for name in names})
    state.validate(family_count)
    return state


def geometry_reservoir_fields(state):
    """Return cell density/moment represented by the oriented links.

    Each link is shared equally by its two adjacent cells.  Dividing its
    physical length by the cell volume gives line density [m^-2].
    """
    qx = np.asarray(state.edge_x_quanta); qy = np.asarray(state.edge_y_quanta)
    dx = float(state.spacing_m); thickness = float(state.section_thickness_m)
    factor = dx/(dx*dx*thickness)
    rho = .5*factor*(
        np.abs(qx)+np.roll(np.abs(qx), -1, axis=1)
        +np.abs(qy)+np.roll(np.abs(qy), -1, axis=0))
    moment = np.zeros(qx.shape+(3,))
    moment[..., 0] = .5*factor*(qx+np.roll(qx, -1, axis=1))
    moment[..., 1] = .5*factor*(qy+np.roll(qy, -1, axis=0))
    return rho, moment


def geometry_plastic_distortion(state):
    """Family-resolved beta_p generated by the retained swept surface."""
    volume = (float(state.spacing_m)**2
              *float(state.section_thickness_m))
    # Surface normal is e_z, hence beta = b tensor e_z * A/V.
    burgers_area = np.sum(np.asarray(state.swept_burgers_area_m3), axis=3)
    result = np.zeros(burgers_area.shape[:-1]+(3, 3))
    result[..., :, 2] = burgers_area/volume
    return result


def propose_plaquette_sweep(state, inventory, alignment, common, systems,
                            orientation_rad, cell, family, burgers_sign,
                            extent):
    """Build one immutable geometry/plastic candidate and its exact ledger.

    The caller owns thermodynamic acceptance.  This routine performs no
    clipping: unavailable reverse line or a moment-realizability violation is
    an inadmissible candidate and raises before publication.
    """
    state.validate(len(systems))
    i, j = (int(cell[0]), int(cell[1])); family = int(family)
    sign_slot = 0 if int(burgers_sign) > 0 else 1
    value = float(extent)
    if not np.isfinite(value) or value == 0.0:
        raise ValueError("geometry sweep extent must be finite and nonzero")
    nx, ny = state.swept_quanta.shape[:2]
    if not (0 <= i < nx and 0 <= j < ny and 0 <= family < len(systems)):
        raise ValueError("geometry event index is outside the represented grid")
    burgers, _, _ = rotated_system_fields(systems, np.asarray(orientation_rad))
    b = (1.0 if sign_slot == 0 else -1.0)*burgers[i, j, family]
    dx = float(state.spacing_m)
    q = np.asarray(state.swept_quanta).copy()
    ba = np.asarray(state.swept_burgers_area_m3).copy()
    q[i, j, family, sign_slot] += value
    ba[i, j, family, sign_slot] += value*dx**2*b
    ex, ey = boundary_of_plaquettes(q)
    ebx, eby = boundary_of_plaquettes(ba/dx**2)
    candidate_geometry = replace(
        state, swept_quanta=q, swept_burgers_area_m3=ba,
        edge_x_quanta=ex, edge_y_quanta=ey,
        edge_x_burgers_m=ebx, edge_y_burgers_m=eby,
        accepted_event_count=np.asarray(int(state.accepted_event_count)+1))
    candidate_geometry.validate(len(systems))

    before_rho, before_kappa = geometry_reservoir_fields(state)
    after_rho, after_kappa = geometry_reservoir_fields(candidate_geometry)
    drho = after_rho-before_rho; dkappa = after_kappa-before_kappa
    density_updates = {}; alignment_updates = {}
    for slot, label in enumerate(("plus", "minus")):
        dline = drho[..., slot]
        dmom = dkappa[..., slot, :]
        dname = f"wall_ordered_{label}_m2"
        old_density = np.asarray(getattr(inventory, dname))
        old_moment = np.asarray(getattr(alignment, dname))
        density_updates[dname] = old_density+dline
        alignment_updates[dname] = old_moment+dmom
        if np.any(density_updates[dname] < -2e-12*np.maximum(old_density, 1.0)):
            raise ValueError("geometry reverse event exceeds represented ordered line")
        density_updates[dname] = np.maximum(density_updates[dname], 0.0)
    candidate_inventory = replace(inventory, **density_updates)
    candidate_alignment = replace(alignment, **alignment_updates)
    candidate_alignment.validate(candidate_inventory, len(systems))

    before_family_beta = geometry_plastic_distortion(state)
    after_family_beta = geometry_plastic_distortion(candidate_geometry)
    dbeta_family = after_family_beta-before_family_beta
    dbeta = np.sum(dbeta_family, axis=2)
    dnye_family = np.stack([
        nye_from_plastic_distortion(dbeta_family[..., a, :, :], dx)
        for a in range(len(systems))], axis=2)
    candidate_common = replace(
        common, beta_p=np.asarray(common.beta_p)+dbeta,
        family_nye_m1=np.asarray(common.family_nye_m1)+dnye_family)
    identity = (np.sum(dnye_family, axis=2)
                -nye_from_plastic_distortion(dbeta, dx))
    node = link_node_balance(ex, ey)
    line_before = float(np.sum(np.abs(state.edge_x_quanta)
                               +np.abs(state.edge_y_quanta))*dx)
    line_after = float(np.sum(np.abs(ex)+np.abs(ey))*dx)
    return (candidate_geometry, candidate_inventory, candidate_alignment,
            candidate_common, {
                "operator": "periodic_plaquette_sweep",
                "cell_ij": [i, j], "family": family,
                "burgers_sign": 1 if sign_slot == 0 else -1,
                "accepted_extent": value,
                "swept_area_m2": value*dx**2,
                "line_length_before_m": line_before,
                "line_length_after_m": line_after,
                "line_length_change_m": line_after-line_before,
                "maximum_node_balance_residual": float(np.max(np.abs(node))),
                "nye_curl_increment_rms_residual_m1": float(
                    np.sqrt(np.mean(identity*identity))),
                "plastic_distortion_increment": dbeta,
                "family_nye_increment_m1": dnye_family,
                "density_increment_m2": drho,
                "alignment_increment_m2": dkappa,
                "post_step_projection_used": False,
            })
