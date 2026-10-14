"""Geometry-aware common-axis structural content on owner interfaces.

For a low-angle tilt interface ``a -> b``, the signed integrated structural
content is ``C = L (theta_b-theta_a)/b``.  Conservative motion changes the
boundary length ``L`` but not ``C`` and therefore enforces ``L theta = const``.
Reaction changes ``C`` and is a separate event.  A graph cycle supplies an
explicit Frank-compatibility audit: independently conserved edge contents need
not remain representable by one orientation per owner after arbitrary geometry
change.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .structural_boundary import (
    StructuralBoundaryParameters, read_shockley_energy_J_m2,
)
from .tensorial_nye import spectral_derivatives


@dataclass(frozen=True)
class StructuralInterface:
    interface_id: str
    owner_a: int
    owner_b: int
    signed_integrated_content: float


@dataclass(frozen=True)
class JunctionDisclinationParameters:
    """Isotropic wedge-disclination upper-bound parameters.

    ``outer_cutoff_m`` is the explicitly declared screening distance.  The
    continuum line energy is ``mu Omega^2 R^2/[16 pi (1-nu)]``.  It is used
    only for common-axis cycle closure defects and is not a 3-D replacement
    for a tensorial generalized-disclination model.
    """

    shear_modulus_Pa: float = 70.0e9
    poisson_ratio: float = 0.34
    outer_cutoff_m: float = 5.0e-7

    def __post_init__(self):
        if not math.isfinite(self.shear_modulus_Pa) or self.shear_modulus_Pa <= 0.0:
            raise ValueError("shear modulus must be positive finite")
        if (not math.isfinite(self.poisson_ratio)
                or not -1.0 < self.poisson_ratio < 0.5):
            raise ValueError("Poisson ratio must lie in (-1, 0.5)")
        if not math.isfinite(self.outer_cutoff_m) or self.outer_cutoff_m <= 0.0:
            raise ValueError("outer cutoff must be positive finite")


@dataclass(frozen=True)
class StructuralNetworkAudit:
    owner_ids: tuple[int, ...]
    fitted_orientation_rad: np.ndarray
    implied_edge_misorientation_rad: np.ndarray
    fitted_edge_misorientation_rad: np.ndarray
    edge_compatibility_residual_rad: np.ndarray
    weighted_rms_compatibility_residual_rad: float
    fundamental_cycle_residual_rad: np.ndarray
    maximum_cycle_residual_rad: float


@dataclass(frozen=True)
class ConservativeGeometryLedger:
    interfaces: tuple[StructuralInterface, ...]
    old_lengths_m: np.ndarray
    new_lengths_m: np.ndarray
    old_misorientation_rad: np.ndarray
    new_misorientation_rad: np.ndarray
    integrated_content_change: np.ndarray
    old_boundary_energy_J_per_m: float
    new_boundary_energy_J_per_m: float
    energy_change_J_per_m: float
    compatibility: StructuralNetworkAudit


@dataclass(frozen=True)
class CompatibilityReactionLedger:
    """Projection of edge content onto one compatible orientation per owner.

    The edge-content change is *not* silently discarded.  Its negative is
    carried by an explicit junction/reaction reservoir, so this operator can
    later be coupled to a crystallographic junction reaction without confusing
    geometric conservation with reaction kinetics.
    """

    before_interfaces: tuple[StructuralInterface, ...]
    candidate_interfaces: tuple[StructuralInterface, ...]
    edge_content_change: np.ndarray
    junction_reservoir_change: np.ndarray
    content_balance_residual: np.ndarray
    before_boundary_energy_J_per_m: float
    candidate_boundary_energy_J_per_m: float
    delta_boundary_energy_J_per_m: float
    before_compatibility: StructuralNetworkAudit
    candidate_compatibility: StructuralNetworkAudit


def pair_interface_length_m(supports, owner_a, owner_b, spacing_m):
    """Symmetric diffuse-contour measure for one owner pair."""
    eta = np.asarray(supports, dtype=float)
    spacing = float(spacing_m)
    if eta.ndim != 3 or not (0 <= owner_a < len(eta)) or not (
            0 <= owner_b < len(eta)) or owner_a == owner_b:
        raise ValueError("valid distinct owner indices are required")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be positive finite")
    if np.max(np.abs(np.sum(eta, axis=0)-1.0)) > 2e-6:
        raise ValueError("supports must form a partition of unity")

    def directed(index, other):
        field = eta[index]
        gx, gy = spectral_derivatives(field, spacing)
        contour = np.sqrt(gx*gx+gy*gy)
        contacts = np.stack([
            np.minimum(field, np.maximum.reduce((
                eta[j], np.roll(eta[j], 1, 0), np.roll(eta[j], -1, 0),
                np.roll(eta[j], 1, 1), np.roll(eta[j], -1, 1))))
            if j != index else np.zeros_like(field)
            for j in range(len(eta))])
        total = np.sum(contacts, axis=0)
        share = np.divide(
            contacts[other], total, out=np.zeros_like(field),
            where=total > 256*np.finfo(float).eps)
        return float(np.sum(contour*share, dtype=np.longdouble)*spacing**2)
    return .5*(directed(owner_a, owner_b)+directed(owner_b, owner_a))


def interface_lengths_m(supports, interfaces, owner_ids, spacing_m):
    lookup = {owner: index for index, owner in enumerate(owner_ids)}
    def pair(item):
        if hasattr(item, "owner_a"):
            return item.owner_a, item.owner_b
        return item.grain_a_id, item.grain_b_id
    return np.asarray([
        pair_interface_length_m(
            supports, lookup[pair(item)[0]], lookup[pair(item)[1]], spacing_m)
        for item in interfaces], dtype=float)


def initialize_structural_interfaces(
        owner_ids, orientations_rad, edge_pairs, lengths_m, burgers_m):
    ids = tuple(map(int, owner_ids))
    orientation = np.asarray(orientations_rad, dtype=float)
    lengths = np.asarray(lengths_m, dtype=float)
    if orientation.shape != (len(ids),) or lengths.shape != (len(edge_pairs),):
        raise ValueError("owner/edge arrays have inconsistent shapes")
    lookup = {owner: index for index, owner in enumerate(ids)}
    result = []
    for index, ((a, b), length) in enumerate(zip(edge_pairs, lengths)):
        if length <= 0.0 or a not in lookup or b not in lookup or a == b:
            raise ValueError("interfaces require positive length and known owners")
        theta = orientation[lookup[b]]-orientation[lookup[a]]
        result.append(StructuralInterface(
            f"edge-{a}-{b}", int(a), int(b), float(length*theta/burgers_m)))
    return tuple(result)


def implied_misorientations_rad(interfaces, lengths_m, burgers_m):
    lengths = np.asarray(lengths_m, dtype=float)
    if lengths.shape != (len(interfaces),) or np.any(lengths <= 0.0):
        raise ValueError("one positive length per interface is required")
    return np.asarray([
        burgers_m*item.signed_integrated_content/length
        for item, length in zip(interfaces, lengths)], dtype=float)


def _fundamental_cycle_residuals(owner_ids, interfaces, edge_values):
    """Deterministic signed closure errors for a spanning-forest cycle basis."""
    ids = tuple(owner_ids)
    parent = {owner: owner for owner in ids}
    graph = {owner: [] for owner in ids}

    def root(owner):
        while parent[owner] != owner:
            parent[owner] = parent[parent[owner]]
            owner = parent[owner]
        return owner

    def path_value(start, stop):
        stack = [(start, None, 0.0)]
        while stack:
            node, previous, total = stack.pop()
            if node == stop:
                return total
            for adjacent, increment in graph[node]:
                if adjacent != previous:
                    stack.append((adjacent, node, total+increment))
        raise RuntimeError("tree path not found")

    residuals = []
    for item, value in zip(interfaces, np.asarray(edge_values, dtype=float)):
        a, b = item.owner_a, item.owner_b
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
            graph[a].append((b, float(value)))
            graph[b].append((a, -float(value)))
        else:
            residuals.append(path_value(a, b)-float(value))
    return np.asarray(residuals, dtype=float)


def audit_network_compatibility(
        owner_ids, interfaces, lengths_m, burgers_m, *, reference_owner=None,
        reference_orientation_rad=0.0):
    ids = tuple(map(int, owner_ids))
    lookup = {owner: index for index, owner in enumerate(ids)}
    if len(lookup) != len(ids):
        raise ValueError("owner identifiers must be unique")
    implied = implied_misorientations_rad(interfaces, lengths_m, burgers_m)
    lengths = np.asarray(lengths_m, dtype=float)
    A = np.zeros((len(interfaces), len(ids)))
    for row, item in enumerate(interfaces):
        A[row, lookup[item.owner_a]] = -1.0
        A[row, lookup[item.owner_b]] = 1.0
    root = ids[0] if reference_owner is None else int(reference_owner)
    if root not in lookup:
        raise ValueError("reference owner is not in the network")
    keep = [index for index in range(len(ids)) if ids[index] != root]
    rhs = implied-A[:, lookup[root]]*float(reference_orientation_rad)
    weighted_A = np.sqrt(lengths)[:, None]*A[:, keep]
    weighted_rhs = np.sqrt(lengths)*rhs
    free, *_ = np.linalg.lstsq(weighted_A, weighted_rhs, rcond=None)
    orientation = np.empty(len(ids))
    orientation[lookup[root]] = float(reference_orientation_rad)
    orientation[keep] = free
    fitted = A@orientation
    residual = fitted-implied
    rms = math.sqrt(float(np.sum(lengths*residual*residual)/np.sum(lengths)))
    cycle_residual = _fundamental_cycle_residuals(ids, interfaces, implied)
    cycle = (float(np.max(np.abs(cycle_residual)))
             if len(cycle_residual) else 0.0)
    return StructuralNetworkAudit(
        ids, orientation, implied, fitted, residual, rms,
        cycle_residual, cycle)


def network_boundary_energy_J_per_m(
        interfaces, lengths_m, burgers_m,
        parameters=StructuralBoundaryParameters()):
    theta = implied_misorientations_rad(interfaces, lengths_m, burgers_m)
    return float(np.sum(
        np.asarray(lengths_m)*read_shockley_energy_J_m2(theta, parameters)))


def network_junction_disclination_energy_J_per_m(
        owner_ids, interfaces, lengths_m, burgers_m,
        parameters=JunctionDisclinationParameters()):
    """Elastic line energy of the fundamental common-axis closure defects."""
    audit = audit_network_compatibility(
        owner_ids, interfaces, lengths_m, burgers_m)
    coefficient = (parameters.shear_modulus_Pa*parameters.outer_cutoff_m**2
                   /(16.0*math.pi*(1.0-parameters.poisson_ratio)))
    return float(coefficient*np.sum(
        audit.fundamental_cycle_residual_rad**2, dtype=np.longdouble))


def conservative_geometry_update(
        owner_ids, interfaces, old_lengths_m, new_lengths_m, burgers_m, *,
        parameters=StructuralBoundaryParameters(), reference_owner=None,
        reference_orientation_rad=0.0):
    old_lengths = np.asarray(old_lengths_m, dtype=float)
    new_lengths = np.asarray(new_lengths_m, dtype=float)
    old_theta = implied_misorientations_rad(interfaces, old_lengths, burgers_m)
    new_theta = implied_misorientations_rad(interfaces, new_lengths, burgers_m)
    old_energy = network_boundary_energy_J_per_m(
        interfaces, old_lengths, burgers_m, parameters)
    new_energy = network_boundary_energy_J_per_m(
        interfaces, new_lengths, burgers_m, parameters)
    audit = audit_network_compatibility(
        owner_ids, interfaces, new_lengths, burgers_m,
        reference_owner=reference_owner,
        reference_orientation_rad=reference_orientation_rad)
    return ConservativeGeometryLedger(
        tuple(interfaces), old_lengths, new_lengths, old_theta, new_theta,
        np.zeros(len(interfaces)), old_energy, new_energy,
        new_energy-old_energy, audit)


def project_content_to_compatible_orientations(
        owner_ids, interfaces, lengths_m, burgers_m, *,
        parameters=StructuralBoundaryParameters(), reference_owner=None,
        reference_orientation_rad=0.0):
    """Return the minimum-weight reaction required by Frank compatibility.

    Conservative motion first leaves each edge content unchanged.  On a graph
    with cycles, the resulting edge misorientations need not be the gradient
    of one scalar owner orientation.  This function uses the same
    interface-length-weighted least-squares projection as the compatibility
    audit, converts the projected misorientations back to signed edge content,
    and books the opposite change to an explicit junction reservoir.

    This is a kinematic projection and ledger, not an automatically accepted
    physical reaction.  A caller must separately supply kinetics and the
    complete energy of the junction reservoir before promoting it.
    """
    lengths = np.asarray(lengths_m, dtype=float)
    if lengths.shape != (len(interfaces),) or np.any(~np.isfinite(lengths)) \
            or np.any(lengths <= 0.0):
        raise ValueError("one positive finite length per interface is required")
    before = audit_network_compatibility(
        owner_ids, interfaces, lengths, burgers_m,
        reference_owner=reference_owner,
        reference_orientation_rad=reference_orientation_rad)
    old_content = np.asarray(
        [item.signed_integrated_content for item in interfaces], dtype=float)
    new_content = lengths*before.fitted_edge_misorientation_rad/float(burgers_m)
    candidate = tuple(
        StructuralInterface(item.interface_id, item.owner_a, item.owner_b,
                            float(content))
        for item, content in zip(interfaces, new_content))
    edge_change = new_content-old_content
    reservoir_change = -edge_change
    residual = edge_change+reservoir_change
    before_energy = network_boundary_energy_J_per_m(
        interfaces, lengths, burgers_m, parameters)
    candidate_energy = network_boundary_energy_J_per_m(
        candidate, lengths, burgers_m, parameters)
    after = audit_network_compatibility(
        owner_ids, candidate, lengths, burgers_m,
        reference_owner=reference_owner,
        reference_orientation_rad=reference_orientation_rad)
    return CompatibilityReactionLedger(
        tuple(interfaces), candidate, edge_change, reservoir_change, residual,
        before_energy, candidate_energy, candidate_energy-before_energy,
        before, after)
