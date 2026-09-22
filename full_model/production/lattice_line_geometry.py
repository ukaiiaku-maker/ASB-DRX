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

try:
    from .tensorial_nye import nye_from_plastic_distortion, rotated_system_fields
except ImportError:  # pragma: no cover - direct production-script execution
    from tensorial_nye import nye_from_plastic_distortion, rotated_system_fields


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


def initialize_closed_swept_surface(state, inventory, alignment, common,
                                    systems, orientation_rad, swept_quanta,
                                    *, continuum_representation_length_m=0.0):
    """Batch-initialize a closed swept surface without executing an event.

    This is an initial-condition constructor, not nucleation or a kinetic
    transaction.  It is algebraically the boundary/map of the supplied
    plaquette chain and deliberately leaves ``accepted_event_count`` at zero.
    """
    state.validate(len(systems))
    if (int(state.accepted_event_count) != 0
            or np.any(np.asarray(state.swept_quanta) != 0.0)):
        raise ValueError("batch surface initialization requires empty geometry")
    q = np.asarray(swept_quanta, dtype=float)
    if q.shape != state.swept_quanta.shape or not np.isfinite(q).all():
        raise ValueError("initial swept surface has an invalid layout")
    dx = float(state.spacing_m)
    burgers, _, _ = rotated_system_fields(systems, np.asarray(orientation_rad))
    signed_burgers = np.stack((burgers, -burgers), axis=3)
    ba = q[..., None]*dx**2*signed_burgers
    ex, ey = boundary_of_plaquettes(q)
    ebx, eby = boundary_of_plaquettes(ba/dx**2)
    geometry = replace(
        state, swept_quanta=q.copy(), swept_burgers_area_m3=ba,
        edge_x_quanta=ex, edge_y_quanta=ey,
        edge_x_burgers_m=ebx, edge_y_burgers_m=eby,
        accepted_event_count=np.asarray(0))
    geometry.validate(len(systems))
    rho_raw, moment_raw = geometry_reservoir_fields(geometry)
    length = float(continuum_representation_length_m)
    rho = apply_physical_reconstruction(rho_raw, dx, length)
    moment = apply_physical_reconstruction(moment_raw, dx, length)
    density_updates = {}; alignment_updates = {}
    for slot, sign in enumerate(("plus", "minus")):
        name = f"wall_ordered_{sign}_m2"
        density = np.asarray(getattr(inventory, name))+rho[..., slot]
        moment_value = (
            np.asarray(getattr(alignment, name))+moment[..., slot, :])
        # Scalar and vector FFT roundoff are independent. Preserve the mapped
        # moment exactly and close only its positive ulp-scale realizability
        # deficit, identically to the physical event path.
        excess = np.linalg.norm(moment_value, axis=-1)-density
        scale = max(float(np.max(np.abs(density))), 1.0)
        if np.max(excess) > 256*np.finfo(float).eps*scale:
            raise ValueError("initialized geometry violates moment realizability")
        density_updates[name] = density+np.maximum(excess, 0.0)
        alignment_updates[name] = moment_value
    initialized_inventory = replace(inventory, **density_updates)
    initialized_alignment = replace(alignment, **alignment_updates)
    initialized_alignment.validate(initialized_inventory, len(systems))
    beta_family_raw = geometry_plastic_distortion(geometry)
    beta_family = apply_physical_reconstruction(beta_family_raw, dx, length)
    beta = np.sum(beta_family, axis=2)
    family_nye = np.stack([
        nye_from_plastic_distortion(beta_family[..., family, :, :], dx)
        for family in range(len(systems))], axis=2)
    initialized_common = replace(
        common, beta_p=np.asarray(common.beta_p)+beta,
        family_nye_m1=np.asarray(common.family_nye_m1)+family_nye)
    return geometry, initialized_inventory, initialized_alignment, initialized_common, {
        "operator": "batch_closed_swept_surface_initializer",
        "initialization_only": True,
        "accepted_physical_event_count": 0,
        "nonzero_plaquette_count": int(np.count_nonzero(q)),
        "maximum_node_balance_residual": float(np.max(np.abs(
            link_node_balance(ex, ey)))),
        "continuum_representation_length_m": length,
    }


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
    # alpha=-Curl(beta_p): the dislocation-line orientation is the negative
    # oriented boundary of the retained positive plastic swept surface.
    moment[..., 0] = -.5*factor*(qx+np.roll(qx, -1, axis=1))
    moment[..., 1] = -.5*factor*(qy+np.roll(qy, -1, axis=0))
    return rho, moment


def physical_reconstruction_kernel(shape, spacing_m, length_m):
    """Positive normalized periodic coarse-graining kernel on cell centers.

    ``length_m`` is a declared continuum representation length, independent of
    mesh spacing and of the phase-interface width.  The discrete kernel has an
    exact unit zero mode, is centrosymmetric/self-adjoint, and is nonnegative.
    A zero length returns the identity map for legacy checkpoint replay.
    """
    nx, ny = map(int, shape)
    spacing = float(spacing_m); length = float(length_m)
    if nx <= 0 or ny <= 0 or spacing <= 0.0 or length < 0.0:
        raise ValueError("reconstruction shape, spacing, and length are invalid")
    kernel = np.zeros((nx, ny), dtype=float)
    if length == 0.0:
        kernel[0, 0] = 1.0
        return kernel
    ix = np.minimum(np.arange(nx), nx-np.arange(nx))*spacing
    iy = np.minimum(np.arange(ny), ny-np.arange(ny))*spacing
    radius = np.sqrt(ix[:, None]**2+iy[None, :]**2)
    normalized = radius/length
    # Compact C2 Wendland reconstruction. Compact support is physical here:
    # exactly zero line outside the declared averaging radius remains an
    # inactive reservoir rather than becoming a Gaussian numerical tail.
    kernel = np.where(
        normalized < 1.0,
        (1.0-normalized)**4*(4.0*normalized+1.0), 0.0)
    kernel /= np.sum(kernel, dtype=np.longdouble)
    return kernel


def apply_physical_reconstruction(field, spacing_m, length_m):
    """Apply the fixed physical geometry-to-continuum map ``K_ell``.

    The same routine is its discrete adjoint because the kernel is real and
    centrosymmetric.  Convolution acts only on the periodic spatial axes and
    preserves every trailing component's integral.
    """
    value = np.asarray(field, dtype=float)
    if value.ndim < 2:
        raise ValueError("reconstructed field requires two spatial axes")
    if float(length_m) == 0.0:
        return value.copy()
    kernel = physical_reconstruction_kernel(
        value.shape[:2], spacing_m, length_m)
    multiplier = np.fft.fftn(kernel, axes=(0, 1)).reshape(
        kernel.shape+(1,)*(value.ndim-2))
    result = np.fft.ifftn(
        np.fft.fftn(value, axes=(0, 1))*multiplier,
        axes=(0, 1)).real
    # FFT roundoff would otherwise turn a compact physical map into tiny global
    # tails and activate nonexistent reservoirs. Remove only values at the
    # floating-point noise scale. The FFT multiplier retains the zero mode to
    # roundoff; independently altering scalar and vector maxima to force a
    # bitwise sum would violate moment realizability. This is numerical cleanup
    # of a compact convolution, not a physical density floor.
    trailing = int(np.prod(value.shape[2:])) if value.ndim > 2 else 1
    flat_result = result.reshape(result.shape[:2]+(trailing,))
    for component in range(trailing):
        scale = max(float(np.max(np.abs(flat_result[..., component]))), 1e-300)
        noise = 256*np.finfo(float).eps*scale
        flat_result[..., component][
            np.abs(flat_result[..., component]) <= noise] = 0.0
    return result


def subcell_geometry_to_continuum(field, spacing_m, length_m,
                                   displacement_m=(0.0, 0.0), *,
                                   return_displacement_derivative=False):
    """Reconstruct and rigidly translate a periodic continuum geometry field.

    Fractional values in ``swept_quanta`` are ensemble weights and are *not*
    coordinates.  This separate map supplies coordinates: it first applies the
    declared physical reconstruction and then evaluates its band-limited
    periodic continuation at ``x-displacement``.  When requested, the two
    returned derivatives are with respect to the physical x/y displacement
    [field unit per metre].  Even-grid Nyquist modes are removed because a
    real-valued Nyquist cosine has no uniquely sampled subcell translate.
    """
    value = apply_physical_reconstruction(field, spacing_m, length_m)
    if value.ndim < 2:
        raise ValueError("subcell geometry requires two spatial axes")
    spacing = float(spacing_m)
    displacement = np.asarray(displacement_m, dtype=float)
    if (spacing <= 0.0 or displacement.shape != (2,)
            or not np.isfinite(displacement).all()):
        raise ValueError("invalid subcell displacement or spacing")
    nx, ny = value.shape[:2]
    kx = 2*np.pi*np.fft.fftfreq(nx, d=spacing)
    ky = 2*np.pi*np.fft.fftfreq(ny, d=spacing)
    spectrum = np.fft.fftn(value, axes=(0, 1))
    if nx % 2 == 0:
        spectrum[nx//2, ...] = 0.0
    if ny % 2 == 0:
        spectrum[:, ny//2, ...] = 0.0
    phase2 = np.exp(-1j*(
        kx[:, None]*displacement[0]+ky[None, :]*displacement[1]))
    phase = phase2.reshape(phase2.shape+(1,)*(value.ndim-2))
    translated_spectrum = spectrum*phase
    translated = np.fft.ifftn(
        translated_spectrum, axes=(0, 1)).real
    if not return_displacement_derivative:
        return translated
    kx_shape = kx.reshape((nx, 1)+(1,)*(value.ndim-2))
    ky_shape = ky.reshape((1, ny)+(1,)*(value.ndim-2))
    derivative_x = np.fft.ifftn(
        -1j*kx_shape*translated_spectrum, axes=(0, 1)).real
    derivative_y = np.fft.ifftn(
        -1j*ky_shape*translated_spectrum, axes=(0, 1)).real
    return translated, np.stack((derivative_x, derivative_y), axis=0)


def geometry_link_nye_mimetic(state):
    """Deposit retained Burgers-weighted links onto adjacent cells.

    This assembly uses the stored link state directly; it does not curl the
    swept surface.  The minus sign is the declared ``alpha=-Curl(beta_p)``
    convention.  The result is family resolved.
    """
    dx = float(state.spacing_m); thickness = float(state.section_thickness_m)
    factor = 1.0/(dx*thickness)
    ex = np.sum(np.asarray(state.edge_x_burgers_m), axis=3)
    ey = np.sum(np.asarray(state.edge_y_burgers_m), axis=3)
    result = np.zeros(ex.shape[:-1]+(3, 3))
    result[..., :, 0] = -.5*factor*(ex+np.roll(ex, -1, axis=1))
    result[..., :, 1] = -.5*factor*(ey+np.roll(ey, -1, axis=0))
    return result


def geometry_surface_nye_mimetic(state):
    """Apply the staggered surface-to-cell boundary map independently.

    The half-cell averaging is the explicit commuting reconstruction between
    the plaquette/edge complex and the cell-centered continuum field.  This
    implementation starts from retained swept Burgers area, not stored links.
    """
    dx = float(state.spacing_m); thickness = float(state.section_thickness_m)
    surface = np.sum(np.asarray(state.swept_burgers_area_m3), axis=3)/dx**2
    ex, ey = boundary_of_plaquettes(surface)
    factor = 1.0/(dx*thickness)
    result = np.zeros(surface.shape[:-1]+(3, 3))
    result[..., :, 0] = -.5*factor*(ex+np.roll(ex, -1, axis=1))
    result[..., :, 1] = -.5*factor*(ey+np.roll(ey, -1, axis=0))
    return result


def geometry_link_nye_spectral_transfer(state):
    """Transfer staggered retained links to the production spectral complex.

    A backward edge incidence has symbol ``(1-exp(-ikh))/h`` whereas the
    production derivative has symbol ``ik``.  The declared multiplier below
    maps the *stored links* to that derivative, including their half-cell
    placement.  Even-grid Nyquist derivatives follow NumPy's real spectral
    convention and are zero.  This is an explicit transfer between two
    discrete structures, not a curl of the retained surface.
    """
    dx = float(state.spacing_m); thickness = float(state.section_thickness_m)
    ex = np.sum(np.asarray(state.edge_x_burgers_m), axis=3)
    ey = np.sum(np.asarray(state.edge_y_burgers_m), axis=3)
    nx, ny = ex.shape[:2]

    def multiplier(n):
        kh = 2*np.pi*np.fft.fftfreq(n)
        denominator = 1.0-np.exp(-1j*kh)
        value = np.zeros(n, dtype=complex)
        active = np.abs(denominator) > 1e-14
        value[active] = 1j*kh[active]/denominator[active]
        if n % 2 == 0:
            value[n//2] = 0.0
        return value

    rx = multiplier(nx).reshape((nx, 1)+(1,)*(ex.ndim-2))
    ry = multiplier(ny).reshape((1, ny)+(1,)*(ex.ndim-2))
    base_x = -ex/(dx*thickness)
    base_y = -ey/(dx*thickness)
    ax = np.fft.ifftn(np.fft.fftn(base_x, axes=(0, 1))*ry,
                      axes=(0, 1)).real
    ay = np.fft.ifftn(np.fft.fftn(base_y, axes=(0, 1))*rx,
                      axes=(0, 1)).real
    result = np.zeros(ex.shape[:-1]+(3, 3))
    result[..., :, 0] = ax
    result[..., :, 1] = ay
    return result


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
                            extent, *, continuum_representation_length_m=0.0):
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
    representation_length = float(continuum_representation_length_m)
    if not np.isfinite(representation_length) or representation_length < 0.0:
        raise ValueError("continuum representation length must be finite and nonnegative")
    drho_raw = after_rho-before_rho
    dkappa_raw = after_kappa-before_kappa
    drho = apply_physical_reconstruction(
        drho_raw, dx, representation_length)
    dkappa = apply_physical_reconstruction(
        dkappa_raw, dx, representation_length)
    density_updates = {}; alignment_updates = {}
    realizability_roundoff_added = {}
    for slot, label in enumerate(("plus", "minus")):
        dline = drho[..., slot]
        dmom = dkappa[..., slot, :]
        dname = f"wall_ordered_{label}_m2"
        old_density = np.asarray(getattr(inventory, dname))
        old_moment = np.asarray(getattr(alignment, dname))
        density_updates[dname] = old_density+dline
        alignment_updates[dname] = old_moment+dmom
        excess = (np.linalg.norm(alignment_updates[dname], axis=-1)
                  -density_updates[dname])
        numerical_scale = max(
            float(np.max(np.abs(density_updates[dname]))), 1.0)
        if np.max(excess) > 256*np.finfo(float).eps*numerical_scale:
            raise ValueError("reconstructed geometry violates moment realizability")
        # Preserve the mapped signed moment (and hence Nye) exactly. Add only
        # the positive scalar ulp deficit caused by independent FFT roundoff.
        # This is ledgered numerical closure, not a physical density floor.
        correction = np.maximum(excess, 0.0)
        density_updates[dname] += correction
        realizability_roundoff_added[label] = correction
        if np.any(density_updates[dname] < -2e-12*np.maximum(old_density, 1.0)):
            raise ValueError("geometry reverse event exceeds represented ordered line")
        density_updates[dname] = np.maximum(density_updates[dname], 0.0)
    candidate_inventory = replace(inventory, **density_updates)
    candidate_alignment = replace(alignment, **alignment_updates)
    candidate_alignment.validate(candidate_inventory, len(systems))

    before_family_beta = geometry_plastic_distortion(state)
    after_family_beta = geometry_plastic_distortion(candidate_geometry)
    dbeta_family_raw = after_family_beta-before_family_beta
    dbeta_family = apply_physical_reconstruction(
        dbeta_family_raw, dx, representation_length)
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
                "continuum_representation": (
                    "fixed_physical_positive_periodic_kernel" if representation_length
                    else "legacy_identity"),
                "continuum_representation_length_m": representation_length,
                "represented_section_thickness_m": float(
                    state.section_thickness_m),
                "raw_geometry_retained_exactly": True,
                "swept_area_m2": value*dx**2,
                "line_length_before_m": line_before,
                "line_length_after_m": line_after,
                "line_length_change_m": line_after-line_before,
                "maximum_node_balance_residual": float(np.max(np.abs(node))),
                "nye_curl_increment_rms_residual_m1": float(
                    np.sqrt(np.mean(identity*identity))),
                "plastic_distortion_increment": dbeta,
                "raw_plastic_distortion_increment": dbeta_family_raw.sum(axis=2),
                "family_nye_increment_m1": dnye_family,
                "density_increment_m2": drho,
                "alignment_increment_m2": dkappa,
                "realizability_roundoff_scalar_line_added_m2": (
                    realizability_roundoff_added),
                "post_step_projection_used": False,
            })
