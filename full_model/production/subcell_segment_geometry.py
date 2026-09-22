"""Persistent physical-coordinate segment geometry for subcell wall events.

The first production path is a closed axis-aligned rectangle because it is the
discriminating V47--V50 fixture.  Its coordinates are physical, not fractional
cell occupancies.  A positive continuous Wendland kernel maps the same four
segments to scalar line and oriented moment.  The enclosed surface supplies
plastic distortion; its curl supplies the authoritative compatible Nye.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .tensorial_nye import (
    nye_from_plastic_distortion, rotated_system_fields, spectral_derivatives,
)


SUBCELL_PREFIX = "v50_subcell_rectangle__"


@dataclass(frozen=True)
class SubcellRectangleGeometry:
    lower_left_m: np.ndarray
    upper_right_m: np.ndarray
    family: np.ndarray
    burgers_sign: np.ndarray
    burgers_vector_m: np.ndarray
    grid_shape: np.ndarray
    spacing_m: np.ndarray
    period_m: np.ndarray
    section_thickness_m: np.ndarray
    representation_length_m: np.ndarray
    line_quadrature_spacing_m: np.ndarray
    accepted_event_count: np.ndarray

    def validate(self, family_count):
        lower = np.asarray(self.lower_left_m, dtype=float)
        upper = np.asarray(self.upper_right_m, dtype=float)
        shape = tuple(int(x) for x in np.asarray(self.grid_shape).reshape(-1))
        period = np.asarray(self.period_m, dtype=float)
        if (lower.shape != (2,) or upper.shape != (2,) or len(shape) != 2
                or period.shape != (2,) or any(x <= 0 for x in shape)
                or np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper))
                or np.any(upper <= lower) or np.any(lower < 0.0)
                or np.any(upper > period)):
            raise ValueError("invalid physical subcell rectangle")
        family = int(self.family); sign = int(self.burgers_sign)
        if not 0 <= family < int(family_count) or sign not in (-1, 1):
            raise ValueError("invalid subcell Burgers owner")
        if np.asarray(self.burgers_vector_m).shape != (3,):
            raise ValueError("subcell Burgers vector must be a three-vector")
        positive = (float(self.spacing_m), float(self.section_thickness_m),
                    float(self.representation_length_m),
                    float(self.line_quadrature_spacing_m))
        if any(not np.isfinite(x) or x <= 0.0 for x in positive):
            raise ValueError("invalid subcell representation metric")
        if not np.allclose(np.asarray(shape)*float(self.spacing_m), period,
                           rtol=2e-14, atol=0.0):
            raise ValueError("subcell period and field grid disagree")
        count = float(self.accepted_event_count)
        if count < 0.0 or not count.is_integer():
            raise ValueError("invalid subcell event count")
        return shape


def subcell_checkpoint_arrays(state, prefix=SUBCELL_PREFIX):
    return {prefix+name: np.asarray(getattr(state, name))
            for name in state.__dataclass_fields__}


def subcell_from_checkpoint_arrays(mapping, family_count,
                                   prefix=SUBCELL_PREFIX):
    names = tuple(SubcellRectangleGeometry.__dataclass_fields__)
    present = [prefix+name in mapping for name in names]
    if not any(present):
        return None
    if not all(present):
        missing = [name for name, ok in zip(names, present) if not ok]
        raise ValueError("incomplete V50 subcell restart: "+", ".join(missing))
    result = SubcellRectangleGeometry(**{
        name: np.asarray(mapping[prefix+name]).copy() for name in names})
    result.validate(family_count)
    return result


def continuous_wendland_kernel_m2(radius_m, length_m):
    """Normalized two-dimensional compact C2 Wendland kernel [m^-2]."""
    radius = np.asarray(radius_m, dtype=float); length = float(length_m)
    q = radius/length
    return np.where(q < 1.0, 7.0/(np.pi*length**2)
                    *(1.0-q)**4*(1.0+4.0*q), 0.0)


def _grid_centers(state):
    nx, ny = state.validate(10**9)
    spacing = float(state.spacing_m)
    return ((np.arange(nx)+.5)*spacing,
            (np.arange(ny)+.5)*spacing)


def _periodic_delta(coordinate, point, period):
    value = coordinate-point
    return value-period*np.rint(value/period)


def _segment_quadrature(p0, p1, maximum_spacing):
    delta = np.asarray(p1)-np.asarray(p0)
    length = float(np.linalg.norm(delta))
    panels = max(1, int(np.ceil(length/float(maximum_spacing))))
    nodes, weights = np.polynomial.legendre.leggauss(4)
    fractions = []
    quadrature_weights = []
    for panel in range(panels):
        left = panel/panels; right = (panel+1)/panels
        fractions.extend(.5*(right-left)*nodes+.5*(right+left))
        quadrature_weights.extend(.5*(right-left)*weights*length)
    points = np.asarray(p0)[None, :]+np.asarray(fractions)[:, None]*delta
    return points, np.asarray(quadrature_weights), delta/length


def subcell_line_fields(state, family_count):
    """Positive scalar line and oriented first moment from actual segments."""
    shape = state.validate(family_count)
    x, y = _grid_centers(state)
    period = np.asarray(state.period_m, dtype=float)
    lower = np.asarray(state.lower_left_m); upper = np.asarray(state.upper_right_m)
    vertices = np.asarray(((lower[0], lower[1]), (upper[0], lower[1]),
                           (upper[0], upper[1]), (lower[0], upper[1])))
    scalar = np.zeros(shape); moment = np.zeros(shape+(3,))
    for index in range(4):
        points, weights, tangent = _segment_quadrature(
            vertices[index], vertices[(index+1) % 4],
            state.line_quadrature_spacing_m)
        for point, weight in zip(points, weights):
            dx = _periodic_delta(x[:, None], point[0], period[0])
            dy = _periodic_delta(y[None, :], point[1], period[1])
            contribution = weight*continuous_wendland_kernel_m2(
                np.sqrt(dx*dx+dy*dy), state.representation_length_m)
            scalar += contribution
            moment += contribution[..., None]*np.asarray(
                (tangent[0], tangent[1], 0.0))
    thickness = float(state.section_thickness_m)
    rho = np.zeros(shape+(int(family_count), 2))
    alignment = np.zeros(rho.shape+(3,))
    slot = 0 if int(state.burgers_sign) > 0 else 1
    rho[..., int(state.family), slot] = scalar/thickness
    # alpha=-Curl(beta): positive swept surface owns the negative CCW boundary.
    alignment[..., int(state.family), slot, :] = -moment/thickness
    return rho, alignment


def _rectangle_cell_area_fraction(state):
    shape = state.validate(10**9); spacing = float(state.spacing_m)
    lower = np.asarray(state.lower_left_m); upper = np.asarray(state.upper_right_m)
    edges_x = np.arange(shape[0]+1)*spacing
    edges_y = np.arange(shape[1]+1)*spacing
    overlap_x = np.maximum(
        np.minimum(edges_x[1:], upper[0])-np.maximum(edges_x[:-1], lower[0]),
        0.0)
    overlap_y = np.maximum(
        np.minimum(edges_y[1:], upper[1])-np.maximum(edges_y[:-1], lower[1]),
        0.0)
    return overlap_x[:, None]*overlap_y[None, :]/spacing**2


def subcell_surface_from_boundary(state, family_count):
    """Construct a swept surface from the independently integrated boundary.

    The positive scalar line remains the physical Wendland line integral.  Its
    oriented first moment supplies a periodic streamfunction through the
    least-squares inverse of ``(-D_y f, D_x f)`` using the same spectral
    derivative as production Nye.  The zero mode is the exact enclosed area.
    This is a boundary-to-surface commuting adapter, not a prescribed Nye
    field.  Nyquist modes are excluded because the real spectral derivative
    cannot represent their derivative on an even grid.
    """
    shape = state.validate(family_count)
    rho, alignment = subcell_line_fields(state, family_count)
    family = int(state.family); slot = 0 if int(state.burgers_sign) > 0 else 1
    thickness = float(state.section_thickness_m)
    raw_moment = alignment[..., family, slot, :2]*thickness
    spacing = float(state.spacing_m)
    kx = 2*np.pi*np.fft.fftfreq(shape[0], d=spacing)
    ky = 2*np.pi*np.fft.fftfreq(shape[1], d=spacing)
    if shape[0] % 2 == 0:
        kx[shape[0]//2] = 0.0
    if shape[1] % 2 == 0:
        ky[shape[1]//2] = 0.0
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    denominator = KX*KX+KY*KY
    qx = np.fft.fftn(raw_moment[..., 0])
    qy = np.fft.fftn(raw_moment[..., 1])
    surface_spectrum = np.zeros(shape, dtype=complex)
    active = denominator > 0.0
    surface_spectrum[active] = (
        1j*KY[active]*qx[active]-1j*KX[active]*qy[active]
    )/denominator[active]
    area_fraction = (np.prod(np.asarray(state.upper_right_m)
                             -np.asarray(state.lower_left_m))
                     /np.prod(np.asarray(state.period_m)))
    surface_spectrum[0, 0] = np.prod(shape)*area_fraction
    surface = np.fft.ifftn(surface_spectrum).real
    # The inverse has only small band-limit ringing.  Apply one affine
    # zero-mode-preserving contraction toward the exact mean when needed to
    # retain a physical [0,1] swept fraction.  Its scale is declared below.
    scale = 1.0
    minimum = float(np.min(surface)); maximum = float(np.max(surface))
    if minimum < 0.0:
        scale = min(scale, area_fraction/(area_fraction-minimum))
    if maximum > 1.0:
        scale = min(scale, (1.0-area_fraction)/(maximum-area_fraction))
    surface = area_fraction+scale*(surface-area_fraction)
    fx, fy = spectral_derivatives(surface, spacing)
    surface_moment = np.stack((-fy, fx), axis=-1)
    absolute = float(np.sqrt(np.mean((raw_moment-surface_moment)**2)))
    reference = float(np.sqrt(np.mean(surface_moment**2)))
    scalar = rho[..., family, slot]
    return surface, {
        "operator": "boundary_streamfunction_commuting_surface",
        "exact_area_fraction": float(area_fraction),
        "observed_area_fraction": float(np.mean(surface)),
        "positivity_affine_scale": float(scale),
        "minimum_surface_fraction": float(np.min(surface)),
        "maximum_surface_fraction": float(np.max(surface)),
        "line_surface_moment_residual_rms_m-1": absolute/thickness,
        "line_surface_moment_reference_rms_m-1": reference/thickness,
        "line_surface_moment_residual_relative": absolute/max(reference, 1e-300),
        "line_continuity_integral_m": np.sum(raw_moment, axis=(0, 1)).tolist(),
        "scalar_line_nonnegative": bool(np.min(scalar) >= 0.0),
        "surface_fraction_bounded": bool(
            np.min(surface) >= -2e-15 and np.max(surface) <= 1.0+2e-15),
        "map_error_budget_relative": 0.05,
    }


def subcell_plastic_and_nye(state, family_count):
    """Plastic surface map and compatible family Nye from the same rectangle."""
    shape = state.validate(family_count); spacing = float(state.spacing_m)
    fraction, _ = subcell_surface_from_boundary(state, family_count)
    beta_family = np.zeros(shape+(int(family_count), 3, 3))
    beta_family[..., int(state.family), :, 2] = (
        np.asarray(state.burgers_vector_m)
        *fraction[..., None]/float(state.section_thickness_m))
    family_nye = np.stack([
        nye_from_plastic_distortion(beta_family[..., family, :, :], spacing)
        for family in range(int(family_count))], axis=2)
    return beta_family, family_nye


def subcell_line_surface_nye(state, family_count):
    """Return independent boundary and swept-surface Nye plus their audit."""
    _, alignment = subcell_line_fields(state, family_count)
    _, family_nye = subcell_plastic_and_nye(state, family_count)
    family = int(state.family); slot = 0 if int(state.burgers_sign) > 0 else 1
    line_nye = np.einsum(
        "i,...j->...ij", np.asarray(state.burgers_vector_m),
        alignment[..., family, slot, :])
    surface_nye = family_nye[..., family, :, :]
    residual = line_nye-surface_nye
    absolute = float(np.sqrt(np.mean(residual*residual)))
    reference = float(np.sqrt(np.mean(surface_nye*surface_nye)))
    _, surface_audit = subcell_surface_from_boundary(state, family_count)
    audit = {
        **surface_audit,
        "line_nye_rms_m-1": float(np.sqrt(np.mean(line_nye*line_nye))),
        "surface_nye_rms_m-1": reference,
        "line_surface_nye_residual_rms_m-1": absolute,
        "line_surface_nye_residual_relative": absolute/max(reference, 1e-300),
        "scientific_compatibility_passed": bool(
            absolute <= surface_audit["map_error_budget_relative"]
            *max(reference, 1e-300)),
    }
    return line_nye, surface_nye, audit


def subcell_face_field_derivative(state, family_count, step_m=None):
    """Same-path discrete shape derivative for the moving +x face.

    This differentiates the actual positive quadrature/map used by production,
    including the two connecting segments.  It is a controlled centered shape
    derivative, not a fitted force or a target field.
    """
    state.validate(family_count)
    h = (min(float(state.spacing_m)/1000.0,
             float(state.representation_length_m)/10000.0)
         if step_m is None else float(step_m))
    if h <= 0.0 or not np.isfinite(h):
        raise ValueError("shape-derivative step must be positive")
    plus = replace(
        state, upper_right_m=np.asarray(state.upper_right_m)
        +np.asarray((h, 0.0)))
    minus = replace(
        state, upper_right_m=np.asarray(state.upper_right_m)
        -np.asarray((h, 0.0)))
    plus.validate(family_count); minus.validate(family_count)
    rho_plus, moment_plus = subcell_line_fields(plus, family_count)
    rho_minus, moment_minus = subcell_line_fields(minus, family_count)
    beta_plus, nye_plus = subcell_plastic_and_nye(plus, family_count)
    beta_minus, nye_minus = subcell_plastic_and_nye(minus, family_count)
    return {
        "step_m": h,
        "scalar_density_derivative_m3": (rho_plus-rho_minus)/(2*h),
        "alignment_derivative_m3": (moment_plus-moment_minus)/(2*h),
        "plastic_distortion_derivative_m-1": (
            beta_plus-beta_minus)/(2*h),
        "family_nye_derivative_m-2": (nye_plus-nye_minus)/(2*h),
        "scheme": "centered_same_path_physical_face_coordinate",
    }


def initialize_subcell_rectangle(
        inventory, alignment, common, systems, orientation_rad, *,
        lower_left_m, upper_right_m, family, burgers_sign, spacing_m,
        section_thickness_m, representation_length_m,
        line_quadrature_spacing_m=None):
    shape = np.asarray(common.orientation_rad).shape
    family = int(family); sign = 1 if int(burgers_sign) > 0 else -1
    burgers, _, _ = rotated_system_fields(systems, np.asarray(orientation_rad))
    # The initializer is restricted to a homogeneous lattice orientation; an
    # evolving nonuniform orientation needs segment-local Burgers ownership.
    bfield = sign*burgers[..., family, :]
    b = np.mean(bfield, axis=(0, 1))
    if np.max(np.abs(bfield-b)) > 2e-12*max(np.linalg.norm(b), 1e-300):
        raise ValueError("subcell rectangle initializer requires homogeneous Burgers field")
    spacing = float(spacing_m)
    geometry = SubcellRectangleGeometry(
        np.asarray(lower_left_m, dtype=float),
        np.asarray(upper_right_m, dtype=float), np.asarray(family),
        np.asarray(sign), np.asarray(b), np.asarray(shape), np.asarray(spacing),
        np.asarray(np.asarray(shape)*spacing), np.asarray(section_thickness_m),
        np.asarray(representation_length_m),
        np.asarray(line_quadrature_spacing_m or representation_length_m/32.0),
        np.asarray(0))
    geometry.validate(len(systems))
    rho, moment = subcell_line_fields(geometry, len(systems))
    density_updates = {}; alignment_updates = {}
    for slot, label in enumerate(("plus", "minus")):
        name = f"wall_ordered_{label}_m2"
        density_updates[name] = np.asarray(getattr(inventory, name))+rho[..., slot]
        alignment_updates[name] = np.asarray(getattr(alignment, name))+moment[..., slot, :]
    new_inventory = replace(inventory, **density_updates)
    new_alignment = replace(alignment, **alignment_updates)
    new_alignment.validate(new_inventory, len(systems))
    beta_family, family_nye = subcell_plastic_and_nye(geometry, len(systems))
    _, _, compatibility = subcell_line_surface_nye(geometry, len(systems))
    new_common = replace(
        common, beta_p=np.asarray(common.beta_p)+np.sum(beta_family, axis=2),
        family_nye_m1=np.asarray(common.family_nye_m1)+family_nye)
    return geometry, new_inventory, new_alignment, new_common, {
        "operator": "physical_subcell_rectangle_initializer",
        "initialization_only": True, "accepted_physical_event_count": 0,
        "independent_line_surface_compatibility": compatibility,
        "line_length_m": float(2*np.sum(
            np.asarray(upper_right_m)-np.asarray(lower_left_m))),
    }


def propose_subcell_face_extension(
        state, inventory, alignment, common, systems, displacement_m):
    """Move the +x face and both connecting endpoints by a physical distance."""
    state.validate(len(systems)); displacement = float(displacement_m)
    if not np.isfinite(displacement) or displacement == 0.0:
        raise ValueError("subcell face displacement must be finite and nonzero")
    candidate = replace(
        state, upper_right_m=np.asarray(state.upper_right_m)
        +np.asarray((displacement, 0.0)),
        accepted_event_count=np.asarray(int(state.accepted_event_count)+1))
    candidate.validate(len(systems))
    height = float(state.upper_right_m[1]-state.lower_left_m[1])
    return _remap_subcell_geometry(
        state, candidate, inventory, alignment, common, systems, {
            "operator": "physical_subcell_rectangle_face_extension",
            "physical_displacement_m": displacement,
            "signed_swept_area_m2": height*displacement,
            "line_length_change_m": 2.0*displacement,
        })


def propose_subcell_translation(
        state, inventory, alignment, common, systems, displacement_m):
    """Rigidly translate the persistent loop through the production map."""
    state.validate(len(systems))
    displacement = np.asarray(displacement_m, dtype=float)
    if (displacement.shape != (2,) or np.any(~np.isfinite(displacement))
            or not np.any(displacement != 0.0)):
        raise ValueError("subcell translation must be a finite nonzero 2-vector")
    candidate = replace(
        state, lower_left_m=np.asarray(state.lower_left_m)+displacement,
        upper_right_m=np.asarray(state.upper_right_m)+displacement,
        accepted_event_count=np.asarray(int(state.accepted_event_count)+1))
    candidate.validate(len(systems))
    return _remap_subcell_geometry(
        state, candidate, inventory, alignment, common, systems, {
            "operator": "physical_subcell_rectangle_rigid_translation",
            "physical_displacement_vector_m": displacement,
            "signed_swept_area_m2": 0.0,
            "line_length_change_m": 0.0,
        })


def propose_subcell_x_face_moves(
        state, inventory, alignment, common, systems, *,
        lower_displacement_m, upper_displacement_m):
    """Move two named x faces in one common geometric transaction."""
    state.validate(len(systems))
    lower = float(lower_displacement_m); upper = float(upper_displacement_m)
    if (not np.isfinite(lower) or not np.isfinite(upper)
            or (lower == 0.0 and upper == 0.0)):
        raise ValueError("shared x-face move requires a finite nonzero displacement")
    active_faces = int(lower != 0.0)+int(upper != 0.0)
    candidate = replace(
        state,
        lower_left_m=np.asarray(state.lower_left_m)+np.asarray((lower, 0.0)),
        upper_right_m=np.asarray(state.upper_right_m)+np.asarray((upper, 0.0)),
        accepted_event_count=np.asarray(
            int(state.accepted_event_count)+active_faces))
    candidate.validate(len(systems))
    height = float(state.upper_right_m[1]-state.lower_left_m[1])
    return _remap_subcell_geometry(
        state, candidate, inventory, alignment, common, systems, {
            "operator": "physical_subcell_rectangle_shared_x_faces",
            "face_displacements_m": {
                "lower_x": lower, "upper_x": upper,
            },
            "active_face_count": active_faces,
            "signed_swept_area_m2": height*(upper-lower),
            "line_length_change_m": 2.0*(upper-lower),
            "common_clock_transaction": True,
        })


def _remap_subcell_geometry(
        state, candidate, inventory, alignment, common, systems,
        geometry_ledger):
    """Apply one candidate geometry through all common state owners."""
    rho0, moment0 = subcell_line_fields(state, len(systems))
    rho1, moment1 = subcell_line_fields(candidate, len(systems))
    density_updates = {}; alignment_updates = {}
    for slot, label in enumerate(("plus", "minus")):
        name = f"wall_ordered_{label}_m2"
        density = (np.asarray(getattr(inventory, name))+rho1[..., slot]
                   -rho0[..., slot])
        mapped_moment = (np.asarray(getattr(alignment, name))
                         +moment1[..., slot, :]-moment0[..., slot, :])
        excess = np.linalg.norm(mapped_moment, axis=-1)-density
        scale = max(float(np.max(np.abs(density))), 1.0)
        if np.max(excess) > 256*np.finfo(float).eps*scale:
            raise ValueError("subcell event violates moment realizability")
        density_updates[name] = np.maximum(
            density+np.maximum(excess, 0.0), 0.0)
        alignment_updates[name] = mapped_moment
    new_inventory = replace(inventory, **density_updates)
    new_alignment = replace(alignment, **alignment_updates)
    new_alignment.validate(new_inventory, len(systems))
    beta0, nye0 = subcell_plastic_and_nye(state, len(systems))
    beta1, nye1 = subcell_plastic_and_nye(candidate, len(systems))
    dbeta_family = beta1-beta0; dnye = nye1-nye0
    dbeta = np.sum(dbeta_family, axis=2)
    new_common = replace(
        common, beta_p=np.asarray(common.beta_p)+dbeta,
        family_nye_m1=np.asarray(common.family_nye_m1)+dnye)
    family = int(state.family); slot = 0 if int(state.burgers_sign) > 0 else 1
    line_increment = np.einsum(
        "i,...j->...ij", np.asarray(state.burgers_vector_m),
        moment1[..., family, slot, :]-moment0[..., family, slot, :])
    surface_increment = dnye[..., family, :, :]
    compatibility_residual = line_increment-surface_increment
    compatibility_absolute = float(np.sqrt(np.mean(
        compatibility_residual*compatibility_residual)))
    compatibility_reference = float(np.sqrt(np.mean(
        surface_increment*surface_increment)))
    return candidate, new_inventory, new_alignment, new_common, {
        **geometry_ledger,
        "plastic_distortion_increment": dbeta,
        "family_nye_increment_m1": dnye,
        "scalar_density_increment_m2": rho1-rho0,
        "alignment_increment_m2": moment1-moment0,
        "independent_line_nye_increment_m1": line_increment,
        "independent_surface_nye_increment_m1": surface_increment,
        "line_surface_event_residual_rms_m-1": compatibility_absolute,
        "line_surface_event_reference_rms_m-1": compatibility_reference,
        "line_surface_event_residual_relative": (
            compatibility_absolute/max(compatibility_reference, 1e-300)),
        "line_surface_event_compatibility_passed": bool(
            compatibility_absolute <= .05
            *max(compatibility_reference, 1e-300)),
        "line_surface_event_error_budget_relative": 0.05,
        "closed_loop_topology": True,
        "post_step_projection_used": False,
    }
