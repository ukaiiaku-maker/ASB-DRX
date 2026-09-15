"""V24 full-elastic kinematics coupled to ledgered wall line supply.

This seam uses the qualified common nonlocal elastic solve and slip kinematics,
but replaces its legacy unlocalized forest-to-wall conversion with explicit
finite-volume transport/capture.  Ordering receives a zero compatibility
coefficient and no orientation-derived target.  An optional, separately
ledgered junction topology route can operate on captured tangle parents.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .arrhenius_kinetics import (
    ActivatedProcess, activated_rate_array_s, exp_floor_enthalpy_j,
)
from .common_tensorial_wall import (
    CommonWallDriving, CommonWallParameters, CommonWallState,
    resolved_driving_components, wall_residual,
)
from .density_state_map import (
    DensityInventory, checkpoint_arrays, derived_density_fields,
    from_checkpoint_arrays,
)
from .extensive_wall import ExtensiveWallParameters, accepted_ordering_step
from .tensorial_nye import rotated_system_fields
from .wall_topology_supply import (
    ReservoirAlignmentState, accepted_junction_topology_step,
    accepted_line_reorientation_step,
    alignment_checkpoint_arrays, alignment_from_checkpoint_arrays,
    apply_signed_ordering_extent, conservative_transport_capture_step,
    validate_junction_alignment,
)


@dataclass(frozen=True)
class V24MechanicalWallState:
    common: CommonWallState
    density: DensityInventory
    reservoir_alignment: ReservoirAlignmentState

    def validate(self, systems, topologies):
        self.common.validate(systems, topologies)
        self.reservoir_alignment.validate(self.density, len(systems))
        validate_junction_alignment(
            self.density, self.reservoir_alignment, topologies)
        pairs = (
            (self.common.mobile_plus_m2, self.density.mobile_plus_m2),
            (self.common.mobile_minus_m2, self.density.mobile_minus_m2),
            (self.common.forest_plus_m2, self.density.forest_plus_m2),
            (self.common.forest_minus_m2, self.density.forest_minus_m2),
            (self.common.wall_plus_m2,
             self.density.wall_tangle_plus_m2+self.density.wall_ordered_plus_m2),
            (self.common.wall_minus_m2,
             self.density.wall_tangle_minus_m2+self.density.wall_ordered_minus_m2),
            (self.common.junction_m2, self.density.junction_m2),
        )
        if any(not np.array_equal(left, right) for left, right in pairs):
            raise ValueError("mechanical and V24 density inventories are not synchronized")


@dataclass(frozen=True)
class V24TopologyKinetics:
    junction_process: ActivatedProcess
    junction_enthalpy_J: float
    critical_stress_Pa: float
    exp_a: float = 2.2
    exp_n: float = 2.5
    exp_floor: float = 0.05
    maximum_junction_fraction_per_step: float = 0.1

    def __post_init__(self):
        if (self.junction_enthalpy_J < 0.0 or self.critical_stress_Pa <= 0.0
                or self.exp_a < 0.0 or self.exp_n < 1.0
                or not 0.0 <= self.exp_floor <= 1.0
                or not 0.0 < self.maximum_junction_fraction_per_step <= 1.0):
            raise ValueError("invalid V24 topology kinetics")


def mechanical_checkpoint_arrays(state):
    """Lossless array payload for the complete V24 mechanical wall state."""
    payload = {"v24_common__"+name: np.asarray(value)
               for name, value in state.common.__dict__.items()}
    payload.update(checkpoint_arrays(state.density, prefix="v24_density__"))
    payload.update(alignment_checkpoint_arrays(state.reservoir_alignment))
    return payload


def mechanical_from_checkpoint_arrays(mapping, systems, topologies):
    common_names = tuple(CommonWallState.__dataclass_fields__)
    missing = [name for name in common_names if "v24_common__"+name not in mapping]
    if missing:
        raise ValueError("incomplete V24 mechanical restart: "+", ".join(missing))
    common = CommonWallState(**{
        name: np.asarray(mapping["v24_common__"+name]).copy()
        for name in common_names})
    density = from_checkpoint_arrays(
        mapping, len(systems), len(topologies), prefix="v24_density__")
    alignment = alignment_from_checkpoint_arrays(
        mapping, density, len(systems))
    result = V24MechanicalWallState(common, density, alignment)
    result.validate(systems, topologies)
    return result


def synchronize_common(state, topologies):
    fields = derived_density_fields(state.density, topologies)
    common = replace(
        state.common,
        mobile_plus_m2=state.density.mobile_plus_m2,
        mobile_minus_m2=state.density.mobile_minus_m2,
        forest_plus_m2=state.density.forest_plus_m2,
        forest_minus_m2=state.density.forest_minus_m2,
        wall_plus_m2=(state.density.wall_tangle_plus_m2
                      +state.density.wall_ordered_plus_m2),
        wall_minus_m2=(state.density.wall_tangle_minus_m2
                       +state.density.wall_ordered_minus_m2),
        junction_m2=state.density.junction_m2,
        wall_order=fields["q_wall_diagnostic"])
    return replace(state, common=common)


def accepted_v24_mechanical_step(
        state, driving, capture_support, systems, topologies,
        common_parameters, extensive_parameters, topology_kinetics, dt_s, *,
        topology_route_enabled=False, maximum_orientation_increment_rad=0.02):
    """Advance mechanics, line transport/capture, ordering, and topology once."""
    state.validate(systems, topologies)
    if extensive_parameters.nye_match_coefficient_J_m != 0.0:
        raise ValueError("V24 production ordering forbids target-Nye driving")
    # The common residual supplies the authoritative full-elastic stress,
    # physical glide speed, and exactly consistent beta/slip/orientation rates.
    mechanics_parameters = replace(
        common_parameters, wall_order_enabled=False,
        extensive_wall_partition_enabled=True, transport_scheme="upwind",
        mobile_correlation_diffusivity_m2_s=0.0)
    drive = resolved_driving_components(
        state.common, driving, systems, topologies, mechanics_parameters)
    # Reuse the already solved elastic fields in the common residual. Passing
    # the raw RSS makes its Taylor reduction identical, and passing speed
    # avoids a second nonlocal solve without changing the constitutive path.
    resolved = CommonWallDriving(
        glide_speed_m_s=drive["speed_m_s"],
        resolved_stress_Pa=drive["raw_stress_Pa"])
    residual = wall_residual(
        state.common, resolved, systems, topologies, mechanics_parameters)
    _, slip_directions, _ = rotated_system_fields(
        systems, state.common.orientation_rad)
    velocity_plus = drive["speed_m_s"][..., None]*slip_directions[..., :2]
    velocity_minus = -velocity_plus
    courant_rate = np.max(
        np.sum(np.abs(velocity_plus), axis=-1)/common_parameters.spacing_m)
    orientation_rate = np.max(np.abs(residual.state_rate.orientation_rad))
    accepted_dt = min(
        float(dt_s), 0.8/max(float(courant_rate), 1e-300),
        maximum_orientation_increment_rad/max(float(orientation_rate), 1e-300))
    rate = residual.state_rate
    # Only the qualified kinematic fields are accepted from this residual.
    # Its legacy density reactions and scalar ordering are deliberately not.
    common = replace(
        state.common,
        slip=state.common.slip+accepted_dt*rate.slip,
        beta_p=state.common.beta_p+accepted_dt*rate.beta_p,
        alignment_m2=state.common.alignment_m2+accepted_dt*rate.alignment_m2,
        family_nye_m1=state.common.family_nye_m1+accepted_dt*rate.family_nye_m1,
        orientation_rad=(state.common.orientation_rad
                         +accepted_dt*rate.orientation_rad),
        temperature_K=(state.common.temperature_K+accepted_dt
                       *residual.plastic_power_W_m3
                       /common_parameters.volumetric_heat_capacity_J_m3_K))
    transported_density, transported_alignment, capture_ledger = (
        conservative_transport_capture_step(
            state.density, state.reservoir_alignment,
            velocity_plus, velocity_minus, capture_support, systems,
            state.common.orientation_rad, common_parameters.spacing_m,
            accepted_dt, topologies))
    working_density = transported_density
    working_alignment = transported_alignment
    topology_ledger = None
    reorientation_ledger = None
    if topology_route_enabled and topologies:
        family_enthalpy = exp_floor_enthalpy_j(
            np.abs(drive["effective_stress_Pa"]),
            topology_kinetics.junction_enthalpy_J,
            topology_kinetics.critical_stress_Pa, topology_kinetics.exp_a,
            topology_kinetics.exp_n, topology_kinetics.exp_floor)
        family_rate_s = activated_rate_array_s(
            topology_kinetics.junction_process, family_enthalpy,
            common.temperature_K[..., None])
        fraction = np.minimum(-np.expm1(-accepted_dt*family_rate_s),
                              topology_kinetics.maximum_junction_fraction_per_step)
        # Thermodynamic affinity selects direction; this first implementation
        # only accepts the favorable tangle->ordered branch. The reverse branch
        # remains available through the detailed-balanced non-reorienting route.
        favorable = (extensive_parameters.ordered_excess_J_m
                     < extensive_parameters.disordered_excess_J_m)
        request_plus = (fraction*working_density.wall_tangle_plus_m2
                        if favorable else np.zeros_like(
                            working_density.wall_tangle_plus_m2))
        request_minus = (fraction*working_density.wall_tangle_minus_m2
                         if favorable else np.zeros_like(
                             working_density.wall_tangle_minus_m2))
        working_density, working_alignment, reorientation_ledger = (
            accepted_line_reorientation_step(
                working_density, working_alignment, request_plus,
                request_minus, (0.0, 0.0, 1.0),
                extensive_parameters.event_length_m, systems,
                common.orientation_rad, accepted_dt, topologies))
        pair_stress = np.stack([
            np.maximum(np.abs(drive["effective_stress_Pa"][..., item.parent_a]),
                       np.abs(drive["effective_stress_Pa"][..., item.parent_b]))
            for item in topologies], axis=2)
        enthalpy = exp_floor_enthalpy_j(
            pair_stress, topology_kinetics.junction_enthalpy_J,
            topology_kinetics.critical_stress_Pa, topology_kinetics.exp_a,
            topology_kinetics.exp_n, topology_kinetics.exp_floor)
        rate_s = activated_rate_array_s(
            topology_kinetics.junction_process, enthalpy,
            common.temperature_K[..., None])
        requests = []
        for index, item in enumerate(topologies):
            first = getattr(working_density,
                f"wall_tangle_{'plus' if item.sign_a > 0 else 'minus'}_m2")[..., item.parent_a]
            second = getattr(working_density,
                f"wall_tangle_{'plus' if item.sign_b > 0 else 'minus'}_m2")[..., item.parent_b]
            fraction = np.minimum(-np.expm1(-accepted_dt*rate_s[..., index]),
                                  topology_kinetics.maximum_junction_fraction_per_step)
            requests.append(fraction*np.minimum(first, second))
        working_density, working_alignment, topology_ledger = (
            accepted_junction_topology_step(
                working_density, working_alignment, np.stack(requests, axis=2),
                systems, topologies, common.orientation_rad, accepted_dt))
    if topology_route_enabled:
        ordered_density, ordered_alignment = working_density, working_alignment
        ordering_thermo = {"disabled_in_explicit_topology_comparator": True}
        ordering_topology = None
    else:
        # Shared EXP-floor/signed-entropy ordering law. With C_FB=0 and a zero
        # target, only the declared extensive free-energy affinity selects direction.
        zero_target = np.zeros(state.common.orientation_rad.shape+(3, 3))
        ordered_density, ordering_thermo, _ = accepted_ordering_step(
            working_density, systems, topologies, common.orientation_rad,
            zero_target, drive["effective_stress_Pa"], common.temperature_K,
            extensive_parameters, accepted_dt)
        extent_plus = (ordered_density.wall_ordered_plus_m2
                       -working_density.wall_ordered_plus_m2)
        extent_minus = (ordered_density.wall_ordered_minus_m2
                        -working_density.wall_ordered_minus_m2)
        ordered_density, ordered_alignment, ordering_topology = (
            apply_signed_ordering_extent(
                working_density, working_alignment,
                extent_plus, extent_minus, systems, common.orientation_rad,
                topologies))
    result = synchronize_common(V24MechanicalWallState(
        common, ordered_density, ordered_alignment), topologies)
    result.validate(systems, topologies)
    return result, {
        "accepted_dt_s": accepted_dt,
        "raw_stress_Pa": drive["raw_stress_Pa"],
        "effective_stress_Pa": drive["effective_stress_Pa"],
        "plastic_power_W_m3": residual.plastic_power_W_m3,
        "transport_capture": capture_ledger,
        "ordering_thermodynamics": ordering_thermo,
        "ordering_topology": ordering_topology,
        "line_reorientation_topology": reorientation_ledger,
        "junction_topology": topology_ledger,
        "legacy_common_density_rates_accepted": False,
        "orientation_target_used": False,
    }
