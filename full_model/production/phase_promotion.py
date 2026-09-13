"""Conservative dislocation-state transfer for full-v34 embryo promotion.

The operation removes only sign-neutral mobile pairs and unsigned forest/wall
content from a resolved embryo core.  Removed line length is stored on a
declared boundary shell.  Signed mobile Burgers content is unchanged exactly;
insufficient shell capacity rejects the operation instead of clipping.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

try:
    from .physical_grains import GrainRecord, GrainTracker
    from .stateful_embryos import EmbryoPopulation, mark_promoted
except ImportError:  # pragma: no cover
    from physical_grains import GrainRecord, GrainTracker
    from stateful_embryos import EmbryoPopulation, mark_promoted


@dataclass(frozen=True)
class PromotionTransferLedger:
    line_content_before_m: float
    line_content_after_m: float
    line_content_removed_from_core_m: float
    line_content_shell_transfer_m: float
    line_content_pair_annihilation_m: float
    line_content_declared_sink_m: float
    line_content_closure_m: float
    maximum_signed_burgers_density_change_m2: float


@dataclass(frozen=True)
class AtomicPromotionLedger:
    embryo_id: int
    child_label: int
    external_work_J: float
    elastic_energy_change_J: float
    bulk_stored_energy_change_J: float
    line_energy_change_J: float
    interface_order_energy_change_J: float
    compatibility_energy_change_J: float
    heat_released_J: float
    other_dissipation_J: float
    energy_closure_J: float
    transfer: PromotionTransferLedger


@dataclass(frozen=True)
class AtomicPromotionResult:
    eta: np.ndarray
    psi_gv: np.ndarray
    Ng: int
    rp: np.ndarray
    rm: np.ndarray
    rho_forest: np.ndarray
    rho_wall: np.ndarray
    rho_gb: np.ndarray
    embryos: EmbryoPopulation
    tracker: GrainTracker
    heat_support: np.ndarray
    ledger: AtomicPromotionLedger


def recrystallized_child_stored_energy_derivative(
        eta_child, parent_stored_energy_J_m3,
        line_energy_coefficient_J_m, target_density_m2):
    """Return dF/deta [J/m3] for a low-energy DRX child in its parent.

    With h(eta)=eta^2(3-2 eta), h'=6 eta(1-eta), and the child
    carrying the declared recrystallized density, the phase derivative is
    -h'(eta) max(E_parent-E_child, 0).
    """
    eta_value = np.asarray(eta_child, dtype=float)
    parent_energy = np.asarray(parent_stored_energy_J_m3, dtype=float)
    coefficient = np.asarray(line_energy_coefficient_J_m, dtype=float)
    if (not np.all(np.isfinite(eta_value)) or np.any(eta_value < 0.0)
            or np.any(eta_value > 1.0)):
        raise ValueError("child phase support must be finite and bounded")
    if (not np.all(np.isfinite(parent_energy)) or np.any(parent_energy < 0.0)
            or not np.all(np.isfinite(coefficient)) or np.any(coefficient < 0.0)
            or not math.isfinite(target_density_m2) or target_density_m2 < 0.0):
        raise ValueError("stored-energy driving inputs must be finite and nonnegative")
    relief = np.maximum(
        parent_energy-coefficient*float(target_density_m2), 0.0)
    return -6.0*eta_value*(1.0-eta_value)*relief


class PromotionEnergyIncreaseError(ValueError):
    """Rejected atomic trial carrying the complete free-energy diagnostic."""

    def __init__(self, delta, free_energy_change_J, tolerance_J,
                 diagnostic_delta=None):
        self.delta = {str(key): float(value) for key, value in delta.items()}
        self.diagnostic_delta = {
            str(key): float(value)
            for key, value in (diagnostic_delta or {}).items()}
        self.free_energy_change_J = float(free_energy_change_J)
        self.tolerance_J = float(tolerance_J)
        terms = ", ".join(
            f"{key}={self.delta[key]:.17g} J" for key in sorted(self.delta))
        super().__init__(
            "atomic promotion would increase total free energy: "
            f"{terms}, total={self.free_energy_change_J:.17g} J, "
            f"tolerance={self.tolerance_J:.17g} J"
            + (", diagnostic_delta=" + ", ".join(
                f"{key}={self.diagnostic_delta[key]:.17g} J"
                for key in sorted(self.diagnostic_delta))
               if self.diagnostic_delta else ""))


def conservative_neutral_density_relief(
        rp, rm, rho_forest, rho_wall, rho_gb, core_mask, shell_mask, *,
        target_core_density_m2, cell_area_m2, represented_thickness_m,
        maximum_density_m2, target_shell_density_m2=None):
    """Return relieved copies plus an exact line/Burgers-content ledger.

    ``rp`` and ``rm`` have shape ``(nx, ny, nslip)`` and units m^-2. Forest
    has the same shape; wall and GB densities have shape ``(nx, ny)``. The
    scalar target is a maximum total core density, but signed mobile content is
    an inviolable lower bound.
    """
    rp0 = np.asarray(rp, dtype=float)
    rm0 = np.asarray(rm, dtype=float)
    forest0 = np.asarray(rho_forest, dtype=float)
    wall0 = np.asarray(rho_wall, dtype=float)
    gb0 = np.asarray(rho_gb, dtype=float)
    core = np.asarray(core_mask, dtype=bool)
    shell = np.asarray(shell_mask, dtype=bool)
    if rp0.ndim != 3 or rm0.shape != rp0.shape or forest0.shape != rp0.shape:
        raise ValueError("mobile and forest populations must share (nx,ny,nslip)")
    if wall0.shape != rp0.shape[:2] or gb0.shape != wall0.shape:
        raise ValueError("wall and GB fields must match the population grid")
    if core.shape != wall0.shape or shell.shape != wall0.shape or np.any(core & shell):
        raise ValueError("core and shell must be disjoint masks on the population grid")
    if not np.any(core) or not np.any(shell):
        raise ValueError("promotion requires nonempty resolved core and shell")
    arrays = (rp0, rm0, forest0, wall0, gb0)
    if any(not np.all(np.isfinite(a)) or np.any(a < 0.0) for a in arrays):
        raise ValueError("all dislocation densities must be finite and nonnegative")
    for value, name in ((target_core_density_m2, "target density"),
                        (cell_area_m2, "cell area"),
                        (represented_thickness_m, "represented thickness"),
                        (maximum_density_m2, "maximum density")):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    rp1, rm1 = rp0.copy(), rm0.copy()
    forest1, wall1, gb1 = forest0.copy(), wall0.copy(), gb0.copy()
    signed0 = rp0 - rm0
    pair_density = 2.0*np.minimum(rp0, rm0)
    signed_floor = np.sum(np.abs(signed0), axis=2)
    pair_total = np.sum(pair_density, axis=2)
    pair_keep_total = np.minimum(
        pair_total, np.maximum(float(target_core_density_m2) - signed_floor, 0.0))
    pair_scale = np.zeros_like(pair_total)
    np.divide(pair_keep_total, pair_total, out=pair_scale, where=pair_total > 0.0)
    # Preserve the signed populations exactly by putting the retained
    # sign-neutral content into the unsigned forest reservoir.  Reconstructing
    # two O(1e14) mobile populations around a much smaller signed difference
    # can change that difference by an ulp even though the algebra is exact.
    # A recovered core need not retain equal positive/negative mobile carriers;
    # the declared residual line content is represented as forest instead.
    rp_core = np.maximum(signed0, 0.0)
    rm_core = np.maximum(-signed0, 0.0)
    forest_core = np.broadcast_to(
        pair_keep_total[:, :, None]/rp0.shape[2], forest0.shape)
    rp1[core] = rp_core[core]
    rm1[core] = rm_core[core]
    forest1[core] = forest_core[core]
    wall1[core] = 0.0

    old_bulk = np.sum(rp0 + rm0, axis=2) + np.sum(forest0, axis=2) + wall0
    new_bulk = np.sum(rp1 + rm1, axis=2) + np.sum(forest1, axis=2) + wall1
    removed_density_integral = float(np.sum(
        (old_bulk - new_bulk)[core], dtype=np.longdouble))
    if removed_density_integral < -1e-8:
        raise RuntimeError("promotion relief attempted to create core line content")
    pair_removed_density_integral = float(np.sum(
        (np.sum(rp0+rm0, axis=2)-np.sum(rp1+rm1, axis=2))[core],
        dtype=np.longdouble))
    if target_shell_density_m2 is None:
        # Strict no-loss route retained as the conservative transfer fixture.
        add_density = removed_density_integral / float(np.count_nonzero(shell))
        if np.any(gb1[shell] + add_density > maximum_density_m2):
            raise ValueError("insufficient boundary-shell capacity for conservative promotion")
        gb1[shell] += add_density
    else:
        target_shell = np.asarray(target_shell_density_m2, dtype=float)
        if target_shell.shape != gb0.shape or not np.all(np.isfinite(target_shell)) \
                or np.any(target_shell < 0.0):
            raise ValueError("target shell density must be finite, nonnegative, and grid matched")
        if np.any(target_shell[shell] > maximum_density_m2):
            raise ValueError("required boundary-shell density exceeds constitutive capacity")
        deficit = np.zeros_like(gb0)
        deficit[shell] = np.maximum(target_shell[shell]-gb0[shell], 0.0)
        needed = float(np.sum(deficit[shell], dtype=np.longdouble))
        if needed > 0.0:
            fraction = min(removed_density_integral/needed, 1.0)
            gb1[shell] += fraction*deficit[shell]

    volume_factor = cell_area_m2*represented_thickness_m
    line_before = float((np.sum(old_bulk, dtype=np.longdouble)
                         + np.sum(gb0, dtype=np.longdouble))*volume_factor)
    line_after = float((np.sum(new_bulk, dtype=np.longdouble)
                        + np.sum(gb1, dtype=np.longdouble))*volume_factor)
    shell_added_density_integral = float(np.sum(
        (gb1-gb0)[shell], dtype=np.longdouble))
    transferred = shell_added_density_integral*volume_factor
    remaining_density_integral = max(
        removed_density_integral-shell_added_density_integral, 0.0)
    pair_fraction = (pair_removed_density_integral/removed_density_integral
                     if removed_density_integral > 0.0 else 0.0)
    pair_annihilation = remaining_density_integral*pair_fraction*volume_factor
    declared_sink = (remaining_density_integral*volume_factor
                     - pair_annihilation)
    # Close the event balance from the directly changed core and shell content.
    # Subtracting two O(domain-total) line lengths loses the much smaller event
    # increment to cancellation on realistic grids and can falsely reject an
    # otherwise conservative transfer.
    closure = (transferred + pair_annihilation + declared_sink
               - removed_density_integral*volume_factor)
    signed_change = float(np.max(np.abs((rp1-rm1) - signed0)))
    tolerance_scale = max(
        abs(removed_density_integral*volume_factor), abs(transferred),
        abs(pair_annihilation), abs(declared_sink), 1e-300)
    tolerance = 512.0*math.ulp(tolerance_scale)
    if abs(closure) > tolerance or signed_change != 0.0:
        raise RuntimeError(
            "promotion transfer failed line/Burgers-content closure: "
            f"closure={closure:.17g} m, tolerance={tolerance:.17g} m, "
            f"signed_change={signed_change:.17g} m^-2")
    return rp1, rm1, forest1, wall1, gb1, PromotionTransferLedger(
        line_before, line_after, removed_density_integral*volume_factor,
        transferred, pair_annihilation, declared_sink,
        closure, signed_change)


def atomic_phase_promotion(
        eta, psi_gv, Ng, rp, rm, rho_forest, rho_wall, rho_gb, *,
        embryos, embryo_id, tracker, phase_support, purity_threshold,
        target_core_density_m2, cell_area_m2, maximum_density_m2,
        represented_thickness_m, boundary_density_target_m2=None,
        minimum_core_area_m2=0.0,
        step, time_s, energy_evaluator: Callable):
    """Construct and validate one all-or-nothing embryo-to-phase transaction.

    ``energy_evaluator`` receives the trial arrays as keyword arguments and
    returns joules under keys ``elastic``, ``bulk_stored``, ``line``,
    ``interface_order``, and ``compatibility``.  The transaction is admissible
    only for nonincreasing total free energy; the decrease is released as heat.
    Inputs are never mutated.
    """
    if Ng >= np.asarray(eta).shape[2]:
        raise ValueError("no available phase slot")
    matches = [record for record in embryos.records if record.embryo_id == embryo_id]
    if len(matches) != 1 or matches[0].status != "promotable":
        raise ValueError("promotion requires one promotable persistent embryo")
    embryo = matches[0]
    if embryo.parent_grain >= Ng or len(tracker.records) != Ng:
        raise ValueError("embryo parent or grain tracker is inconsistent")
    support = np.asarray(phase_support, dtype=float)
    fields0 = np.asarray(eta, dtype=float)
    if support.shape != fields0.shape[:2] or not np.all(np.isfinite(support)) \
            or np.any((support < 0.0) | (support > 1.0)):
        raise ValueError("phase support must be finite, bounded, and grid matched")
    if not 0.0 < purity_threshold < 1.0:
        raise ValueError("promotion purity threshold must lie in (0,1)")
    core = support >= purity_threshold
    shell = (support > 1.0-purity_threshold) & (~core)
    if not np.any(core) or not np.any(shell):
        raise ValueError("embryo lacks resolved core or interface shell")
    if (not math.isfinite(minimum_core_area_m2) or minimum_core_area_m2 < 0.0):
        raise ValueError("minimum core area must be finite and nonnegative")
    if np.count_nonzero(core)*cell_area_m2 < minimum_core_area_m2:
        raise ValueError("embryo pure core is below the declared resolved area")
    if embryo.history and embryo.history[-1].radial_velocity_m_s <= 0.0:
        raise ValueError("embryo is not positively growing")

    original = dict(eta=fields0, psi_gv=np.asarray(psi_gv, dtype=float), Ng=int(Ng),
                    rp=np.asarray(rp, dtype=float), rm=np.asarray(rm, dtype=float),
                    rho_forest=np.asarray(rho_forest, dtype=float),
                    rho_wall=np.asarray(rho_wall, dtype=float),
                    rho_gb=np.asarray(rho_gb, dtype=float))
    old_energy = energy_evaluator(**original)
    required = ("elastic", "bulk_stored", "line", "interface_order", "compatibility")
    if any(key not in old_energy or not math.isfinite(float(old_energy[key])) for key in required):
        raise ValueError("energy evaluator omitted a finite required term")

    eta1 = fields0.copy()
    active = eta1[:, :, :Ng]
    simplex = np.sum(active, axis=2)
    if np.max(np.abs(simplex-1.0)) > 1e-10:
        raise ValueError("parent phase fields do not satisfy the simplex")
    eta1[:, :, :Ng] = active*(1.0-support[:, :, None])
    eta1[:, :, Ng] = support
    if np.max(np.abs(np.sum(eta1[:, :, :Ng+1], axis=2)-1.0)) > 2e-15:
        raise RuntimeError("trial child insertion violated the phase simplex")
    psi1 = np.asarray(psi_gv, dtype=float).copy()
    psi1[Ng] = embryo.orientation_rad
    rp1, rm1, forest1, wall1, gb1, transfer = conservative_neutral_density_relief(
        rp, rm, rho_forest, rho_wall, rho_gb, core, shell,
        target_core_density_m2=target_core_density_m2,
        cell_area_m2=cell_area_m2,
        represented_thickness_m=represented_thickness_m,
        maximum_density_m2=maximum_density_m2,
        target_shell_density_m2=boundary_density_target_m2)
    trial = dict(eta=eta1, psi_gv=psi1, Ng=Ng+1, rp=rp1, rm=rm1,
                 rho_forest=forest1, rho_wall=wall1, rho_gb=gb1)
    new_energy = energy_evaluator(**trial)
    if any(key not in new_energy or not math.isfinite(float(new_energy[key])) for key in required):
        raise ValueError("trial energy evaluator omitted a finite required term")
    delta = {key: float(new_energy[key])-float(old_energy[key]) for key in required}
    diagnostic_delta = {
        key: float(new_energy[key])-float(old_energy[key])
        for key in sorted(set(old_energy) & set(new_energy) - set(required))
        if math.isfinite(float(old_energy[key])) and math.isfinite(float(new_energy[key]))
    }
    free_change = sum(delta.values())
    tolerance = 512.0*math.ulp(max(*(abs(float(old_energy[k])) for k in required), 1e-300))
    if free_change > tolerance:
        raise PromotionEnergyIncreaseError(
            delta, free_change, tolerance, diagnostic_delta)
    heat = max(-free_change, 0.0)
    closure = -(sum(delta.values()) + heat)
    if abs(closure) > tolerance:
        raise RuntimeError("atomic promotion energy ledger does not close")

    promoted_records = tuple(
        mark_promoted(record, step, time_s) if record.embryo_id == embryo_id else record
        for record in embryos.records)
    embryos1 = EmbryoPopulation(embryos.next_id, promoted_records)
    parent = tracker.records[embryo.parent_grain]
    child = GrainRecord(
        label=Ng, orientation_rad=embryo.orientation_rad,
        parent_label=embryo.parent_grain,
        lineage=f"{parent.lineage}/embryo-{embryo.embryo_id}",
        birth_time_s=float(time_s), source_embryo_id=embryo.embryo_id,
        embryo_promoted=True)
    tracker1 = GrainTracker(tracker.records + (child,), tracker.time_s)
    ledger = AtomicPromotionLedger(
        embryo.embryo_id, Ng, 0.0, delta["elastic"], delta["bulk_stored"],
        delta["line"], delta["interface_order"], delta["compatibility"],
        heat, 0.0, closure, transfer)
    return AtomicPromotionResult(
        eta1, psi1, Ng+1, rp1, rm1, forest1, wall1, gb1,
        embryos1, tracker1, support/float(np.sum(support)), ledger)
