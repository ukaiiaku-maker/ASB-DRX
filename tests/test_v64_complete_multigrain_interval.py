from dataclasses import replace

import numpy as np

from full_model.analysis.run_v58_three_grain_production import (
    initialize_network_state, network_interfaces,
)
from full_model.production.arrhenius_kinetics import ActivatedProcess, EV_J
from full_model.production.common_tensorial_wall import CommonWallParameters
from full_model.production.coupled_multigrain_interval import (
    CoupledMultigrainContext, advance_complete_multigrain_interval, make_endpoint,
)
from full_model.production.multigrain_common_state import PhysicalTransferLaw
from full_model.production.multigrain_production import (
    MultiGrainFrontKinetics, MultiGrainProductionRuntime,
)
from full_model.production.tensorial_nye import bcc_four_family_systems


def context(front_enabled=True):
    spacing = 5e-6/8
    wall = CommonWallParameters(
        spacing_m=spacing, elastic_iterations=2,
        mobile_correlation_diffusivity_m2_s=0.0,
        maximum_fraction_per_step=.1, volumetric_heat_capacity_J_m3_K=3.8e6,
        thermal_diffusivity_m2_s=.15/3.8e6, bath_rate_s=0.0)
    kinetics = MultiGrainFrontKinetics(
        ActivatedProcess("test", 1e8, 0.0, 1e8),
        PhysicalTransferLaw(.55, .08, .02), .35*EV_J, 1e9,
        2.0, 1.5, .10, wall.burgers_m**3, wall.burgers_m,
        virtual_fraction=2e-4, maximum_fraction_per_step=.015,
        closure_fraction=.05, maximum_backtracks=14)
    return CoupledMultigrainContext(
        bcc_four_family_systems(), (), wall, kinetics, spacing, 4.96e-10,
        4e4, 1e-8, 10,
        {"topologies": (), "systems": bcc_four_family_systems(),
         "phase_barrier_J_m3": 5e6, "phase_gradient_J_m": 5e-7,
         "boundary_line_energy_J_m": wall.line_energy_J_m,
         "boundary_junction_energy_J_m": wall.junction_energy_J_m,
         "reference_temperature_K": 900.0},
        front_enabled=front_enabled)


def parent(ctx):
    state = initialize_network_state(
        8, 900.0, 4, spacing_m=ctx.spacing_m,
        interface_width_m=3.125e-7)
    runtime = MultiGrainProductionRuntime(network_interfaces(state))
    return make_endpoint(state, runtime, .012, ctx)


def test_complete_interval_advances_physical_and_load_clocks_once():
    # The small n=8 fixture exercises the exact coupled clock with the front
    # channel disabled; resolved-front acceptance is covered by the production
    # checkpoint tests and the V64 n48/n64 smoke trajectories.
    ctx = context(front_enabled=False)
    before = parent(ctx)
    result = advance_complete_multigrain_interval(before, 1e-10, ctx)
    assert np.isclose(result.endpoint.runtime.ledger.physical_time_s, 1e-10)
    assert np.isclose(result.endpoint.load-.012, 4e-6)
    assert np.isclose(result.endpoint.runtime.ledger.applied_shear_strain, 4e-6)
    assert result.audit["mechanical_consumed_interval_s"] == 1e-10
    assert before.runtime.ledger.physical_time_s == 0.0


def test_disabled_front_channel_has_exact_zero_front_heat():
    ctx = context(front_enabled=False)
    result = advance_complete_multigrain_interval(parent(ctx), 1e-10, ctx)
    assert result.integrals["front_heat_J"] == 0.0
    assert np.all(result.integrals["front_heat_J_by_cell"] == 0.0)
    assert result.audit["front_subintervals"] == 0
