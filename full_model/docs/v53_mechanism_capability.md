# V53 production mechanism capability

This table separates software capability from what the V53 bulk and prepared-
geometry calculations actually exercise.  `Implemented` means an authoritative
production routine exists.  It does not mean the mechanism has produced the
named physical phenomenon.

| Mechanism | Implemented | Enabled in V53 bulk/geometry | Directly exercised there | Physically demonstrated there | Authoritative routine and evolved state |
|---|---:|---:|---:|---:|---|
| Existing-grain-boundary migration with conservative defect processing | Yes | No | No | No | `moving_front.advance_front`; `SparseFrontState` inside `I3State.common_front.front`, invoked by `run_i3_cycle` only when `I3Controls.front_enabled` is true |
| Intragranular signed-dislocation organization | Yes | Yes in bulk | Yes | Organization/flow only; no independently recognized new boundary | `v24_mechanical_wall.accepted_v24_mechanical_step`; signed family reservoirs in `MechanicalWallState.density`, common `beta_p`, and compatible Nye |
| Independent intragranular boundary recognition | Implemented as a diagnostic candidate | No | No | No | `intragranular_subgrain.recognize_subgrain`; its one-grain state is not the V53 bulk state and recognition does not allocate a production grain |
| Grain representation/promotion and subsequent growth | Existing boundaries only | No | No | No new grain | `moving_front.initialize_existing_boundary_front` and `advance_front`; no V53 nucleation/promotion path is enabled |
| Plastic-work and reaction-heat production | Yes | Yes | Yes; scalar interval ledgers retained | Energy accounting, not localization | `accepted_v24_mechanical_step` and the accepted Mura balance; common temperature in `MechanicalCommonState` |
| Thermal transport and temperature-dependent flow/recovery | Yes in the full production driver | Bulk uses temperature-dependent mechanics but not the monolithic ASB transport experiment | Not as an ASB causal pair | No | `drx_full_v34_recovery.py`, including `_v30_thermal_dissipation_channels`, the thermal update, and `causal_temperature_ablation` |
| Spatial plastic-power concentration and shear-band persistence | Fields/classifier implemented | No accepted-trajectory power field in V53 checkpoints | No | No ASB | `drx_full_v34_recovery.py` saves `asb_last_plastic_power_W_m3` and `asb_last_heat_production_W_m3`; `_asb_band_metrics` and the V37/V41 postprocessors classify connected support and persistence |
| Prepared two-face line geometry | Yes, default off | Yes only in the V53 geometry branch | Pending handoff stage | At most manufactured mechanism verification | `accepted_subcell_x_faces_shared_clock` through `coupled_geometry_clock`; this is neither a grain boundary nor spontaneous DRX |

The next independent mechanism calculation is a current-source ASB causal pair
at 30,000 s^-1 with positive conductivity.  Both cases start from the same
deterministic one-grain heterogeneous initialization, disable grain nucleation,
and differ only by whether the flow-rate evaluation uses the evolving
temperature or the declared `freeze_flow` control.  This repeats a previously
informative configuration without transferring its qualification across source
commits.

