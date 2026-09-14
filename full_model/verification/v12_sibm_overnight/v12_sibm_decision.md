# Directive v12 SIBM post-processing decision

## Decision

`FULL_MODEL_SIBM_GROWTH_MECHANISM_NOT_SUPPORTED`

The negative decision is validity-led, not a fitted or tuned outcome. Every seeded trajectory already lacks an independently resolved parent pure core in its step-0 campaign checkpoint. The two unseeded controls begin with a resolved pair but lose the parent pure core by step 250. A diffuse pair zero contour can persist after this loss and therefore cannot by itself establish SIBM of two physical grains.

All phase-simplex, population nonnegativity, line, signed-Burgers, energy/heat, and no-label/no-orientation-allocation checks pass at the audited checkpoints. Those conservation results do not cure the pair-identity failure.

## Cross-case findings

- Baseline minus matched-control tip change at their common horizon: -6.267996e-07 m.
- Window trajectories agree within one 128-grid cell: True (secondary evidence only).
- Grid relative spread at the common horizon: 80.225%; provisional 5% test: False.
- Mobility displacement ordering: False.
- Manufactured criticality sign split passed: False.
- Temperature and rate endpoints remain confounded by invalid pair identity and different evolving thermomechanical histories; neither is promoted as an interpretable trend.

## Primary classifications

- `SIBM_CRITICALITY_CONTROL_FAILED`: E1_subcritical_fixture, E2_supercritical_fixture
- `SIBM_FLAT_BOUNDARY_MIGRATION_ONLY`: A2_no_bulge, D2_grid192_control
- `SIBM_PAIR_IDENTITY_LOST`: A1_baseline_bulge, A3_window_2um, A4_window_3p5um, A5_radius_0p5um, A6_radius_1um, B1_mobility_0p3, B2_mobility_3, B3_temperature_1000K, B4_temperature_1200K, B5_rate_300, B6_rate_3000, D1_grid192_bulge, D3_grid256_bulge

## Required next scientific action

Construct or select a source state whose parent and child retain resolved pure cores after seeding, make loss of either core a clean per-case stop, and rerun only a compact bulge/control/criticality qualification before any new range campaign. The present matrix must remain implementation-falsification and negative-mechanism evidence.
