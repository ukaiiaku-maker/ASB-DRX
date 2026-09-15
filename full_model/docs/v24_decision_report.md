# V24 decision report: topological wall supply, dynamic SIBM, and ASB persistence

Date: 2026-09-15

This report applies the fixed V24 criteria without fitting to phase-field
outcomes. All V23 evidence remains frozen, all HPC3 result archives were
checksum verified after retrieval, and no unrelated job was modified.

## Decisions

| Question | Decision | Scientific pass |
|---|---|---:|
| Does a mechanically generated orientation gradient acquire sufficient conserved signed boundary inventory and persist as a LAGB? | `KINEMATIC_TRANSITION_BAND_WITH_INSUFFICIENT_BOUNDARY_INVENTORY` | no |
| Does a favorable planar low-defect child advance under its own stored-energy difference with valid controls? | `FULL_DYNAMIC_PLANAR_SIBM_SIGN_OR_COUPLING_FAILURE` | no |
| Does seed-43 localization satisfy the strict ASB persistence and refinement definition? | `STRICT_ASB_NOT_OBSERVED` | no |

These are mechanism-specific negative decisions, not a rejection of the full
campaign premise. They identify the closures that must change before broader
parameter or material studies are defensible.

## Signed topological line supply

Every signed mobile, forest, tangle, and ordered reservoir carries a bounded
line-direction first moment. Capture, non-reorienting transfer, finite-segment
reorientation, and junction reactions have separate ledgers. The tensorial
balance is reported as

`d(alpha)/dt + Curl(J_alpha) = R_topology`.

Finite-segment forward and reverse reactions share the common EXP-floor rate
and satisfy `k_f/k_r = exp(-Delta F_event/(k_B T))`. Reverse events consume the
stored turning-node and curvature inventories. No orientation-derived target,
Nye matching coefficient, grain allocation, or multi-hit response is active.

Across 16/24/32 grids, the full-elastic transition band reaches only
0.91--0.95 degrees local plateau misorientation. Transport/capture supplies
0.014--0.038 of the required integrated Frank--Bilby inventory. Explicit
reorientation supplies 3.05--3.51 times the required magnitude, but in the
wrong tensorial direction: the minimum local residual remains above 1.0.
Neither route produces a qualifying persistent segment.

Machine-readable result:
`full_model/verification/v24_mechanical_supply_local.json`.

## Full-driver planar SIBM

The first zero-artificial-pressure matrix used a common phase geometry, exact
zero seeded curvature, equal/favorable/reversed density contrasts, and a
mobility-off control. Hard phase, label, and front-ledger invariants pass, but
the equal, favorable, and reversed active branches all advance by many
interface widths. The mobility-off control is stationary.

An audit found that the inherited bicrystal checkpoint also carried 62.7%
macroscopic strain and approximately 465 MPa stress. A single bounded repair
reset clock, total strain, and plastic strain before common equilibration; it
did not change mobility or stored-energy parameters. The repaired active
branches still move 19.76, 19.96, and 18.45 interface widths in the same net
direction. The reversed branch therefore contradicts its negative initial
pressure. The complete phase-owned functional also applies compatibility
content asymmetrically to the child, so equal assigned density is not an equal
complete-energy control.

Production SIBM remains rejected until that phase ownership/sign coupling is
repaired. The legacy full driver also lacks exact all-defect freezing and the
submitted matrix exercises the final all-channel state rather than all five
sequential full-driver stages; those limitations prohibit promotion even if a
future directional matrix succeeds.

Machine-readable results:
`full_model/verification/v24_planar_sibm_result.json` and
`full_model/verification/v24_planar_sibm_unloaded_result.json`.

## Strict ASB persistence

The unchanged conjunctive thresholds are active fraction at most 0.25,
matched adiabatic temperature excess at least 50 K, post-peak softening at
least 20%, effective width at least two diffuse-interface widths, continuous
persistence at least 1 microsecond, and 5% grid/timestep refinement.

- The 64-grid history reaches 36.32% maximum softening and 252.1 K maximum
  matched heating. Sixteen snapshots satisfy the full instantaneous
  conjunction, but the longest continuous interval is only 0.210 microseconds.
- The 128-base history reaches 108.6 K matched heating but only 18.00% maximum
  softening, so it has zero conjunctively qualifying snapshots.
- The half-step 128 history also reaches only 18.00% maximum softening and has
  zero conjunctively qualifying snapshots.

Because neither 128 history qualifies, onset/width refinement cannot be
established. The strict ASB claim remains false; the strong localization and
heating near miss is retained as evidence.

Machine-readable result:
`full_model/verification/v24_strict_asb_result.json`.

## Next scientific work

1. Repair phase-energy ownership so equal parent/child states are a genuine
   fixed point and pressure reversal reverses the full variational velocity.
   Add exact defect-freeze and sequential full-driver activation controls.
2. Replace the prescribed out-of-plane reorientation product with a
   stress/topology-selected product direction or explicit junction network
   capable of closing the local Frank--Bilby vector, without target fitting.
3. Do not extend the present seed-43 ASB rate bracket. First identify why the
   128-grid post-peak response saturates at 18%; any subsequent bracket must
   retain the fixed persistence and refinement criteria.

The configured regression suite passes 429 tests. At the final queue audit,
no V24 campaign job remained active; unrelated job `55950433` was untouched.
