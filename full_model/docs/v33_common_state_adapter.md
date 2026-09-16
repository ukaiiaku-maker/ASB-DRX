# V33 common-state production adapter

## Decision

Milestones I0 and I1 pass in the actual full-v34 production driver. I2 is
partly verified (complete signed-reservoir fixture and bitwise production
restart), but its complete moving-front energy acceptance is still open. I3 is
not attempted. The adapter is therefore an integration advance, not a DRX or
material-validation result.

The compact 32-square matrix is recorded in
`full_model/verification/v33_common_state_decision.json`. In its combined case
the accepted two-step trajectory has maximum slip `1.42678e-9`, maximum plastic
distortion `1.76575e-9`, and nonzero accepted swept volume
`9.06878e-34 m3`. The front line ledger closes to `1.14678e-16` relative to
processed content and signed closure is zero. A segmented restart compares 95
authoritative fields bitwise with the continuous trajectory.

## Ownership table

| Mutable quantity | Authoritative owner | Legacy/read-only view | Commit rule |
|---|---|---|---|
| parent/child/wake support, pair geometry, first-arrival history | `SparseFrontState` embedded in `CommonFrontState` | phase fields and coupled-front runtime | accepted phase/front proposal only |
| mobile `+/-`, forest `+/-`, wall `+/-` by BCC family | phase-supported `CommonWallState` owners | sparse `rp/rm/forest/wall` totals | Mura increment then complete front transfer in one adapter transaction |
| signed junction populations | phase-supported owner arrays | none | transferred, stored, annihilated, or sunk with the same accepted sweep |
| boundary excess | signed reservoir-class arrays plus junction array | scalar line and family-signed sparse boundary views | incremented only by accepted front passage |
| alignment/line direction state | phase-supported `alignment_m2` | global common view | support-weighted commit |
| slip and plastic distortion | phase-supported `slip` and `beta_p` | `gamma_slip`, `eps_p` | accepted Mura update and front history transfer |
| lattice orientation | phase-supported orientation plus declared phase geometry | `psi_lat` | synchronized after accepted phase/front update |
| family and total Nye | curl of reconstructed authoritative plastic distortion | family view supplied to common Mura operator | recomputed after support weighting; includes interface product-rule term |
| intrinsic HAGB content | phase-field gradient/interface functional | none | never inserted into plastic-excess inventory |
| temperature/thermal energy | phase-supported temperature and physical heat ledger | global `T` | committed with Mura/front transaction; prescribed-temperature control is explicit |
| external work, defect/elastic storage, dissipation and heat | existing common-Mura and coupled-front physical ledgers | diagnostic CSV/JSON | no numerical compatibility term may enter physical heat |

An absent material owner has zero extensive line inventory. Scalar sparse
fields are rebuilt from signed owners and cannot write back when the adapter is
active.

## Discrete Nye reconstruction

For material supports `w_a` and phase-owned plastic distortions `beta_a`, the
adapter reconstructs

`beta = sum_a w_a beta_a`

and applies the production spectral curl to that reconstructed field. Thus the
computed incompatibility contains both weighted bulk curl and the discrete
counterpart of `grad(w_a) x beta_a`. The recorded `interface_nye_m1` is the
difference between this result and a support-weighted sum of owner Nye fields.
It is plastic excess incompatibility, not intrinsic HAGB content.

## Accepted transaction

1. Snapshot immutable common owners, front geometry, loading, time, and ledgers.
2. Evaluate the common Mura proposal over the physical interval.
3. Project its accepted increment into supported owners and rebuild sparse
   compatibility views.
4. Evaluate the phase/front proposal from that state.
5. For an accepted sweep, transfer every signed line class and junction
   population, update kinematics, boundary excess, temperature, history, and
   ledgers, then reconstruct Nye from the committed plastic distortion.
6. Publish once. Rejected front trials retain the pretrial phase and material
   state.

The implementation is a controlled split, not a monolithic solve. I0 verifies
its front-off, Mura-off, and prescribed-temperature limits. I1 verifies actual
simultaneous activity. The next required repair is to include nonzero junction
and kinematic-transfer energy in the front acceptance decision before claiming
I2, followed by coupling-error and grid refinement for I3.

