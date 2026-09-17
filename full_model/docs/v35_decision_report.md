# V35 recurrent common-state and physical-time decision

## Outcome

V35 closes the implementation blocker that prevented a second complete
Mura/front cycle. Parent, child, and processed-wake owners now persist their
extensive signed scalar inventories and reservoir-resolved line moments; the
front transfers both through the same accepted partition. The next Mura step
is reconstructed from those owners without a target-Nye inversion, silent
zero initialization, or owner reset.

Two forced production cycles pass on one common 2 ns clock, including a
nonzero second sweep. Midpoint restart is bitwise exact, a rejected second
front rolls back exactly to the Mura-only state, advance/retreat/revisit
preserves wake moments, and vanishing support preserves inactive history. A
separate 100-interval alternating stress test accepts 100/100 events and
commits; it is schema endurance with enlarged forced front exposure, not a
physical-time result. Its 1000-interval projection exceeded the 30-minute
local boundary and was not launched locally.

The scientific completion gate remains open. The pressure-free complete-state
control is a valid arrest under the retained generic kinetics, the physical
front horizon is still short, and no material calibration, DRX, new grain,
LAGB maturation, or strict ASB is claimed.

## Decisions

| Branch | Decision | Boundary |
|---|---|---|
| Recurrent owner state | `RECURRENT_OWNER_STATE_IMPLEMENTED` | Two consecutive Mura/front cycles, exact restart/rollback, retreat/revisit, vanishing support, and 100-cycle schema endurance pass. |
| Common physical clock | `TWO_CYCLE_COMMON_CLOCK_VERIFIED` | Both operators cover the same accepted 2 ns interval in each of two forced cycles. |
| Pressure-free front | `VALID_PHYSICAL_ARREST` | External work is zero. Both actual irreversible directional events are downhill and receive equal barrierless rates, hence zero drift. |
| Mura fractional events | `FEASIBLE_EXTENT_OPERATOR_DISCRIMINATING` | Later checkpoint family 3 accepts extent 0.0625 although the full event is uphill; an exact-time short continuation differs from frozen V34 while preserving invariants. |
| Phase proposal | `QUALIFIED_SEPARATED_PHASE_CONTROL` | IMEX and topology switches are independent; inactive control is bitwise identical; active-set-off reproduces the archived unauthorized island. Scope remains existing-boundary/no-nucleation. |
| Physical event measure | `DIMENSIONALLY_QUALIFIED_NOT_CALIBRATED` | Counts use real interface area divided by b²; velocity and areal site density are mesh/patch/thickness independent. b³,b remains generic. |
| Thermal causality | `FLOW_TEMPERATURE_CAUSAL_EFFECT_ESTABLISHED_MATRIX_RUNNING` | Full law and corrected authoritative flow freeze are complete; recovery-group, thermostat, and conduction controls continue. No strict-ASB claim. |

## Recurrent state and compatibility

The V35 checkpoint schema owns parent/child/wake `DensityInventory` and full
`ReservoirAlignmentState` fields in addition to common slip, plastic
distortion, orientation, temperature, phase support, boundary excess, and
front history. Support-weighted reconstruction uses only present owners.
Transfers of alignment and scalar line use the identical accepted front
fraction. The independently assembled reservoir Nye field remains a check on
the plastic-distortion curl; it is not constructed as the residual needed to
close that curl.

The common-clock two-cycle result uses 200 MPa applied pressure only as a
forced verification. Its signed sweeps are `1.23493e-27 m3` and
`1.23436e-27 m3`. The 100-cycle stress result uses deliberately enlarged
exposure to exercise topology/history and is not labeled physical time.

## Pressure-free stored-energy result

The geometric proposal probe is now explicitly separated from physical work
and final kinetics. It only constructs a topology-valid finite candidate; the
candidate's complete directional energies replace the probe before rate and
publication decisions. The probe contributes zero work and zero stored
energy.

For the current high-defect/low-defect bicrystal at zero applied pressure, the
actual A→B and B→A event changes are `-1.1070e-23 J` and `-4.0884e-24 J`.
They are not microscopic reverses because both process line content. Both are
downhill, so the retained Metropolis pair assigns the same barrierless rate.
Net velocity and published volume are exactly zero. This is a physically
consistent arrest for the current generic law, not evidence that stored
energy is absent and not a numerical rejection.

## Mura family extent

The frozen V34 selector tested each family only at full finite extent. V35
samples the same constitutive ray at `s=1 ... 1/2048`, selects the largest
admissible sampled extent, and reevaluates the joint event. At step 1451,
families 2 and 3 remain genuinely uphill throughout. At step 2971, family 2
remains uphill but family 3 becomes downhill below the full-event overshoot
and selects `s=0.0625`.

Matched continuations from the byte-identical step-2971 checkpoint end at the
same physical time `4.202103e-6 s`. Both remain hard-invariant clean. The
feasible-extent state differs from frozen V34 by `-1.2847e13 m^-2` in maximum
signed density and `-0.00603 degree` in orientation span after 13 intervals.
This establishes a real trajectory consequence but not long-time convergence.

## Phase and kinetic measure

The phase integrator switch and topology active-set switch are independent.
For smooth n64 migration the active set does not activate and on/off eta,
density, and temperature are bitwise identical. At the archived n128
discriminator it activates 48 times and changes 596 pixels; disabling it
reproduces `UNAUTHORIZED_PHASE_ISLAND`. The restriction is therefore a named
existing-boundary topology model, not an integrator repair or a general
no-nucleation theorem.

The physical site measure is `N=A_front/b²`, with directional expected events
`N r± dt`, swept volume `(N+−N−)b³`, and velocity `b(r+−r−)`. A 15-case,
zero-applied-pressure screen over 900/1100/1300 K and stored-energy pressures
of ±600, ±200, and 0 MPa has exact sign symmetry, maximum detailed-balance
residual `1.11e-16`, zero patch residual, and thickness-independent normalized
velocity. It is a bounded hypothesis screen, not calibration.

## Provenance

The old `59f4d12` trajectory failed at step 1451. The repaired `e7aa16e`
trajectory resumed that exact checkpoint, remained valid through history step
4650, and disappeared without a terminal record; checkpoint step 4553 is its
latest exact restart. These histories are not relabeled as V35 trajectories.

Machine records are `v35_recurrent_state_decision.json`,
`v35_mura_feasible_extent.json`, `v35_mura_matched_time_comparison.json`,
`v35_mura_trajectory_reconciliation.json`,
`v35_phase_kinetic_decision.json`, and
`v35_parameter_hypothesis_registry.json`.

## Corrected thermal causality

At matched step 2500 (`8.3367 us`, nominal strain 0.2501), freezing only the
authoritative flow temperature at 900 K raises stress by `267.66 MPa`, lowers
`Tmax` by `528.06 K` (`1281.10 K` versus `1809.16 K`), reduces softening from
`0.14053` to `0.05754`, and increases the frozen activity fraction from
`0.43043` to `0.97282`. The corrected ablation closes first law to
`1.54e-11` relative and its Burgers, line, and energy residuals remain below
`7.8e-17`, `4.8e-17`, and `1.5e-18`.

This establishes that temperature-dependent EXP-floor flow feedback is needed
for the observed broad-to-more-localized evolution in this trajectory. It does
not establish strict ASB: the full-law case still fails the unchanged
localization threshold. The corrected recovery-temperature group, exact
thermostat, and finite-conduction controls remain active and are not inferred
from this one intervention.

## Remaining work

The next scientifically justified step is not to force motion. It is to finish
the remaining recovery-group, thermostat, and conduction comparisons,
preserve the pressure-free arrest as a baseline, and derive or constrain a
directional kinetic closure that distinguishes non-reverse downhill
transactions before spending a long physical horizon. The generic b³,b rate
family remains useful for bounded sensitivity, not for a material prediction.

The final merged local suite passes `693` tests in `370.81 s` with
`PYTHONPATH=src:.`. This count includes the recurrent-state, common-clock,
pressure-free, feasible-extent, phase-switch, and physical event-measure tests.
