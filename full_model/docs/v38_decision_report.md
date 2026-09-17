# V38 informative-horizon and common-state decision report

## Evidence boundary

V38 continues the pushed V37 execution checkpoint `9b9c684` and preserves the
earlier manifest checkpoint `bd366c7` in its earlier role. The first V38
infrastructure repair is `1388e8f`; the production and diagnostic repair is
`4b4a4af`; the front overlap driver is `1f568c1`; and the first common-state
multirate driver is `ea06103`. Every numerical JSON records its exact source
commit. The V37 rate screen remains a rate screen, not a trajectory or a
material calibration.

## Front parameter semantics and selection

The production channel now reports four distinct quantities:

- `front_event_volume_m3`, the geometric volume used for physical event counts;
- `front_jump_length_m`, which also fixes physical site area;
- `front_channel_pressure_Pa = |Delta F|/V_event`;
- `front_activation_volume_m3 = -dG_front*/dP`, evaluated analytically from
  the EXP-floor barrier.

The old `moving_front_activation_volume_b3` input remains readable as an
explicit compatibility spelling for the geometric event volume. It no longer
silently names that volume as a barrier derivative. A finite-difference test
verifies the analytical derivative.

The raw 17-case ranking was retrieved and asserted. `availability_mid` is
exactly one half of the baseline fixed-state velocity and is relabeled a
slower-availability control. `shape_a_high` is the selected existing
intermediate candidate. `shape_n_low` remains the fastest hypothesis, but it
changes pressure sensitivity and is not described as a speed multiplier.

## Affordable physical front horizon

The bounded owner-moment operation is a legitimate minimum-change projection
onto the exact common first moment and the owner constraints
`|kappa_owner| <= rho_owner`. It does not reconstruct a target from Nye. The
optimized solver removes converged bulk points while retaining the reference
map and fully polarized analytical limit. At n128 it is state-equivalent to
`2.22e-16` maximum absolute difference and is 2.69 times faster for the
measured operation.

A matched 1 ms, zero-applied-work, predeformed-hold overlap compared 5 us,
25 us, 100 us, 250 us, and 1 ms front macro intervals. The 1 ms result differs
from the 5 us reference by at most 0.285% across accepted displacement,
signed sweep, processed/boundary/sink line, and complete-energy change. It is
therefore the largest tested interval passing the unchanged 5% tolerance.
The reference advances 3.995 nm in 1 ms at n16; a quarter of the 0.40 um
interface width is consequently an approximately 25 ms, 25-macro-interval
target under the evolving `shape_n_low` hold hypothesis. This is a scheduling
endpoint, not a constant-speed prediction.

## Equal-owner residual

At fixed 100 nm spacing and 0.40 um interface width, increasing periodic slab
length from 3.2 to 4.8 to 6.4 um gives Mura-off displacements
`2.203e-13`, `2.116e-13`, and `2.113e-13` m. The residual plateaus instead of
decaying with interface separation. Recurrent Mura reduces it by only about
`2.67e-15` m at the two larger sizes. The evidence therefore rejects the
finite-separation interaction explanation and supports a local diffuse-phase
energy/representation (capillary) residual. No fitted drift or equal-state
projection is used.

## Ordered-line timestep repair

The exact-source legacy replay identifies the first divergent channel:

| integrated channel | relative 100 ps/50 ps difference |
|---|---:|
| captured line | `7.66e-12` |
| Mura stretching | `1.98e-8` |
| locking/unlocking | `4.82e-5` |
| gross ordering turnover exposure | `9.73e-5` |
| net tangle-to-ordered transfer | `9.31e-1` |

The old ordering cap accepted at most one capped extent per outer Mura step and
discarded the remaining reaction exposure. A frozen-field reaction-only replay
reproduced the discrepancy, excluding capture timing as its first cause.

The repair conservatively subcycles the reversible ordering channel over the
complete elapsed time, advances signed line alignment with every scalar
transfer, and reports actual time-averaged rates. No reaction time is
discarded. In the repaired replay, net ordering-transfer disagreement is below
`1e-4`; final ordered line is `0.04144` versus `0.04181` m per metre thickness,
or 0.88% relative. Both refinements select the same near-extinction response,
while total line, Burgers/Nye consistency, and owner alignment bounds remain
closed. This passes the provisional 5% ordered-response requirement for this
matched horizon. Long macro intervals that neither resolve the channel nor
reach the declared stationary tolerance fail explicitly.

## Thermal diagnostics and fields

Current-source restart checkpoints now save three distinct fields:

- absolute shear rate (historical activity only);
- signed work-conjugate plastic power `sum(tau_resolved * gamma_dot)`;
- the actual irreversible heat-production field supplied to the thermal
  equation after physical partitioning and process-zone regularization.

Postprocessing preserves historical absolute-temperature participation but
adds `T-T0` diagnostics with the declared positive-excess weight
`max(T-T0,0)`, signed moments, variance, peak-minus-mean, negative-anomaly
fraction, and axis profiles. Periodic translation/rotation and uniform 900 K
background fixtures pass. A uniform field is labeled
`NO_LOCALIZED_COMPONENT`; it is not assigned a zero-width band.

The frozen V37 conduction source predates the new power-field checkpoint keys,
so its rerun can supply corrected temperature-rise metrics but cannot be
relabeled as independently field-ledgered power evidence. That evidence starts
with current-source V38 trajectories.

## Common-state integration decision

A monolithic 1 ms common step was rejected correctly: Mura accepted only
0.584 ms and the front did not publish, so forcing the requested clock would
have created unledgered time. A Strang multirate implementation now advances
Mura/thermal to the half-clock, evaluates one full front macro interval from
the midpoint state, and closes the second Mura/thermal half-clock.

Its first 1 ms preflight exposed a sharper numerical terminal after the front:
the ordering channel did not reach its declared stationary tolerance within
8192 resolved 0.5 ps substeps. The driver refuses to discard the remainder or
coarsen that channel silently. Thus a coherent current-source state and clock
implementation now exist, but the informative integrated trajectory is not
yet qualified. The next correction must be an implicit or independently
qualified asymptotic ordering solve; reverting to the outer-step cap is not
authorized.

## Current decisions

- Front mechanism-specific names and raw-rate ranking: **passed**.
- n128 bounded-moment map equivalence and acceleration: **passed**.
- Front-only 1 ms macro overlap at n16: **passed**, spatial promotion pending.
- Equal-owner residual attribution: **local diffuse-interface residual**.
- Ordered-channel matched-time accuracy: **passed for the audited horizon**.
- Thermal diagnostic semantics and current-source field output: **passed**;
  V37 frozen-source power fields remain unavailable by provenance.
- Coherent informative common-state trajectory: **not yet passed** because the
  resolved explicit ordering channel reaches a stiffness terminal.
- DRX, a persistent compatible LAGB, calibrated material response, and strict
  ASB are **not claimed**.

The merged current-source regression boundary passes **729 tests** in
178.06 seconds.
