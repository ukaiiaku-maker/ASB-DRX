# V51 decision report

## Decision

V51 qualifies the repaired elementary event measure, a boundary/surface map
below the provisional 5% compatibility threshold on n32/n64/n128, atomic
fail-closed publication, and a scoped two-face shared clock. The protected
loading and same-initial-state hold calculations both completed their exact
52-interval horizon with valid checksums and full requested durations. Their
response is predominantly elastic but exhibits strongly increasing plastic
activity under loading. No stress peak, DRX, persistent LAGB, strict ASB, or
material calibration is demonstrated.

Classification:
`EVENT_MEASURE_AND_GEOMETRY_QUALIFIED_BULK_PAIR_COMPLETE_SPATIAL_REFINEMENT_OPEN`.
Fixtures pass, but the overall scientific gate remains open because no
qualified restriction/initialization operator exists for a matched evolved
bulk companion grid. No spatial-convergence claim is made.

## Event measure and compatible geometry

For the retained one-defect climb convention, V51 uses one count everywhere:

`N_event = N_site |Delta x| / ell_event = |Delta N_species|`.

For the 10 nm n32 qualification event this is 63636.8296707 events. The same
count enters the affinity and ledger, giving `1.85112559336e8 s-1`,
`0.0459079147 m/s`, and `2.178273629e-7 s`. The V50 mixed maximum gives
130072.840791 events and `9.13652011867e7 s-1`; it is retained only as a
labeled comparator. Undefined counts fail without an arbitrary floor. A
separately declared volume-preserving area-event branch remains available.

The scalar positive physical line map is unchanged, preserving the V50 energy
benchmark. A new surface streamfunction is integrated from the independently
oriented boundary with the same spectral derivative used by production Nye.
The zero mode preserves area; a recorded affine contraction only enforces the
bounded surface fraction. Independent initial/event relative discrepancies
are:

| grid | initial | 10 nm event increment |
|---:|---:|---:|
| 32 | 0.4590% | 2.4340% |
| 64 | 0.00615% | 0.1511% |
| 128 | 0.000162% | 0.01052% |

The actual reservoir-moment increment is now compared with swept-surface Nye
inside the production audit. An n16 event with 52.47% candidate mismatch is
rolled back exactly with no event published; n32 publishes and passes. Two
accepted coupled cycles with a checkpoint/restart retain the same ownership.

The n128 10 nm ordered-gradient quotient is correctly labeled a secant:
`-3.5125064e-8 N`. Centered local derivatives at 1 and 0.5 nm are
`-3.51159380e-8 N` and `-3.51159350e-8 N`, close to the continuous
`-3.51163714e-8 N`. The sampled noninteger-translation energy variation
`1.186e-17 J` remains visible; no fitted mesh potential is subtracted.

## Shared clock and overshoot

The two-face transaction evolves both physical faces from one immutable state.
With fixed synthetic rates of `1e8` and `2e8 s-1`, both see a common `1e-7 s`
exposure; elapsed time is not the `2e-7 s` sum of individual clocks. The joint
energy includes cross terms, face order is permutation invariant, two half
clocks reproduce one full clock, signed exchange closes, and nonnegative event
counts sum to 47345.801275. This is a fixed-rate homogeneous-scope result, not
qualification of state-dependent heterogeneous face sampling.

The real face transaction now treats a blocked finite endpoint as a search
problem on the same immutable state. A labeled chemical-reservoir capability
control rejects 50, 25, 12.5, 6.25, and 3.125 nm proposals and accepts the
largest tested connected downhill displacement, 1.5625 nm. This control
demonstrates the numerical capability; it is not an intrinsic calibration.

## Completed physical horizon

Both branches reach `25.390625 us`; loading adds exactly 0.005078125
engineering shear (the planning milestone was 0.005). Loading used frozen V49
source `37065c25e02507da1bc25491574c3ef6fbb01966`; hold used V50 source
`181b28b5a3bb46a92571aa8c07d0a97646679390`. All 104 segments used their full
requested durations, so the conditional V50 driver repair and optional
geometry path are inactive. A direct cheap same-state/full-duration overlap
test confirms equality of the active equations; no expensive replay is
needed.

At the common endpoint, loading minus hold is +407.888346 MPa stress,
+0.000500254 engineering plastic shear, +0.304976 K mean temperature, and
+0.360119 K peak temperature. Plasticity consumes 9.851% of the added shear.
The loading interval plastic rate rises monotonically from 3.642 to
67.040 s-1, reaching 33.52% of the imposed 200 s-1 rate, while the hold rate
declines to 3.474 s-1. The stress remains rising; no peak is observed. The
elastic identity `G(Delta gamma-Delta gamma_p)` matches the observed stress
increment within `6.6e-7 Pa`. Final first-law residuals are
`-3.09e-27 J` (loading) and `-4.29e-27 J` (hold).

The front is disabled, so this pair cannot demonstrate recrystallization.
Loading family-Nye RMS is `41.207 m-1`, versus `6.172 m-1` for hold, but the
orientation spans remain very small and no prepared-loop evidence is relabeled
as a newly formed boundary.

## Numerical uncertainty and performance

A copied evolved n128 loading endpoint was advanced by one full macro and two
half macros without modifying production output. Full/half relative L2
differences are 0.02757% in plastic distortion, 0.02814% in family Nye, and
`9.84e-6%` in temperature; stress differs by 15.39 kPa. Halving costs 1216 s
versus 502 s for the full macro, so the full macro is retained for this
observable scope.

The profile localizes 484.94 s to ordering, 456.12 s to GMRES, and 442.70 s to
matvec/FFT work; state validation is 0.335 s and I/O is negligible. Categories
overlap and are not summed. This identifies the ordering linear solve as the
next optimization target without changing tolerances or kinetics.

A companion-grid bulk comparison was not fabricated: the repository lacks a
qualified physical restriction/initialization of this evolved n128 checkpoint
to another grid. Geometry itself has a genuine fixed-physical-scale n32/n64/
n128 comparison, but that does not establish convergence of the bulk fields.

## Verification and next decision

The canonical suite first exposed one historical V43 assertion that expected a
false hard pass from the represented-plaquette Curl identity. It was converted
to an explicit negative/quarantine assertion; no production criterion was
relaxed. The affected suite passes 9/9. The complete final suite passes
823/823 with the command `PYTHONPATH=src:. python -m pytest -q tests`.

The next bounded work is to define and verify a conservative common-physical-
field restriction/prolongation for evolved signed reservoirs, plastic
distortion, temperature, and phase. Only then should a bulk companion-grid
continuation be used for a spatial claim. Independently, state-dependent local
stress/temperature sampling must replace the synthetic shared-face rates
before interpreting heterogeneous geometry kinetics.
