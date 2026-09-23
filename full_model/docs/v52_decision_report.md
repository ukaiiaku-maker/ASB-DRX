# V52 completion decision report

## Decision

V52 is complete at its achieved scope. The frozen n128 loading continuation
reached 104 intervals (50.78125 microseconds), and the same analytic-origin
n192 companion history reached 32 intervals (15.625 microseconds). All matched
comparisons at intervals 1, 8, 16, and 32 satisfy their time, load,
configuration, and initialization-provenance preconditions and pass the
provisional 5% selected-observable criterion.

Classification:
`PHYSICAL_FLOW_APPROACH_COMPLETE_MATCHED_N192_THROUGH_INTERVAL_32_GEOMETRY_OPTION_QUALIFIED_WITH_CLIMB_STRESS_HYPOTHESIS`.

This does not qualify the later n128 response spatially beyond 15.625
microseconds. It does not demonstrate DRX, a persistent LAGB, strict ASB, or a
material calibration.

## n128 physical continuation

The n128 trajectory completed from the retained interval-52 state without a
load-origin or energy-baseline reset. At 50.78125 microseconds it reaches
engineering shear 0.03015625, plastic shear 0.0043226251, mean stress
2.30177598 GPa, mean temperature 1102.77858 K, and peak temperature
1103.40046 K. Plastic shear is 42.5612% of the shear added beyond the preload.
The last-interval plastic rate is 197.7683/s, or 98.884% of the imposed
engineering rate.

The preregistered descriptive flag—plastic rate at least 80% of the imposed
rate for five accepted intervals—passes, with 24 consecutive qualifying
intervals. It is not a universal flow or material criterion. Stress remains
maximal at the endpoint, so no peak is resolved and no additional peak fork is
warranted. The historical 15.3915 kPa near-state difference remains only a
candidate local scale, not a global error bound.

All available incremental first-law checks pass. The endpoint cumulative
first-law residual is -3.32e-27 J and the maximum absolute cumulative residual
is 4.18e-27 J. The front remains disabled; no grain or recrystallization claim
is available from this trajectory.

## Matched n128/n192 spatial evidence

The n192 state was evaluated directly from the common deterministic analytic
initializer. A same-grid n128 audit compares all 177 restart payload fields,
including signed scalar inventories, alignment moments, parent/child/wake
owners, phase and runtime histories, exactly. Initial clock and load metadata
also match. There was no evolved-state interpolation or remeshing.

Relative differences in selected mean-response increments are:

| interval | time (us) | stress | plastic shear | mean T | peak T |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.48828125 | 7.88e-9 | 4.25e-7 | 4.87e-7 | 1.71e-7 |
| 8 | 3.90625 | 2.56e-9 | 1.11e-7 | 3.95e-7 | 7.52e-7 |
| 16 | 7.8125 | 3.35e-8 | 1.10e-6 | 1.49e-6 | 2.17e-6 |
| 32 | 15.625 | 3.13e-7 | 5.54e-6 | 6.16e-6 | 5.60e-6 |

At interval 32, common-band half-width-32 relative L2 differences are 0.0110%
for beta_p, 0.3855% for family Nye, 0.0125% for temperature rise, and 3.4439%
for wall-ordered-plus density. The n192 fine-only fractions above the n128
Nyquist are 0.000730%, 0.3550%, 0.000892%, and 0.0641%, respectively. These
are selected mean and spectral-structure statements, not equality of every
field. The n128 response from 15.625 to 50.78125 microseconds remains spatially
unqualified.

The four n192 stages reused the accepted prefix: 1, 8, 16, then 32 intervals.
The final n192 checkpoint checksum is verified. Git source identities changed
when analysis-only comparison code was committed during the staged history;
the production directory and physical continuation driver are byte-identical
between those commits, which is recorded explicitly in the decision artifact.

## Solver overlap and performance

The retained evolved-state overlap measures 552.536 s for the composed oracle
and 204.480 s for the fused operator, a 2.70215x speedup on this host.
Plastic distortion, phase, family Nye, temperature increment, and selected
observables agree exactly; reservoir totals agree to at worst 2.19e-16
relative. The overlap did not compare every reservoir spatial field or every
history variable, and the speedup is not presented as universal.

## Optional shared-face geometry

The signed chemical coefficient now retains the sign of z*b_z*L/Omega;
nonnegative event counts alone use an absolute value. Negative-b_z and
nonzero-chemical-potential tests agree with the accepted shared ledger. A
stalled face can remain fixed while the other advances on the common physical
clock.

On a nonuniform temperature and mechanically nonzero resolved-stress field,
the two rigid faces have unequal rates. Against order-128 quadrature, order-64
rate errors are 6.23e-8 and 5.81e-8 relative. The existing one-microsecond
whole/split displacement difference remains 0.3363%, repeated geometry
restart is exact, and the minimal
geometry--Mura/ordering/thermal--geometry alternation is restart exact.

The optional kinetics currently use resolved glide stress as a declared climb
activation hypothesis. The tested geometry move has zero elastic-energy
increment even though sampled stress is nonzero. Therefore a calibrated
climb-resolved activation variable and a nonzero conjugate mechanical-work
geometry case remain open; V52 does not overstate this branch as a material
climb law.

## Recovery and verification

The original n128 worker was never interrupted. The obsolete waiting manager
was replaced through a PID/start-time/cwd/output-ownership audit. The new
controller uses a logical-output lock, resumable journal, direct/module-form
process recognition, checksum validation, staged child identities, bounded
infrastructure recovery, valid-terminal handling, and staged cost decisions.
The first n192 launch exposed a metadata-schema defect before any interval was
accepted; the initializer was repaired and the controller resumed without
repeating completed n128 work.

The final canonical suite passes 846/846 tests in 333.47 s. The authoritative
machine-readable decision is
`full_model/verification/v52_completion_decision.json`. Raw n128 and n192
checkpoints remain in their local results directories; copied manifests and
all report-level checksums are retained under `full_model/verification`.
