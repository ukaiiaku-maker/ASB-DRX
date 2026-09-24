# V54 mechanism-first ASB and existing-boundary DRX decision

## Decision

V54 completed both production anchors locally with hard numerical validity and the required causal/control separation. The scoped scientific result is negative for strict ASB and for substantial existing-boundary DRX growth at the tested conditions. This is not a universal rejection of either mechanism.

| Mechanism | Implemented | Enabled | Exercised | Demonstrated |
|---|---:|---:|---:|---|
| Thermal-feedback ASB pathway | yes | yes | yes | causal broad/nonpersistent feedback; **not strict ASB** |
| Prepared misoriented-boundary pathway | yes | yes | yes | atomic motion and defect processing; **not substantial geometric growth** |
| Spontaneous intragranular grain birth | no | no | no | not claimed |

## ASB causal pair

The n128 production pair advanced from one exact shared prefix in alternating physical-horizon blocks to 8.3366667 microseconds and audited applied strain 0.2501. The feedback member used evolving temperature in flow and recovery. The `freeze_flow` control held only the flow temperature at 900 K while recovery, conduction, heat evolution, and exports remained active. Runtime routing, exact clocks, affine loading history, parameter differences, independent dissipation channels, first law, Burgers balance, and line balance all pass.

At the endpoint, feedback gives 2.70404 GPa mean stress, 1167.47 K mean temperature, 1225.72 K peak temperature, a 58.253 K peak-minus-mean contrast, and plastic-power participation 0.90178. The frozen-flow control gives 2.94933 GPa, 1178.13 K, 1196.29 K, 18.159 K, and participation 0.98361. Thus thermal feedback reduces stress by 245.29 MPa relative to the control and increases peak-minus-mean contrast by 40.094 K. It does not produce a qualifying persistent connected band: no localization episode passes the inherited strict criterion, and both maximum and terminal qualifying durations are zero/undefined.

Classification: `VALID_BROAD_OR_NONPERSISTENT_THERMAL_FEEDBACK`.

## Existing-boundary DRX anchor

The production n64 calculation used a 3.2 micrometre domain, 400 nm interface width, 20 degree declared parent/child misorientation, measured 18.544 degree pure-core contrast, 0.35 child line fraction, 1100 K initial temperature with thermal evolution, 100 s^-1 declared loading rate, 2 ns increments, qualified midpoint loading, and `compatible_dealiased` Mura transport. Both bulk evolution and the physical moving-front pathway were active. External migration pressure/work remained zero; intrinsic boundary energy came from the phase barrier and gradient terms.

The paired enabled/disabled trajectories both completed 500 intervals (1.000 microsecond). The enabled front moved 3.82047e-13 m, or 0.0015405 Burgers vectors, swept 1.21277e-27 m3 of new material with zero revisit, processed 1.16419e-12 m of line, and stored 2.91748e-14 m at the boundary. The one-Burgers-vector, predeclared substantial-motion threshold is 2.48e-10 m, so the observed motion is real but negligible geometrically. The disabled path has exactly zero sweep, displacement, and processing; equal-complete-state and zero-exposure controls have zero physical sweep, and contrast reversal changes the displacement sign. Loading and complete-front energy ledgers pass cumulatively and interval-by-interval.

Classification: `VALID_ATOMIC_MOTION_WITH_NEGLIGIBLE_GEOMETRIC_GROWTH`. This supports a working prepared-boundary transaction and defect-processing pathway; it does not establish meaningful DRX growth or spontaneous grain birth.

## Numerical repair and provenance

The first DRX worker encountered an exact representation invariant at accepted interval 23: independently advanced scalar compatibility views had accumulated a cancellation-sized difference from the authoritative split tangle/ordered reservoirs. The repair reconstructs those views exactly from their authoritative owners; it does not alter the reservoirs or loosen tolerances. The accepted prefix was retained and resumed with an explicit source transition from `35f123c` through repair `4aac7f2` to numerical source `028b8d6`.

ASB numerical source is `bb468ab`; DRX numerical source is `028b8d6`; the final analysis and canonical-test source is `19a5aac`. At that frozen source, `PYTHONPATH=src:. python -m pytest -q tests` passes all 872 tests in 462.90 seconds. The initial invocation without `PYTHONPATH=src:.` stopped at collection and executed no tests; it is an environment error, not regression evidence.

Machine-readable decisions are in `full_model/verification/v54_mechanism_decision.json`; immutable paths and SHA-256 hashes are in `full_model/verification/v54_campaign_manifest.json`. Full-resolution result records and figures remain under `/Users/sdillon/HPC3/campaigns/asb-drx-v54-20260924`.

## Next discriminating work

For ASB, retain the matched intervention and choose one registered physical change based on the observed broad feedback—prefer the 0.75 micrometre heterogeneity radius at otherwise unchanged conditions to test transport-length control before increasing the rate. For prepared-boundary DRX, the measured rate is exposure-limited rather than invalid; extend physical time with an accuracy-qualified multirate route before changing front kinetics. Independently, connect current-state orientation-plateau/Frank–Bilby recognition to a neutral promotion interface before making any intragranular DRX claim.
