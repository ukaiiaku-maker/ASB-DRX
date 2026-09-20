# V45 scalable ordering and matched physical-horizon decision

## Decision

V45 completes the retained n128 endpoint and passes temporal refinement, but
does not pass selected spatial refinement. The current classification is
`CURRENT_SOURCE_TEMPORAL_PASSED_SPATIAL_FAILED`. A longer physical-response
horizon was therefore not launched. This is a numerical dependency failure,
not a physical rejection of wall formation, DRX, or ASB.

The V44 solution-improvement claim is corrected to
`COMPATIBLE_TRANSPORT_IMPLEMENTED_MATCHED_SPATIAL_IMPROVEMENT_UNESTABLISHED`.
The previously contrasted 8.42% and 0.0264% values belong to different macro
stages. At matched first-macro stages, V43 and V44 errors are essentially the
same. Raw V44 evidence and source identities remain preserved.

## Ordering kinetics and scalable dispatch

For each Burgers family and sign, the bounded coordinate is

`q_s = rho_ordered_s / (rho_tangle_s + rho_ordered_s)`, with `0 <= q_s <= 1`,

and the unchanged finite-rate law is

`qdot_s = -a(x) tanh[l_e (mu_ordered_s-mu_tangle_s)/(2 k_B T)]`.

The chemical-potential action contains the full periodic FFT gradient and Nye
terms. No local Jacobian sparsity is asserted. Directional differences and the
adjoint symmetry of the FFT Hessian pass. The transient backend uses a bounded
matrix-free exponential Rosenbrock action. At attempt exposure `1e-6` it
matches dense BDF. Dense finite evolution at exposure `1e-3` and the bounded
convex endpoint at `1.000001e-3` differ by at most `1.55e-12` relative in the
tested signed reservoirs. Production therefore uses the finite backend below
that measured overlap and the convex asymptotic solve at or above it.

The endpoint stationary remainder is reported separately and is not called a
finite-time error. Earlier projected-Newton, trust-region backward-Euler, and
adaptive explicit candidates are retained as quarantined numerical failures,
not silently presented as successful solvers.

## Exact retained-state completion

The retained V44 `after_front.npz` checkpoint was validated by SHA-256 and
advanced through exactly four accepted second-half Mura substeps of
`9.765625e-7 s` each. The missing exposure is `3.90625e-6 s`; wall time was
153.80 s. The front was not repeated and its heat was not deposited again.
Each substep has an atomic checkpoint. The final payload SHA-256 is
`f5216337fbd4ac1a1b47c18ac02b109e312ed0c246aaf41617fd0a9ecddc4953`.

A full analytic-start n128/sub4 regeneration agrees with the resumed state
across 177 arrays to `5.87e-16` maximum relative-to-pair-peak difference. The
only non-bitwise differences are roundoff-level ordered-alignment values.

## Matched numerical accuracy

All values below use the same 3.2 micrometre periodic domain, coefficients,
loading, compatible de-aliased Mura operator, and 7.8125 microsecond macro
horizon. The Fourier comparison uses centered modes `[-24,24]^2`, coefficients
normalized by `n^2`, and is accompanied by the unfiltered whole-field RMS.

| Comparison | Curl Fourier error | Curl RMS error | Decision |
|---|---:|---:|---|
| n128, sub4 vs sub8 | 0.7994% | 1.3737% | pass |
| n192, sub4 vs sub8 | 0.7768% | 3.0340% | pass |
| sub8, n128 vs n192 | 7.9756% | 17.9885% | fail |

The spatial discrepancy is absent initially. After the first Mura half it is
3.7976% in the selected band and 6.0026% in whole-field RMS; the front changes
neither materially. The second Mura half increases it to the final values.
Thus the remaining error is localized to spatial Mura/wall evolution, not the
front transaction or temporal integrator.

The band audit sharpens this diagnosis. At the final stage, modes `[-8,8]^2`
differ by only 0.4575%. Error rises to 2.9338%, 7.9756%, 14.9379%, and 36.4246%
as half-width increases through 16, 24, 32, and 48. The n192 field retains
substantial energy above the n128 common range. Output filtering was not used
for the pass/fail decision.

## Ordered first-generation residual

Changing the n16 implicit residual tolerance from `2e-8` to `2e-10` changes
the first generated ordered line by exactly zero at reported precision, and
the complete reaction clock closes. Nevertheless, the selected sub8 ordered
inventory is `1.24498e-5 m` at n128 and `3.66160e-6 m` at n192; its cross-grid
field error remains about 75.2% in the selected Fourier measure and 64.0% in
RMS amplitude. It remains a tolerance-insensitive, spatially unresolved
near-extinction residual, not a physical wall signal.

## Geometry measure and clock

A fixed 10 nm physical boundary displacement over a fixed 0.8 micrometre
boundary gives `8e-15 m^2` swept area on n16/n32/n64. Weighted line length and
explicit geometric line energy are grid invariant. Reversing site enumeration
changes only roundoff; splitting every patch extent produces the same physical
arrays. A rejected geometry proposal commits zero time, area, and heat; a
subsequent compatible continuum step advances `1e-12 s`, and a second geometry
event is then evaluated from the evolved state.

The signed exchange convention is explicit: positive `trace(dbeta_p)` denotes
positive represented plastic-volume exchange with a spatially equilibrated
point-defect reservoir. Zero reservoir work does not mean zero exchanged
matter.

The coupled continuum energy does not converge when a one-cell lattice line
is deposited into the continuum wall-gradient energy. Its geometric line
energy converges, but the singular continuum-gradient contribution does not.
Geometry kinetics is therefore not promoted until geometry-owned line energy
is separated from, or regularized consistently into, the continuum field.

## Scope and next action

Numerical identities, restart completion, temporal accuracy, and the explicit
geometry measure pass within their stated scopes. Selected spatial solution
accuracy and coupled geometry-energy convergence fail. DRX, a persistent
LAGB, strict ASB, and material calibration are not claimed.

The final regression boundary passes 764 tests in 193.91 s.

The next exact task is to resolve or physically regularize high-wave-number
Mura wall content and separate geometry-owned line energy from the singular
continuum gradient term. Only after those two spatial dependencies pass should
the original longer physical-response horizon resume.

Primary records:

- `full_model/verification/v45_ordering_qualification.json`
- `full_model/verification/v45_resumed_n128_endpoint.json`
- `full_model/verification/v45_refined_current_accuracy.json`
- `full_model/verification/v45_spatial_spectrum.json`
- `full_model/verification/v45_ordered_residual_sensitivity.json`
- `full_model/verification/v45_dynamic_geometry_measure.json`
- `full_model/verification/v45_campaign_controller.json`
