# V46 state-qualified ordering and physical spatial representation decision

## Decision

V46 repairs the two production representations that blocked V45 and recovers
the original 62.5 microsecond common-state benchmark.  The campaign result is
`RESOLVED_REPRESENTATION_PASSED_COMMON_SHORT_WAVE_STRONG_NORM_FAILED`.  This is a numerical and representation
qualification, not experimental validation and not a claim of spontaneous
LAGB formation, DRX, strict ASB, or material calibration.

The scientific production source is
`096159de1478bb259e445c15c11765d9149d7fe0`.  Atomic continuation machinery was
added at `dbed107cbafc470175d1e30b02716576399ac2bc`; it does not change the
production equations.  Later commits add postprocessing and attributable
evidence only.

## State-qualified ordering

For every active sign/family coordinate

`q_s = rho_ordered_s / (rho_tangle_s + rho_ordered_s)`, `0 <= q_s <= 1`,

the retained kinetic law gives the necessary finite-speed condition

`|q_s(t+H)-q_s(t)| <= integral_t^(t+H) a_s(x,tau) d tau`.

Production now evaluates this condition componentwise with the local attempt
exposure before accepting a convex asymptotic endpoint.  An inactive or exact
zero pool has no fraction coordinate and retains its exact semantics.  The
counterexample beginning at `q=0.25` with exposure `0.001` correctly rejects
the unreachable equilibrium `q=0`; finite integration ends at
`q=0.24900000014189905` and certifies the kinetic accuracy of that fixture.
The bound is recorded only as necessary, not as a complete finite-time error
certificate.

The production obstacle solve also now qualifies the state actually published
using its projected physical KKT residual.  Its ADMM penalty is `L/128`, chosen
from the conditioning of the declared quadratic operator.  This replaces the
old `L/2048` value that left active capture families short of the obstacle KKT
point at the iteration cap.  It is not fitted to phase-field outcomes.

The first attempted n128 long step exposed this issue: the former asymptotic
solve was rejected and its exponential fallback spent more than five minutes
in an FFT-backed action without reaching a checkpoint.  Failed states are
retained under the `v46-checkpointed` result tree.  After the KKT repair, every
published n128/n192 benchmark operation uses the qualified bounded-convex
endpoint with zero accessibility violation.  No elapsed physical time or
event scale was discarded.

## Compatible continuum representation

The first causal spatial mismatch was the binary capture support followed by
one-cell captured-line deposition.  V46 replaces that pair, for the compatible
production route, with:

- exact fractional physical cell coverage at the trap boundary;
- the entry measure `max(w_destination-w_source, 0)` from the accepted signed
  transport direction; and
- a positive, normalized, periodic compact-Wendland representation of fixed
  physical length 400 nm for capture into the continuum wall reservoir.

The 400 nm length is a previously missing coarse-graining assumption.  It was
selected from the existing 450 nm trap half-width, not adjusted until an output
passed.  The pre-registered sweep remains visible: the half-width-24 captured
line discrepancy falls from 12.0424% at zero representation length through
11.1918%, 9.3321%, and 5.3791% at 50, 100, and 200 nm, to 2.6904% at 400 nm and
1.4429% at 800 nm.  The selected production curl–Nye rate discrepancy is
0.0264%, and no field in the declared causal audit exceeds 5%.

The map preserves the zero mode and nonnegativity.  If local capacity is
insufficient after mapping, the complete capture event receives one scalar
extent reduction instead of cellwise clipping that would alter its spatial
moment.  Mesh size, the 400 nm phase-interface width, atomistic core radius,
and this capture representation length remain distinct quantities.

## Geometry-to-continuum energy representation

Retained links and swept surfaces remain exact topology.  The continuum uses
one fixed physical, positive, periodic map `K_ell` for scalar line, Burgers
moment, plastic distortion, and Nye from compatible curl.  The representation
length is 400 nm and the modeled out-of-plane section thickness is fixed at
one micrometre for all grids.  The existing continuum line/logarithmic and
gradient functional owns represented self energy once; no duplicate unresolved
core term is added in this minimal architecture.

This separation is consistent with the established use of non-singular line
representations and with deriving continuum density energies by coarse
graining discrete configurations.  It is a declared model-scope completion,
not an output filter.  The discrete map is self-adjoint to `1.21e-15` relative
residual, commutes with the compatible curl to `8.18e-16` relative RMS, and
preserves its zero mode exactly.  Only an ULP-scale positive scalar correction
needed to close moment realizability is added and ledgered.

At n32/n64 the stored total energy differs by 0.0882%, below the 5% criterion.
The nearly cancelling translation increment differs by 59.6% relatively,
although its absolute n32/n64 discrepancy is `3.59e-16 J`, or `5.43e-8` of the
stored-energy scale.  V46 therefore qualifies stored energy but does not claim
relative convergence of that tiny incremental observable.

The repaired transaction performs two consecutive nonzero geometry updates
from evolved states, advancing two microseconds in total.  It also rejects an
energetically unfavorable update with exact rollback and zero consumed time.
The declared equilibrated point-defect reservoir work in this capability
fixture is an uncertain parameter, not a material calibration or an artificial
production pressure.

## Physical-time production continuation

The repaired n128/n192 pair starts from source-identical analytic states on a
3.2 micrometre periodic domain.  One macro contains sixteen accepted Mura
substeps and one front transaction.  The first macro at 7.8125 microseconds
passes the selected curl–Nye comparison: 0.4607% complex error in centered
half-width 24, 0.1903% whole-field RMS-amplitude difference, and 2.6699% strong
norm error on the full n128 common band.  This authorized the original
eight-macro horizon.

At 62.5 microseconds the matched comparison is:

| Curl–Nye observable | Relative n128/n192 difference | 5% decision |
|---|---:|---|
| centered Fourier half-width 24 | 1.3765% | pass |
| whole-field RMS amplitude | 0.7867% | pass |
| strong norm on common half-width 63 | 6.3213% | fail |

Both trajectories contain 128 Mura operations and eight front transactions,
including the source-identical first macro.  All event and family scales are
exactly 1.0, every state-accessibility check passes, and the maximum projected
ordering KKT residual is `9.30e-12`.  The physical clocks
close at exactly 62.5 microseconds.  No long endpoint is published from an
unconverged operation.

The selected physically resolved observable therefore passes through the
recovered horizon, while convergence of every shorter common-grid mode does
not.  The machine controller keeps `scientific_gate_passed=false` for the
broad spatial-completion claim and records the selected resolved-observable
pass separately.  No threshold was weakened and no coordinate shift was
optimized.

## Verification, provenance, and claim boundary

The final merged regression boundary passes 772 tests in 262.15 seconds.
Large result arrays remain local and are addressed by manifest and checkpoint
SHA-256 values.  The V45 seven-file evidence set was verified at startup and
is not overwritten.  Superseded V46 attempts remain retained with their
original source identities.

The 400 nm representation is supported as a fixed, resolved coarse-graining
choice for this arbitrary model, not calibrated to a material.  The repaired
prepared-loop fixture demonstrates representation and transaction capability;
it does not demonstrate spontaneous wall formation.  No resolved orientation
plateau plus independent signed circuit and grain lifecycle has been shown, so
LAGB and DRX remain unclaimed.  Strict ASB and experimental/material validation
also remain outside this V46 result.

Primary records:

- `full_model/verification/v46_ordering_applicability.json`
- `full_model/verification/v46_mura_suboperator_audit.json`
- `full_model/verification/v46_geometry_representation.json`
- `full_model/verification/v46_checkpointed_spatial_accuracy.json`
- `full_model/verification/v46_original_horizon_accuracy.json`
- `full_model/verification/v46_campaign_manifest.json`
- `full_model/verification/v46_campaign_controller.json`

Scientific context for the declared representation follows Cai, Arsenlis,
Weinberger, and Bulatov, *J. Mech. Phys. Solids* 54 (2006),
doi:10.1016/j.jmps.2005.09.005, and Zaiser, *Phys. Rev. B* 92 (2015),
doi:10.1103/PhysRevB.92.174120.  These references motivate consistent
non-singular/coarse-grained representations; they do not prescribe or calibrate
the campaign's 400 nm length.
