# V47 force accuracy, fixed-scale refinement, and physical-response decision

## Decision

V47 is a mixed but decision-grade advance.  It confirms that the V46
short-wave discrepancy is real, supplies a finite n256 fixed-scale result,
repairs the finite-time ordering dispatch, and converts geometry chemical work
from arbitrary proposal credit to a physical signed exchange.  It also shows
that the geometry affinity/rate and ordered-density refinement remain
unqualified.  The aggregate classification is
`KINEMATICS_REFINED_ORDERING_REPAIRED_GEOMETRY_RATE_UNQUALIFIED`.

This is neither a material calibration nor a claim of spontaneous LAGB
formation, DRX, or strict ASB.  All new calculations ran locally and the
unrelated DDD campaign was observed but not modified.

## Frozen V46 evidence and raw spectral union

The retained n128/n192 final checkpoint hashes and continuation-manifest hashes
match the V46 manifest.  Direct FFTs of the raw compatible curl--Nye arrays
reproduce the reported physical RMS values, 25.58629031375322 and
25.78918069860933 m^-1.  Explicit zero extension on the union of the discrete
spectral spaces gives:

- fine-only tail squared-norm fraction: 2.138819%;
- fine-only tail RMS fraction: 14.624702%;
- union difference relative to fine full RMS: 15.912703%; and
- Pythagorean identity residual: 3.16e-16.

The coarse Nyquist contribution is reported separately and is negligible.
This is a difference between two discrete reconstructions, not an error against
an exact continuum solution and not a thermodynamic energy fraction.

## Fixed 400 nm representation

One finite-time Mura interval, 0.48828125 microseconds, was run from the same
declared analytic initial state at n128, n192, and n256.  The physical domain
remained 3.2 micrometres and the representation length remained 400 nm.  All
three intervals closed exactly with unit event scale and the finite RK2
ordering route.

For n192 versus n256, curl--Nye differs by 0.00893% on half-width 24,
0.04493% on half-width 63, and 0.12279% on the larger half-width-95 band.
Plastic distortion and temperature-rise differences are below 0.005% on these
bands.  Ordered density does not qualify: it differs by 8.80% on half-width 63
and 9.95% on half-width 95.  The result therefore qualifies short-horizon
curl/kinematic observables only.  A 62.5 microsecond n256 trajectory was not
run: profiling plus finite-time ordering made it inappropriate alongside the
active local DDD calculation.

The retained long-horizon n128/n192 result remains unchanged: half-width-24
curl--Nye passes at 1.3765%, while the common half-width-63 strong norm fails at
6.3213%.  The new short result does not erase that later-time failure.

A source-frozen late-state restart fork advances the same 3.90625 microsecond
Mura exposure with eight and sixteen accepted substeps.  The half-width-63
relative differences are 0.00207% for curl--Nye, 0.000501% for plastic
distortion, 0.00340% for temperature rise, 0.0000368% for ordered density, and
1.06e-6% for total density.  Both clocks close exactly.  This qualifies the
late local temporal resolution; it does not qualify the earlier ordered-density
spatial error.

## Geometry affinity and chemical reservoir

The V46 geometry artifact actually used a fixed 3.2 micrometre represented
section thickness.  The V46 report/manifest statement of 1 micrometre was an
attribution error.  There is no production-source difference between geometry
artifact source `9ecd31da6a714e94c03b37346ba46b8a6802fa59` and V46 scientific
source `096159de1478bb259e445c15c11765d9149d7fe0`; the retained artifact is
relabeled, not rescaled.  The common-state benchmark independently uses a
two-Burgers-vector represented thickness, 4.96e-10 m, and is unaffected.

Production now evaluates the ordered-gradient event increment with the exact
quadratic polarization identity rather than subtracting two large endpoint
energies.  Endpoint subtraction and the direct increment are both retained so
arithmetic cancellation is visible.  A complete-loop rigid translation has
exactly zero total-energy change and its ordered density agrees with an exact
one-cell roll to 4.15e-16 relative L2, establishing the zero-self-force control.

The nonzero local force is not yet qualified.  At n64 it varies by 5.1398%
over fractional extents 0.025--0.2, just outside the fixed 5% bound.  More
importantly, the active translation increment changes from -2.4373e-16 J at
n64 to +1.24895e-16 J at n128.  The dominant ordered-gradient contribution
also changes sign.  Agreement of the large stored-energy background is not
used to promote this force.

For new physical-reservoir cases,

`Delta N = trace(Delta beta_p) dx^2 h s / Omega`,
`W_chem = mu Delta N`,

with named vacancy species, signed stoichiometry `s`, atomic volume `Omega`,
and work-on-system sign.  Legacy per-extent credit is mutually exclusive with
this path and remains available only to replay frozen fixtures.  The tested
negative, zero, and positive chemical potentials change complete affinity with
the expected sign.  They all retain the same 2e5 s^-1 proposal rate, exposing
the remaining defect: the present EXP-floor rate is stress-driven and complete
affinity acts only as an atomic accept/reject guard.  Geometry rate is therefore
explicitly unqualified.

## Finite-time ordering applicability

The exact scalar tanh counterexample passes accessibility and has zero
equilibrium KKT residual but reaches 0.3473761134414577 rather than 0.5.  Actual
V46 states show why a state-specific guard is necessary:

| state | finite versus retained asymptotic ordered-density L2 |
|---|---:|
| first nonzero Mura | 29.0447% |
| post-front | 2.4677% |
| late macro | 0.02837% |

Production now requires the initial-to-equilibrium distance to be no more than
5% before using the convex asymptotic endpoint.  For the declared convex
projected gradient flow, distance to equilibrium is non-increasing, making the
initial distance a conservative finite-horizon bound.  The first nonzero state
has an 82.096% bound and automatically dispatches to finite RK2.  The late
state has a 1.4028% bound and retains the qualified asymptotic route.  The
accessibility, projected KKT, endpoint-distance, and finite-integration fields
remain separate in the ledger.

For very long exposures that begin outside the bound, projected RK2 advances
the physical rate until the evolving state first enters the same 5% contraction
ball around a KKT-qualified endpoint.  Only then may that endpoint represent
the remaining clock.  Integrated substeps, requested substeps, switch distance,
and certified remainder time are separate ledger fields; no reaction time is
discarded.  The retained V39 restart tests explicitly keep their historical
equilibrium dispatcher so they remain checkpoint-semantic fixtures rather than
silently becoming new physical-response evidence.

## Physical interpretation

Postprocessing the completed V46 n128 trajectory gives 0.000108324 mean
plastic shear, 0.30630 K mean heating, and 0.72339 K peak-minus-mean
temperature at 62.5 microseconds.  The eight accepted front transactions move
either contour by at most 2.1352e-11 m, only 5.338e-5 of the 400 nm interface
width.  They process 6.4886e-11 m of line with 1.41e-23 m maximum closure
residual.  This is valid bulk recovery/heating with tiny existing-boundary
motion, not appreciable migration or DRX.

A bounded current-source pair starts from that same final state and advances
one finite-ordering interval.  Continued loading at 100 s^-1 adds 2.4414e-5
shear relative to hold.  Its final effective resolved-stress RMS is 1.2536 MPa
higher, its mean temperature is 3.204e-5 K higher, and its plastic shear is
4.754e-15 higher.  The front channel is deliberately inactive in this compact
bulk pair; no existing-boundary migration, DRX, or ASB claim is made.

## Verification and claim boundary

The local 8-versus-16 late-state temporal result passes the 5% threshold.  The
final merged regression passes 778 tests in 301.28 seconds and is recorded in
the V47 controller and manifest.  Large checkpoint arrays stay under
`full_model/production/results-local`; tracked evidence records their absolute
paths and SHA-256 values.

Qualified scopes are raw-union accounting, short-horizon curl/kinematic
refinement, rigid-translation symmetry, physical chemical-work normalization,
state-qualified ordering dispatch, V46 physical postprocessing, and the bounded
bulk response pair.  Unqualified scopes are the active geometry force/rate,
ordered-density spatial refinement, the full-horizon n256 field, spontaneous
wall/LAGB formation, DRX, strict ASB, and material calibration.
