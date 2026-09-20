# V44 compatible transport and geometric-rate decision

## Decision

V44 is a qualified numerical and representation advance, not a completed
physical-horizon qualification.  The compatible nonlinear product repairs the
specific first-Mura spatial discrepancy isolated by V43.  Full temporal
qualification remains failed/incomplete: the completed n192 endpoint differs
by 15.93% in curl-Nye RMS amplitude between one and four substeps, and the n128
four-substep second half did not complete within a declared 90-minute local
bound because the ordering subsystem requested a dense numerical BDF
Jacobian.  No DRX, persistent LAGB, strict ASB, or material calibration is
claimed.

## Compatible production transaction

The V44 production option is
`mura_transport_operator="compatible_dealiased"`.  It forms each nonlinear
product with 3/2 Fourier padding before multiplication.  The same accepted
swept fields

\[
  s^\pm_h = \mathcal D_{3/2}(v^\pm\times\kappa^\pm)
\]

generate the moment update, family plastic flow, and Nye update:

\[
 \dot\kappa^\pm=-\operatorname{Curl}_h s^\pm_h,\qquad
 J^p_a=b_a\otimes(s^+_{h,a}-s^-_{h,a}),\qquad
 \dot\alpha_a=-\operatorname{Curl}_hJ^p_a.
\]

The scalar population uses the conservative flux associated with the declared
`alpha=-Curl(beta_p)` convention.  Line stretching needed to maintain
`|kappa| <= rho` is separately ledgered.  Capture only repartitions the common
accepted line/moment event and creates no second total-Nye source.  Negative
spectral trials fail and return to the existing extent controller; no physical
density floor, output smoothing, clipping of a finite negative state, or
post-step Nye projection is used.

Atomic tests verify periodic scalar balance below `1e-12 m`, alignment balance
below `1e-12 m`, nonnegativity, and exact beta/Nye closure.  The legacy mixed
operator remains selectable solely for frozen comparisons.

## Spatial and temporal evidence

All resolved calculations start from the analytic initial state and retain the
V43 physical coefficients.

| comparison | first Mura curl complex | first Mura RMS amplitude | final curl complex | final RMS amplitude |
|---|---:|---:|---:|---:|
| n128/n192, one step per half | 0.0264% | 0.00043% | 4.304% | 7.399% |
| n128/n192, four substeps | 3.260% | 4.591% | incomplete | incomplete |

The first-stage error is therefore repaired below the provisional 5% threshold
for both time choices.  The one-step final complex coefficients also fall below
5%, but their RMS amplitude does not.  Same-grid one/four-substep comparisons
at the first half are 4.94% complex and 2.21% RMS for n128, and 4.83% complex
and 6.70% RMS for n192.  At the completed n192 final endpoint they are 4.69%
complex and 15.93% RMS.  Time accuracy is not settled.

The n128 refined trajectory published its analytic initial, first-Mura, and
front stages, then spent 90 minutes in the second-half ordering solve.  The
interruption traceback localizes the cost to SciPy BDF's dense numerical
Jacobian for a globally coupled spectral-gradient ordering residual.  This is
not attributed to the compatible Mura flux and is not treated as a converged
endpoint.  The next numerical task is a matrix-free or Fourier-diagonal
finite-time ordering integrator with the same energy and bounds, followed by
completion of this exact missing endpoint.

## Ordered near-extinction reservoir

The ordered line is created in the first Mura operation and then remains
nearly unchanged:

| grid | one-step first-half line | four-substep first-half line |
|---|---:|---:|
| n128 | `1.2448790947e-5 m` | `1.2448790947e-5 m` |
| n192 | `3.6595123126e-6 m` | `3.6595123126e-6 m` |

The exact invariance under time subdivision and approximately cubic grid
scaling persist.  A mechanism replay records active transfer cells, chemical
potential range, negative free-energy rate, zero discarded time, dispatch,
tolerances, and solver evaluations.  This reservoir is classified as a
separate first-operation near-extinction residual and is not a physical wall
signal.

## Independent geometry/continuum connection

The stored swept plaquettes and stored Burgers-weighted links are assembled
independently.  Their mimetic boundary/deposition reconstructions agree
exactly.  An explicit staggered-link transfer maps the edge incidence symbol
`(1-exp(-ikh))/h` to the production spectral symbol `ik`; it agrees with the
spectral curl of retained swept distortion to `2.7e-16`, `4.3e-16`, and
`5.3e-16` relative RMS on n16/n32/n64.  Neither side is overwritten from the
other.

This audit also corrected the physical sign: because
`alpha=-Curl(beta_p)`, dislocation-line orientation is the negative oriented
boundary of a positive swept plastic surface.

## Geometric kinetics and physical time

Plaquette extent is declared as a fractional ensemble weight at a represented
site; each ledger records cell area, swept area, requested time, accepted time,
rate exposure, active extent cap, and remaining time.  The low-barrier
four-direction response accepts five events, including same-state dyadic
partial extents after full-proposal rejection, and advances only
`4.01164e-12 s` of a requested `1e-9 s`.  No represented direction then
advances.  It is classified
`CONSTRAINED_ARREST_NO_REPRESENTED_PATH`, not horizon completion or general
physical pinning.

Non-volume-preserving sweeps are explicitly classified as climb with material
exchange.  V44 assumes a spatially equilibrated point-defect reservoir with
zero chemical reservoir work.  Its validity limit is explicit: vacancy
diffusion transients are not represented.

## Evidence and next action

- `full_model/verification/v44_compatible_transport_decision.json`
- `full_model/verification/v44_geometry_rate_audit.json`
- `full_model/verification/v44_ordered_first_generation_n16.json`
- `full_model/verification/v44_campaign_controller.json`

Next: replace the dense-Jacobian finite-time ordering BDF path with a
matrix-free/FFT-compatible bounded integrator, complete the retained n128
second half, and repeat the final same-grid temporal comparison.  Broader
organization, DRX, or ASB campaigns are not justified before that closure.

## V45 interpretation addendum

The V44 compatible transport operator remains an internally conservative,
de-aliased candidate, but the earlier spatial-improvement wording compared
different trajectory stages. The V43 value `0.0841592749725368` belongs to
`interval2_one_step.after_first_mura`; the V44 value
`0.0002638702041757045` belongs to the analytic-start first macro.

At matched named first-macro stages, V43/V44 cross-grid complex curl errors
are `0.0002636062097406558`/`0.0002638702041757045` after the first Mura half
and `0.043001734780054555`/`0.04304400535716677` at the final stage. Therefore the
current claim is
`COMPATIBLE_TRANSPORT_IMPLEMENTED_MATCHED_SPATIAL_IMPROVEMENT_UNESTABLISHED`.
The raw V44 numbers and their original source identities remain unchanged;
this addendum retracts only the unmatched solution-improvement inference.
