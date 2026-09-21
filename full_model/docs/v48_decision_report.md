# V48 decision report

## Decision

V48 repairs two production defects and qualifies a third mechanism at the
implementation level:

- the projected-Heun finite-time route is no longer a production accuracy
  reference;
- transport, locking, ordering, owner publication, and front diagnostics now
  share one signed-reservoir defect functional;
- represented geometry uses a finite EXP-floor barrier multiplied by a
  complete-event, event-normalized downhill activity.

The campaign currently supports `IMPLEMENTATION_VERIFIED` for all three,
`CONTROLLED_PHYSICAL_RESPONSE` for the short n16 loading/hold continuation,
and no DRX, LAGB, strict-ASB, or material-calibration claim.  The fixed-scale
matched-strip geometry increment remains scientifically unresolved because its
n32/n64/n128 sign changes.

## Finite-rate ordering

The actual production residual supplies a direct false-stationarity example.
On the compact nonuniform spectral fixture, projected Heun returns the initial
ordered field exactly at attempt exposure 0.01 even though both stages are
nonzero and cancel; the adaptive Rosenbrock endpoint agrees with the independent
dense BDF endpoint.  Production uses the adaptive route and disables the
unproved Euclidean-distance asymptotic handoff.

Local error acceptance does not certify its own global endpoint.  The ledger
therefore records local control separately and leaves the per-solve global
certificate false.  The retained n128 comparison is run from a detached source
and compares default and four-times-tighter tolerances on the exact same
pre-ordering density hash.  At the first nonzero n128 state the endpoints agree
to `2.23e-16` relative L2, while the repaired solution changes `29.0%` from the
initial ordered field and projected RK2 changes exactly zero.  This is an
independent endpoint qualification, not a per-solve self-certificate; its
quantitative record is `v48_ordering_first_nonzero.json`.

The same frozen-source comparison also qualifies the post-front and late-macro
states.  Their maximum default-versus-tightened differences are respectively
`2.15e-16` and `2.62e-16` relative L2, while their finite changes from the
initial ordered fields are `2.47e-2` and `2.84e-4`.  Accuracy is therefore
qualified for the tested observable at all three retained states.  Efficiency
is not: the late default and tightened solves required `7594.6 s` and
`6417.4 s`, so this backend is not yet suitable for broad resolved-grid sweeps.

## Common energy and load clock

The V47 endpoint discrepancy was primarily a diagnostic-functional mismatch.
Repricing the retained V46 endpoints changes the reported defect increment from
`+4.4286875458e-14 J` to `-4.9499624790e-18 J`.  This does not retroactively
qualify the historical cumulative first law because its complete external-work
history was not retained.

Tracing a current-source transaction exposed an additional real ownership gap:
the downhill mobile-to-forest locking release was neither stored nor converted
to heat.  It is now priced with the common extensive functional, rejected
atomically if uphill without a source, and deposited as heat if downhill.  A
compact fixed-strain transaction closes total internal energy to
`4.42e-28 J`.  The endpoint-loading test closes the initial-to-midpoint load,
fixed-midpoint constitutive update, and midpoint-to-endpoint load work path to
better than `2e-11` relative.

The eight-interval n16 response reaches `3.90625e-6 s`, an additional tensor
shear `e12=3.90625e-4` (engineering shear `7.8125e-4`).  At the endpoint,
continued loading minus hold gives:

- mean shear stress: `+6.9264e7 Pa`;
- engineering plastic shear: `+3.8771e-6`;
- mean temperature: `+2.1631e-3 K`.

Its maximum interval first-law residual is `9.06e-15` relative.  This is a
controlled short-horizon capability result at n16, not a spatially refined
physical prediction.

## Geometry force and kinetics

The n64 same-path event is quadratic in fractional extent to
`8.01e-14` relative fit residual.  Derivatives of that fit and centered
same-state finite differences agree within `6.6e-11` relative.  This confirms
that the former 5.14% secant spread was ordinary path curvature rather than a
derivative failure.  Integer-cell translation remains exact; the declared
fractional-occupancy translation path changes energy by at most
`2.85e-7` of stored energy.

The geometry activity uses complete energy per physical event, not complete
mesh-patch energy in a Boltzmann exponent.  Nonzero-barrier controls span
blocked, moderate, and saturated affinities at 800/1100/1400 K.  Four accepted
events from successively evolved states advance `4.4451e-10 s`.  The intrinsic
zero-chemical-work control is blocked without mutation.  A separately labeled
vacancy-reservoir hypothesis advances; it was selected to discriminate the law,
not fitted to a phase-field outcome.  Its actual reverse edge reverses signed
exchange and affinity exactly and is rejected without mutation.

The event rate and finite extent are closed as a scalar fixed point because the
complete endpoint affinity depends on the extent actually advanced.  The
ledger retains the independently recomputed endpoint rate, iteration count,
and relative residual rather than silently reporting the initial probe rate.

The fixed 10 nm matched-strip increments remain
`-6.0308e-16`, `-2.4373e-16`, and `+1.2489e-16 J` at n32/n64/n128.
Consequently geometry rate implementation is verified but that physical force
observable is not numerically qualified.

## Regression and claim boundary

The final scoped regression contains 789 passing tests.  No test threshold was
relaxed.  The V39 historical asymptotic fixture explicitly opts into its legacy
heuristic; current production does not inherit that switch.

The proper V48 classification is:

`FINITE_RATE_AND_COMMON_ENERGY_REPAIRED_GEOMETRY_RATE_VERIFIED_MATCHED_FORCE_UNRESOLVED`

No spontaneous nucleation, persistent new LAGB, DRX, strict ASB, or material
calibration is claimed.
