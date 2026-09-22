# V50 decision report

## Decision

V50 repairs the conditional shortened-interval loading defect and qualifies a
real persistent subcell rectangle through the production common-state map. The
single-face geometry, force, physical event clock, matter exchange, heat,
restart, rollback, second evolved event, and noninteger rigid translation pass.
The retained V49 n128 loading calculation remains valid and running under its
immutable V49 source. V50 does not yet complete the selected bulk horizon, its
hold comparator, simultaneous same-state multi-segment clocks, or the detailed
evolved-ordering profile. No DRX, LAGB, strict-ASB, or material-calibration
claim is made.

The campaign classification is
`PRODUCTION_SUBCELL_SINGLE_FACE_QUALIFIED_PHYSICAL_HORIZON_RUNNING`.

## Loading and restart repair

The old conditional branch could evolve a candidate at the requested midpoint,
shorten the accepted duration, and merely reprice that candidate at the new
midpoint. V50 instead treats every shortened candidate as an unpublished
numerical probe. It returns to the identical pretrial state, rebuilds the load
at the shortened interval's midpoint, and re-evolves. Selection is bounded to
16 retries. Publication additionally requires an independent incremental
first-law check with absolute tolerance `1e-26 J` and relative tolerance
`1e-9`; a ledger residual is never deposited as compensating heat.

Driver tests cover two consecutive shortenings, equivalence to an explicit
small step, exact restart across the seam, and the analytical old-branch
counterexample of `+44.55 J/m3`. Exact restarts inherit their original load
origin. A declared protocol transition instead derives the new origin and
initial tensor shear from the recorded evolved endpoint, retains cumulative
work and the initial-energy baseline, and records its parent checkpoint. The
driver refuses an implicit load jump or an unsupported non-shear endpoint.

## Production physical geometry

The new state owns the rectangle's physical lower/upper coordinates, Burgers
family and sign, physical Burgers vector, field grid, period, section thickness,
fixed 400 nm representation length, quadrature spacing, and accepted event
count. A normalized positive Wendland kernel maps all four actual segments to
scalar line density and oriented moment. The enclosed physical surface maps
plastic distortion, and the authoritative family Nye is the curl of that same
plastic distortion. Coordinates persist in checkpoints.

For a 10 nm extension of an initially 0.8 by 0.8 micrometre rectangle, the
independent continuous reference is `+3.5116371418e-16 J`, or an ordered-gradient
force of `-3.5116371418e-8 N`. Production results are:

| grid | gradient increment (J) | relative error | force (N) |
|---:|---:|---:|---:|
| 32 | 3.6094979442e-16 | 2.7868% | -3.6094979442e-8 |
| 64 | 3.5233603511e-16 | 0.33384% | -3.5233603511e-8 |
| 128 | 3.5125063977e-16 | 0.024754% | -3.5125063977e-8 |

Thus the production sign is positive at every tested grid and the n128
component is well inside the provisional 5% comparison. Offset and independent
line-quadrature variations are retained in the JSON evidence. The same-path
shape derivative gives line-length derivative `1.99996176` and near-zero
integrated oriented derivative (`2.68e-13` in its absolute discrete units).

The persistent production map also translates the loop by `(0.37, -0.23)`
cells and reverses it without changing its declared line length or net swept
area. The reverse beta residual is `1.32e-23`; the absolute density residual is
`2.44e-4` against density values of order `1e12 m-2`. Its small finite sampled
translation-energy variation (`1.19e-17 J`) is reported, not subtracted by a
fitted mesh-position potential.

## Physical event and clock

For declared climb exchange, the event independently compares

`Delta N = z b_n L Delta x / Omega`

with the trace of the swept plastic-distortion volume. The first accepted
10 nm contraction gives `-63636.8296706771` versus
`-63636.8296706778` defects, with residual `6.84e-10`, and exact heat-plus-
complete-energy closure. The intrinsic positive extension is atomically
blocked with zero consumed time and zero heat. A contraction is accepted, an
exact restart is performed, and a second evolved contraction is accepted.

The fixed-rate control uses `r_site=2.5e8 s-1` and
`ell_event=2.48e-10 m`, giving exactly `0.062 m/s`; the clock is independent of
field spacing. Splitting the same 10 nm contraction into 1, 2, 4, and 8 events
preserves endpoint and summed complete energy to roundoff. The last two elapsed
times differ by `2.98e-5` relative. Separate tests exercise nonzero EXP-floor
barriers at 0.25 and 1 GPa and the high-stress floor limit, with the production
ledger rate equal to an independent Arrhenius evaluation. Activation entropy
continues to enter once through the existing production rate function.

This qualifies one active physical face. It does not yet qualify two
simultaneous independent segments in one common state; serially calling the
single-face transaction is not promoted as evidence for that requirement.

## Retained physical trajectory

At the final V50 snapshot, PID 82602 matches the V49 executable, working
directory, output root, start identity, and immutable source
`37065c25e02507da1bc25491574c3ef6fbb01966`. Its newest published prefix is
18/52 intervals. Every segment consumed its full requested duration, so the
conditional V49 shortening defect has not been exposed in this prefix. The
latest checkpoint checksum matches its manifest.

The valid prefix reaches `8.7890625 us`, `0.0017578125` additional engineering
shear, mean shear stress `1.93365 GPa` at interval 18, and remains a
loading-only, front-disabled result.
The last-four-interval median is 448.94 s, projecting about 4.24 h for the
remaining 34 intervals. The machine-readable progress record contains every
endpoint's stress, total/plastic shear, mean/peak temperature, Krylov count,
wall time, and cumulative first-law residual.

No duplicate loading solve was launched. A component-resolved evolved-ordering
profile was deferred because the retained n128 solve is the permitted heavy
local calculation. Its per-interval cost and iteration histories are already
being retained; detailed copied-state profiling follows completion.

## Verification and remaining decisions

The final source passes 15/15 focused V50 tests in 84.44 s and the canonical
suite passes 811 tests in 568.49 s using
`PYTHONPATH=src:. python -m pytest -q tests`. A preliminary bare invocation
failed collection because `src/asb_drx` was not on the interpreter path; this
environment failure is explicitly recorded and was not counted as a model
failure.

The next automatic sequence is:

1. Continue read-only audits of each V49 macro and preserve the last valid
   full-duration prefix if shortening ever appears.
2. Postprocess the completed 52-interval loading horizon.
3. Continue the valid one-interval hold checkpoint, which shares the loading
   initial state, to the same 52-interval horizon under the frozen V50 source.
   The no-load-jump transition path remains available for a separate recovery
   exposure; it is not substituted for this matched causal control.
4. Profile one copied evolved ordering interval, then choose a numerical
   optimization only if it preserves overlap.
5. Add a common-state multi-segment owner/transaction and the finite-proposal
   downhill-to-uphill subdivision test before making broader geometry-clock
   claims.
6. Perform the smallest matched timestep/companion-grid comparisons needed for
   the bulk observables actually interpreted.

The full physical horizon and causal loading/hold comparison remain unfinished;
accordingly `scientific_gate_passed` is false despite the positive production
geometry result.
