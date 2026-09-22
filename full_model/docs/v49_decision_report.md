# V49 decision report

## Decision

V49 passes the local ordering-workspace, active-set Jacobian, batch geometry
initialization, rigid subcell-map, physical event-clock, and restart/load-origin
software gates. It does not pass the scientific partial-segment geometry-force
gate: the actual 10 nm strip-extension increment remains negative at n32/n64
and positive at n128. No DRX, persistent LAGB, strict-ASB, or material
calibration claim is made.

## Ordering

The adaptive full/half-step Jacobian now derives its projection mask from the
trial duration being differentiated. A duration-crossing regression compares
the resulting full and half-step actions with directional finite differences.
The fixed Nye basis, stress-dependent attempt field, and thermal factor are
cached once per ordering solve. A diagonal approximation to the shifted
spectral-gradient block is used only as a GMRES preconditioner; the Krylov
residual retains the exact FFT operator.

On the compact nonuniform stiff transient, preconditioning reduces GMRES work
from 290 to 154 iterations (46.9%). Wall time changes from 3.88 to 3.55 s in
the frozen evidence run, while the endpoints differ by 1.83e-15 relative L2
and both consume the same 4.713455835e-10 s physical clock. This qualifies the
operator reuse and preconditioner locally; it is not evidence of a comparable
speedup on every n128 evolved state.

The three immutable V48 n128 records remain external raw calculation evidence.
Their SHA-256 values are retained in `v49_qualification.json`. The directive's
referenced review JSON and independent script were not attached, so that
provenance gap is explicit rather than reconstructed.

## Geometry representation

Fractional plaquette occupancy is now explicitly an ensemble event weight,
not a rigid subcell coordinate. A separate band-limited periodic continuum map
implements rigid displacement and its analytic shape derivative. Integer-cell
translation, integral preservation, L2 invariance, and directional derivative
tests pass. Because Fourier interpolation is not intrinsically positivity
preserving, this operator is qualified as a symmetry/shape-derivative oracle;
it is not silently substituted for the physical scalar-line owner.

The closed swept-surface fixture is initialized in one algebraic transaction.
It retains the exact boundary, Burgers-weighted surface, density, moment,
plastic distortion, and compatible Nye while recording zero accepted physical
events. The batch and historical sequential initializers agree on the tested
state. This reduces an n128 fixture-plus-strip calculation to about 12 s.

Actual current-source 10 nm strip-extension energies are
`-6.0307941e-16`, `-2.4373144e-16`, and `+1.2489498e-16 J` at n32/n64/n128.
The sign change is not hidden by the new rigid-translation oracle. A future
production event needs subcell coordinates for the moved segment itself before
that force can be promoted.

## Physical clock and continuation

The geometry constitutive rate is a per-site frequency. Its declared clock is

`physical velocity = physical event jump * event frequency`,

with numerical extent rate equal to physical velocity divided by grid spacing.
The default jump is the Burgers magnitude unless a separate positive physical
jump is declared. The transaction ledger independently reports jump,
displacement, numerical extent, swept area, physical event count,
microscopic jumps per active site, active-site measure, and elapsed time.
Synthetic spacing refinement preserves velocity and doubles numerical extent
when spacing halves, as required.

The V49 continuation driver persists load origin, initial tensor shear,
protocol, rate, cumulative external work, and the initial internal-energy
baseline. Restart arguments are checked against these fields. A shortened
accepted constitutive step is advanced as a physical subinterval with new
midpoint/end loads rather than treated as a fatal clock mismatch. Continuous
and restarted smoke paths pass. A resolved long-horizon result is a separate
pending calculation, not inferred from the smoke test.

## Regression and classification

The canonical suite passes 796 tests in 236.44 s. The present classification
is:

`ORDERING_CLOCK_AND_SUBCELL_ORACLE_QUALIFIED_PARTIAL_SEGMENT_FORCE_UNRESOLVED`

