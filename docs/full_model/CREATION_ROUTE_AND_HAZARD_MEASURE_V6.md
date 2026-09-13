# Full-v34 creation route and physical hazard measure

## Selected creation route: precursor creation (Route B)

The patched stateful path uses one stochastic microscopic creation process and
one subsequent collective-coordinate growth process. They do not pay the same
barrier twice.

The local event rate is

`lambda = nu_eff exp[-(H_EXP-floor(p) - T DeltaS*)/(kB T)]`,

where the favorable bulk driving pressure `p` has units `J m^-3 = Pa`, the
EXP-floor enthalpy has units J/event, and `nu_eff` has units events/site/s.
Constant activation entropy enters this expression once. A successful event
creates a precursor at the declared minimum phase-field-resolved radius.

The classical circular free energy

`DeltaF(R) = 2 pi R gamma h - pi R^2 DeltaPsi h + compatibility/orientation`

is not included in the stochastic exponential on this route. It establishes
thermodynamic feasibility, the classical critical-radius diagnostic, and the
sign of subsequent continuous growth/shrinkage. The candidate cutoff is applied
to the EXP-floor kinetic free barrier, not again to the classical barrier.

Thus `H0`, `DeltaS*`, and fixed attempt frequency describe precursor creation;
interface energy and stored-energy relief describe post-creation evolution.
They are not independently tuned to pay two versions of one formation barrier.

The immutable-v34 regression retains its legacy classical barrier-crossing
hazard when `use_expf_embryo_creation=false`.

## Area-integrated hazard

The production stateful path interprets the kinetic rate as `s^-1` per physical
site. Multiplication by a declared site density `n_site [m^-2]` gives hazard
density `lambda_A [m^-2 s^-1]`. A grid cell contributes

`dLambda = lambda_A dx dy dt`.

The domain sum is a dimensionless expected event count. It converges under mesh
refinement and is invariant to translating a support field across the Eulerian
grid. One persistent domain Poisson clock carries residual exposure. Its
threshold is redrawn only after a realized event. Completed, deferred,
discarded, transferred, and newly initialized exposure are ledgered separately.
