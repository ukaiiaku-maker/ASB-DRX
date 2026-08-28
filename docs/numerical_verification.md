# Numerical verification

## Net-flow and recovery integration

The active continuum flow uses the odd forward-minus-unloaded-reverse EXP-floor
rate. Mechanics is advanced by a matrix-free backward-Euler Newton--GMRES solve
with exact antiplane projection in each Jacobian-vector product. The two points
that were unresolved by the signed explicit one-way law (850 and 950 K,
45000 s^-1, density ratio 2) both reached 0.9 applied shear in 300 steps with no
halving, at most three Newton iterations, and final nonlinear residuals below
`2e-14`.

Dynamic recovery is integrated by the exact exponential solution at the old
operator-split temperature. Its decrease in stored line energy is a separately
recorded heat source. An unloaded isothermal test checks exponential density
decay, exact stored-energy release, bath heat, and global ledger closure. The
finite-wavenumber operator including recovery independently matches a centered
finite-difference Jacobian.

## Timestep and grid refinement

The first nonlinear refinement uses the provisional analytical-boundary
condition 950 K, 4500 s^-1, density ratio 2, and target shear 0.3. It is a
deterministic generic test, not a material validation. Diagnostic retention is
fixed at 75 equal strain intervals for every member, independently of the
internal timestep.

Timestep refinement uses 75, 150, and 300 imposed steps on a 16 by 16 grid.
Grid refinement uses 16, 24, and 32 points with 300 imposed steps. The final
150-to-300 timestep comparison has maximum relative change 0.873%, controlled
by net density change. The final 24-to-32 grid comparison has maximum relative
change 0.00124%, controlled by child-order area fraction. Final/peak stress,
maximum temperature, matched temperature excess, density change, and child
order fraction are all included in the gate. Both comparisons are below the
provisional 5% threshold and preserve the nonlocalized classification. No
member halved its timestep; the implicit flow solve required at most four
Newton iterations.

The machine-readable result is
[`output/local_refinement.json`](../output/local_refinement.json). The run took
62 seconds locally, below the threshold for an extended HPC3 calculation.

## Remaining convergence gate

This test does not establish convergence of localization onset or band width,
because no member localized. Those observables must be refined separately if a
continuous collective coupling creates a candidate band. Likewise, child-order
fraction is not yet a physical DRX fraction; the embryo/orientation gate is a
separate workstream.
