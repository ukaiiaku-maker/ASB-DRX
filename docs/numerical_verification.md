# Numerical verification

## Gate B0 signed-polarization fixture

The B0 numerical and kinematic fixture passes locally and in checksum-verified HPC3 run
`20260910T203608Z-a4ac913-1fd2e3` / job `55923138`. Sixteen targeted tests
verify exact homogeneous Gate A reduction, signed/family/crystal symmetries,
frame covariance, manufactured Nye content, packet transport, reaction
balances, false-wall controls, finite nonlinear wavelength selection,
refinement, restart, and absence of grain allocation.

The generic fixture selects a `2.000 micrometer` GND-bearing wall with zero
scalar total-density contrast. Maximum timestep and grid changes are 0.1521%
and `3.422e-7%`. All local/HPC decisions agree and primary fields agree to
`1.5e-13` on global scale. This does not calibrate a material wavelength or
promote the modulation to a physical wall or LAGB. Its original scientific
pass interpretation was superseded on 2026-09-11 because patterning is an
unloaded prescribed signed-density spinodal with fixed total density and no
dynamic Gate A coupling.

## Gate A BCC rotation material point

Substantive Gate A passed locally and on HPC3 using the published Bertin BCC
constitutive architecture. HPC3 run `20260909T044301Z-9ed6550-896a31`, job
`55843151`, passed all nine targeted tests in 31 seconds and was fetched with
verified checksums. The `1e-3` to `5e-4` true-strain-step comparison changes
final stress by 0.1243%, total density by 0.1621%, and attractor angle by
0.1723%, all below the declared 5% threshold.

All pass/fail checks are identical on macOS and HPC3. Seven of eight paths
agree to about `1e-12` in primary observables. The near-symmetric `[101]`
tension path has bounded transient sensitivity (0.592% stress, 0.791% total
density, 0.00143 degree angle) and reconverges below `5e-8` relative in final
stress/density. Full-precision platform outputs are retained separately.

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
