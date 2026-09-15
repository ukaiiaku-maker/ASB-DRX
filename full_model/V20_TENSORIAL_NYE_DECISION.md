# V20 tensorial Nye and on-trajectory decision

V20 stops at a scientific prerequisite, not at a computational bottleneck:
`V20_COMMON_OPERATOR_REPAIR_REQUIRED_NO_HPC`.

The frozen V19 claim remains exactly
`NO_PHYSICAL_LAGB_PRECURSOR_OBSERVED_AT_32x32_AND_10_PERCENT_STRAIN`.
Nothing here upgrades that local negative observation into a constitutive or
material no-go.

## What the V19 trajectory actually did

Both V19 branches were replayed from their checkpointed parameter dictionaries
with exact spatial checkpoints every 1% strain. Only output cadence and
evolution-neutral V20 diagnostics changed. The signed and unsigned family wall
budgets close below `4.4e-14` and `9.9e-16` relative, respectively.

All efficiency-weighted channel exposures during the 10 microsecond trajectory
are below unity. Capture reaches only about `0.0181`, junction transfer
`0.00303`, release `1.31e-4`, annihilation `5.46e-5`, and wall-order relaxation
`0.224`. Aggregate wall polarization is only `2.19e-7` to `2.35e-7`, although
the family-resolved absolute signed source is nonzero. The direct V19
classification is therefore
`SIGNED_CONTENT_PRESENT_BUT_NOT_SPATIALLY_ORGANIZED`, with short kinetic
exposure and missing physical junction/nonlocal topology as contributors.

The prior projected finite-mode result cannot supply the missing exposure. The
production order law relaxes toward an algebraic wall-fraction/polarization
target, whereas the projected model follows the derivative of a wall free
energy. Production capture is also a one-sided function of newly arrived
discrete advective flux, whereas the projection treats capture as a local
differentiable reaction. There is no common Jacobian, so the prior eigenvalues
and cumulative growth estimate are invalid as predictions of V19. This rules
out `ADEQUATE_LINEAR_EXPOSURE_NO_NONLINEAR_RESPONSE` and prevents a nonlinear
falsification claim.

## Repairs completed locally

The full driver now has an opt-in four-family BCC 2.5-D state carrying full
three-dimensional Burgers vectors and normals, full plastic distortion,
first-order family alignment, and a family-resolved Nye tensor that retains
connection terms when crystal orientation varies. The same accepted slip event
updates slip, plastic distortion, alignment, and Nye. A 10-step production
smoke gives dual-Nye residual `1.90e-15` and discrete divergence residual
`5.77e-16`; it retains one grain.

An independent simple-tilt Frank--Bilby fixture converges from `3.46%` residual
at 128 to `1.73%` at 256. A high unsigned-density, sign-balanced control has
zero Nye. Frame covariance and a sessile two-parent Frank-rule junction with
line-node closure pass.

The production mechanics now calls a shared periodic nonlocal elastic solver.
Zero-mode, sign reversal, elastic energy, and work-conjugacy tests pass. Against
the same 1000-step pre-refactor V19 replay, the shared solver changes the most
sensitive wall field by at most `4.31e-13` relative. A V20 20-step continuous
run and 8+12 segmented run are bitwise identical for all density, thermal,
plastic, tensorial-Nye, wall, and orientation states. This test also exposed and
repaired an old signed/scalar staggered-checkpoint load defect without moving
the reconciliation relative to continuous execution.

The configured local suite passes 377 tests.

## Remaining hard prerequisite

The tensorial state and explicit junction topology are qualified foundations,
but the production wall reactions still use the quarantined V19 symmetric
scalar transfer. Density transport has not yet been made one discretely common
flux update with the tensorial alignment state, and the explicit Frank-rule
junction products are not yet active in production. Consequently Stages B and
D are not complete, the aggregate V20 fixture is false, and no finite-mode
growth exposure can yet be calculated honestly.

No HPC3 campaign is authorized. A long run becomes eligible only after a
single differentiable production/projected operator passes the remaining flux,
topology, restart, and regression tests and predicts
`G_finite >= 8` on an actual loading path.
