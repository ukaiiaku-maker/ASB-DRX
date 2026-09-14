# V13 valid-pair SIBM requalification decision

## Decision

The v12 scientific classification is amended to:

- `V12_SOURCE_PAIR_AND_POST_SEED_GEOMETRY_INVALID_FOR_SIBM_QUALIFICATION`
- `FULL_MODEL_SIBM_GROWTH_MECHANISM_UNRESOLVED`

The v12 trajectories remain immutable historical evidence. They do not reject
SIBM because their selected parent phase had no connected pure core or usable
normal depth after seeding, and the seed tip was too close to another boundary.

V13 passes the isolated geometry/operator preflight, but full-model
requalification is paused at a protected conservation failure. A near-static
common-state equilibration of the canonical pair raises
`signed boundary residual exceeds removable line content` in
`moving_front.advance_front`. No parameter retuning and no HPC submission were
performed after that failure. The full-model SIBM growth mechanism therefore
remains unresolved rather than supported or rejected.

## Actual post-seed pair precondition

Pair identity is recomputed from the post-seed phase field before scientific
step 1. All criteria use metres and the actual parent and child fields. The
validator requires connected pure cores for both phases, minimum core inradius,
available normal depth on both sides, adequate connected HAGB length, clearance
from other boundaries and triple junctions, and adequate active-window
clearance. An invalid pair emits `sibm_invalid_post_initialization_pair.json`
and terminates without advancing a scientific timestep.

The retired v12 source is now rejected by this precondition for four explicit
reasons: missing parent connected pure core, inadequate parent core inradius,
zero parent normal depth, and inadequate seed-tip clearance.

## Boundary displacement and pinned-cap energetics

The seed no longer overwrites a disk. It displaces the signed-distance contour
of the selected parent/child interface and reconstructs only that pair's
diffuse equilibrium profile while preserving their local sum. A zero-amplitude
operation is bitwise exact and no label or orientation is allocated.

For cap height \(a>0\) and half chord \(b>0\), the implemented geometry is

\[
R=\frac{a^2+b^2}{2a},\qquad
\theta=4\tan^{-1}(a/b),\qquad
L=R\theta,
\]

\[
A=\frac{R^2}{2}(\theta-\sin\theta),\qquad
\kappa=R^{-1}.
\]

With boundary energy \(\gamma\), stored-energy pressure \(p_s\), compatibility
pressure \(p_c\), drag pressure \(p_d\), and represented thickness \(t\), the
declared excess energy is

\[
E(a)=t\{\gamma[L(a)-2b]-(p_s-p_c-p_d)A(a)\}.
\]

The code evaluates the analytical derivative
\(dE/da=t[\gamma\,dL/da-(p_s-p_c-p_d)dA/da]\). Its finite-difference audit
passes. At \(a=0.6\ \mu\mathrm{m}\), \(b=1.5\ \mu\mathrm{m}\), and
\(\gamma=0.5\ \mathrm{J\,m^{-2}}\), the actual-geometry critical stored
pressure is \(2.2988505747\times10^5\ \mathrm{Pa}\).

## Local evidence

- The canonical compact bicrystal passes the physical post-seed validator on
  128 and 192 grids.
- Flat favorable and reversed driving pressures translate with opposite,
  correct signs; mobility-off is stationary; an unpinned bulge smooths.
- A two-step canonical full-driver run closes the line and energy ledgers,
  preserves signed Burgers content, and allocates no label or orientation.
- Short pinned subcritical and supercritical runs are not accepted as a
  criticality split because initial diffuse-profile relaxation dominates.
- Common-state equilibration then exposes the hard signed-content ledger
  failure above. This is the stopping condition specified by Directive v13.
- The matched v32/full-v34 ASB regression remains preserved.

Machine-readable results are in `v13_valid_pair_preflight.json` and
`v13_full_model_requalification.json`. The next authorized action is to repair
the phase/moving-front common-state handoff, prove restart-exact pinned
criticality locally, and only then submit the compact single-job HPC3 bundle.
