# V18 intragranular DRX decision

## Decision

Accepted at the limited claim level:

`SINGLE_CRYSTAL_DEFORMATION_GENERATES_A_COMPATIBLE_LAGB_PRECURSOR`

Not yet accepted:

`INTRAGRANULAR_SUBGRAIN_TO_PHYSICAL_DRX_GRAIN_PATH_SUPPORTED`

The stronger claim remains false because the calculation demonstrates a
qualified common-functional handoff, not sustained supercritical moving-front
growth and persistence after loading heterogeneity is removed.

## What was repaired

The production phase interpolation previously used only `A_E(T) rho`, although
the density equation used the complete v34 line + positive logarithmic +
ordering + low-density potential. Parent, child, and other phase-owned states
now use the same complete functional, with the previous pointwise temperature
dependence retained. GND/orientation and boundary-residual terms remain
separate and are not counted again in the scalar density energy.

The V15 coupled-neutral scalar pressure root is retired: production startup
rejects that option. V14/V15 files remain frozen numerical diagnostics.

## One-grain result

The representative calculation starts with one label, one orientation, zero
internal boundary, and zero applied continuation pressure. A smooth stress
concentration produces differential slip; it does not prescribe an orientation
or wall wavelength. The orientation field is derived from that slip. Four-family
signed wall populations evolve separately by conservative mobile-to-wall
capture toward the kinematic incompatibility. Wall order and neutral recovery
use separate EXP-floor activated processes with independent entropy parameters.

At 1.5 ms on the 128-square grid:

- independently measured misorientation: `12.9239 deg`;
- equivalent plateau radius: `1.30754 um`;
- boundary order mean / closure: `0.67743 / 1.0`;
- Frank--Bilby relative residual: `3.7606%`;
- interior/boundary orientation-gradient RMS: `6.38497e4 / 2.60478e5 m^-1`;
- interior/exterior total density: `6.49665e15 / 7.60000e15 m^-2`.

Thus the interior is a resolved low-gradient, lower-defect-energy orientation
plateau enclosed by an ordered signed wall before the 15-degree HAGB
classification threshold. The 96-to-128 grid comparison changes the tracked
misorientation/radius metrics by at most `0.740%`; halving the timestep also
stays below the provisional 5% threshold.

At 64 square, the candidate remains qualified at every sampled state from
`1.2` through `1.6 ms`; its misorientation increases monotonically from
`10.5250` to `14.0334 deg` while the independent Frank--Bilby residual falls.
This supplies a finite persistence interval rather than a one-frame label.

The one-grain fixture has no allocation operation. A separate handoff accepts
only the qualified recognition record, inherits its measured interior
orientation exactly, creates a two-phase simplex with zero residual, lowers
the common defect energy, and invokes the existing v14 conservative front
transfer. Its line closure and signed-Burgers change are exactly zero in the
front ledger.

## Negative controls

- With wall ordering kinetically disabled, the orientation/density pattern is
  not recognized and no lower-energy subgrain develops.
- With a spatially uniform differential-slip field, no orientation plateau or
  signed wall is recognized.
- With neutral recovery disabled, the LAGB geometry can form but has no stored
  energy advantage; the phase handoff rejects it.
- SIBM remains a distinct resolved-HAGB path using the same common defect
  thermodynamics. No scalar pressure was inserted and no pathway-specific
  thermodynamic coefficient was retuned.

## Balance and scope

The one-grain line ledger closes to `1.95e-16` relative and the maximum family
signed-content drift is `1.22e-19` relative. Exact restart is verified across
slip, all mobile/forest/wall populations, wall order, time, and cumulative
recovery state. The full repository suite passes 362 tests at this checkpoint.

This is a generic mechanism qualification under a declared heterogeneous
loading fixture. It is not material calibration, a mesh-converged grain-size
prediction, a universal DRX onset, or an ASB map. The v32 ASB regression remains
separate and unchanged.
