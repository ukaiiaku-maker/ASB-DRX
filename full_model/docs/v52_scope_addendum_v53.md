# V52 scope addendum recorded by V53

V52 remains a valid, immutable numerical checkpoint.  This addendum narrows
two interpretations without changing any V52 artifact or trajectory.

## Flow measures

The imposed tensor shear rate is 100/s and the imposed engineering shear rate
is 200/s.  At interval 104, the three reported plastic fractions answer
different questions:

- 42.5612316% is plastic shear divided by all shear added after the preload;
- 73.5187690% is the plastic fraction added from interval 52 through 104;
- 98.88414194% is the final interval's plastic-rate/imposed-rate ratio.

The last quantity is a descriptive rate-balance observation, not a universal
yield criterion.  The corresponding controlled-trajectory tangent estimate,

`G * (1 - engineering_plastic_rate / 200/s)`,

is not an intrinsic fitted hardening modulus.  V52 resolves neither a stress
peak nor a microstructural steady state.  Its n192 spatial support ends at
interval 32, before the later high-flow regime.

## Prepared-geometry elastic null

The retained subcell rectangle maps its swept xy surface into the third
plastic-distortion column,

`beta_family[..., family, :, 2] = b * surface_fraction / thickness`.

The V52 elastic helper symmetrizes only `beta_p[..., :2, :2]`.  Therefore an
isolated retained rectangle move changes no input read by that helper.  At
fixed driving its elastic-energy increment is identically zero, not merely
small in the selected fixture.  The appropriate classification is

`IN_PLANE_ELASTIC_ENERGY_HAS_A_NULL_FOR_THE_RETAINED_XY_SURFACE_GEOMETRY_INCREMENT`.

V52's positive line/gradient energy, signed species exchange, finite-rate
clock, quadrature, restart, and ownership evidence remain valid within their
declared scope.  The resolved glide stress used by its optional geometry
kinetics remains an uncalibrated barrier-modulation comparator; it did not
supply the omitted conjugate elastic work.

V53 introduces a default-off, z-invariant full-tensor elasticity branch.  Any
energy difference obtained by repricing a prepared V52 state with that new
functional is a representation comparison.  It is not deposited as heat and
does not retroactively change the V52 bulk trajectories.

No V52 result establishes spontaneous grain formation, a persistent LAGB,
DRX, strict ASB, calibrated climb kinetics, or a production material
calibration.
