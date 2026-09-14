# V15 coupled-neutral fixed-point decision

## Purpose and unchanged physics

V15 tested whether the qualified V14 moving-front operator admits a fully
coupled neutral pinned-cap state before any mesh or long-horizon campaign. No
mobility, transmission, storage, sink, line-energy, interface-energy, or defect
parameter was retuned. The physical cap, chord, domain, interface width, and
represented thickness remain the V14 values.

Signed applied pressure is now represented without making an absolute phase
energy negative. For applied pressure `P`, the nonnegative offsets are

`E_parent^P = max(P,0)`, `E_child^P = max(-P,0)`,

so `E_child^P-E_parent^P=-P` for either sign. This is an algebraic
representation repair; it leaves the common-functional derivative unchanged.

## Declared amplitude direction and pressure root

The amplitude mode is the centered derivative of the same pinned-cap seed
family used by the full model,

`v_a = [eta(a+1 nm)-eta(a-1 nm)]/(2 nm)` at `a=0.6 um`,

restricted to the active parent/child interface. It is stored in the restart
and survives checkpointing exactly. At every coupled-neutral construction step,
the driver evaluates the complete phase derivative (gradient, multi-order
barrier, common stored energy, compatibility, and rho/eta term) at zero and
`1 MPa`. Since pressure enters linearly, it applies

`P_c = -D_a F(0) (1 MPa)/(D_a F(1 MPa)-D_a F(0))`.

The analytical and centered finite-difference derivatives agree to
`5.59e-7` relative at the audited endpoint. A 20-step continuous trajectory is
bitwise identical to a 10+10 restart (`maximum_scaled_difference=0`).

## Material-state relaxation

The frozen V14 state was not a material fixed point: its density changed by
about `5e12 m^-2` per `0.1 us` step. Holding phase mobility and front processing
off while retaining the full constitutive equations reduced the mean total
density from about `2.47e17` to `9.63e16 m^-2` over approximately `60 ms`.
The interface was then returned to the original `0.1 us` phase timestep before
the coupled-neutral tests. This preparatory state is not itself promoted as a
coupled fixed point.

## Decisive result

Two consecutive 240-step coupled holds start from exact checkpoints. Their cap
drifts are individually below `0.02 cell` (`-0.003224` and `-0.002433` cell),
and every line, signed-Burgers, line-energy/heat, nonnegativity, simplex, and
no-allocation invariant passes. Nonetheless:

- the energetic neutral pressure changes from `0.565142 MPa` to
  `0.075599 MPa`, a `-0.489543 MPa` shift;
- the processed-line increment rises from `4.33212e-7 m` to `2.47427e-6 m`;
- the second/first processing-increment ratio is `5.71146`;
- the late velocity is statistically nonzero; and
- local virgin sweep accelerates while the net cap remains nearly stationary.

Thus small net contour drift is masking irreversible local state evolution.
The front reactions change the energetic neutral point and amplify spatial
shape modes; they do not approach zero processing or stationary cycling. A
single scalar pressure root of the declared cap functional therefore does not
produce a coupled fixed point for this operator even at 128 square.

Machine decision:

`V15_COUPLED_NEUTRAL_FIXED_POINT_FAILED_AT_128`

The numerical fixture is valid, finite-difference verified, conservative, and
restart exact, so `fixture_passed=true`. The scientific fixed-point gate is
false. Under Directive v15 this prohibits 192/256 threshold extrapolation,
long canonical branches, a polycrystal SIBM claim, and the overnight HPC3
bundle. Claim level remains 2: canonical pinned sign discrimination only.
