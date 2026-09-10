# Gate B signed spatial transport and wall patterning

## Status and scientific boundary

This gate implements a reduced one-dimensional periodic continuum-dislocation
model with four BCC `1/2<111>` Burgers families and explicit positive/negative
populations. It is coupled to the Gate A kinematics by the same Burgers-family
basis, signed Orowan flux, plastic distortion, plastic spin, and an exact
homogeneous reduction to the quarantined Bertin material-point step.

The finite-wavelength mechanism follows the signed-density continuum structure
developed by Groma, Zaiser, and Ispánovity, *Physical Review B* 93, 214110
(2016), [DOI 10.1103/PhysRevB.93.214110](https://doi.org/10.1103/PhysRevB.93.214110),
and the deterministic follow-up by Wu et al., *Physical Review B* 98, 054110
(2018), [DOI 10.1103/PhysRevB.98.054110](https://doi.org/10.1103/PhysRevB.98.054110).
Those works identify signed GND/total densities, long-range internal stress,
local gradient stresses, and nontrivial mobility as the pattern-forming
ingredients. The present coefficients are declared generic verification
fixtures; they are not transferred material parameters.

The model contains no phase fields, embryos, grain labels, boundary mobility,
or DD multi-hit memory. `collective_scale` is fixed to zero and a nonzero value
is rejected. Passing this gate establishes a physical GND-bearing
orientation-gradient precursor, not a grain boundary, DRX, polygonization, or
ASB.

## State, geometry, and units

For Burgers family `a=1,...,4` and sign `s=+,-`, the mobile and locked line
densities are

\[
\rho^a_s,\rho^{a,J}_s\quad[\mathrm{m}^{-2}],\qquad
\kappa^a=\rho^a_+-\rho^a_-\quad[\mathrm{m}^{-2}].
\]

The four unit slip directions are the Gate A `<111>` directions. Each has a
declared perpendicular unit normal `m^a`; `s^a dot m^a=0`. With periodic
coordinate `x` and Burgers magnitude `b`, the compatible family slip is fixed
by

\[
\partial_x\gamma^a=-b\kappa^a,
\quad \beta^p=\sum_a\gamma^a s^a\otimes m^a,
\quad \alpha=-\operatorname{Curl}\beta^p.
\]

Thus `gamma` and `beta^p` are dimensionless and the Nye tensor `alpha` has
units `m^-1`. For a one-dimensional gradient direction `n`,

\[
\alpha=b\sum_a\kappa^a s^a\otimes(n\times m^a).
\]

The equality is evaluated independently from the discrete curl and is a sign
and units test. Lattice orientation is the exact exponential of
`-skw(beta^p)` in the reduced small-elastic-strain limit. Under a rigid frame
rotation `Q`, the implementation verifies `alpha' = Q alpha Q^T`.

## Flux, plastic slip, and reactions

The mobile balances are

\[
\partial_t\rho^a_s+\partial_xJ^a_s
=S^a_{mult,s}-S^a_{ann,s}-S^a_{lock,s},
\]

with locking received by `rho^{a,J}_s`. The signed normalized flux `j_q` is
mapped to physical line fluxes by

\[
J^a_+=\frac{\rho_0}{2}j_q^a,\qquad
J^a_-=-\frac{\rho_0}{2}j_q^a,
\qquad
\dot\gamma^a=b(J^a_+-J^a_-)=b\rho_0j_q^a.
\]

Periodic finite-volume upwinding separately verifies transport of a
nonnegative packet at prescribed velocity. Multiplication adds equal `+/-`
pairs, annihilation removes equal compatible `+/-` content, and locking
transfers each sign unchanged from mobile to a separate junction reservoir.
Consequently multiplication and annihilation change total line length but not
net Burgers content; locking changes neither total line length nor net Burgers
content.

For every reaction step the machine ledger records, in `m^-1`, mobile content
before/after, pair multiplication, pair annihilation, mobile-to-junction
transfer, junction content before/after, both balance residuals, and net signed
content change. Nonnegativity is preserved by exact bounded reaction
fractions. No junction topology or reaction coefficient is claimed calibrated.

## Internal stress and nonlinear wavelength selection

Define the dimensionless signed field `q^a=kappa^a/rho_0`. The baseline
chemical/internal-stress potential is

\[
\mu^a=-a q^a+g(q^a)^3-c\partial_{xx}q^a+h\psi^a,
\qquad -\partial_{xx}\psi^a=q^a-\langle q^a\rangle,
\]

\[
j_q^a=-M\partial_x\mu^a,qquad
\partial_tq^a=-\partial_xj_q^a.
\]

Here `M [m^2 s^-1]`, `c [m^2]`, `h [m^-2]`, and `a,g` are dimensionless.
The reported internal stress is `-tau_ref mu [Pa]`. The local term acts on
signed excess, the gradient term is a short-range backstress regularizer, and
the inverse-Laplacian term is a reduced long-range elastic penalty. This is not
a Cahn--Hilliard double well in scalar total density: each family total remains
uniform in the baseline pattern test, Taylor strength retains its positive
square-root density dependence, and only signed incompatibility patterns.

Linearization about `q=0` gives, for nonzero wavenumber,

\[
\lambda(k)=M(ak^2-ck^4-h).
\]

The unstable band is bounded at both small and large `k`, and the fastest mode
is

\[
k_*=\sqrt{a/(2c)},\qquad \Lambda_*=2\pi/k_*.
\]

The cubic term saturates the finite-amplitude nonlinear wall, while the
gradient and long-range terms prevent grid-scale collapse and macrophase
coarsening. The ETD step treats the exact linear dispersion and the cubic term
explicitly. The generic fixture declares `Lambda_*=2 micrometers` in a
`16 micrometer` periodic domain; this is a verification length, not a measured
material wall spacing.

## Wall controls and acceptance

The wall classifier is conjunctive: resolved spectral concentration, nonzero
Nye/GND fraction, and nonzero orientation gradient are all required.

- A spatial total-density band with `rho_+=rho_-` is classified
  `total_density_band_only`, has zero Nye content, and cannot pass.
- An alternating signed/dipolar structure has zero domain-integrated Burgers
  content but nonzero local `kappa`, Nye tensor, and orientation gradient. It
  passes only when a finite resolved structure-factor peak is also present.
- A diffuse signed fluctuation without sufficient spectral concentration is
  not a wall.
- Gate B has exactly one crystal and exposes no grain-label or phase-field
  state; its physical grain count is identically one.

The provisional numerical gate compares 128/256 grids and time steps
`0.004/0.002 s` at fixed `1 s` duration. Wall spacing, GND RMS, orientation
gradient RMS, and structure-factor peak fraction must each change by less than
5%. Continuous and checkpointed trajectories must be bitwise identical within
one execution environment.
