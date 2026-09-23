# V53 z-invariant full-tensor geometry mechanics

## Scope and constraint

The extension retains the periodic two-dimensional xy grid and adds three
displacement components.  Fluctuations satisfy

`u = u(x,y)` and `partial_z u = 0`.

The prescribed mean strain is a symmetric 3x3 tensor.  The present prepared
geometry verification uses prescribed mean `E_33 = 0`; this is plane strain,
not plane stress.  In particular, `sigma_33` is generally nonzero.  The
extension is default-off, so the retained in-plane n128/n192 cohort continues
to use its original equations.

## Elastic functional and units

The recoverable elastic energy is

`E_el = integral_V 0.5 * e_el : C : e_el dV`,

with

`e_el = E_bar + sym(grad_xy u) - sym(beta_p) - e_fixed`.

Strain and plastic distortion are dimensionless, stress and stiffness are Pa,
energy density is J/m3, and integrated energy is J.  The retained isotropic
Voigt-average Lamé coefficients are constructed from the existing `c11`,
`c12`, and `c44`; no modulus is fitted in V53.

For each nonzero Fourier wave vector `k=(kx,ky,0)`, equilibrium is solved from

`[mu |k|^2 I + (lambda+mu) k tensor k] u_hat`
`    = -i k_j (C : e_star)_ij`,

where `e_star = sym(beta_p) + e_fixed`.  The zero Fourier mode is the
prescribed mean strain.  The implementation reports the residual of
`div(sigma)=0` independently.

## Geometry conjugacy

At fixed total strain, a geometry coordinate `q_r` must satisfy

`dE_el/dq_r = - integral_V sigma : sym(d beta_p/dq_r) dV`.

The retained xy-surface rectangle changes `beta_i3`; its conjugate components
therefore include `sigma_13`, `sigma_23`, and `sigma_33`.  V53 differentiates
the actual positive subcell map and oriented face convention.  It does not
rotate the BCC Burgers vector, delete `beta_i3`, or introduce a target force.

The total geometry affinity retains separate elastic, logarithmic/line,
ordered-gradient, and chemical-reservoir contributions.  No outcome-dependent
self-energy is subtracted.  The difference between the old projected elastic
functional and the new full-tensor functional at the same prepared state is a
model representation difference, not physical heat.

## Event and activation quantities

Four quantities remain distinct:

1. The signed complete affinity is the accepted energy change per microscopic
   face event, including elastic, stored-line/gradient, and signed chemical
   work.
2. The geometric event measure is one exchanged species per climb event per
   physical face.  The coefficient `c = z b_z L / Omega` retains its sign;
   only nonnegative event counts use an absolute value.
3. The optional stress-like barrier input is the magnitude of the effective
   glide resolved shear stress sampled on the face.  It is explicitly an
   uncalibrated non-glide barrier-modulation hypothesis.
4. For the EXP-floor enthalpy, the activation volume is

   `-dG*/dsigma = H0 (1-f) a n r^(n-1) exp(-a r^n) / sigma_c`,

   for positive `r=sigma/sigma_c`, and zero for nonpositive input.  It has
   units J/Pa = m3 and is neither the geometric event measure nor the complete
   affinity.

Activation entropy is included once in `G*=H*-T DeltaS*` and is independent of
the stress derivative above.

## Common physical clock

The coupled prepared-geometry macro uses

`B(H/2) -> G(H) -> B(H/2)`.

The physical elapsed time is `H`, not `2H`.  Bulk and geometry exposures are
reported separately.  Actual accepted bulk substeps fill both half exposures,
and geometry rates are reevaluated on evolving shared states.  Both stalled
faces consume the geometry exposure as an identity while bulk evolution
continues.  If an exposure cannot be filled after a valid trial prefix, the
production common-clock wrapper rolls the complete macro back; no partially
mutated state is published with zero elapsed time.
