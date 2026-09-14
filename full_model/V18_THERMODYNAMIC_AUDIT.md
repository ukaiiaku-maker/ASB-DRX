# V18 common defect-energy and Hessian audit

## Scope

V14 and V15 remain frozen moving-front numerical diagnostics. Their scalar
continuation-pressure root is not a production mechanism. Production startup
now rejects `sibm_coupled_neutral_feedback=true`.

The prior driver used the full v34 potential for the density equation but only
`A_E(T) rho` in the common phase interpolation. V18 repairs that mismatch:
phase-owned parent and child states now use the same complete local density
functional (line, positive logarithmic, inherited ordering, and soft low-density
branches). GND/orientation and boundary-residual energies remain separately
declared compatibility terms, so they are not folded into the scalar density
energy a second time.

## Logarithmic coefficient

The retained coefficient is

`C_log(T) = 0.06 A_E(T) > 0`, with units J/m.

It is classified as the existing phenomenological configurational/correlation
branch. Its curvature is `C_log/rho > 0`; it cannot create a spinodal alone.
It is not claimed simultaneously as elastic outer-cutoff self energy. With
`R proportional to rho^(-1/2)`, the latter generates a *negative* coefficient
on `rho log rho`; its density-independent linear part would also overlap the
declared `A_E rho`. No elastic-cutoff logarithm is enabled in V18, preventing
both sign conflation and double counting.

The mathematical energy uses the exact `rho log rho -> 0` limit. A floor is
accepted only when evaluating the singular chemical potential; it neither
changes the energy nor sources density.

## Ordering and total Hessian

The old scalar Gaussian ordering dip is the fully ordered (`q_w=1`) limit of
the new explicit wall-order functional. In the monolithic full model, the
existing organized-wall reservoir fraction `rho_wall/rho` is the equivalent
order state used by phase ownership. At zero organized-wall fraction, density
is a disordered tangle and receives no ordering credit. A bounded order barrier and a positive
wall-reservoir partition penalty distinguish mobile/forest density from an
ordered wall. The inherited ordering dip appears exactly once.

The machine-readable audit reports Hessian eigenvalues in normalized variables
`(rho_m/rho_scale, rho_f/rho_scale, rho_w/rho_scale, q_w)`. Negative curvature,
where present, is attributed to the coupled ordering branch—not to the positive
logarithmic term. The square-gradient wall term is nonnegative and has no
preferred wavenumber; it therefore regularizes interfaces without prescribing
a wall spacing.

## Qualification status

Stages 1 and 2 are implementation-complete subject to the full regression
suite. This does not qualify intragranular DRX. A one-grain/no-boundary spatial
fixture must next demonstrate compatible signed walls, an independently
measured orientation plateau, and persistence before any phase support can be
allocated.
