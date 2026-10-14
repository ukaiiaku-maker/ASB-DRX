# V66 literature-to-operator basis

This note records which statements are used as operator constraints and which
remain hypotheses.  Literature does not override the immutable campaign
mechanics: Peierls, Taylor, Hall--Petch, thermal activation, and the EXP-floor
Arrhenius rate remain active.

## Arrhenius mechanics and Hall--Petch events

Azizi and Dillon, `1-s2.0-S1359645426005628-main.pdf`, write a mechanism rate
as

`strain_rate_i = delta_strain_i * nu0 * N_i(M) *
 exp[-(H_i* - T S_i* - phi_i sigma v_i*)/(kT)]`.

Their Hall--Petch pile-up scaling uses `N_HP = rho b^3/d`,
`phi_HP = sqrt(d/l_dd)`, and `delta_strain = b/l_dd`; their boundary/TJ
nucleation scaling uses `N_nuc = b^2/d^2`, `phi_nuc = sqrt(d/b)`, and
`delta_strain = b/d`.  These are fixed-state event-rate models.  They do not
specify the signed post-event line transaction, heat, or boundary structural
state.  V66 therefore uses the published scaling for event opportunity and the
immutable EXP-floor form for activation, while keeping the sign-balanced pair
source and structural absorption transaction explicitly identified as closure
hypotheses.

## Recovery, polygonization, and RX route

Rollett et al., `ROLLETT_9780080982694.pdf`, distinguish concurrent recovery
processes: annihilation, rearrangement into subgrain walls, subgrain growth,
and misorientation evolution.  The text supports pre-existing subgrains as RX
precursors, abnormal growth under stored-energy gradients, and increasing
misorientation during growth through an orientation gradient.  It also warns
that subgrain growth without an orientation gradient can lower mean
misorientation.  V66 consequently requires an explicit compatible signed-line
ledger for rotation and uses selective cell elimination, rather than a generic
grain-label double well or imposed orientation.

## Structural boundary kinematics

Zhang and Xiang, *Motion of grain boundaries incorporating dislocation
structure*, arXiv:1710.01856, represent signed boundary-dislocation arrays and
derive boundary motion and rotation from their motion/reaction.  The relevant
constraints are:

- Frank compatibility for a 2-D tilt boundary is
  `theta n + sum_j rho_j b_j = 0`;
- misorientation is therefore computed from signed structural content, not an
  independently evolved angle;
- conservative boundary motion preserves total boundary line content and, for
  their shrinking circular/elliptic examples, `L theta` is constant;
- dislocation reaction changes structural content and supplies a distinct
  sliding/rotation contribution;
- the long-range interaction maintains Frank compatibility, while the local
  energy is a generalized Read--Shockley form.

The V66 common-axis rule `delta theta = b delta Lambda` and its geometric
dilution are the straight, low-angle scalar specialization of these
constraints.  They are not a general 3-D Frank--Bilby law.  Arbitrary boundary
normals and rotations must not be claimed until the vector compatibility
residual is a production state and is energetically controlled.

## Bulk signed-density energetics and organization

Zaiser, *Local density approximation for the energy functional of
three-dimensional dislocation systems*, arXiv:1508.03652, separates:

- self/correlation energy with the dislocation-density logarithm;
- nonlocal Hartree energy of excess/GND density; and
- a local quadratic correction in excess density whose variation gives a back
  stress.

The paper explicitly cautions that local backstress is a correction to, not a
replacement for, long-range GND interaction.  This supports the V18
`rho log rho` repair and signed Nye coupling.  It does not justify treating a
scalar density double well as dislocation thermodynamics.

Groma, Ispanovity, and Zaiser, *Dislocation patterning in a 2D continuum theory
of dislocations*, arXiv:1601.07831, show that Taylor-like `sqrt(rho)` flow
resistance can supply an instability only in the flowing regime, while a
correlation-derived density-diffusion stress (or a higher-order correction) is
needed to select a finite wavelength.  Their fastest wavelength scales with
`1/sqrt(rho)`; without the diffusion-like coefficient `A`, all wave numbers can
grow and no maximum is selected.  They also require a nontrivial threshold
mobility.

This explains the scoped V66 H3 negative result: the present local
junction/forest mode is nearly wave-number independent.  A future H3
production candidate must add and audit a correlation-derived signed flux and
the nonlocal GND interaction; adding scalar Cahn--Hilliard separation or tuning
a grid-scale diffusion coefficient would not test the cited mechanism.

## Current decision

H1/H2 (compatible absorption plus selective inherited-subgrain growth) remain
the primary RX path.  H3 is a separately testable organization path, not a
device for forcing nucleation.  Boundary/TJ-only and matched random sources
remain independent source-location hypotheses because the literature rate law
does not determine a post-event source geography or signed transfer kernel.
