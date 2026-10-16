# V67 ordered-wall neutral-pair recovery

## Physical role

The production wall reservoirs carry positive and negative density
`rho_w+^a` and `rho_w-^a` (m^-2) for each BCC Burgers family `a`. Their signed
difference supplies Burgers/Nye content; their sum stores line energy. Equal
opposite-sign wall content is therefore redundant for the declared signed
structure and may be removed by thermally activated climb recovery. This is
the continuum counterpart of polygonization by annihilation of redundant
content followed by rearrangement of the surviving excess into lower-energy
arrays.

The recovery transaction is not grain nucleation. It does not alter phase
support, orientation, slip, plastic distortion, or any interface reservoir.
It is used immediately before the separately conserved Frank--Bilby reaction
cone, which can reorganize the signed survivors without inventing line.

## Local candidate and kinetics

For owner support `eta`, wall order `q`, and family `a`, the accessible pair is

`p^a = min(rho_w+^a, rho_w-^a)`.

Only cells with `eta > 0.75` and `q >= 0.20` participate. The conjugate line
affinity is evaluated from the same production Helmholtz functional:

`A^a = max(mu_w+^a + mu_w-^a, 0)` (J m^-1).

One shared Arrhenius EXP-floor law is used in loading and retained-state holds:

`r^a = nu exp[-max(Q_floor, Q - A^a l*)/(k_B T)]` (s^-1),

`f^a = 1 - exp(-r^a Delta t)`.

The candidate removes `f^a p^a` from both signs. Parameters are inherited from
the existing wall-recovery scale rather than fitted to the phase-field result:
`Q = 1.20 eV`, `Q_floor = 0.05 eV`, `nu = 1e7 s^-1`, and
`l* = b = 2.48e-10 m`. There is no strain-rate switch and no per-condition
parameterization.

## Exact balances and energy publication

For every owner and family,

`Delta integral eta(rho_w+^a-rho_w-^a) dV = 0`.

If `P` is the support-weighted removed pair measure, then

`Delta integral eta(rho_w+^a+rho_w-^a) dV + 2 P = 0`.

The cold candidate is evaluated by the complete common-state Helmholtz
functional. An uphill finite step is backtracked and is never published. For a
downhill accepted step, `-Delta F` is deposited as local heat over the reaction
activity and the independently recomputed internal-energy balance must satisfy
the existing 5% hard tolerance. This makes energy release an audit result, not
an input chosen to close the ledger.

## Qualification and claim boundary

Atomic tests cover nonnegativity, exact signed and total-line balances,
support-weighted diffuse owners, positive/negative sign exchange, Burgers-family
permutation, zero-GND/high-total-density recovery, wall-order gating, heat, and
the first law. A coupled-interval test proves production persistence and ledger
ownership. The source checkpoint `6d45315` passes 1165 canonical tests.

These tests qualify the transaction, not DRX. The active production
discriminator must still show whether recovery makes the surviving content
form a finite interior common-orientation region and whether that structure is
retained or grows during a no-reset hold.
