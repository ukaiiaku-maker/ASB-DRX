# V48 numerical, energy, and geometry definitions

## Finite extensive ordering

For each active Burgers family and sign, the evolved coordinate is

\[
q_s=\rho^{\rm ordered}_s/(\rho^{\rm tangle}_s+
\rho^{\rm ordered}_s),\qquad 0\le q_s\le1 .
\]

The signed-family pool in the denominator is fixed during ordering.  The
physical rate is unchanged from V47,

\[
\dot q_s=-a_s(T,\tau)\tanh\left[
\frac{\ell_e(\mu^{\rm ordered}_s-\mu^{\rm tangle}_s)}{2k_BT}
\right],
\]

with the EXP-floor Arrhenius attempt rate `a_s`.  V48 integrates this equation
with an exact-FFT-JVP Rosenbrock--Euler method and step doubling.  A trial of
duration `h` is compared with two trials of duration `h/2`; only the two-half
state is accepted, and a rejected numerical trial advances neither state nor
physical clock.  Local error control and independent endpoint qualification
are separate ledger fields.  The old packed-coordinate Euclidean endpoint
distance is not used as a production remainder certificate.

## Authoritative defect Helmholtz density

Transport, locking, ordering, geometry, owner publication, and front
diagnostics now use the same signed-reservoir density:

\[
f_d=\Gamma\rho+c_\rho\rho\ln(\rho/\rho_0)
+c_f\rho_f+c_t\rho_{w,t}+c_o\rho_{w,o}
+\frac{C_\alpha}{2}|\alpha_o-\alpha_*|^2
+\frac{\kappa_o}{2}|\nabla\rho_{w,o}|^2+f_j .
\]

Here densities have units m⁻²; line coefficients have units J m⁻¹; the Nye
tensor has units m⁻¹; and `f_d` has units J m⁻³.  The represented cell energy
is `f_d dx dy h_section`.  The legacy scalar-order functional remains only for
historical fixtures lacking reservoir-resolved owners.

The physical total is defect Helmholtz plus boundary excess, recoverable
elastic, phase local/gradient, and thermal internal energy.  Numerical
compatibility penalties are excluded.  For a fixed-mean-strain constitutive
interval, elastic release, defect storage, and generated heat are internal
conversions and must sum to zero.  A load ramp is Strang-like: load from the
endpoint to the midpoint at fixed state, evolve at the fixed midpoint, then
load from midpoint to endpoint at the evolved state.  The two elastic-energy
changes are the external work for that split path.

## Affinity-coupled represented geometry

For a complete copied-state event in direction `d`, work on the system is
positive and

\[
\Delta\Phi_d=\Delta F_d-W_{{\rm mech},d}-\mu\Delta N_d,
\qquad A_d=-\Delta\Phi_d .
\]

The coarse mesh patch is not one atomic event.  Its event count is the
absolute exchanged species count for a climb-like event, or swept area divided
by `b²` for a volume-preserving sweep.  The event-normalized affinity is
`A_event=A_d/N_event`.  V48 uses the declared deterministic dissipative law

\[
r_d=r_{\rm EXP}(T,\tau)\max\left[
\tanh\left(\frac{A_{\rm event}}{2k_BT}\right),0\right].
\]

This activity vanishes at neutrality, rejects an uphill direction without
mutation, retains the nonzero EXP-floor barrier and signed activation entropy
once, and does not assert local detailed balance for two distinct irreversible
outgoing endpoints.  Actual reverse edges must be evaluated from their own
copied states and signed material exchanges.
