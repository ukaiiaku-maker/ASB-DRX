# V23 extensive density and wall-state authority

## Decision boundary

V22 is frozen at `185ee86`. Its wall classification remains
`WALL_ORDER_FUNCTIONAL_STILL_UNPHYSICAL`. V23 does not reinterpret the
normalized V22 order field as evidence of a boundary.

## One density inventory

All dislocation populations are line length per material volume, in m^-2.
There is no implicit 2-D thickness conversion. Thus, on a uniform grid,

\[
 L/t_z=\sum_{ij}\rho_{ij}\,\Delta x\Delta y .
\]

For each of four BCC Burgers families and each sign, the authoritative state is

\[
 \rho_m^\pm,\quad \rho_f^\pm,\quad
 \rho_t^\pm,\quad \rho_o^\pm,
 \qquad \rho_w^\pm=\rho_t^\pm+\rho_o^\pm .
\]

Junction `j_c` is a reaction extent in m^-2. It contributes
`m_c j_c` to total line, where `m_c` is the product-line multiplicity. Therefore

\[
 \rho_{tot}=\sum_{a,s}(\rho_{m,a}^s+\rho_{f,a}^s+
 \rho_{t,a}^s+\rho_{o,a}^s)+\sum_c m_cj_c.
\]

The legacy arrays `rho`, `rho_wall`, and `q_wall_v19` are derived views only:
`rho=rho_tot`, `rho_wall=sum(rho_t+rho_o)`, and
`q_wall_v19=rho_o/(rho_t+rho_o)`. They must never be added to the extensive
state or included in an energy a second time. No factor of 1e3 is authorized.

For an old V22 restart only, its scalar `q` may partition the already-existing
wall line into `rho_o=q rho_w` and `rho_t=(1-q)rho_w`. This one-time conversion
creates no line. A native V23 restart must contain every signed tangle and
ordered reservoir; partial reconstruction is rejected.

## Energy channels

The V23 local/nonlocal wall functional contains only declared physical
channels:

\[
 f=E_l\rho_{tot}+C_\rho\rho_{tot}\ln(\rho_{tot}/\rho_{ref})
 +E_t\rho_t+E_o\rho_o
 +\frac{K_\alpha}{2}\|\alpha_o-\alpha_{kin}\|^2
 +\frac{K_g}{2}\sum_{a,s}|\nabla\rho_{o,a}^s|^2+f_j,
\]

where

\[
 \alpha_o=\sum_a(\rho_{o,a}^+-\rho_{o,a}^-)
 b_a\otimes l_a.
\]

Every listed coefficient is nonnegative. The positive `rho ln rho` term is
included once through total density. There is no negative density-only well,
scalar-order barrier, absent-wall penalty, or direct label allocation.

## Conservative ordering kinetics

For each sign/family, `rho_t <-> rho_o` uses the same line pool and event
length `ell_event`. With `Delta mu=mu_o-mu_t`,

\[
 k_+/k_-=\exp[-\Delta\mu\ell_{event}/(k_BT)],\qquad
 \dot\rho_o=k_+-k_-,\qquad \dot\rho_t=-\dot\rho_o.
\]

The shared bounded split uses `1 +/- tanh(Delta mu ell/(2 kBT))`. It gives
`Delta mu * dot(rho_o) <= 0`, conserves line and Burgers content exactly, and
uses the declared event length to convert J/m chemical potential to J/event.
The accepted-step map limits the fraction transferred from any donor; its JVP
differentiates that exact map, including the active scale.

## Present authorization

The density ledger and extensive exchange fixtures are locally qualified. A
manufactured two-degree BCC tilt wall now closes an independently calculated
Frank--Bilby circuit to below `6e-16` on 16, 32, and 64 grids at fixed physical
wall width. This required separating the sessile ordered-line direction from
the inherited V22 glide-line direction; the latter has a 0.627 relative
projection residual and is rejected.

The long 20--50% strain wall campaign is still not authorized. The extensive
operator must first be coupled to the mechanically heterogeneous accepted
trajectory and demonstrate unseeded nonlinear formation with the required
RSS, slip-gradient, Curl-beta, Nye, and orientation diagnostics. Manufactured
closure is a fixture, not scientific wall formation.
