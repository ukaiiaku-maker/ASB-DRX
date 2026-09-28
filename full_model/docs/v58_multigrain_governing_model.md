# V58 multi-grain common-owner production model

## State and ownership

For persistent material grains `g=1,...,G`, the dimensionless supports obey

`w_g >= 0`, `sum_g w_g = 1`.

Each grain owns signed mobile, forest, and wall line densities
`rho[g,r,a,+/-]` in m^-2, junction density in m^-2, slip and plastic
distortion (dimensionless), alignment in m^-2, family Nye tensors in m^-1,
orientation in radians, and retained hardening variables. Temperature is a
single common Eulerian field in K; it is copied to the owner records only for
schema compatibility and may not evolve as independent dormant histories.

An interface component has a persistent ID and grain pair. It owns first-pass
and revisit support, blocked signed line and junction inventories (m^-2), the
unresolved laboratory-frame Burgers vector by reservoir (m^-1), exposure (s),
periodic winding, signed/absolute sweep, work, and heat.

## Reconstruction and Nye audit

Intensive common fields are reconstructed as

`q_bar = sum_g w_g q_g`,

while the authoritative incompatible tensor is evaluated after reconstructing
plastic distortion,

`alpha = -Curl(sum_g w_g beta_p,g)`  [m^-1].

The independent audit retains the terms

`alpha = sum_g w_g[-Curl(beta_p,g)] + sum_g[-grad(w_g) x beta_p,g] + R_h`,

where `R_h` is the discrete spectral product-representation residual. The
support-gradient term is explicit interface content. Separately,

`R_owner = sum_g w_g[-Curl(beta_p,g) - sum_a alpha_g,a]`

measures pre-existing owner-reservoir inconsistency. Neither term is
distributed by `|slip|` into family kinetics.

## Physical transfer and Burgers closure

Competing interface proposals are formed from one immutable accepted state and
are scaled cellwise so no donor spends more than its support. Transmitted
neutral pairs remain neutral. Signed donor content is projected into the
receiving crystal's rotated BCC Burgers basis. The receiver coefficients are
limited by the transmitted line-length budget. The remaining laboratory-frame
Burgers vector is stored on the interface, and any scalar line not representable
after reprojection is also stored there. Neutral blocked content is split among
boundary storage, annihilation, and a declared neutral external sink. A signed
external sink is forbidden until it carries its own explicit Burgers vector.

The transaction independently closes:

- support exactly;
- nonnegative total line inventory;
- receiver-basis signed Burgers content plus interface residual;
- fresh and revisit material histories;
- simultaneous donor capacity at junctions.

## Complete functional

The final shared state is priced once. Nonlinear defect storage is integrated
over persistent material owners before spatial ownership is summed,

`F_defect = integral sum_g w_g f_defect(q_g) dV`,

not as `f_defect(sum_g w_g q_g)`. The latter creates undeclared cross-grain
correlation energy in a diffuse interface and is not conjugate to owner-local
kinetics. Plastic distortion is instead reconstructed first because it is the
source of one common elastic boundary-value problem.

The complete functional is

`F = F_line/log/order + F_signed-junction + F_boundary-excess`

`    + F_elastic + F_phase-local + F_phase-gradient`.

The multiphase local/gradient terms contain intrinsic grain-boundary energy.
Explicit boundary excess contains only trapped dislocation/junction content,
so intrinsic boundary energy is not charged twice. Periodic eigenstrain
equilibrium supplies recoverable elasticity. Constraint and normalization
penalties are excluded from physical storage and heat.

Thermal internal energy is

`U_th = integral C_v (T-T_ref) dV` [J].

For front motion, independent mobility dissipation is evaluated as driving
pressure times realized swept volume. It is not assigned from the final energy
residual. An adiabatic event deposits that dissipation into the common
temperature; a prescribed-temperature event exports it through the declared
thermostat. Publication requires both

`Delta F - W_ext + E_sink <= tolerance`

and

`Delta U - W_ext + E_thermostat + E_sink = 0`

within the declared numerical tolerance. Decisions are SHA-256 bound to the
exact pre-state, candidate, functional configuration, and interval.

## Kinetics and physical clocks

Both directions of every active interface are priced from the whole shared
state. Directional pressure lowers the retained EXP-floor activation enthalpy

`H*(p) = H0[f + (1-f) exp(-a (p/p_c)^n)]`.

The Arrhenius prefactor and activation entropy are counted once. The selected
velocity determines the requested support transfer over the actual interval;
joint capacity then limits all incident boundaries simultaneously. A finite
event that misses the energy audit is halved and rebuilt atomically.

The existing signed-wall residual advances dislocation/plastic owners. A
single equilibrated stress tensor is solved from the reconstructed plastic
distortion. Each owner's rotated BCC Schmid tensors resolve that same stress,
so support-weighted plastic increments are conjugate to the common elastic
energy while grain orientations remain distinct. Only supported material
ages; an explicit active mask prevents dormant depleted cells from limiting
the global timestep. Production line transport uses the conservative upwind
flux to preserve positivity; spectral derivatives remain authoritative for
Nye and gradient energies. Support-weighted heat is reconstructed once and
thermal diffusion is applied once to the common temperature with the exact
periodic Fourier propagator. Strain-controlled external work is independently
integrated from the equilibrated mean stress. If a mechanical interval exceeds
5% relative first-law error, it is rolled back and recursively divided in both
physical time and applied strain; the outer clock advances only after every
leaf and their cumulative balance qualify.

## Current scope

The initialized three-grain junction is prepared existing-boundary DRX, not
spontaneous grain birth. The baseline uses zero applied migration pressure,
the retained V55 density contrast (`4e17`, `1e17`, `3e17` m^-2), a 5 micrometre
periodic domain, a 0.3125 micrometre diffuse-interface width, continued strain,
four BCC families, physical recovery/heat, and the same equations for all
rate/temperature cases. Strict ASB additionally requires persistent localized
plastic power/heat, stress response, causal thermal controls, and selected
spatial/time refinement; a hotspot or endpoint contrast alone is insufficient.

The earlier `cea9371` n32 checkpoint through step 25 is retained as superseded
evidence only. It accumulated history using independent whole-domain owner
elastic solves and a post-mixture nonlinear defect energy. Restarting that
state under the common-stress functional produces a real initial relaxation
and is therefore prohibited by the v2 checkpoint configuration binding.
