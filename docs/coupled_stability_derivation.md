# Finite-wavenumber coupled stability operator

## Scope

This is a frozen-time linearization of the active net EXP-floor, antiplane,
storage, heat, and binary phase equations. It predicts infinitesimal growth or
decay about a spatially homogeneous interior phase mixture. It is not a DRX
nucleation criterion: a pure parent lies on the constrained phase-simplex
boundary, and crossing a finite interfacial barrier remains a finite-amplitude
problem.

For each nonzero Fourier wave vector, the perturbation state is

\[
x=(\delta\gamma^p,\delta T,\delta\rho_0,\delta\rho_1,\delta\phi)^T,
\qquad \eta_0=1-\phi,\quad\eta_1=\phi.
\]

## Exact antiplane elimination

Periodic mechanical equilibrium projects the elastic distortion transverse to
the wave vector. A perturbation in the represented x-directed plastic shear
therefore gives

\[
\delta\sigma_x=-G P_{xx}\delta\gamma^p,
\qquad P_{xx}=\frac{k_y^2}{k_x^2+k_y^2}.
\]

The orientation is physical to this scalar antiplane model: a mode varying in
x is mechanically compatible (`Pxx=0`), whereas a band varying in y has the
maximum local stress feedback (`Pxx=1`). The imposed-loading zero mode is a
different problem and is excluded.

## Net flow tangents

Each grain rate is the forward activated rate minus the unloaded reverse rate,
evaluated at local stress `sigma/q(rho)`. The reverse subtraction is essential:
it makes zero stress an exact equilibrium and removes the discontinuity in the
old signed one-way continuum law. Analytical derivatives include the density
dependence of both the Taylor factor and the reverse rate. The mixture rate is

\[
R=h(1-\phi)r_0+h(\phi)r_1,
\]

with all five derivatives retained. Centered finite differences independently
verify the stress, temperature, and density rate tangents.

## Storage and heat branches

Away from the thermodynamic cap, `rho_i_dot=K r_i` and the mechanical heat rate
is `(sigma-E_line K)R`. When requested line storage exceeds plastic work, the
active smooth branch is `Keff=sigma/E_line`, so mechanical heat is identically
zero and the stress derivative of `Keff` enters both density equations. At
`sigma=E_line K` the limiter is nonsmooth and no unique Jacobian exists; the
implementation rejects that state instead of reporting a misleading tangent.

Thermal diffusion contributes `-alpha |k|^2 delta T`. Phase relaxation heat is
the local dissipation `M D^2/2`, where `D=mu_1-mu_0`; its linearization is
retained for a nonstationary frozen base.

## Phase sector

For the binary simplex, the projected Allen-Cahn equation is

\[
\dot\phi=-\frac{M}{2}D,
\]

where

\[
D=2W\phi(1-\phi)(1-2\phi)
 +E_{line}[\rho_1h'(\phi)-\rho_0h'(1-\phi)]
 +2\kappa |k|^2\delta\phi.
\]

This distinguishes an interior phase/order instability from creation of a new
orientation. The latter requires the separate embryo/orientation gate.

## Numerical verification

The complete analytical 5 by 5 operator is compared column-by-column with a
centered finite difference of an independently evaluated frozen-mode RHS.
Tests cover both antiplane orientation limits, uncapped and capped storage, and
explicit rejection of the cap kink. Eigenvalues are computed after a diagonal
similarity scaling to avoid loss of precision from mixing shear, kelvin, and
dislocation-density units; the scaling does not change the eigenvalues.
