# Gate B1 driven signed CDD development

## Current disposition

Gate B1 is implemented as a new, dynamically Gate-A-coupled development model,
but the physical wall gate is **not passed**. Ten compact core tests pass. The
first nonlinear study found finite intermediate signed modes only at a load
increment that fails timestep refinement; those fields are rejected as wall
evidence. No HPC3 verification has been submitted and Gate C remains blocked.

The Bertin BCC-Ta constants remain a quarantined reference fixture. The CDD
coefficients `A=D=1` and the cross-family elastic scale are generic
dimensionless closure choices, not a material calibration. The DD collective
closure is exactly disabled.

## State, kinematics, and units

For each BCC Burgers family `a=1,...,4`, the primary spatial fields are mobile
and locked positive/negative line densities

\[
\rho_a^\pm,\rho_{a,L}^\pm\;[\mathrm{m}^{-2}],\qquad
\rho_a=\rho_a^++\rho_a^-,\qquad
\kappa_a=\rho_a^+-\rho_a^-.
\]

Every cell also owns the Gate A `F^p`, initial orientation, temperature,
strain, and time. Gate A supplies the Cauchy stress, resolved and effective
family shear stresses, current MRSSP normals, lattice orientation, shear
rates, and density source. The signed Orowan rate is

\[
\dot\gamma_a=b(\rho_a^+v_a^+-\rho_a^-v_a^-),\qquad
\dot F^p=L^pF^p,\qquad L^p=\sum_a\dot\gamma_a s_a\otimes m_a.
\]

The declared Nye measure is `alpha=-Curl(Fp-I)` in `m^-1`; division by `b`
gives an equivalent GND density in `m^-2`. This finite-deformation proxy and
its sign are retained as a precursor measure, not a Frank--Bilby boundary
inventory.

## Transport and stress reduction

The separately conservative mobile balances are

\[
\partial_t\rho_a^\pm+\partial_x(\rho_a^\pm v_a^\pm)=S_a^\pm.
\]

Following Groma, Zaiser, and Ispanovity, PRB 93, 214110 (2016), the local
correlation stresses use

\[
\tau_a^{back}=-D\mu b\,\frac{\partial_x\kappa_a}{\rho_f},\qquad
\tau_a^{diff}=-A\mu b\,\frac{\partial_x\rho_a}{\rho_a}.
\]

An anisotropic one-dimensional cross-family elastic reduction applies a
positive interaction matrix to `kappa` with Fourier kernel `-i/k`; it introduces
no selected length. Gate A provides the homogeneous signed speed. Above its
Taylor threshold, the derivative of the selected EXP/power/drag branch gives
the finite differential mobility. For positive drive the sign structure is

\[
v_a^+=v_a^0+M_a(\tau^{sc}+\tau^{back})
 +M_a(\kappa_a\tau_f/\rho_a+\tau^{diff}),
\]
\[
v_a^-=-v_a^0-M_a(\tau^{sc}+\tau^{back})
 +M_a(\kappa_a\tau_f/\rho_a+\tau^{diff}),
\]

with the corresponding resolved-stress sign for reverse loading. Mean/back
stress reverses with line sign; polarization-friction and diffusion do not.
The unloaded state has zero flux. A secant mobility trial was rejected because
division by vanishing overdrive produced a nonphysical threshold singularity.

## Source and transfer ledger

Gate A's frozen-step density ODE is integrated exactly in `sqrt(rho)`. Its
gross multiplication integral is evaluated on the same exact trajectory and
gross removal follows from `Delta rho = generation-removal`. Multiplication
adds equal opposite-sign pairs. Annihilation removes equal pairs and is capped
by the minority population, so it cannot erase signed Burgers content. Any
unavailable removal is explicitly recorded as `annihilation_limited_m_inv`.

Same-family locking, unlocking, and cross-family association are separate
sign-preserving mobile/locked transfers. Cross-family association does not
claim a physical BCC junction product or topology. The step ledger closes
mobile plus locked line content and each family's signed inventory. The
periodic boundary flux is zero.

## Linear stability and emergent scale

For a homogeneous flowing state, the implemented diagnostic evaluates the
positive branch of the published one-dimensional (`k_y=0`) operator:

\[
(\lambda+A k^2)(\lambda+D k^2)+\beta k^2=0,
\quad
\beta=(\dot\gamma'+2\alpha')(\dot\gamma'-\alpha').
\]

Here physical wave number is `sqrt(rho_f) k`. Growth is disabled outside the
flowing regime. The instantaneous Gate A state supplies the overdrive,
temperature-dependent modulus, forest density, and therefore `dot gamma'` and
the fastest discrete mode. No `selected_wavelength_m` exists in Gate B1. The
onset snapshot in the 16 micrometer development domain predicts modes 25--28,
or approximately 0.57--0.64 micrometer, and the formula directly gives the
similitude scaling `Lambda proportional to 1/sqrt(rho)`.

## Evidence and unresolved convergence

The core suite verifies zero-load stability, exact homogeneous Gate A
reduction, independent total/signed perturbations, removal of a Fourier mode,
positive conservative packet transport, sign parity of correlation stresses,
line/Burgers ledgers, nonzero Nye response to differential signed content, a
balanced high-density false-wall control, exact checkpoint/restart, finite-mode
driven dispersion, and absence of phase/grain-label state.

The nonlinear probes are deliberately not accepted:

- at grid 64 and final strain 0.008, increments `5e-4`, `2.5e-4`, and
  `1.25e-4` selected signed modes 9, 19, and 4 respectively;
- GND/total ratios were `4.83e-5`, `1.39e-2`, and `6.38e-5`;
- the middle increment also developed much larger total-density contrast;
- grid 128 at increment `2.5e-4` regenerated modes 29--43 after removing the
  predicted mode, but total-density power contaminated the Nyquist range and
  the signed peak remained diffuse.

This is a failed timestep/grid convergence gate, not a parameter target. The
next implementation task is a convergent coupled integrator that refreshes
gradient velocities and Gate A feedback within transport substeps. Only then
may the seed/domain/density/temperature matrix and one HPC3 bundle run.

Gate B1 has no phase field, grain label, boundary object, or physical grain
increment. Polygonization, LAGB recognition, DRX, and ASB are out of scope.

