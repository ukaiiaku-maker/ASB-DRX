# Gate B1 driven signed CDD development

## Current disposition

Gate B1 is implemented as a new, dynamically Gate-A-coupled development model,
but the physical wall gate is **not passed**. Eleven compact core tests pass.
The first nonlinear signal failed timestep refinement and was rejected. A
coupled internal refresh limit now makes requested-step refinement agree to
`1.6e-14` on global field scale. The converged trajectory does not form a wall,
and the full implemented operator selects the domain mode. No HPC3 verification
has been submitted and Gate C remains blocked.

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
Taylor threshold, the derivative of the selected power/drag branch gives
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

For a homogeneous flowing state, the reduced reference diagnostic evaluates the
positive branch of the published one-dimensional (`k_y=0`) operator:

\[
(\lambda+A k^2)(\lambda+D k^2)+\beta k^2=0,
\quad
\beta=(\dot\gamma'+2\alpha')(\dot\gamma'-\alpha').
\]

Here physical wave number is `sqrt(rho_f) k`. Growth is disabled outside the
flowing regime. The instantaneous Gate A state supplies the overdrive,
temperature-dependent modulus, forest density, and therefore `dot gamma'` and
the fastest discrete mode. No `selected_wavelength_m` exists in Gate B1, and
the formula directly gives `Lambda proportional to 1/sqrt(rho)`.

The acceptance diagnostic separately finite-differences the complete
semi-discrete state map in the physical 12-dimensional subspace: eight signed
populations and four Gate-A MRSSP slip amplitudes. It includes density sources,
finite-volume flux, `F^p`/orientation feedback, and the multiplicative plastic
update. At the onset snapshot this full operator has positive growth but its
fastest mode is mode 1. The reduced finite mode is therefore not a prediction
of the implemented coupled model.

## Evidence and unresolved convergence

The core suite verifies zero-load stability, exact homogeneous Gate A
reduction, independent total/signed perturbations, removal of a Fourier mode,
positive conservative packet transport, sign parity of correlation stresses,
line/Burgers ledgers, nonzero Nye response to differential signed content, a
balanced high-density false-wall control, exact checkpoint/restart, reduced
finite-mode dispersion, requested-step partition invariance, and absence of
phase/grain-label state.

The nonlinear probes are deliberately not accepted:

- at grid 64 and final strain 0.008, increments `5e-4`, `2.5e-4`, and
  `1.25e-4` selected signed modes 9, 19, and 4 respectively;
- GND/total ratios were `4.83e-5`, `1.39e-2`, and `6.38e-5`;
- the middle increment also developed much larger total-density contrast;
- grid 128 at increment `2.5e-4` regenerated modes 29--43 after removing the
  predicted mode, but total-density power contaminated the Nyquist range and
  the signed peak remained diffuse.

The splitting failure is corrected by refreshing the coupled update at a
maximum strain interval of `1.25e-4`, while an exactly homogeneous state remains
one exact Gate-A step. Requested increments from `5e-4` through `1.25e-4` now
follow the same internal trajectory. At strain 0.015, three converged seeds
remain homogeneous/balanced, with GND/total ratios only `0.82e-6` to `1.38e-6`.
A fixed-total-strain hold relaxes the perturbation.

The next task is to resolve the long-wave physical-slip/Gate-A coupling, most
likely by supplying mechanically equilibrated spatial stress. Only after the
full operator has a finite growing mode may the incommensurate-domain,
density-similitude, temperature/rate, wall-width, and persistence matrix run.

Gate B1 has no phase field, grain label, boundary object, or physical grain
increment. Polygonization, LAGB recognition, DRX, and ASB are out of scope.
