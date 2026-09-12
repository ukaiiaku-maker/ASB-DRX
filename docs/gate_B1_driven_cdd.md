# Gate B1 driven signed CDD development

## Current disposition

Gate B1 is implemented as a new, dynamically Gate-A-coupled development model,
but the physical wall gate is **not passed**. Thirteen compact core tests pass.
The first nonlinear signal failed timestep refinement and was rejected. A
coupled internal refresh limit now makes requested-step refinement agree to
`1.6e-14` on global field scale. A provisional nonzero one-dimensional
long-range kernel made the full operator stable. Enforcing the derived
`k_y=0`, `T(k)=0` reduction restores a finite growing mode without fitting a
wavelength. The completed HPC3 scientific matrix rejects grid convergence and
intrinsic domain-independent spacing. Gate C remains blocked.

The Bertin BCC-Ta constants remain a quarantined reference fixture. The CDD
coefficients `A=D=1` are generic
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

For the declared one-dimensional wave vector, the published anisotropic
long-range factor `T(k)` is exactly zero. A nonzero cross-family `-i/k` kernel
is retained only as an ablation; any tested nonzero amplitude suppresses the
instability and is not the baseline. Cross-family forest resistance remains.
Gate A provides the homogeneous signed speed. Above its
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
The periodic one-dimensional mechanical reduction uses the spatially common
resolved traction supplied by Gate A; local density, temperature, MRSSP, `F^p`,
and orientation still control each cell's response.

## Source and transfer ledger

Gate A's frozen-step density ODE is integrated exactly in `sqrt(rho)`. Its
gross multiplication integral is evaluated on the same exact trajectory and
gross removal follows from `Delta rho = generation-removal`. Multiplication
adds equal opposite-sign pairs. Annihilation removes equal pairs and is capped
by the minority population, so it cannot erase signed Burgers content. Any
unavailable removal is explicitly recorded as `annihilation_limited_m_inv`.
Gate A controls the exact spatially averaged family-reservoir source. Each
family's gross pair source is distributed uniformly, as in the conservative
nonzero-wave-number stability derivation, while preserving its exact integral
and exact homogeneous reduction. Thus the very fast material-point relaxation
cannot masquerade as a spatial wall instability.

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
semi-discrete map on the Nye-compatible manifold. Every active signed-density
Fourier perturbation carries the quadrature slip/`F^p` perturbation required by
`partial_x gamma_a=-b kappa_a`; independent incompatible `F^p` waves are
excluded. It includes finite-volume flux, averaged Gate-A reservoir sources,
MRSSP/orientation feedback, and the multiplicative plastic update. At the
onset snapshot families 0 and 1 are active, so the physical operator is four
dimensional. With the superseded long-range amplitude `0.25`, every tested mode
was damped. In the strictly derived `T=0` baseline, the full compatible operator
selects finite mode 6 with growth `4.45e7 s^-1` on the 64-point audit grid.
This change follows the declared reduction and is not a wavelength fit.

## Evidence and extended-matrix decision

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

Resolution-invariant Fourier noise with mode 6 deleted produces mode 10 at both
64 and 128 points (1.6 micrometers), but the completed 128/256 comparison does
not converge. The selected wavelength changes by 10%; GND/total, total-density
contrast, orientation-gradient RMS, and mean wall FWHM change by 95%, 84%, 96%,
and 620%, respectively. The 256-point result is diffuse GND polarization rather
than the coarse-grid GND-rich precursor. Across 13, 16, and 19 micrometer boxes,
the selected mode remains 10 and the wavelengths are 1.3, 1.6, and 1.9
micrometers. This is box locking, not a demonstrated intrinsic length scale.

HPC3 run `20260912T011600Z-2c0bf13-edff97` / Slurm `55949943` completed all 38
staged tests and the full matrix. It passed timestep refinement, three-seed
formation, density similitude, temperature/rate execution through Gate A,
unload characterization, line balance, and exclusion of labels/phases. It
failed only the scientific grid and incommensurate-domain criteria. The
machine-readable result therefore reports `fixture_passed=true`,
`scientific_gate_passed=false`, and classification
`DRIVEN_CDD_PRECURSOR_PRESENT__SCIENTIFIC_MATRIX_FAILED`. This negative result
is the authoritative Gate B1 checkpoint; no coefficient or tolerance was
retuned after inspection.

## Failure diagnosis and next admissible revision

A selected-mode full-operator audit on a homogeneous 256-point state shows
that modes 32, 64, 96, 120, and 127 are strongly damped throughout the sampled
loading path. The mode-127 nonlinear peak is therefore not supported by the
homogeneous dispersion relation. The present finite-volume implementation
computes correlation stresses at cell centers and then averages the resulting
velocities to faces; this has an odd/even null at Nyquist after nonlinear
harmonics form. A local Lax--Friedrichs trial removed that null but generated
thousands of CFL substeps near strain 0.011, demonstrating that numerical
dissipation alone is not a valid repair; the trial was reverted.

The incommensurate-domain setup has a second defect: `spectral_noise_modes=12`
uses the same random Fourier mode numbers in every box, so the initial physical
wavelengths scale with domain length. The persistent mode-10 outcome therefore
cannot distinguish intrinsic selection from memory of the seed. A replacement
verification must use one declared physical random-field spectrum across box
sizes and a conservative staggered discretization of the back/diffusion fluxes,
then repeat local refinement before any HPC3 submission. Filtering the
mode-127 result or adding a fitted wavelength is prohibited.

Gate B1 has no phase field, grain label, boundary object, or physical grain
increment. Polygonization, LAGB recognition, DRX, and ASB are out of scope.
