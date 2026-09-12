# Mission v3 decision-driven campaign

Status: adopted 2026-09-12.

The supplied decision memo (SHA-256
`d9ee87c800a180f1789ff47c3ee8bb6005ef53b5fc2103ad08b12d31117af7c1`)
and Mission v3 directive (SHA-256
`f3f521241d0b11ebfac37d3c44fa0ca621aee92a9d13af1bd3cff8059a1984f5`)
supersede the serial interpretation of the campaign. Gates control claim
strength; only failed hard invariants prohibit using an affected result.
Scientific diagnostic failures select the next discriminating experiment.

## Arrhenius backbone

The v3 implementation separates the enthalpic EXP floor

\[
\Delta H^*=H_0(T)\{f_H+(1-f_H)\exp[-A(|\tau|/\tau_c(T))^n]\}
\]

from signed activation entropy, `Delta G*=Delta H*-T Delta S*`. Event rates
use entropy exactly once. A shared mechanism entropy preserves directional
symmetry, zero net rate at zero stress, and nonnegative `tau*rate`. Constant
entropy and attempt frequency are deliberately reported through their
identifiable product `nu0 exp(Delta S*/kB)`. Negative free barriers stop with a
request for an explicit barrierless/drag transition.

Five isolated families cover zero, positive, negative, and bounded
temperature-dependent entropy while retaining the same enthalpy law. These are
structural comparisons, not fitted material properties.

## Correlation-flux candidates

For each signed family, correlation stresses are placed directly at face
`i+1/2`:

\[
\tau_b=-D\mu b\frac{\kappa_{i+1}-\kappa_i}{\Delta x\,\rho_{f,i+1/2}},
\qquad
\tau_d=-A\mu b\frac{\rho_{i+1}-\rho_i}{\Delta x\,\rho_{i+1/2}}.
\]

The positive and negative correlation fluxes are
`F+=rho+ M (tau_b+tau_d)` and `F-=rho- M (-tau_b+tau_d)`. Their divergence
telescopes exactly on the periodic grid. Candidate 1 uses arithmetic face
densities. Candidate 2 uses logarithmic means, which obey the discrete chain
rule `L(a,b)[log(b)-log(a)]=b-a` and is preferred for subsequent
free-energy/dissipation integration. Both candidates have a nonzero, damped
Nyquist symbol and second-order low-mode convergence. Neither is yet coupled
to staggered plastic slip/`Fp`; therefore no repaired Gate B1 claim is made.

## Physical perturbation process

The replacement generator specifies a Gaussian physical power spectrum with a
correlation length and physical wavenumber cutoff. Refinements of the same box
are exact restrictions at common nodes once the cutoff is resolved. Different
boxes sample the same physical spectrum and contain different mode counts,
rather than receiving identical Fourier mode numbers. The correlation length
is a fixture/noise parameter and may not be interpreted or fitted as a wall
wavelength.

## Current decision

The entropy-separated Arrhenius kernel and both isolated staggered fluxes may
advance to compact cross-platform verification. The logarithmic face form is
the leading integration candidate because of its discrete chain rule, but full
signed Orowan compatibility, positivity, stiffness treatment, and nonlinear
Gate B1 convergence remain required before it can replace the preserved
cell-centered no-go model.

## Isolated polygonization fixture

The v3 polygonization state separates mobile and wall-captured positive and
negative populations by Burgers family. Distinct Arrhenius mechanisms govern
capture, climb annihilation, and wall ordering. Capture is sign-preserving;
climb removes equal opposite-sign pairs; both retain total signed Burgers
content. Removed line energy is reported explicitly as released energy.

For the declared simple-tilt fixture, wall excess over resolved width `w`
defines the angle through `2 sin(theta/2)=b w |rho_w+ - rho_w-|`. Thus a
balanced high-density wall has zero angle, and an orientation cannot be
created by maturity or a label. This fixture has one physical grain and no
allocation surface. It tests the downstream mechanism independently but
cannot support an integrated LAGB or DRX claim until driven CDD supplies a
qualified wall inventory.

The subsequent kinematic fixture stores signed densities at cells and plastic
slip at faces. For signed line flux `J=F+-F-`, it advances
`kappa_dot=-div(J)` and `gamma_dot_face=b J`. Therefore
`div(gamma_face)+b kappa=0` is preserved algebraically under the same flux,
including the Nyquist mode. Cell-centered slip for Gate-A calls is a
second-order average of adjacent face values; the staggered quantity remains
the authoritative Nye-compatible state. This resolves the placement question
without yet changing the preserved Gate B1 implementation.

## Pre-integration thermodynamic repair

The selected logarithmic stress discretization has **not** been promoted from
the logarithmic-mean identity alone. A separate integrable candidate uses

\[
 \psi_c=\mu b^2\left[A\sum_a\rho_a
 (\ln(\rho_a/\rho_*)-1)+{D\over2}{\sum_a\kappa_a^2\over\rho_F}\right],
\]

where `rho_a=rho_a+ + rho_a- + 2 rho_floor`,
`kappa_a=rho_a+ - rho_a-`, and `rho_F=sum_a rho_a`. Thus `psi_c` has units
`J m^-3` and its population derivative has units `J m^-1`. With right-face
flux

\[
 F_{a,f}^{\pm}=-{M_{a,f}\mathcal L_{a,f}^{\pm}\over b}
 {\mu_{a,i+1}^{\pm}-\mu_{a,i}^{\pm}\over\Delta x},
\]

the periodic finite-volume update gives exactly

\[
 {d\mathcal F_h\over dt}=-\sum_{a,f}{M_{a,f}\mathcal L_{a,f}^{\pm}\over b}
 {\left(\Delta\mu_{a,f}^{\pm}\right)^2\over\Delta x}\le0.
\]

Finite-difference differentiation of the energy, the algebraic dissipation
identity, exhausted-family zero flux, and floor refinement are unit tested.
The floor remains numerical regularization and must converge away. The prior
face-stress logarithmic form remains an ablation; only the integrable form
enters the new coupled candidate.

The authoritative signed density is now mobile plus locked plus wall content.
Sign-preserving transfer among those reservoirs leaves Nye content and face
slip unchanged. Flux-step clipping is explicitly ledgered, and the accepted
integrated regime requires exactly zero correction. Face-to-cell slip and the
adjoint cell-to-face traction projection satisfy an exact plastic-work
identity. Production EXP-floor configurations now require `shape_n >= 1`, so
the zero-stress activation volume is finite.

## First integrated Arrhenius--CDD increment

`integrated_cdd_v3.py` owns four-family mobile, locked, and wall signed
populations, authoritative face slip, cellwise `Fp`, rotation, and temperature.
It combines physical positive/negative Arrhenius line flux with the variational
correlation flux, advances density and slip with the identical face flux, maps
slip to a multiplicative `Fp` increment, and closes external, elastic,
plastic, correlation, line, and heat terms. Its complete checkpoint restarts
bitwise. A homogeneous state reduces exactly to the corresponding local
forward-minus-reverse Arrhenius/Orowan increment.

This is a numerical-integration checkpoint, not a wall result. Grain and phase
allocation remain absent, the collective DD law remains disabled, and the
failed Gate B1 matrix remains authoritative.

The polygonization fixture now has a convex disordered/ordered wall free
energy with configurational mixing. Its Arrhenius mobility relaxes toward the
thermodynamic minimum in either direction. A new independent Frank--Bilby
residual accepts a kinematic orientation jump supplied separately from wall
content; the earlier self-substitution residual remains only an implementation
identity.

## Integrated one-dimensional mechanism decision

The stiff logarithmic-density part now uses a backward-Euler Fourier solve,
while the nonlinear polarization potential and physical Arrhenius flux remain
explicit. The resulting effective face flux reconstructs the complete density
increment, so it also advances authoritative slip without breaking Nye
compatibility. In a stiff fixture the explicit method halves its interval and
the IMEX method accepts it whole; both remain filter-free and dissipative.

The same accepted interval now includes bounded pair multiplication,
equal-sign-pair annihilation, locking, unlocking, and wall capture. Each
process has a separate ledger; only multiplication and annihilation change
total line content, and none changes total signed Burgers content.

A factorized local screen spans signed entropy, three positive backstress/
diffusion ratios, three densities, 64/128/256 grids, 13/16/19 micrometer boxes,
zero/below/driven/unload stresses, three physical-noise seeds, and two noise
correlation lengths. All complete frozen eight-population spectra are damped
through Nyquist; every nonlinear noise case reduces total-density contrast.
This is a structural result: constant-stress Arrhenius flux supplies advection,
and the positive variational energy supplies diffusion. Local mode-independent
reactions cannot select a finite wavelength. Therefore this baseline cannot
produce a spontaneous wall over the screened positive-energy parameter class.
The next justified branch is state-dependent friction plus cross-family
internal/reaction coupling, not a broader search of the same damped equations.

## Cross-family escalation architecture

A primary-literature audit rejects a purely scalar cross-family locking-rate
patch as the production escalation. Starkey, Hochrainer, and El-Azab introduce
junction-point density and graph connectivity so open slip-system lines remain
collectively closed and continue to satisfy `alpha=curl(beta_p)`,
`div(alpha)=0`, and Frank's junction rule
([JMPS 2022, 104685](https://doi.org/10.1016/j.jmps.2021.104685)). The related
vector-density reaction implementation treats annihilation, glissile and
sessile junctions, and cross slip while explicitly enforcing line continuity
([JMPS 2021, 104327](https://doi.org/10.1016/j.jmps.2020.104327)). Multiple-slip
CDD studies also identify junction formation as central to predicted wall
orientation rather than merely a scalar hardening increment
([IJP 2018](https://doi.org/10.1016/j.ijplas.2017.09.015)).

Accordingly, the next production candidate will add signed vector density and
an explicit junction-point/topology reservoir with stoichiometric Burgers
incidence. State-dependent friction may be screened alongside it, but scalar
locked density alone remains a kinetic ablation. This is a 2-D-capable state
extension; it is not calibrated from the old DD multi-hit histories.
