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
