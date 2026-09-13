# Mission-v3 hard-consistency repair and first topology decision

Status: local decision-grade checkpoint, 2026-09-12.

## Provenance boundary

The validated integrated-control source remains `a445735`. The fetched result
was recorded by `4099ada`; the topology architecture checkpoint was
`32ccb56`. Both requested diffs over `a445735..32ccb56` are empty for `src`,
`tests`, and `tools`, including `integrated_cdd_v3.py`. Therefore HPC3 job
`55972295` validates exactly `a445735`; no later scientific source was
incorrectly attributed to it.

The present repair changes scientific source after that boundary and is only
locally verified. It does not inherit the earlier HPC validation and does not
need an HPC run until a computational result survives the local discriminating
analysis.

## Physical rates, fluxes, and content

The Arrhenius API now separates event frequency from physical velocity:

\[
\Gamma^+-\Gamma^-\ [\mathrm{s^{-1}}],\qquad
v=\ell_{\rm event}(\Gamma^+-\Gamma^-)\ [\mathrm{m\,s^{-1}}].
\]

The spatial operator uses only

\[
F^\pm=\rho^\pm v^\pm\ [\mathrm{m^{-1}s^{-1}}],\qquad
\dot\gamma=b(F^+-F^-)\ [\mathrm{s^{-1}}].
\]

All one-dimensional content ledgers use
`sum(rho_i * dx)` and therefore report `m^-1`. The old generalized
`net_rate_s_inv` remains available only for dimensionless local event
increments and is explicitly forbidden for CDD flux construction.

## Declared small-strain antiplane kinematics

The integrated control no longer claims multiplicative finite strain. Its
authoritative kinematic field is additive plastic distortion
`beta_p`. With laboratory slip dyads

\[
P_{a,i}=R_iP_a^cR_i^T,
\]

the accepted increment is

\[
\Delta\beta^p_i=\sum_a\Delta\gamma_{a,i}P_{a,i}.
\]

Lattice rotation is independent of `polar(beta_p)` and advances from elastic
spin:

\[
R_i^{n+1}=\exp(\Delta W-\Delta W_i^p)R_i^n,
\quad
\Delta W_i^p=\operatorname{skw}(\Delta\beta_i^p).
\]

For simple antiplane shear, `Delta W_xz=Delta gamma/2` and
`Delta W_zx=-Delta gamma/2`. A pure-elastic manufactured test and a
nonuniform rotating-crystal test verify this convention.

The same cellwise weight `w_ai=(P_ai)_xz` is used for resolved shear,
plastic-distortion accumulation, macroscopic plastic shear, plastic work, and
the adjoint face traction. Thus the face/cell work identity is tested for the
actual rotated crystal, not a fixed crystal-frame surrogate.

## Stored-energy ledger

The energy is split into mobile correlation/entropy, total-signed long-range
polarization, mobile line, junction/locked, and wall terms:

\[
\Psi=\Psi_m+\Psi_{\rm long}[\kappa_m+\kappa_J+\kappa_w]
 +e_m\rho_m+e_J\rho_J+e_w\rho_w.
\]

The chemical potential for mobile transport is the derivative of this same
energy while stationary content is held fixed. The timestep reports each
component change, total transport free-energy change, and reaction free-energy
change. Scalar locking/capture laws remain bounded numerical fixtures; their
residual heat cannot be promoted to physical junction dissipation.

## Minimal vector/topology model

`vector_topology_cdd_v3.py` is a two-dimensional-capable, lowest-order
alignment model. It carries eight nonnegative mobile species (positive and
negative populations of four BCC Burgers families), one explicit product
species per declared junction reaction, junction point density, and a shared
local line tangent. For reaction `r: a+b <-> j`, the incidence column is

\[
S_{ar}=S_{br}=-1,\qquad S_{jr}=+1,
\]

and the product is constructed as `b_j=b_a+b_b`. Consequently

\[
B S=0
\]

to roundoff and the reaction source preserves the Nye tensor pointwise. This
fixed-alignment restriction is intentionally a falsification model; it is not
a general CDD topology representation.

Forward normalized mass action and reverse release are

\[
r_r=k_f\rho_a\rho_b/\rho_* - k_r p_r,
\qquad
\frac{k_f}{k_r}=\exp[-\Delta G_r/(k_BT)].
\]

The forward transition state uses the EXP-floor mechanism. A reverse barrier
that would be nonpositive is rejected pending an explicit drag branch.
Glissile and sessile products are distinct; reverse release is explicit.
Cross-slip, climb topology, endpoint transport, and multiple line orientations
are not yet implemented and are not claimed.

## Complete isothermal dispersion operator and decision

For the aligned state, the Fourier Jacobian contains every term in the minimal
closure:

\[
L(k)=J_{\rm reaction}-ik_\parallel V-k^2D
-ik_\parallel\rho_0\frac{\partial v}{\partial p}.
\]

State-dependent friction is not a multiplier. It enters the physical stress
scale as

\[
\tau_{\rm eff,a}=\operatorname{sign}(\tau_a)
\max\left(|\tau_a|-\sum_r h_{ar}p_r,0\right),
\]

and velocity remains the positive-mobility EXP-floor glide law.

The bounded local screen spans four positive diffusivities and four positive
forest coefficients. Fifteen of sixteen combinations are damped over the
declared continuum interval. The sole positive case has its maximum exactly at
the shortest admitted wavelength, `2.184e-7 m`; extending the symbol exposes a
peak near `1.34e-8 m`, below the continuum validity scale. There is therefore
no robust interior finite-wavenumber branch.

Decision:

`MINIMAL_INSTANTANEOUS_JUNCTION_FRICTION_CLOSURE_REJECTED`

The next discriminating model is delayed junction memory together with at
least a second line-orientation moment. No nonlinear wall simulation and no
HPC submission are justified from this closure. Wall, DRX, ASB, integrated
mechanism, and predictive-validation flags remain false.

## Delayed-memory escalation

The next linear operator adds one memory state per junction,

\[
\dot m_r=(p_r-m_r)/\tau_r,
\]

and makes the same physical forest resistance respond to `m_r`. No spatial
length is introduced. A 324-case screen spans three temperatures, three
stresses, three positive forest strengths, three positive diffusivities, and
four memory times. Twenty-seven cases have a positive interior continuum mode,
with wavelengths from `2.75e-7` to `1.02e-6 m` and growth rates from about
`4` to `101 s^-1`. They span all three temperatures, two stresses, multiple
diffusivities, and multiple relaxation times, but occur only at the largest
screened forest coefficient, `1e-6 Pa m2`.

This identifies a bounded finite-mode region; it does not yet establish a
robust mechanism. The forest-strength threshold and the fixed-alignment
restriction are now the main vulnerabilities. The justified next action is a
compact nonlinear falsification with common physical noise, grid/domain
refinement, and full balance checks, alongside replacement of the shared
tangent by a second line-orientation moment.

The memory result `output/v3_memory_dispersion.json` was regenerated against
source `6260a4b`; SHA-256
`273696218d058455852466df189be6fa38162a1210b9ac902674aad099957942`.

Local evidence: 237 canonical tests pass. The instantaneous machine result
`output/v3_topology_dispersion.json` records separate fixture and scientific
fields against source `3f4712c`. Its SHA-256 is
`cbb4103631d291eae783e5bf0a303a27d6b1098205585c7c82d9ec76a3a931e0`.
