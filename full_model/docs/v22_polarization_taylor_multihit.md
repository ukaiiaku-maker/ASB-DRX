# V22 polarization-gated wall, Taylor resistance, and multi-hit operator

## Claim boundary

The frozen V21 evidence supports only
`V21_CURRENT_OPERATOR_DOMINATED_BY_UNIFORM_ORDERING_AND_HAS_NO_SIGNED_FINITE_MODE`
and `INTRAGRANULAR_DRX_MECHANISM_UNRESOLVED`. The V21 moving-front result is
`SIBM_NUMERICAL_SIGN_REGRESSION_PASSED`; it is not physical SIBM evidence.

## State and units

For each BCC Burgers family (a), the common state carries nonnegative signed
mobile, forest, and wall line densities
(ho^{\pm}_{m,a},\rho^{\pm}_{f,a},\rho^{\pm}_{w,a}) in m\(^{-2}\), explicit
junction extents (j_c) in m\(^{-2}\), dimensionless wall order (q), and a
dimensionless coordination state (c_{\rm mh}\in[0,1]). Slip, plastic
distortion, alignment, family Nye tensors, orientation, and temperature retain
the V21 definitions. Stress is Pa, velocity is m s\(^{-1}\), energy density is
J m\(^{-3}\), and each line reaction rate is m\(^{-2}\) s\(^{-1}\).

The objective signed wall tensor and polarization are

\[
 \alpha_w=\sum_a(\rho^+_{w,a}-\rho^-_{w,a})\,b_a\otimes l_a,
 \qquad
 \Pi_w={\|\alpha_w\|\over b\rho_w+\epsilon_\rho},
 \qquad \rho_w=\sum_a(\rho^+_{w,a}+\rho^-_{w,a}).
\]

Both (b_a) and (l_a) rotate with the lattice. No orientation-derived wall
target is used.

## Compared gates and selected functional

Two bounded gates are implemented and tested:

\[
 g_{\rm product}={\rho_w\over\rho_w+\rho_{1/2}}
 {\Pi_w^2\over\Pi_w^2+\Pi_{1/2}^2},
\]

and

\[
 g_{\rm joint}={z^2\over z^2+z_{1/2}^2},\qquad
 z={\rho_w\over\rho_{\rm ref}}\Pi_w.
\]

The smaller joint-rational form is the default. Both pass the absent-wall,
dense sign-balanced tangle, polarity-reversal, and rigid-frame controls. With
the physical-wall-initiated interpolation \(h(q)=q(2-q)\), the local
order/partition energy is

\[
 f_q=Wq^2(1-q)^2+C_0(1-g)q^2-A_0gh(q)
 +{C_p\over2\rho_*}[\rho_w-g h(q)\rho_*]^2.
\]

The gradient term is \(\kappa_q|\nabla q|^2/2\). At \(g=0\),
\(\partial^2f_q/\partial q^2|_0=2(W+C_0)>0\), independently of total density.
At sufficient signed polarization, \(h'(0)=2\) lets the physical gate—not
order-field noise—initiate ordering, and the negative \(A_0g h\) branch can
make an ordered wall favorable. At \(g=0\) this source vanishes exactly. The
non-ordering defect branch remains
\(E_l\rho+C_\rho\rho\ln(\rho/\rho_{\rm ref})\), whose second derivative is
\(C_\rho/\rho>0\). All wall chemical potentials include the exact derivative
of \(g\) with respect to each signed family population.

## Arrhenius--Taylor glide

The objective family resistance is

\[
 \tau_{T,a}=\alpha_T\mu b\sqrt{\rho_{f,a}+c_w\rho_{w,a}
 +\sum_c h_{ac}m_cj_c}.
\]

The differentiable friction map used by the same nonlinear residual and JVP is

\[
 \tau_{\rm eff}=\tau\left[1-{\tau_T\over
 \sqrt{\tau^2+\tau_T^2+\delta_T^2}}\right].
\]

It is odd, is zero at zero drive, never reverses the drive, and reduces exactly
to \(\tau\) when \(\alpha_T=0\). The EXP-floor activation enthalpy is evaluated
at \(|\tau_{\rm eff}|\), and signed velocity uses
\(\tanh(\tau_{\rm eff}/\tau_c)\). Thus both
\(\tau\dot\gamma\ge0\) and \(\tau_{\rm eff}\dot\gamma\ge0\). Raw mechanical
power supplies the energy ledger; Taylor friction contributes to heat rather
than a hidden stored-energy reservoir.

## Multi-hit ablation

The disabled state is an exact regression. When enabled,

\[
 \dot c_{\rm mh}=(1-c_{\rm mh})\Gamma_{\rm coll}
 -c_{\rm mh}/\tau_{\rm mh},\qquad
 \Gamma_{\rm coll}=s_{\rm coll}{\sum_c\Gamma^{\rm turnover}_c\over\rho_{\rm ref}}.
\]

Each selected competing rate is multiplied by
(\exp(\lambda_i c_{\rm mh})), with (|\lambda_i|\le5). The baseline values
favor wall/junction reactions and reduce annihilation, but are an explicitly
labeled ablation rather than DD calibration. The same global accepted-step
limiter preserves (0\le c_{\rm mh}\le1); no term changes the Burgers or line
stoichiometry.

Every reversible reservoir exchange uses the attempt-bounded split

\[
 k_+=k_A[1-\tanh(\Delta\mu/2E_r)],\qquad
 k_-=k_A[1+\tanh(\Delta\mu/2E_r)].
\]

Thus \(0\le k_\pm\le2k_A\) and
\(k_+/k_-=\exp(-\Delta\mu/E_r)\) exactly. This replaces the V21 unbounded
symmetric-exponential factor that stalled an exhausted reservoir while
preserving the thermodynamic bias, EXP-floor attempt frequency, and reaction
stoichiometry.

## Finite-time tangent ownership

Instantaneous symbols differentiate the common residual. Finite-time analysis
instead differentiates the complete accepted-step map, including its global
positivity/bound active set, by centered JVP. Products of these accepted-map
tangents along the actually advanced trajectory yield propagator singular
values. Signed gain is projected onto normalized plus/minus differences of the
mobile, forest, and wall reservoirs; the scalar (q) coordinate is excluded.
Centered boundary derivatives are evaluated at a declared small interior
audit state \(q=c_{\rm mh}=10^{-4}\); exact-zero stability is tested directly
from the Hessian and nonlinear residual.

Wall order uses a directional bound-degenerate Onsager mobility. If the raw
Allen--Cahn rate is positive it is multiplied by \(1-q\); if negative it is
multiplied by \(q\). This retains
\(\dot q\,\delta F/\delta q\le0\), permits a physical polarized-wall source at
\(q=0\), and makes outward motion vanish continuously at either bound. It
prevents a roundoff-scale order pixel from producing a Zeno timestep without
clipping an interior thermodynamic force. Coordination uses its analytic
invariant-interval kinetics plus the \(10^{-12}\) numerical active set.

## Current qualification boundary

Local thermodynamic, dissipation, balance, symmetry, JVP, and bitwise restart
fixtures pass. These fixtures authorize a long nonlinear campaign but do not
constitute a scientific DRX result. Grain/phase allocation remains disabled.

## Long-campaign decision and rejected threshold candidate

The completed 64/128-grid, 10/12.7-micrometre campaign returned nearly uniform
order in every ordering-enabled branch, with no one-degree Frank--Bilby wall
candidate. The order-disabled control remained exactly at zero. The V22
decision is therefore `WALL_ORDER_FUNCTIONAL_STILL_UNPHYSICAL`; this rejects
the present ordering handoff, not intragranular DRX in general.

Replacing only \(h(q)=q(2-q)\) by \(h(q)=q^2(3-2q)\) was tested and rejected.
Exact \(q=0\) became invariant, but a fixed \(10^{-6}\) perturbation still
bootstrapped to nearly uniform order by four-percent strain without a physical
wall. That candidate was not retained in production. A subsequent repair must
threshold the physical gate itself or introduce a thermodynamically ledgered
physical wall-formation trigger; changing mobility or relying on exact-zero
initialization is not acceptable.
