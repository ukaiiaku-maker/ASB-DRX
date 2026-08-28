# Analytical EXP-floor strength law and collective-transition hypothesis

Status: governing-equation baseline, 2026-08-28. Legacy programs and data are context only and do not calibrate or validate this derivation. Dislocation dynamics is not a parameter source.

## 1. EXP-floor activation barrier

Use the local resolved activation stress `tau >= 0` and

`G(tau,T) = G0(T) [ f + (1-f) exp(-a (tau/tau_c(T))^n) ]`,

with `G0>0`, `tau_c>0`, `a>0`, `n>0`, and `0<=f<1`. Thus `G(0,T)=G0(T)` and `G(infinity,T)=f G0(T)`. The floor is physical only if the high-stress residual barrier is supported by the selected target; it is not a numerical clamp.

The activation volume is positive:

`V*(tau,T) = -dG/dtau = [G0(1-f) a n/tau_c] (tau/tau_c)^(n-1) exp[-a(tau/tau_c)^n]`.

Temperature laws for `G0(T)` and `tau_c(T)` remain explicit, differentiable functions. Exponential reference laws are a convenient first candidate,

`G0(T)=G_ref exp[-g_T (T-T_ref)/T_ref]`,

`tau_c(T)=tau_ref exp[-s_T (T-T_ref)/T_ref]`,

but their parameters must be optimized and tested against an authoritative strength dataset rather than inherited from old files.

## 2. Independent-node rate law

Let `X=(2 rho)^(-1/2)` be a forest-spacing scale and `q=b/X=b sqrt(2 rho)`. The analytical baseline is

`gamma_dot = eta0 q^p exp[-G(tau,T)/(k_B T)]`,

`sigma = q tau`,

where the second equation is the Taylor stress-concentration mapping used to define the macroscopic strength curve. This is a dissipative kinetic law, not a free-energy density and not a DRX criterion.

At imposed `gamma_dot`, define the normalized required barrier

`h(q,T,gamma_dot) = (k_B T/G0) [ln(eta0/gamma_dot) + p ln q]`.

For `f<h<1`, inversion of the EXP-floor barrier gives

`y=(h-f)/(1-f)`,

`tau=tau_c [-ln(y)/a]^(1/n)`,

`sigma(q)=q tau_c [-ln(y)/a]^(1/n)`.

The boundary cases `h>=1` and `h<=f` are outside this interior inverse and must be reported as zero-stress/athermal-limit or residual-floor-limit cases, not silently capped.

## 3. Closed-form analytical peak

At fixed `T` and `gamma_dot`, stationarity of `sigma(q)` yields

`(h-f) [-ln((h-f)/(1-f))] = p k_B T/(n G0)`.

Define

`D(T) = p k_B T/[n G0(T)(1-f)]`.

An interior maximum exists only if `0 < D <= 1/e`. The maximum uses the principal Lambert branch:

`W = W_0(-D)`,

`y_peak = exp(W) = -D/W`, with `1/e <= y_peak < 1`,

`h_peak = f + (1-f)y_peak`,

`q_peak = exp{[ln(gamma_dot/eta0) + G0 h_peak/(k_B T)]/p}`,

`rho_peak = q_peak^2/(2 b^2)`,

`tau_peak = tau_c [-ln(y_peak)/a]^(1/n)`,

`sigma_peak = q_peak tau_peak`.

The `W_{-1}` solution is the lower-density stationary minimum, not the strength maximum. At fixed temperature,

`rho_peak proportional to gamma_dot^(2/p)` and `sigma_peak proportional to gamma_dot^(1/p)`.

These scaling relations are direct falsification tests. If measured peak rate sensitivity does not follow them over a proposed regime, the independent-node law or Taylor concentration mapping is incomplete there.

The peak has no thermodynamic implication by itself. In particular, `d sigma/d rho < 0` is not a storage-energy spinodal and cannot nucleate a crystallographic orientation.

## 4. Parameter optimization

Parameters are divided before fitting:

| Group | Quantities | Treatment |
|---|---|---|
| Known physical | `k_B`, crystal geometry; `b` after material selection | fixed with provenance |
| Barrier shape | `G_ref`, `tau_ref`, `f`, `a`, `n`, temperature coefficients | optimized against strength/rate/temperature data |
| Kinetic scale | `eta0`, `p` | jointly optimized but tested for strong compensation |
| Collective | stress-transfer kernel, threshold distribution, memory time, reset law | absent from baseline; introduced only if collective observables require it |
| PF/thermal | GB energy/mobility, heat capacity, conductivity, storage/recovery | calibrated from separate authoritative sources |

Fit raw stress observations, not pre-extracted peak labels alone where full curves are available. Use transformed bounded parameters, a robust likelihood with reported measurement uncertainty, multi-start global-to-local optimization, profile likelihood or posterior covariance, and held-out temperature/rate conditions. Penalize no data point by changing equations between conditions. Report parameter correlations and practical non-identifiability, especially `eta0`--`G0`, `a`--`tau_c`, and `f`--high-stress coverage.

The analytical peak provides inexpensive residuals and gradients for optimization. Numerical root finding is reserved for extensions that break the closed form.

## 5. Transparent nodes and a possible collective link

“Transparent” is interpreted here as a node that can be crossed/unzipped/sheared and then repins or transfers stress, rather than an indefinitely impenetrable obstacle. Transparency alone does not imply multi-hit behavior.

Let each node `i` have EXP-floor hazard

`lambda_i = nu_i exp[-G_i(tau_i + z_i,T)/(k_B T)]`,

and let an event at node `j` deliver a transient stress increment `Delta tau_ij(t)` through an elastic transfer kernel. Linearizing the induced event probability over a memory interval gives a branching matrix

`B_ij = integral_0^infinity lambda_i [exp(V_i* Delta tau_ij(t)/(k_B T))-1] dt`,

where `V_i*=-dG_i/dtau` is supplied analytically by the EXP-floor law. Its spectral radius

`R = spectral_radius(B)`

is a mathematically defined collective susceptibility. For `R<1`, the mean triggered cluster size of the linear branching approximation is proportional to `1/(1-R)`. `R -> 1` marks loss of validity of the independent-node approximation and requires nonlinear saturation, finite-system, and stress-conservation terms. It is not a grain-birth gate.

If a single transferred increment is subthreshold but several increments accumulated within the relaxation time can activate a node, multi-hit behavior becomes a first-passage problem for the shot-noise state

`z_dot_i = -z_i/tau_r + sum_events_j Delta tau_ij delta(t-t_j)`.

An integer Poisson-tail formula follows only after restrictive assumptions: identical increments, independent parent events, fixed memory window, and a sharp threshold. The production candidate should therefore evolve `z_i` or a reduced distribution of it and derive any effective hit order from the stress-transfer/threshold model. Density may enter through node spacing, coordination, barrier distribution, and transfer kernel; no arbitrary density switch is introduced.

## 6. Research assessment

The collective hypothesis is plausible but not yet established for the intended material:

- Xu and Picu found a stress- and temperature-dependent transition from individual unzipping to correlated jerky bypass, and concluded that higher moments of obstacle density matter: [Phys. Rev. B 76, 094112 (2007)](https://doi.org/10.1103/PhysRevB.76.094112).
- Sobie et al. found that large-scale effective bypass barriers differ from unit obstacle barriers because dislocation morphology and cooperative processes matter: [J. Mech. Phys. Solids 105, 150-160 (2017)](https://doi.org/10.1016/j.jmps.2017.05.003).
- Ovaska, Laurson, and Alava showed that pinning strength changes collective dislocation dynamics from jamming through depinning-like behavior to quenched activity: [Scientific Reports 5, 10580 (2015)](https://doi.org/10.1038/srep10580).
- Rizzardi, Derlet, and Maaß found fat-tailed intermittency while obstacles remain shearable and exponential scale-dependent behavior for incoherent obstacles: [Phys. Rev. Materials 6, 073602 (2022)](https://doi.org/10.1103/PhysRevMaterials.6.073602).
- Aissaoui et al. report that dislocation density controls avalanche cutoff and triggering-stress distributions, with `Delta gamma_max proportional to b/sqrt(rho)`: [Phys. Rev. Materials 10, 053603 (2026)](https://doi.org/10.1103/6xmp-9cvm).

These works motivate the branching/shot-noise research route, but none identifies its parameters for this campaign. The route must be compared against the independent baseline using macroscopic transients, burst statistics if available, and held-out conditions. If no target data discriminate it, the collective extension remains an uncertainty/ablation rather than production physics.

Complete single-glider DDD context was subsequently located in `/Users/sdillon/Taylor_DDD` and `/Users/sdillon/Taylor_DDD_arrhenius_native`; see `taylor_ddd_context_audit.md`. The persistent-contact event histories provide the missing structural objects needed to test the branching/shot-noise reduction: contact identity and residence, neighboring load-bearing lengths, signed force work, hazard accumulation, release/reset, and swept strain. They can falsify the reduction and determine whether scalar `R` is adequate, but they do not supply production parameters.

The first HPC3 structural test (`20260828T130729Z-1a147e0-710710`) found much stronger multi-hit clustering in the higher-density reduced history, but zero spectral-radius proxy for every sampled native one-step contact operator. Only 11 next-audit-step redistribution samples were available, all in the densest native case. Thus the available histories do not establish the causal feedback assumed by `R`; the independent EXP-floor law remains the baseline and the collective construction remains an unparameterized ablation.

## 7. Immediate verification gates

1. Symbolic identities and dimensions of the barrier, inverse, activation volume, and peak.
2. HPC3 numerical checks of forward/inverse closure, Lambert peak stationarity, branch selection, and rate scalings.
3. Identifiability study on synthetic observations generated from declared parameters; this tests the optimizer, not physical validity.
4. Selection and provenance of the real material/strength dataset.
5. Baseline fit and held-out validation before any collective or phase-field coupling.
