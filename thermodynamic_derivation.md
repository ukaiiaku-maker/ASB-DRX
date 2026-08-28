# Candidate thermodynamic and kinetic architecture

Status: first derivation for evidence review; no production solver is authorized yet.

## State and balances

All densities below are line length per volume, m^-2. Slip-system resolution is retained where mechanics/evidence requires it.

| State | Type and admissible range | Balance/evolution form |
|---|---|---|
| displacement `u` or elastic strain `eps_e` | constrained mechanical field | `rho_m u_ddot = div sigma` when inertia matters; otherwise `div sigma=0`, with `eps_e = sym grad u - eps_p - eps_th`. |
| slip `gamma^a` / plastic strain `eps_p` | accumulated nonconserved kinematic state | `gamma_dot^a = G^a(tau_eff^a,T,rho_m^a,rho_f^a,q_DD,...)`; `eps_p_dot=sum_a gamma_dot^a sym(s^a tensor n^a)`. |
| mobile signed densities `rho_m^{a+/-}` | nonnegative, transported/generated/annihilated | conservative transport plus multiplication, immobilization, remobilization, annihilation, and boundary flux; signed difference supplies GND consistency. |
| forest/immobile `rho_f^a` | nonnegative, generated/annihilated | storage from mobile activity and junctions minus recovery/remobilization. Not a mobile carrier. |
| organized wall/cell `rho_w` | nonnegative, nonlocally organized/generated/recovered | gradient/nonlocal flux plus organization source and dynamic/diffusive recovery. |
| signed GND/Nye content `alpha` | kinematically constrained/transported | derived from plastic distortion curl where resolved; transport and boundary residual ledger must preserve Burgers content. |
| GB residual content `rho_GB` or interfacial Burgers vector `b_GB` | interface-supported, generated/relaxed/transported with GB | balance of blocked/transmitted incident content, accommodation, migration transport, and relaxation. |
| phase fields `eta_i` | nonconserved, `0<=eta_i<=1`, partition/purity constraint | constrained Allen--Cahn gradient flow from the free energy; inactive labels excluded/retired. |
| crystallographic orientation `R` or minimal coordinates | nonconserved on orientation manifold | grain-interior anchoring plus GB/orientation gradient flow; new orientation requires finite support/provenance. |
| temperature `T` | positive, energy balance | `rho_mass c_p T_dot = div(k grad T)+q_pl-q_stored-q_other-q_bath` with boundary flux. |
| collective DD state `z` or phase probabilities `p_j` | normalized internal state, conditionally stochastic | validity-bounded renewal/phase-type master equation or locked event sampler. It produces completion flux/statistics, never a grain Boolean. |
| embryo objects, if Route B | finite set with positive size and enumerated status | interfacial/stored-energy driven growth/shrinkage plus stochastic attempt history; promoted only through persistent `eta` support. |

Reservoir identities remain distinct unless a later derivation and data justify reduction.

## Candidate Helmholtz free energy

For domain `Omega` and GB/interfacial support represented diffusely,

`F = integral_Omega [ f_el(eps_e,T,R) + f_th(T) + f_dis(rho_f,rho_w,alpha,T) + f_ord(eta,R,T) + f_grad(grad eta, grad R, grad rho_w, alpha,T) + f_cpl ] dV + F_emb`.

1. `f_el = 1/2 eps_e : C(T,R,eta) : eps_e` [J m^-3]. Variation gives stress and elastic driving forces.
2. `f_th` is a thermodynamically consistent thermal reference whose derivative gives entropy/heat capacity; it is not an arbitrary temperature polynomial.
3. A baseline dislocation energy is `sum_r A_r(T) rho_r`, with `A_r ~ c_r mu(T)b^2` [J m^-1]. Logarithmic screening/core corrections may be used only over calibrated ranges. Mobile-carrier dissipation is not double-counted as stored energy.
4. Any nonconvex wall/organization term must derive from competition between local storage/entropy and an independently motivated interaction/ordering energy. A provisional generic form is `f_w = A_w rho_w + k_B T rho_w[ln(rho_w/rho_ref)-1] + f_int(rho_w,T)` plus `kappa_w |grad rho_w|^2/2`. Its Hessian must be plotted over the full calibrated domain. A negative eigenvalue denotes density organization only, not crystallographic DRX.
5. `f_ord` is a multi-order-parameter grain energy with minima at valid grain states and penalties for mixed interiors. `f_grad` sets GB energy and width; coefficients are calibrated jointly to measured/literature GB energy/mobility and resolved by the mesh.
6. Orientation energy lives on the crystal-symmetry quotient and reproduces a calibrated misorientation-dependent boundary energy (e.g. low-angle Read--Shockley limit with high-angle saturation). It must not allow density cleanup with zero misorientation to count as a new grain.
7. Couplings such as lowered dislocation storage inside a recrystallized phase enter through explicit energetic terms and variational derivatives, with the released energy routed to heat/interface work/content ledgers. No instantaneous low-density reset is admissible.
8. `F_emb`, if explicit embryos are selected, is the same bulk/interface energy evaluated for finite geometry, not a second ad hoc barrier. For an isolated circular 2-D nucleus per unit depth, `Delta F(R)=2 pi R gamma_gb - pi R^2 Delta f`, `R_c=gamma_gb/Delta f`; signs and rates become verification targets.
9. The locked collective state receives an energetic term only if DD/literature establishes a stored configurational energy and conjugate force. Otherwise it is a dissipative kinetic internal state.

The Arrhenius--Taylor flow stress `sigma_AT(rho,T,edot)` is excluded from `F`. A plastic-work integral would require a precise strain-like conjugate variable and path-independent potential; those conditions are not established and plastic flow is dissipative/path dependent.

## Variations and dissipation

- `sigma = partial f_el / partial eps_e`.
- Phase/orientation chemical forces are variational derivatives `mu_eta_i=delta F/delta eta_i`, `mu_R=delta F/delta R` with constraints projected appropriately.
- An organization chemical potential is `mu_w=partial f_dis/partial rho_w - div(partial f_grad/partial grad rho_w)`.
- Allen--Cahn/Onsager laws use positive-semidefinite mobilities: `eta_dot=-L_eta P(mu_eta)`, orientation rate `=-L_R P_R(mu_R)`, and conserved organization flux `J_w=-M_w grad mu_w` where conservation applies.
- Plastic flow, defect reactions, GB transmission, recovery, and collective transitions may be explicit constitutive kinetics, but each must satisfy nonnegative dissipation after accounting for stored-energy changes.
- The thermal source is the residual of mechanical power after every recorded stored/interface/residual/accommodation channel, rather than an independently tuned Taylor--Quinney factor that double counts storage.

An unloaded, isothermal, closed relaxation discretization must satisfy `F^{n+1} <= F^n + tolerance`. A convex-splitting, discrete-gradient, or accepted-step energy check will be selected before production.

## DD closure alternatives

### A. Conditional first-moment surface

A table or uncertainty-aware surrogate maps all DD-supported covariates `x=(rho components,T,tau,edot,length,microstructure,...)` to mean completion intensity and amplitude statistics. It is transparent and minimal but cannot reproduce non-exponential waiting times, over/under-dispersion, or serial correlations unless those observables are irrelevant on continuum timescales. Queries outside the convex validity envelope stop or warn explicitly.

### B. Stateful renewal/phase-type closure

A condition-dependent semi-Markov or phase-type state `p_0...p_K` advances through hit/relaxation phases with rates fitted only to DD event histories. The completion flux is a transition out of terminal phases, after which the state resets according to the fitted renewal law. Low-density exponential/Poisson behavior and coordinated multi-hit behavior must emerge from fitted transition structure, not a density switch. This can preserve age, correlation time, dispersion, and restart state.

Selection rule: fit both on common calibration conditions; choose A only if held-out mean, survival/hazard, CV/Fano, amplitude, and correlations relevant to PF are statistically adequate. Otherwise choose the smallest B model supported by likelihood/information criteria and held-out diagnostics. PF results never participate in this selection.

## Candidate couplings (hypotheses, not gates)

Each coupling must have independent evidence, limiting behavior, and an ablation:

- collective completion flux changes immobilization/storage partition;
- collective state drives wall/GND organization flux or correlation length;
- event amplitudes create intermittent plastic power while preserving the mean/work ledger;
- correlation length informs a nonlocal stress/organization kernel;
- collective stress redistribution modifies resolved slip/GB loading;
- finite-amplitude thermal/structural fluctuations alter embryo sampling frequency, without direct birth.

At zero coupling, the locked DD state remains statistically correct and the continuum reduces to its baseline material-point/phase-field limits.

## DRX routes to compare

### Route A: order-parameter finite fluctuations

Use a sufficiently expressive orientation/order-parameter basis and thermodynamically consistent stochastic forcing whose fluctuation amplitude is independently calibrated. Determine whether finite support of a new orientation crosses the interfacial barrier and survives. Do not create a label through a topology split.

### Route B: explicit physical embryos

Persist ID, position/support/shape, trial orientation and misorientation, parent/lineage, birth time/strain/age, stored-energy relief, interfacial energy, total `Delta F`, barrier/attempt history, integrated viability, growth rate, status, and RNG lineage. Couple the object into phase fields continuously; promotion is the appearance and survival of physical order-parameter support. The isolated circular-nucleus limit, complete content/energy ledger, collision/overlap behavior, checkpointing, and retirement are mandatory.

## Timescale and stability work before 2-D sweeps

Compute loading, elementary-event, renewal completion/correlation, storage/organization, dynamic/diffusive recovery, embryo nucleation/growth, GB migration, thermal diffusion/bath, localization growth, and elastic-wave times. Compare `L/c_s` with all evolution times to select quasi-static versus inertial mechanics. Linearize homogeneous mechanics/plastic/storage/collective/thermal equations and calculate the finite-wavenumber Jacobian/dispersion relation. Separate structural eigenmodes from thermal localization modes and use them to choose smoke/ladder conditions, not to tune regime labels.
