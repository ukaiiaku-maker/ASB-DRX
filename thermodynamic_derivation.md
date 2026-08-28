# Candidate thermodynamic and kinetic architecture

Status: first derivation plus executable verification kernel; no production solver is authorized yet.

The material-agnostic kernel in `src/asb_drx/thermodynamics.py` implements the two-state grain-energy sign convention, its discrete variational derivative, an energy-checked periodic Allen--Cahn step, distinct conservative dislocation reservoirs, an exact incremental work ledger, and the sharp-interface circular-nucleus limit. Generic fixture values test dimensions and invariants only. They are not production coefficients and cannot be inherited by a later material configuration.

For the symmetric local barrier `W eta^2(1-eta)^2` and gradient penalty `kappa |grad eta|^2/2`, the planar diffuse-interface energy is `gamma=sqrt(2 kappa W)/6`. The 2-D verification uses this derived `gamma` in `R_c=gamma/Delta f`, initializes diffuse circular support on a periodic square grid, and preregisters shrink/grow signs plus final grid/timestep changes below 5%. Checkpoint/restart covers every variable in this deliberately limited deterministic kernel (`eta`, time, and accepted-step count); it does not satisfy the later full coupled restart requirement.

## State and balances

All densities below are line length per volume, m^-2. Slip-system resolution is retained where mechanics/evidence requires it.

| State | Type and admissible range | Balance/evolution form |
|---|---|---|
| displacement `u` or elastic strain `eps_e` | constrained mechanical field | `rho_m u_ddot = div sigma` when inertia matters; otherwise `div sigma=0`, with `eps_e = sym grad u - eps_p - eps_th`. |
| slip `gamma^a` / plastic strain `eps_p` | accumulated nonconserved kinematic state | `gamma_dot^a = G^a(tau_eff^a,T,rho_m^a,rho_f^a,z,...)`; the baseline uses the EXP-floor analytical law; `eps_p_dot=sum_a gamma_dot^a sym(s^a tensor n^a)`. |
| mobile signed densities `rho_m^{a+/-}` | nonnegative, transported/generated/annihilated | conservative transport plus multiplication, immobilization, remobilization, annihilation, and boundary flux; signed difference supplies GND consistency. |
| forest/immobile `rho_f^a` | nonnegative, generated/annihilated | storage from mobile activity and junctions minus recovery/remobilization. Not a mobile carrier. |
| organized wall/cell `rho_w` | nonnegative, nonlocally organized/generated/recovered | gradient/nonlocal flux plus organization source and dynamic/diffusive recovery. |
| signed GND/Nye content `alpha` | kinematically constrained/transported | derived from plastic distortion curl where resolved; transport and boundary residual ledger must preserve Burgers content. |
| GB residual content `rho_GB` or interfacial Burgers vector `b_GB` | interface-supported, generated/relaxed/transported with GB | balance of blocked/transmitted incident content, accommodation, migration transport, and relaxation. |
| phase fields `eta_i` | nonconserved, `0<=eta_i<=1`, partition/purity constraint | constrained Allen--Cahn gradient flow from the free energy; inactive labels excluded/retired. |
| crystallographic orientation `R` or minimal coordinates | nonconserved on orientation manifold | grain-interior anchoring plus GB/orientation gradient flow; new orientation requires finite support/provenance. |
| temperature `T` | positive, energy balance | `rho_mass c_p T_dot = div(k grad T)+q_pl-q_stored-q_other-q_bath` with boundary flux. |
| collective stress memory `z`, optional | stress-like dissipative internal state | relaxing shot noise driven by event-to-event elastic stress transfer. It is absent from the independent baseline and never creates a grain Boolean. |
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
9. The optional collective stress-memory state receives an energetic term only if an independent derivation establishes stored configurational energy and a conjugate force. Otherwise it is a dissipative kinetic internal state.

The Arrhenius--Taylor flow stress `sigma_AT(rho,T,edot)` is excluded from `F`. A plastic-work integral would require a precise strain-like conjugate variable and path-independent potential; those conditions are not established and plastic flow is dissipative/path dependent.

## Variations and dissipation

- `sigma = partial f_el / partial eps_e`.
- Phase/orientation chemical forces are variational derivatives `mu_eta_i=delta F/delta eta_i`, `mu_R=delta F/delta R` with constraints projected appropriately.
- An organization chemical potential is `mu_w=partial f_dis/partial rho_w - div(partial f_grad/partial grad rho_w)`.
- Allen--Cahn/Onsager laws use positive-semidefinite mobilities: `eta_dot=-L_eta P(mu_eta)`, orientation rate `=-L_R P_R(mu_R)`, and conserved organization flux `J_w=-M_w grad mu_w` where conservation applies.
- Plastic flow, defect reactions, GB transmission, recovery, and collective transitions may be explicit constitutive kinetics, but each must satisfy nonnegative dissipation after accounting for stored-energy changes.
- The thermal source is the residual of mechanical power after every recorded stored/interface/residual/accommodation channel, rather than an independently tuned Taylor--Quinney factor that double counts storage.

An unloaded, isothermal, closed relaxation discretization must satisfy `F^{n+1} <= F^n + tolerance`. A convex-splitting, discrete-gradient, or accepted-step energy check will be selected before production.

For the homogeneous finite-loading verification, `Delta tau=G(Delta gamma-Delta gamma_p)`. Using the increment-average stress makes external work minus the exact elastic-energy change equal the plastic work. The executable material point partitions that residual into the separately computed line-energy increase and heat; it has no fitted Taylor--Quinney factor. Forest storage is a visible provisional source and a step is rejected if it requests more stored energy than plastic work. Recovery, wall organization, multiple slip, thermoelasticity, conduction, and spatial equilibrium are intentionally absent from this limit and remain later coupled gates.

## Independent and collective kinetic alternatives

### A. Independent EXP-floor baseline

Use the analytical barrier, activated rate, inverse, and Lambert-W peak in `analytical_strength_derivation.md`. Optimize its parameters only against an authoritative material strength/rate/temperature dataset. This is the production default unless discriminating observations reject it.

### B. Stress-transfer branching and multi-hit memory

Let an event at node `j` transfer a transient stress `Delta tau_ij(t)` to node `i`. The EXP-floor activation volume supplies the linear response in a branching matrix, while a relaxing shot-noise stress `z_i` represents subthreshold increments accumulating into a multi-hit first-passage event. Density acts only through derived spacing, connectivity, threshold distributions, and the transfer kernel.

Selection rule: retain A unless held-out transient, burst, or correlation observables relevant to the continuum reject it and favor B after complexity is penalized. Phase-field outcomes never select the kinetic alternative.

## Candidate couplings (hypotheses, not gates)

Each coupling must have independent evidence, limiting behavior, and an ablation:

- collective completion flux changes immobilization/storage partition;
- collective state drives wall/GND organization flux or correlation length;
- event amplitudes create intermittent plastic power while preserving the mean/work ledger;
- correlation length informs a nonlocal stress/organization kernel;
- collective stress redistribution modifies resolved slip/GB loading;
- finite-amplitude thermal/structural fluctuations alter embryo sampling frequency, without direct birth.

At zero collective coupling, the model reduces exactly to the analytical independent-node material-point and phase-field limits.

## DRX routes to compare

### Route A: order-parameter finite fluctuations

Use a sufficiently expressive orientation/order-parameter basis and thermodynamically consistent stochastic forcing whose fluctuation amplitude is independently calibrated. Determine whether finite support of a new orientation crosses the interfacial barrier and survives. Do not create a label through a topology split.

### Route B: explicit physical embryos

Persist ID, position/support/shape, trial orientation and misorientation, parent/lineage, birth time/strain/age, stored-energy relief, interfacial energy, total `Delta F`, barrier/attempt history, integrated viability, growth rate, status, and RNG lineage. Couple the object into phase fields continuously; promotion is the appearance and survival of physical order-parameter support. The isolated circular-nucleus limit, complete content/energy ledger, collision/overlap behavior, checkpointing, and retirement are mandatory.

## Timescale and stability work before 2-D sweeps

Compute loading, elementary-event, renewal completion/correlation, storage/organization, dynamic/diffusive recovery, embryo nucleation/growth, GB migration, thermal diffusion/bath, localization growth, and elastic-wave times. Compare `L/c_s` with all evolution times to select quasi-static versus inertial mechanics. Linearize homogeneous mechanics/plastic/storage/collective/thermal equations and calculate the finite-wavenumber Jacobian/dispersion relation. Separate structural eigenmodes from thermal localization modes and use them to choose smoke/ladder conditions, not to tune regime labels.
