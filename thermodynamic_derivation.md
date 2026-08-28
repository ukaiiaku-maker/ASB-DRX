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

The next spatial limit is a periodic 1-D shear layer under common shear stress, which is the quasistatic force-balance solution for simple shear without body force. Local EXP-floor rates depend on local temperature and forest density; the volume-average plastic rate drives the finite-loading stress update. Local plastic work is partitioned before a conservative periodic heat-diffusion update. The explicit diffusion step enforces `alpha Delta t/Delta x^2 <= 1/2`; positivity alone is not an adequate stability test. A homogeneous layer must reduce to the material point, and both mechanical and mean thermal ledgers must close. This model can test thermal-feedback structure, but no localization it produces is physical ASB until properties, boundaries, length scales, perturbations, and convergence are independently validated.

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

The material-agnostic classifier in `src/asb_drx/grains.py` now enforces a deliberately narrower verified subset of that contract. A label has resolved support only where it is both the dominant order parameter and above a purity threshold, its support exceeds an explicitly supplied area criterion, and the support is one four-connected component on the periodic grid. A root becomes active only after the persistence interval. A descendant becomes a promoted recrystallized grain only after the same persistence interval, valid parent-prefixed lineage, and a minimum misorientation evaluated modulo an explicitly supplied scalar symmetry order. Loss of resolved support removes it immediately from physical counts and retires its immutable provenance after a grace interval. Rejected and retired records cannot silently reactivate.

This classifier is measurement infrastructure, not nucleation physics. Its area, purity, persistence, misorientation, and symmetry settings must ultimately derive from the resolved diffuse-interface scale, temporal convergence, and the selected material/crystal symmetry. The generic verification fixture values cannot be transferred to a production material. Full orientation-manifold dynamics, stochastic trial generation, energetic acceptance, collisions, and production multi-order-parameter evolution remain open.

### Constrained multi-order verification kernel

The next isolated kernel represents `N` allocated grain fields on the pointwise simplex `eta_i >= 0`, `sum_i eta_i = 1` with

`F = integral [ W sum_(i<j) eta_i^2 eta_j^2 + sum_i g_i h(eta_i) + (kappa/2) sum_i |grad eta_i|^2 ] dV`,

where `h(eta)=eta^3(10-15 eta+6 eta^2)`. The endpoint-flat interpolation is essential: a lower bulk energy for an already allocated child must drive an interface but must not spontaneously create that child from an exactly pure parent. The unconstrained chemical potentials are

`mu_i = 2 W eta_i sum_(j != i) eta_j^2 + g_i h'(eta_i) - kappa laplacian(eta_i)`.

The Onsager update uses the tangent projection `mu_i - mean_j(mu_j)`, followed only by roundoff-level simplex normalization and an energy/nonnegativity acceptance gate. For a planar boundary between two equal-bulk phases, reduction to one independent field gives `gamma = sqrt(kappa W)/3`. If phase 1 is lower by `Delta f > 0`, the two-dimensional circular sharp-interface limit is `R_c=gamma/Delta f` and `R_dot=M_eff(Delta f-gamma/R)`. The generic tests must recover the sign on each side of `R_c`, pointwise simplex conservation, energy decrease, label-permutation symmetry, pure-parent invariance, exact restart, and tracker coupling with fixed label count.

This construction still omits crystallographic boundary-energy anisotropy, a full orientation quotient/manifold, triple-junction calibration, elastic coherency, evolving stored-energy coupling, stochastic candidate creation, and collision/coalescence rules. Passing it establishes a constrained variational baseline only.

### Explicit dislocation stored-energy coupling

The next coupling removes the arbitrary binary bulk offset by assigning each allocated grain a fixed, explicit dislocation density `rho_i` and stored line energy `e_line`, so `g_i=e_line rho_i`. The child driving energy is therefore

`Delta f_(parent->child) = e_line (rho_parent-rho_child)`.

For the binary simplex, `h(eta_parent)+h(eta_child)=1`, so adding the same density to both grains changes the energy reference but not the projected dynamics. This common-offset invariance is a required test. The densities remain provenance/configuration entries during a phase step: boundary motion continuously changes the volume-weighted stored energy rather than applying an instantaneous density reset.

For each accepted phase step, split the total free energy per out-of-plane depth as `F=E_stored+E_interface/order`. With `Delta F<=0`, the isolated isochoric ledger is

`Delta E_stored + Delta E_interface/order + Q = 0`, `Q=-Delta F >= 0`.

The dissipated free energy is routed to temperature through `Delta T=(Q/A)/(rho_m c_p)`, where the supplied volumetric heat capacity is written directly as `rho_m c_p`. This is a closed relaxation ledger only. It does not yet include external mechanical work during the same step, evolving intragranular reservoirs, recovery/annihilation products, latent heat, conduction, or temperature-dependent mobility. Those channels must be combined without double counting in the coupled solver.

### Candidate-admission barrier (distinct from EXP-floor slip)

Candidate admission uses the classical interfacial/stored-energy competition only as an auditable baseline. It is not the EXP-floor barrier governing dislocation escape. For a cylindrical embryo of represented thickness `t`, boundary energy `gamma`, and stored-energy driving `Delta f>0`,

`Delta G(R)=t(2 pi R gamma - pi R^2 Delta f)`,

`R_c=gamma/Delta f`, `Delta G*=pi t gamma^2/Delta f`, and the excess energy returns to zero at `2 R_c`. Given an independently supplied areal attempt rate `I0`, the Poisson probability over eligible area `A` and interval `Delta t` is

`p=1-exp[-I0 A Delta t exp(-Delta G*/(k_B T))]`.

The decision kernel receives, rather than generates, a uniform random draw. It reports distinct rejection reasons for sub-resolution support, subcritical radius, insufficient symmetry-reduced misorientation, and the thermal draw. It does not allocate an order-parameter field. This separation preserves RNG provenance and prevents a candidate bookkeeping event from becoming a physical grain by definition.

Neither `I0` nor the represented thickness may be fitted opportunistically to a desired DRX onset. Both require a material/geometry interpretation and independent evidence. A production stochastic formulation must also demonstrate mesh/time-step invariant event intensity, spatial eligibility, overlap handling, detailed energy accounting across barrier crossing, exact RNG restart, and held-out onset statistics.

### Shared thermomechanical/phase ledger

The first combined verification limit is a binary aggregate: mechanics and temperature are homogeneous, while the two order parameters occupy a periodic 2-D domain. At the beginning of a step, the phase fractions are `phi_i=<h(eta_i)>`; the binary identity `h(eta_0)+h(eta_1)=1` makes `sum_i phi_i=1`. Each grain uses the same common stress but its own forest density in the independent EXP-floor law. The mean plastic increment is `Delta gamma_p=sum_i phi_i Delta gamma_p_i`, giving the exact finite-loading update

`Delta tau=G(Delta gamma_applied-Delta gamma_p)`.

The mechanical substep raises each grain density by its explicit storage law, partitions the corresponding grain-weighted plastic work into stored line energy and mechanical heat, and updates the shared temperature. The phase substep then uses those updated densities as `g_i=e_line rho_i`; its free-energy loss supplies phase heat. If phase-energy acceptance reduces the timestep, the mechanical substep is discarded and recomputed at that same smaller interval.

Let `E_s` and `E_I` denote domain-averaged stored-dislocation and interface/order energies. The combined accepted step must satisfy

`W_ext = Delta E_elastic + Delta E_s + Delta E_I + Q_mechanical + Q_phase`,

`rho_m c_p Delta T = Q_mechanical + Q_phase`.

This direct initial-to-final ledger prevents the density-storage increment and the phase-consumed stored energy from being counted twice. Required limiting tests are exact reduction to the existing material point for a pure parent, exact reduction to stored-energy phase relaxation at zero stress/loading, preservation of a zero child field, one shared accepted time increment, and bitwise restart of all current coupled state.

The binary aggregate is not spatial ASB mechanics: it has no displacement field, local stress redistribution, conduction, heterogeneous temperature, recovery, multiple slip, or physical boundary conditions. It verifies coupling algebra before those mechanisms are introduced.

### Periodic 2-D common-stress thermo-phase limit

The next spatial limit places temperature `T(x)`, two forest-density fields `rho_i(x)`, effective plastic shear, and the two order parameters on the same periodic 2-D grid while retaining the quasistatic simple-shear result that shear stress is spatially uniform. Each cell's effective plastic increment is `sum_i h(eta_i) Delta gamma_p_i(tau,rho_i,T)`, and the domain mean closes the finite-loading stress update. Mechanical heat is local; periodic conduction uses the explicit two-dimensional Fourier bound `alpha Delta t/Delta x^2 <= 1/4`.

The phase chemical potentials use spatial stored energies `e_line rho_i(x) h(eta_i)`. After the mechanical density update, the projected phase step must lower the updated free energy. Its exact global free-energy decrement is routed to heat; the spatial heat shape is assigned in proportion to the local Onsager dissipation proxy `sum_i |P mu_i|^2` and normalized to preserve that exact total. This distribution is a declared numerical closure, not a calibrated microscopic heat-source law.

The same global work identity as the aggregate applies, and periodic conduction must conserve mean thermal energy. Required tests are reduction to the established common-stress shear layer for a uniform pure parent, conductive damping with mean conservation, global closure, zero-child invariance, and exact restart of every current spatial field.

This remains quasistatic common-stress mechanics. A physical ASB gate additionally requires boundary/loading realism, perturbation-spectrum controls, sustained localization relative to homogeneous/isothermal/phase-disabled controls, mesh-converged band width and onset, and displacement-resolved or otherwise justified mechanical equilibrium.

## Localization observables and acceptance rule

Let `q_ij = |dot(gamma_p,ij)|` on an `N`-cell 2-D grid. The plastic participation fraction is `f_q = (sum q_ij)^2/(N sum q_ij^2)`. It is one for homogeneous flow and decreases as plastic activity concentrates. Effective widths are inverse-participation widths of the two coordinate marginals, `w_x = dx (sum q_x)^2/sum(q_x^2)` and analogously for `w_y`; the reported band width is `min(w_x,w_y)`. Thermal evidence is the maximum local temperature excess relative to a matched nonlocalizing control, not relative to the initial temperature. Mechanical softening is measured against the running absolute stress peak.

An ASB candidate is accepted only when all four conditions hold for a declared number of consecutive accepted steps: concentrated plastic flow, positive temperature excess over the matched control, post-peak stress softening, and a finite width exceeding a declared multiple of the phase-field interface width. A refinement gate separately requires both onset and width to agree within a declared relative tolerance. The executable generic fixture uses `f_q <= 0.4`, temperature excess `>= 20 K`, softening `>= 0.1`, width `>= 3` interface widths, persistence of three steps, and 5% refinement agreement. These are explicit test thresholds, not calibrated material criteria.

## Timescale and stability work before 2-D sweeps

Compute loading, elementary-event, renewal completion/correlation, storage/organization, dynamic/diffusive recovery, embryo nucleation/growth, GB migration, thermal diffusion/bath, localization growth, and elastic-wave times. Compare `L/c_s` with all evolution times to select quasi-static versus inertial mechanics. Linearize homogeneous mechanics/plastic/storage/collective/thermal equations and calculate the finite-wavenumber Jacobian/dispersion relation. Separate structural eigenmodes from thermal localization modes and use them to choose smoke/ladder conditions, not to tune regime labels.
