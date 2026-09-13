# Decision log

## ADR-0001: Independent redevelopment

Status: accepted, 2026-08-27.

Use a new package and derive its state, energy, kinetics, and bookkeeping independently. Preserve v32--v34 as evidence/regressions. Reason: v34 candidate records and label counts cannot represent persistent physical embryos/grains, and the old Arrhenius--Taylor interpretation is rejected.

## ADR-0002: DD closure is blocked pending raw evidence

Status: accepted, 2026-08-27.

Do not fit `m(rho,T)` or reuse the v34 Poisson tail/domain-count closure as the new DD artifact. The located first-avalanche v6 code specifies a stochastic hazard/cascade simulation at a default 300 K, 2e4 s^-1, and one geometry, but no raw event datasets were found. A simulated closure cannot validate itself or establish the asserted transition.

## ADR-0003: Compare renewal memory against a first-moment table

Status: proposed.

Option A is a validity-bounded tabulated/surrogate conditional event intensity. Option B is a stateful phase-type/semi-Markov renewal closure whose internal phases reproduce waiting-time shape, dispersion, correlations, and completion flux. Select the smallest representation that passes held-out DD statistics; expect Option B only if higher-order observables reject a memoryless description.

## ADR-0004: DRX representation remains open

Status: proposed.

Route A asks whether stochastic finite-amplitude perturbations in a fixed orientation/order-parameter basis can produce persistent new support without label splitting. Route B uses explicit finite-amplitude embryo objects with a complete interfacial/stored-energy ledger and promotion only after order-parameter support develops. Select only after isolated-nucleus and free-energy-barrier analysis.

## ADR-0005: Campaign-specific HPC3 namespace

Status: accepted, 2026-08-27.

Use `/pub/sdillon1/codex-runs/asb-drx-independent` remotely and `hpc3-results/asb-drx-independent` locally. A worktree-local runner configuration overrides the unrelated parent campaign configuration without modifying it.

## ADR-0006: Gate 0 no-go; do not manufacture missing evidence

Status: accepted, 2026-08-27.

Stop before DD fitting and production implementation. The exhaustive project inventory, adjacent DDD repository search, and campaign remote-storage search did not locate raw DD event trajectories supporting the asserted Poisson-to-multi-hit transition. The material target is also ambiguous: Fe/BCC prose conflicts with Cr metadata, and the validation folders mix multiple alloys. Substituting synthetic avalanche outputs, choosing a material by filename majority, or tuning v34 candidate parameters would violate the immutable constraints.

## ADR-0007: Legacy controls are mechanism controls, not physical validators

Status: accepted, 2026-08-27.

Retain v32 as an ASB-like numerical regression, v33 as a false-grain structural negative control, and v34 as a zero-candidate bookkeeping failure. Only v32 reproduces its finite diagnostics to numerical precision. v33 reproduces label explosion with unchanged topology but not its exact birth count. v34 again produces no candidates at all and its detailed trajectory diverges. None establishes physical DRX or mesh-converged ASB, and no legacy parameter was tuned.

## ADR-0008: Supersede DD gating and legacy regression status

Status: accepted, 2026-08-28; supersedes ADR-0002, ADR-0003, ADR-0006, and the regression role in ADR-0007.

Dislocation dynamics will not parameterize the model. Legacy programs, values, and outputs are context only and do not gate the new model. The campaign proceeds from governing analytical equations; physical calibration still requires an authoritative target dataset.

## ADR-0009: EXP-floor analytical baseline

Status: accepted, 2026-08-28.

Use `G=G0(T)[f+(1-f)exp(-a(tau/tau_c(T))^n)]` with an independent-node activated rate law as the baseline. Its inverse and Lambert-W strength peak are defined in `analytical_strength_derivation.md`. The strength peak is a kinetic prediction, not a DRX trigger or free-energy instability.

## ADR-0010: Collective response is a derived ablation

Status: proposed, 2026-08-28.

If target observations require collective transparent-node behavior, derive it from a stress-transfer branching matrix and relaxing multi-hit shot-noise state. Do not introduce an arbitrary density threshold, prescribed hit order, or DD-fitted switch. Promote the extension only after it outperforms the independent baseline on held-out discriminating observations.

## ADR-0011: Use complete single-glider DDD for structural falsification only

Status: accepted, 2026-08-28.

The located `Taylor_DDD` persistent-contact simulations may test whether the proposed contact graph, elastic transfer, branching susceptibility, and multi-hit memory are structurally adequate. They do not set the EXP-floor or collective production parameters. The immutable source landmark is native ExaDiS commit `fb7610b`; result-file hashes and limitations are recorded in `taylor_ddd_context_audit.md`.

## ADR-0012: Do not promote a transparent-node multiplier

Status: accepted, 2026-08-28.

Primary literature supports history-dependent first/repeat forest crossings and shows that glissile or shearable junctions can remain hardening agents. It also shows that obstacle character changes collective-event statistics. These results motivate the existing conditioning/shot-noise ablation but do not identify its transfer kernel, reset time, or target-material parameters. No scalar multi-hit multiplier enters the baseline. Reconsider only with causal event parentage and held-out transient/burst observations that reject the independent EXP-floor law.

## ADR-0013: Reuse the single-glider DDD parameter set as a generic fixture

Status: accepted, 2026-08-28; narrows ADR-0008 and ADR-0011 by explicit user authorization.

Use the complete EXP-floor single-glider campaign's declared constants as one internally consistent, non-material parameter fixture. This does not make the DDD trajectory a calibration target or establish applicability to a materials class. The exact mapping retains `H=0.50 eV`, `S=-9 k_B`, `tau_c=14.5 GPa`, `f=0.20`, `a=6.65607`, `n=2.15276`, `eta0=1e12 s^-1`, `p=4`, `b=2.48e-10 m`, `G=80 GPa`, and the DDD geometry `q=2 b sqrt(rho)`. The PF and thermal constants remain declared generic fixtures.

The arbitrary regime boundary is the closed-form independent-law strength maximum `rho=rho_peak(T, rate)`. Densities above it are labeled only `post_peak_collective_candidate`; this is not a transparent-node, ASB, DRX, or material boundary. The DDD driver's `analytical_peak_density_m2=1e18` is excluded because inspection shows it is a hard-coded assignment, not an evaluation of the governing equations.

## ADR-0014: Do not launch the boundary array with frozen common stress

Status: accepted, 2026-08-28.

The preregistered single-job smoke at 950 K, `45000 s^-1`, and `rho/rho_peak=1` passes the coupled ledger and 16²/32² refinement check but remains essentially homogeneous despite `26.74 K` temperature excess over its matched control. This is expected from the current common-stress verification kernel, which lacks local elastic redistribution. Do not spend an array on a model structurally unable to establish the required localization mechanism. Add and verify local mechanics, including isolated limits, energy closure, restart, and refinement, before the sparse boundary campaign.
## ADR-0015: Adopt v2 Burgers-resolved physics addendum

Status: accepted, 2026-08-29; reclassifies the scalar implementation without
changing its verified baseline results.

- **Input:** `/Users/sdillon/DRX-ASB/CODEX_INDEPENDENT_DD_PF_DRX_ASB_CAMPAIGN_v2.md`,
  SHA-256 `37142ee8029b4f461cbdbfa326c58632a7d4fd2988ff951af3ccef5e0d9dc2da`.
- **Decision:** retain the integrated scalar antiplane model as a numerical and
  thermodynamic baseline only. Do not call its density bands walls or its phase
  labels crystallographic DRX.
- **Required successor:** Burgers/sign-resolved transported populations,
  plastic distortion and spin, physical orientation, Nye/GND content,
  boundary Burgers content, and Frank--Bilby-constrained recognition/dynamics.
- **DD decision:** existing completed single-glider outputs cannot identify a
  wall source or rotation law. Hit order and clustering alone cannot be used as
  surrogates. Gates G/H remain no-go.
- **Execution rule:** the user's later direct instruction permits small local
  verification; extended calculations continue on HPC3. Existing unrelated
  local and HPC3 jobs remain untouched.
- **Gate semantics:** passing an analytical fixture is not passing its
  scientific gate. Machine-readable outputs carry both fields separately.

## ADR-0016: Implement Gate A as a quarantined published Ta reference

Status: accepted, 2026-09-08.

Implement Bertin et al.'s four-Burgers-vector BCC pencil-glide material point
with its published Ta parameters solely to verify plastic-spin rotation,
attractor selection, family-density redistribution, and inactive-family
relaxation. Enforce the paper's temperature/rate envelope. Do not combine this
fit with the generic EXP-floor model or describe it as production calibration.
Gate A may pass on qualitative published orientation outcomes, exact restart,
mechanism ablations, and timestep refinement; lack of tabulated MD trajectories
must remain an explicit quantitative-validation limitation.

## ADR-0017: Use signed GND transport for the Gate B wall precursor

Status: superseded, 2026-09-11, by ADR-0018. Retained as the Gate B0 fixture
decision and not accepted as physical wall formation.

Use four positive/negative BCC Burgers-family population pairs and obtain
plastic slip from signed line flux. Define the Nye tensor only as the declared
curl of compatible plastic distortion. Adopt a reduced Groma-type conserved
signed-GND transport with local dry-friction drive, GND-gradient backstress,
long-range elastic penalty, and cubic finite-amplitude saturation. This gives
a documented finite-wavenumber dispersion relation without assigning a double
well to scalar total density or a negative Taylor slope.

The `2 micrometer` wavelength and all transport coefficients are generic
verification fixtures, not material calibration. The DD collective scale is
exactly zero. A passed Gate B state remains one crystal with no phase or label
allocation; polygonization and Frank--Bilby boundary recognition require later
gates.

## ADR-0018: Reclassify the prescribed signed-polarization mode as Gate B0

Status: accepted, 2026-09-11; supersedes the scientific interpretation of
ADR-0017 without altering commit `dbc45cb` or HPC3 job `55923138`.

The conserved potential contains a negative quadratic in signed polarization,
prescribes its fastest wavelength through `selected_wavelength_m`, fixes total
density by reconstruction, and evolves without applied stress, temperature,
MRSSP geometry, or Gate A feedback. It therefore verifies signed kinematics,
Nye compatibility, conservation, ETD finite-mode evolution, controls, restart,
and reproducibility only. Record `numerical_fixture_passed=true`,
`kinematic_and_balance_fixture_passed=true`, and
`scientific_gate_passed=false`.

Gate B1 must use separately transported density-weighted positive/negative
fluxes in a driven Gate-A-coupled state, have no unloaded instability, allow
independent total/signed density evolution, and predict rather than prescribe
spacing scaling. Polygonization remains blocked.

## ADR-0019: Reject the first Gate B1 nonlinear pattern as unconverged

Status: accepted, 2026-09-11.

Implement the Groma--Zaiser positive/negative flux sign structure with actual
Gate A stress, temperature, MRSSP geometry, orientation, density source, and
plastic-distortion feedback. Use Gate A's finite differential mobility above
the Taylor threshold; reject the singular secant-mobility trial. Keep pair
generation/removal and locked transfers separately ledgered and preserve net
Burgers content when a scalar Gate A sink exceeds the available minority sign.

The reduced flowing-state density operator predicts an emergent finite mode and
no wavelength parameter exists. The original nonlinear signal failed
strain-increment refinement and was rejected. A declared coupled internal
refresh limit corrects that splitting artifact, but the converged trajectory
remains homogeneous. After imposing common periodic resolved traction,
homogenizing only the exact Gate-A reservoir source, and restricting the
linearization to the Nye-compatible active-family manifold, every resolved
mode is damped. Leave `physical_CDD_wall_gate_passed=false`, do not submit HPC3,
and resolve the constitutive mismatch between Gate-A differential mobility and
the ideal dry-friction CDD instability before running the robustness matrix.
Gate C remains blocked.

## ADR-0020: Use the strict one-dimensional long-range projection for Gate B1

Status: accepted as the one-dimensional baseline; Gate B1 rejected by extended
verification, 2026-09-11.

For the declared `k_y=0` CDD wave vector, use the published `T(k)=0`
long-range factor. Retain the provisional cross-family `-i/k` kernel only as an
ablation: amplitudes from 0.05 through 0.25 suppress the full instability and
are not a derived one-dimensional baseline. This is a geometric reduction, not
coefficient tuning.

Use resolution-invariant random Fourier initial fields and delete the predicted
fastest mode. The full Nye-compatible active-family operator then selects
finite mode 6, while nonlinear 64/128 calculations independently select mode
10. Do not pass Gate B1 from these local results: require the single extended
128/256 job plus seed, domain, density-similitude, temperature/rate, wall-width,
balance, and unload controls. Gate C stays blocked until the machine result.

The single extended run `20260912T011600Z-2c0bf13-edff97` completed those
controls without retuning. The fixture and precursor checks pass, but the
128/256 observables are far outside the 5% limit and the selected mode remains
10 in all three incommensurate boxes, so wavelength scales with box length.
Record `fixture_passed=true`, `scientific_gate_passed=false`, and
`physical_CDD_wall_gate_passed=false`. Treat the observed structures as a
driven GND-rich precursor and diffuse-polarization response, not a converged
wall. Gate C remains blocked pending a constitutive mechanism that produces a
grid- and domain-independent length without fitting the phase-field outcome.

## ADR-0021: Reject a face-dissipation patch and redesign the Gate B1 flux fixture

Status: accepted, 2026-09-11.

The failed 256-point response has a near-Nyquist total-density peak, but
selected-mode linearizations at strains 0.005, 0.010, and 0.015 strongly damp
modes 32 through 127. Replacing averaged face velocities with a Rusanov flux
removes the odd/even algebraic null but becomes severely CFL-stiff and is not a
scientific correction; revert it. Separately, the domain test's fixed Fourier
mode count reproduces the same mode number in different boxes and therefore
does not supply a common physical perturbation spectrum.

Before another gate run, formulate backstress and diffusion contributions as
conservative staggered face fluxes, verify their discrete dispersion against
the continuum relation through Nyquist, and construct a box-independent
physical random-field ensemble. Do not add filtering, artificial viscosity, a
preferred wavelength, or a coefficient fitted to these PF/CDD outcomes.

## ADR-0022: Advance logarithmic staggered flux and face-slip kinematics

Status: accepted for integration development, 2026-09-12.

Mission-v3 comparison run `20260912T180135Z-556e986-552929` verifies two
conservative face-stress formulations through the 256-point Nyquist mode. Both
remove the old odd/even null and converge to the continuum low-mode symbol.
Advance the logarithmic-mean form because its exact discrete chain rule is the
stronger basis for a free-energy/dissipation proof; do not interpret this as a
wall-mechanism selection.

Store authoritative signed populations at cells and accumulated plastic slip
at faces. Updating `kappa_dot=-div(F+-F-)` and
`gamma_dot_face=b(F+-F-)` preserves the discrete Nye compatibility relation
under the identical flux. A second-order face average may supply cell slip to
Gate A, but may not replace the staggered state in compatibility diagnostics.
The next integration must restore Gate A homogeneous reduction and all source,
lock, and energy ledgers before any nonlinear wall comparison.

## ADR-0023: Replace the logarithmic stress heuristic with an integrable flux

Status: accepted for integrated numerical development, 2026-09-12.

The logarithmic mean alone did not prove entropy stability for ADR-0022's
separately normalized backstress/diffusion stresses. Introduce a distinct
correlation free energy containing signed-family logarithmic density and a
forest-normalized polarization penalty, differentiate it analytically, and
drive each signed population by its chemical-potential difference. This gives
an exact nonpositive discrete dissipation identity on the periodic grid. Keep
the former arithmetic and logarithmic stress forms as numerical ablations.

Make total mobile+locked+wall signed content authoritative for Nye
compatibility, ledger all positivity corrections, and reject rather than use
any integrated interval with active clipping. Use mutually adjoint face/cell
projections for slip and traction. Restrict the production EXP-floor exponent
to `n >= 1` at zero stress.

Advance a new four-family integrated state containing every signed reservoir,
face slip, `Fp`, rotation, and temperature. The first implementation qualifies
homogeneous Arrhenius/Orowan reduction, multiplicative kinematics, energy/work
closure, unloaded correlation relaxation, and exact restart. It does not yet
qualify nonlinear wall formation: local reaction integration and an IMEX
stiffness comparison precede the HPC bounded matrix. Keep all mechanism and
integrated-claim booleans false.

Replace polygonization's irreversible maturity increase by relaxation toward
the minimum of an explicit convex disordered/ordered wall free energy, and add
an independent Frank--Bilby residual whose orientation jump is not derived
from the tested wall inventory. No grain allocation is authorized.

## ADR-0024: Reject the positive-energy constant-mobility 1-D wall mechanism

Status: accepted locally; cross-platform verification pending, 2026-09-12.

Treat the exact logarithmic-density diffusion implicitly and retain the
nonlinear polarization potential explicitly. The returned conservative flux
must update density and staggered slip identically. Integrate bounded
Arrhenius pair multiplication, annihilation, locking, unlocking, and wall
capture with separate line and signed-content ledgers.

The complete frozen eight-population symbol shows that physical Arrhenius flux
is advective under fixed resolved stress, while the positive variational
correlation energy damps every nonzero mode. Mode-independent local reactions
do not introduce finite-wavenumber selection. The factorized local screen is
damped through Nyquist for all tested entropy signs, positive coefficient
ratios, densities, grids, domains, and load controls; physical-noise nonlinear
steps likewise reduce contrast.

Reject this model family as a spontaneous wall mechanism without rejecting it
as a verified transport/relaxation operator. Do not broaden the same parameter
search or make a wall claim. Verify the negative decision once on HPC3, then
advance to state-dependent friction and cross-family internal/reaction
coupling. Preserve positive free energy and do not introduce a fitted
wavelength or negative signed-density quadratic.

## ADR-0025: Require topology-preserving junction state for cross-family CDD

Status: accepted architecture, implementation pending, 2026-09-12.

Do not promote a scalar density-product locking rate to the production
cross-family mechanism. Primary vector-CDD formulations preserve line
continuity and Frank's rule by introducing junction-point/endpoint density and
explicit reaction topology. Add a signed vector density per slip family and a
junction incidence state whose reaction stoichiometry conserves the total Nye
tensor and Burgers content. Carry sessile and glissile products separately.

State-dependent friction remains an admissible coupled hypothesis, and the
existing scalar locked reservoirs remain useful balance fixtures, but neither
may stand in for junction topology. Build this extension in a 2-D-capable
module and first verify topology, objectivity, energy, and homogeneous
reduction before nonlinear wall searches. Old DD multi-hit histories remain
structural context only and do not set reaction rates or memory.

## ADR-0026: Repair units, reduced kinematics, work projection, and energy state

Status: accepted locally, 2026-09-12.

Separate Arrhenius event frequency from physical glide velocity and require an
explicit event length for every CDD flux. Report spatially integrated 1-D line
content in `m^-1`. Reclassify the integrated solver as small-strain antiplane:
accumulate additive plastic distortion and evolve lattice rotation from total
spin minus plastic spin. Use the same cellwise rotated dyads for resolved
stress, plastic distortion, macroscopic plastic shear, and work projection.

Include stationary signed content in long-range energy and its mobile chemical
derivative. Ledger mobile correlation, long-range, mobile-line, junction, wall,
transport, and reaction free-energy changes separately. The scalar reservoir
reactions remain fixtures. The repaired canonical suite passes 236 tests.

## ADR-0027: Reject the minimal instantaneous junction-friction closure

Status: accepted locally, 2026-09-12.

Implement the smallest aligned vector/topology state with eight signed mobile
species, explicit junction products, a stoichiometric incidence matrix,
pointwise Frank-rule/Nye conservation, detailed-balance release, and
junction-dependent forest resistance inside the EXP-floor stress scale.
Linearize the complete isothermal reaction--transport model before nonlinear
simulation.

Across the bounded screen, fifteen of sixteen cases are damped. The only
positive case maximizes at the declared shortest continuum wavelength, and
the extended symbol peaks below continuum validity. This is boundary-seeking,
not an intrinsic continuum wall spacing. Reject the instantaneous aligned
closure, do not run it nonlinearly or on HPC3, and advance to delayed junction
memory plus a higher-order line-orientation state. All wall/DRX/ASB and
predictive flags remain false.

## ADR-0028: Advance delayed junction memory to nonlinear falsification

Status: finite-mode region identified locally, 2026-09-12.

Add one kinetic memory state per junction, `m_dot=(p-m)/tau`, and let the
already declared junction forest resistance respond to memory instead of the
instantaneous product density. This adds a timescale but no wavelength. The
complete reaction--transport--memory symbol preserves the zero-mode Frank
invariant.

A 324-case bounded screen produces 27 positive interior continuum modes with
wavelengths from 0.275 to 1.02 micrometers over three temperatures, two load
levels, multiple diffusivities, and multiple relaxation times. The region
exists only at the highest screened forest coefficient. Record
`finite_mode_region_identified=true`, while keeping mechanism, wall, DRX, ASB,
and predictive flags false. Advance to a compact nonlinear falsification and
a higher-order line-orientation state; do not run a broad HPC campaign.
