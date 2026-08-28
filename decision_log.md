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
