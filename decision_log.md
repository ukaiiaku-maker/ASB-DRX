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
