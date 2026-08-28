# Local antiplane coupled boundary campaign

## Scope and disposition

This campaign integrates the verified periodic antiplane equilibrium operator with the generic single-glider EXP-floor law, finite elastic loading, forest storage, local plastic-work heating, periodic heat transport, and two-order-parameter phase relaxation. The DDD data do not parameterize this coupled model. The DDD-derived file is used only for the user-authorized generic single-glider EXP-floor parameterization; rates above 4.5 s^-1 are analytical extrapolations. The results are a deterministic mechanism screen, not material validation, DRX evidence, or a converged ASB regime map.

## Coupled update

For each accepted increment, the scalar antiplane elastic problem minimizes the periodic elastic energy for the imposed mean shear and the local plastic shear. Its discrete Fourier solution projects every nonzero compatible mode transverse to the discrete wave vector. Even-grid Nyquist derivatives are assigned zero so real fields remain Hermitian and the midpoint work identity closes for broadband fields.

The old equilibrated local stress drives the signed EXP-floor shear rate separately in each order-parameter field. The interpolated grain rates update the local plastic shear, and equilibrium is solved again at the new imposed shear. Forest storage requests the generic rate `K |delta gamma|`; if its line energy exceeds the available local midpoint plastic work, all grain-wise storage increments at that grid point are scaled by a common factor. This is an explicit thermodynamic limiter, and its activity is reported rather than hidden.

Plastic work not stored as line energy becomes heat. Constant-property periodic conduction is advanced by the exact Fourier heat propagator. The phase fields use the projected variational force on the pointwise simplex, accept only nonincreasing discrete phase energy, and route released stored/interface energy to heat. A 128-machine-epsilon phase-energy floor converts normalization-only last-bit motion into a phase no-op instead of endless timestep halving.

Thermal and matched isothermal controls share every accepted time interval. Diagnostics are retained every 0.0009 applied shear; the three-sample persistence criterion therefore means 0.0027 applied shear, independent of internal adaptive substeps. Timestep, limiter, and ledger extrema are aggregated over every internal substep. A condition is marked `numerically_unresolved` after 20,000 accepted substeps rather than being assigned a physical class.

## Verification

- The final complete local suite passes 123 tests.
- Uniform fields reduce to the prior common-stress kernel.
- Broadband even-grid antiplane midpoint work closes to floating-point precision.
- Periodic heat diffusion matches the exact spectral decay and preserves the mean.
- Isothermal and phase-disabled controls are exact.
- Continuous and checkpoint/restart trajectories agree bitwise.
- The full state derives equilibrated stress on load; stress is not independent checkpoint state.
- The difficult 850 K, 4.5 s^-1, density-ratio-0.5 HPC3 smoke reached exactly 0.9 applied shear in 3,070 accepted steps and passed all 20 staged tests.

## Sparse matrix result

The final matrix used a 16 by 16 periodic grid, a 16 micrometre square domain, target applied shear 0.9, temperatures 850/950/1050 K, rates 4.5/450/45000 s^-1, density ratios 0.5/1/2 relative to the analytical EXP-floor peak, and one deterministic initial perturbation.

| Quantity | Result |
|---|---:|
| Conditions | 27 |
| Completed to target shear | 25 |
| Numerically unresolved | 2 |
| Classified localized | 0 |
| Maximum temperature | 1259.68 K |
| Maximum matched-control temperature excess | 256.90 K |
| Maximum post-peak softening fraction | 0.6639 |
| Minimum retained active fraction | 0.4756 |
| Maximum absolute global ledger residual | 1.29e-6 J m^-3 |
| Maximum absolute thermal ledger residual | 1.18e-6 J m^-3 |

The unresolved points are 850 K and 950 K at 45000 s^-1 and density ratio 2. Both exceeded 20,000 accepted substeps before reaching 0.9 shear. The corresponding 1050 K point completed, consistent with temperature changing the stiffness of the post-peak EXP-floor response; this is a numerical observation, not yet a physical boundary claim.

All 25 completed points failed the plastic-concentration requirement at the fixed retained-strain cadence. Several points showed strong heating and softening, but those responses remained too spatially distributed for the conjunctive ASB classification. This is the central screening result: local antiplane stress redistribution plus the present deterministic perturbation, storage, heat, and phase relaxation is not sufficient to generate an ASB in this matrix. A collective/multi-hit closure or another physically justified localization mechanism remains a hypothesis to derive and test, not a switch to add.

Forest storage limiting occurred in five completed low-rate records. The largest instantaneous limited area was 4.6875% of the grid; the largest count was 819 internal steps. The limiter is therefore material to selected low-stress trajectories and must remain an explicit diagnostic in later calibration.

The phase field changed continuously but did not nucleate a new label. The initial state already contains a generic diffuse child-order perturbation with about 0.199 area-averaged support. Its evolution is not a physical grain count and is not classified as DRX.

## HPC3 provenance

- Single-case gate: run `20260828T175315Z-c70b816-2b0175`, Slurm `55643489`, completed in 41 s, peak memory 359.32 MB.
- Final matrix: run `20260828T183409Z-a8ee6af-ffbc1a`, Slurm `55644087`, completed in 17:34 with 99.34% CPU efficiency and 260.08 MB peak memory.
- Scientific source commit recorded by the matrix: `8f820117f64e56d8147115435669fbd78d3ddd01`.
- Final staging commit: `a8ee6af`.
- Combined deterministic initial-state SHA-256: `7111cc050c1ae41a1bb93f8999003aa2825bcf74d83fa814fc215843a6922f6b`.
- Generic EXP-floor source artifact SHA-256: `14a7a3c7341da5f7d991c229af5efe7d2a4e1cb2ada4597b2cdad44efd8b2b2b`.
- Source archive SHA-256: `93b30cddd0f07f175a5046225f630e3338886bde54978739a387d19759645b1b`.
- Result archive SHA-256: `331ccc9ded665073d419d712188bdb696aa5d7406177a979e047d9e2b254897c`.
- Matrix JSON SHA-256: `d280d7cb84dbc7f370eafb17a951b2e45a9a5edf6d7d4efe99cbc9cb04de0b9d`.

The first full-matrix attempt failed at a phase-stationarity roundoff edge case. The second was killed at the 2 GB limit because internal substeps retained full states. A third run was deliberately cancelled after 2:24 when the original 100,000-step screening guard was found too permissive. No failed or cancelled result was used scientifically; their diagnoses led respectively to the energy floor, bounded retention, and the explicit 20,000-substep unresolved policy.

The verified fetched result is [local_boundary_matrix.json](../hpc3-results/asb-drx-independent/20260828T183409Z-a8ee6af-ffbc1a/work/output/local_boundary_matrix.json).

## Next gate

Do not infer a localized/nonlocalized boundary through the two unresolved points. The next numerical task is a timestep/integrator study specifically for the high-rate density-ratio-2 branch, followed by 16/32-grid refinement at any candidate localization boundary. The next physics task is to derive and independently constrain any collective multi-hit coupling before adding it to storage, intermittency, or spatial correlation; the present matrix provides the no-collective baseline.
