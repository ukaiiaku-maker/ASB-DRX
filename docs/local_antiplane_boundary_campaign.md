# Local antiplane coupled boundary campaign

## Scope and disposition

This campaign integrates the verified periodic antiplane equilibrium operator with the generic single-glider EXP-floor law, finite elastic loading, forest storage, local plastic-work heating, periodic heat transport, and two-order-parameter phase relaxation. The DDD data do not parameterize this coupled model. The DDD-derived file is used only for the user-authorized generic single-glider EXP-floor parameterization; rates above 4.5 s^-1 are analytical extrapolations. The results are a deterministic mechanism screen, not material validation, DRX evidence, or a converged ASB regime map.

## Coupled update

For each accepted increment, the scalar antiplane elastic problem minimizes the periodic elastic energy for the imposed mean shear and the local plastic shear. Its discrete Fourier solution projects every nonzero compatible mode transverse to the discrete wave vector. Even-grid Nyquist derivatives are assigned zero so real fields remain Hermitian and the midpoint work identity closes for broadband fields.

The continuum rate is the odd forward activated rate minus the unloaded reverse rate. This preserves the source one-way DDD event law while making zero stress an exact continuum equilibrium. A matrix-free backward-Euler Newton--GMRES solve advances plastic flow and antiplane equilibrium together; the exact Fourier projection is used in every Jacobian-vector product. Forest storage requests the generic rate `K |delta gamma|`; if its line energy exceeds the available local midpoint plastic work, all grain-wise storage increments at that grid point are scaled by a common factor. This is an explicit thermodynamic limiter, and its activity is reported rather than hidden.

The generic dynamic-recovery law is fixed once from two explicitly arbitrary neutral-boundary anchors, rather than fitted to old data. Recovery is integrated exponentially, and every decrease in stored line energy is recorded as a separate heat source. Its parameters are `Q_rec approximately 1.138 eV` and `tau_rec(950 K) approximately 1.856 s`; these are numerical design values, not material calibration.

Plastic work not stored as line energy becomes heat. Constant-property periodic conduction is advanced by the exact Fourier heat propagator. The phase fields use the projected variational force on the pointwise simplex, accept only nonincreasing discrete phase energy, and route released stored/interface energy to heat. A 128-machine-epsilon phase-energy floor converts normalization-only last-bit motion into a phase no-op instead of endless timestep halving.

Thermal and matched isothermal controls share every accepted time interval. Diagnostics are retained every 0.0009 applied shear; the three-sample persistence criterion therefore means 0.0027 applied shear, independent of internal adaptive substeps. Timestep, limiter, and ledger extrema are aggregated over every internal substep. A condition is marked `numerically_unresolved` after 20,000 accepted substeps rather than being assigned a physical class.

## Verification

- The final complete suite passes 147 tests.
- Uniform fields reduce to the prior common-stress kernel.
- Broadband even-grid antiplane midpoint work closes to floating-point precision.
- Periodic heat diffusion matches the exact spectral decay and preserves the mean.
- Isothermal and phase-disabled controls are exact.
- Continuous and checkpoint/restart trajectories agree bitwise.
- The full state derives equilibrated stress on load; stress is not independent checkpoint state.
- The difficult 850 K, 4.5 s^-1, density-ratio-0.5 HPC3 smoke reached exactly 0.9 applied shear in 3,070 accepted steps and passed all 20 staged tests.

## Historical signed-one-way sparse matrix result

The matrix below predates the net-rate, implicit-flow, and recovery corrections. It is retained only as a numerical baseline and must not be mixed with the active v2 campaign. It used a 16 by 16 periodic grid, a 16 micrometre square domain, target applied shear 0.9, temperatures 850/950/1050 K, rates 4.5/450/45000 s^-1, density ratios 0.5/1/2 relative to the old one-way analytical peak, and one deterministic initial perturbation.

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

## Active v2 net-flow/recovery matrix

The corrected matrix uses the same 27 generic conditions, grid, domain, target
strain, deterministic perturbation, and classification rule, but it is a new
model version: the boundary is the net EXP-floor peak, flow is implicit, zero
stress satisfies detailed balance, and analytically constrained recovery is
active. All 27 conditions reached exactly 0.9 applied shear in exactly 1,000
accepted steps. No step was halved, no condition was numerically unresolved,
and the nonlinear flow solve required at most eight Newton iterations with
maximum reported residual `9.20e-12`.

| Quantity | v2 result |
|---|---:|
| Conditions completed | 27 / 27 |
| Numerically unresolved | 0 |
| Classified localized | 0 |
| Minimum active plastic fraction | 0.999975 |
| Maximum temperature | 1270.40 K |
| Maximum matched temperature excess | 268.36 K |
| Maximum post-peak softening fraction | 0.4093 |
| Maximum storage-limited area | 0 |
| Maximum absolute global ledger residual | 2.66e-6 J m^-3 |
| Maximum absolute thermal ledger residual | 1.11e-6 J m^-3 |

Every record failed the plastic-concentration criterion; 12 also failed the
post-peak-softening criterion. Heating and recovery therefore change the mean
response substantially but do not generate a localized band from the present
deterministic perturbation. Eighteen conditions have a net density decrease and
nine a net increase. This is consistent with recovery competing with storage,
but the arbitrary recovery anchors prevent a materials interpretation.

HPC3 run `20260828T232914Z-f5889ab-7987e9`, Slurm `55646836`, completed in
20:15 with 93.42% CPU efficiency and 116.57 MB peak memory. The clean staging
commit is `f5889ab`; scientific source commit is `a927f28`. The fetched archive
is checksum verified. Source archive SHA-256 is
`ada39876a7e448598e13064cd64302173b15995e08c797ef27e2d0258dbc3782`,
result archive SHA-256 is
`74ed17e8570f0e514224a6533e48bd37b92d110ccf6e59a7e4604cfddef3b83b`,
and matrix JSON SHA-256 is
`db354365994c6badcff40c5290020053d7b506af83934a760fc1bfbb77e558d9`.
The verified result is
[local_boundary_matrix.json](../hpc3-results/asb-drx-independent/20260828T232914Z-f5889ab-7987e9/work/output/local_boundary_matrix.json).

The immediately preceding job `55646831` was cancelled after 26 seconds,
before its first record, because it still carried the obsolete v1 output schema.
It produced no accepted scientific result.

## Active v2 gate

The formerly unresolved 850/950 K, 45000 s^-1, density-ratio-2 points now reach 0.9 shear locally in 300 steps with zero halving and at most three Newton iterations. A near-boundary nonlinear refinement at 950 K and 4500 s^-1 passes the provisional 5% target: 0.873% maximum final timestep change and 0.00124% maximum 24-to-32-grid change.

Continuous sequential-hit, rearming-contact, and shot-noise closures were compared. Only shot noise can represent clustering, but the current native DDD audit does not identify its signed transfer, memory time, or spatial kernel, so no collective closure is promoted. A separate physical embryo/orientation gate now prevents phase labels or collective events from being counted as DRX. The active v2 sparse matrix is staged independently and embeds its recovery design and schema in the output.
