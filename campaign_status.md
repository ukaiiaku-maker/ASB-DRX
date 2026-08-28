# Campaign status

Updated: 2026-08-28 (America/Los_Angeles)

## Current gate: material-agnostic coupled-mechanism verification

The 2026-08-27 DD-data no-go is superseded by the clarified scope: DD will not parameterize the model, and old programs/data are context only. Their inventory and HPC3 reproductions remain an audit record but are not new-model evidence or gates.

- Repository remote verified as `https://github.com/ukaiiaku-maker/ASB-DRX.git`.
- Isolated branch/worktree created from remote `main` at `a5dd798096e3896f319d314e8e4c60f5b277e589`.
- Source evidence root located at `/Users/sdillon/DRX-ASB` (33,358 files before filtering; about 20 GB).
- Six supplied files located at the evidence-root top level with upload suffixes normalized away.
- Legacy v32, v33 controls, and v34 sources/results located under `recrysyallization_PF-2D/shear_banding`.
- HPC3 aliases and runner verified. Existing unrelated local and Slurm campaigns were observed and left untouched.
- An EXP-floor barrier, independent-node rate law, inverse, and closed-form rate--temperature strength peak have been derived in `analytical_strength_derivation.md`.
- A material-agnostic analytical kernel and verification tests have been added under `src/asb_drx/` and `tests/`; numerical execution is restricted to HPC3.
- HPC3 run `20260828T120911Z-9d9e7c4-1bdf8a` (job `55637582`) passed all five analytical tests and was fetched with verified checksums.
- HPC3 identifiability run `20260828T131741Z-9caa154-ffbe2c` (job `55637767`) passed seven tests with verified retrieval. Strength-only peaks expose the exact scale compensation; independent peak density restores the tested five-parameter local rank.
- Literature motivates, but does not parameterize, a collective transparent-node hypothesis based on stress-transfer branching and multi-hit shot-noise memory.
- Complete single-glider Taylor DDD context was located in `/Users/sdillon/Taylor_DDD` and `/Users/sdillon/Taylor_DDD_arrhenius_native`. Commit `fb7610b` contains a passing native ExaDiS persistent-contact gate; later EXP-floor campaigns contain event histories across density and temperature. These are now structural evidence, not parameter sources.
- HPC3 structural run `20260828T130729Z-1a147e0-710710` (job `55637740`) passed both collective-context tests with verified retrieval. Higher density has substantially more multi-hit clustering, but all sampled native one-step contact operators have zero spectral-radius proxy and only 11 redistribution samples exist. No causal collective law or production parameter is established.
- HPC3 thermodynamic run `20260828T132308Z-07d589a-1e7eac` (job `55637784`) passed six material-agnostic tests with verified retrieval: discrete variational consistency, monotone relaxation, conservative reservoir transfer, exact work partition, nucleus-limit signs, and range rejection.
- HPC3 spatial run `20260828T132810Z-a4a0bf0-5c7bfe` (job `55637801`) passed nine tests with verified retrieval. The diffuse 2-D nucleus shrinks/grows on the correct sides of the derived critical radius, final grid/timestep changes are 0.384%/0.0021%, and the complete current limited state restarts bitwise exactly.
- HPC3 material-point run `20260828T133324Z-6920914-866ee2` (job `55637814`) passed ten tests with verified retrieval. Finite elastic loading, EXP-floor plastic flow, stored line energy, and residual heat close the incremental work ledger exactly without a tuned heat fraction.
- HPC3 shear-layer run `20260828T133943Z-ae2fdf9-d7f5b0` (job `55637821`) passed fourteen tests with verified retrieval. The common-stress layer reduces to the material point, conserves/damps heat correctly, closes both ledgers, and restarts exactly. An earlier unstable explicit-diffusion run was rejected and corrected by enforcing the Fourier bound.
- HPC3 grain-metric run `20260828T134840Z-5e8fabc-192eaf` (job `55637844`) passed ten tests with verified retrieval. Allocated labels, periodic topology components, resolved support, physical grains, and promoted recrystallized grains are distinct metrics; promotion requires resolved persistent support, lineage, and symmetry-reduced misorientation, while retirement preserves provenance. The preceding packaging-only failure omitted an imported module and executed no grain test.
- HPC3 multi-order run `20260828T135602Z-d1de48a-eb7ef1` (job `55637866`) passed all seventeen multi-order/grain tests with verified retrieval. The projected Onsager evolution preserves the pointwise simplex to roundoff, decreases the declared free energy, leaves an exactly pure parent unchanged, is label-permutation symmetric, recovers the analytical circular-nucleus growth signs, restarts exactly for its complete current state, and feeds the tracker without allocating labels.
- HPC3 stored-energy coupling run `20260828T140503Z-8fb6236-7dab82` (job `55637879`) passed fourteen tests with verified retrieval. The grain-growth driving force is the explicit line-energy/density difference; common density offsets do not alter binary dynamics; a pure parent neither resets nor heats; and every free-energy decrement closes between stored energy, interface/order energy, and heat.
- First-pass legacy audit and candidate thermodynamic architecture drafted.
- Full evidence inventory completed: 33,358 files, 21,536,785,369 bytes, no hash errors.
- Campaign-specific HPC3 smoke job `55633650` completed, fetched, and checksum-verified.
- Exact-source HPC3 legacy controls completed and were fetched with verified checksums: v32 job `55633674`, v33 job `55633691`, and v34 job `55633694`.
- v32 reproduces all finite numerical diagnostics within `atol=1e-10`, `rtol=1e-9`; 18 degenerate zero-variance correlations differ only as `NaN` versus roundoff near `1e-17`.
- v33 reproduces the false-grain mechanism but not its exact trajectory: 165 hazard births and 177 allocated labels versus 12 unchanged topology components. These are rejected as physical grains.
- v34 produces zero active, new, or promotable candidates and zero births throughout. This reproduces the stored zero-candidate failure, not the requested candidate-without-promotion premise. Its detailed trajectory is not numerically reproducible from the supplied source/configuration.

## Active scientific boundaries

1. The analytical kernel and first optimizer-identifiability gate pass on HPC3. Physical fitting must not estimate both stress scale and attempt rate from strength peaks alone; it requires independent peak density or an authoritative fixed scale.
2. Physical calibration cannot begin until one authoritative target material and strength/rate/temperature dataset are selected.
3. The collective extension is an ablation, not baseline physics. Existing DDD histories show density-dependent clustering but do not resolve causal parentage or a nonzero feedback operator; higher-cadence evidence is required before reconsideration.
4. Phase-field production work remains gated by free-energy/dissipation review and separately sourced material/GB/thermal data.
5. The isolated thermodynamic/diffuse-nucleus gates and physical-grain classifier invariants pass, including their limited-state restart checks. Multi-order-parameter orientation dynamics, energetic nucleation, coupling, and production-state restart remain unverified.
6. The homogeneous thermomechanical ledger passes, but spatial mechanics/heat transport and localization remain unimplemented and unverified.
7. Periodic 1-D heat transport/common-stress mechanics now pass generic controls. Physical boundary conditions, multidimensional equilibrium, calibrated localization, and DRX coupling remain unverified.
8. Constrained isotropic multi-order dynamics and their tracker coupling pass generic controls. Crystallographic orientation-manifold dynamics, material-scaled anisotropic GB properties, energetic candidate generation, collisions, and coupled thermomechanical DRX remain unverified.
9. Fixed per-grain dislocation stored energy now drives the isolated phase relaxation with a closed heat ledger. Evolving spatial reservoirs under deformation/recovery, simultaneous mechanical work, conduction, temperature-dependent GB mobility, and nucleation rates remain unverified.

## Required external resolutions

1. Resolve whether the production target is Fe, Cr, or a named alloy and identify the authoritative strength/rate/temperature dataset.
2. Clarify whether “transparent Taylor pinning nodes” means shearable forest junctions, solute/precipitate obstacles, or another obstacle class; this changes the transfer kernel and reset law.

Symbolic and software work may proceed. Physical parameter optimization and predictive claims may not proceed without the target dataset.
