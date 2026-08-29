# Campaign status

Updated: 2026-08-28 (America/Los_Angeles)

## Current gate: material-agnostic coupled-mechanism verification

### Active redevelopment update

- The v2 mission/physics addendum was adopted on 2026-08-29 from
  `CODEX_INDEPENDENT_DD_PF_DRX_ASB_CAMPAIGN_v2.md`, SHA-256
  `37142ee8029b4f461cbdbfa326c58632a7d4fd2988ff951af3ccef5e0d9dc2da`.
  It reclassifies the scalar/two-reservoir implementation below as a verified
  baseline rather than the production DRX architecture.
- Read-only DD organization audit finds event timing, contact, force, barrier,
  and clustering information, but no complete Burgers-sign/family, reaction,
  Nye/GND, wall-structure, or lattice-rotation observables. No DD-to-wall law
  or locked collective closure is scientifically identifiable.
- The five required addendum design documents and four quarantined A--D
  analytical/kinematic fixtures are now present. The fixtures test invariants
  without claiming the scientific gates: each deliberately reports
  `scientific_gate_passed=false` until its missing calibration/benchmark exists.
- The full local regression suite passes 151 tests after adding these fixtures.

- Continuum flow now uses forward-minus-unloaded-reverse EXP-floor kinetics and
  a matrix-free backward-Euler antiplane solve. The two old stiff/unresolved
  high-rate cases reach 0.9 shear with no timestep halving.
- The full frozen-time finite-wavenumber operator is a verified 5 by 5 system
  for plastic shear, temperature, two density reservoirs, and binary order,
  including antiplane orientation, both storage-cap branches, phase heat, and
  recovery.
- Governing-equation analysis proved that storage alone cannot create a
  temperature/rate post-peak boundary. A single generic Arrhenius recovery law
  was therefore constrained by two declared arbitrary neutral points without
  retuning the DDD-derived flow parameters.
- The nonlinear near-boundary refinement passes the provisional 5% gate:
  0.873% maximum final timestep change and 0.00124% maximum 24-to-32-grid
  change. The condition remains nonlocalized, so band onset/width convergence
  is still not established.
- Sequential-hit and rearming-contact closures are underdispersed and cannot
  reproduce the audited high-density event CV above one. Shot-noise
  self-excitation remains a future ablation, not production physics, because
  its causal transfer and memory parameters are unidentified.
- A checkpointed physical embryo/orientation gate is implemented. Phase labels
  cannot become DRX without a promoted embryo that is distinct, supercritical,
  beyond the zero-excess radius, persistent, positively driven, and supported
  by a resolved pure phase field.
- The complete local suite currently passes 147 tests.
- Active v2 HPC3 run `20260828T232914Z-f5889ab-7987e9` / job `55646836`
  completed and fetched with verified checksums. All 27 sparse-matrix conditions
  reached 0.9 shear in 1,000 steps with no halving or unresolved point; none
  localized because the minimum active plastic fraction remained 0.999975.
  Maximum temperature was 1270.40 K and maximum matched-control excess was
  268.36 K. The preceding 26-second job `55646831` was cancelled before its
  first record to correct an obsolete output schema and is not used.

The 2026-08-27 DD-data no-go is superseded by the clarified scope: DD will not parameterize the model, and old programs/data are context only. Their inventory and HPC3 reproductions remain an audit record but are not new-model evidence or gates.

- Repository remote verified as `https://github.com/ukaiiaku-maker/ASB-DRX.git`.
- Isolated branch/worktree created from remote `main` at `a5dd798096e3896f319d314e8e4c60f5b277e589`.
- Source evidence root located at `/Users/sdillon/DRX-ASB` (33,358 files before filtering; about 20 GB).
- Six supplied files located at the evidence-root top level with upload suffixes normalized away.
- Legacy v32, v33 controls, and v34 sources/results located under `recrysyallization_PF-2D/shear_banding`.
- HPC3 aliases and runner verified. Existing unrelated local and Slurm campaigns were observed and left untouched.
- An EXP-floor barrier, independent-node rate law, inverse, and closed-form rate--temperature strength peak have been derived in `analytical_strength_derivation.md`.
- A material-agnostic analytical kernel and verification tests have been added under `src/asb_drx/` and `tests/`; small verification runs may execute locally and extended campaigns use HPC3.
- HPC3 run `20260828T120911Z-9d9e7c4-1bdf8a` (job `55637582`) passed all five analytical tests and was fetched with verified checksums.
- HPC3 identifiability run `20260828T131741Z-9caa154-ffbe2c` (job `55637767`) passed seven tests with verified retrieval. Strength-only peaks expose the exact scale compensation; independent peak density restores the tested five-parameter local rank.
- Literature motivates, but does not parameterize, a collective transparent-node hypothesis based on stress-transfer branching and multi-hit shot-noise memory.
- A 2026-08-28 primary-literature refresh confirms that repeat-pass conditioning is plausible and that glissile/shearable junctions may still harden. It does not provide a transferable multi-hit factor, stress-transfer kernel, or reset time; the collective extension remains outside the baseline.
- Complete single-glider Taylor DDD context was located in `/Users/sdillon/Taylor_DDD` and `/Users/sdillon/Taylor_DDD_arrhenius_native`. Commit `fb7610b` contains a passing native ExaDiS persistent-contact gate; later EXP-floor campaigns contain event histories across density and temperature. These are now structural evidence, not parameter sources.
- HPC3 structural run `20260828T130729Z-1a147e0-710710` (job `55637740`) passed both collective-context tests with verified retrieval. Higher density has substantially more multi-hit clustering, but all sampled native one-step contact operators have zero spectral-radius proxy and only 11 redistribution samples exist. No causal collective law or production parameter is established.
- HPC3 thermodynamic run `20260828T132308Z-07d589a-1e7eac` (job `55637784`) passed six material-agnostic tests with verified retrieval: discrete variational consistency, monotone relaxation, conservative reservoir transfer, exact work partition, nucleus-limit signs, and range rejection.
- HPC3 spatial run `20260828T132810Z-a4a0bf0-5c7bfe` (job `55637801`) passed nine tests with verified retrieval. The diffuse 2-D nucleus shrinks/grows on the correct sides of the derived critical radius, final grid/timestep changes are 0.384%/0.0021%, and the complete current limited state restarts bitwise exactly.
- HPC3 material-point run `20260828T133324Z-6920914-866ee2` (job `55637814`) passed ten tests with verified retrieval. Finite elastic loading, EXP-floor plastic flow, stored line energy, and residual heat close the incremental work ledger exactly without a tuned heat fraction.
- HPC3 shear-layer run `20260828T133943Z-ae2fdf9-d7f5b0` (job `55637821`) passed fourteen tests with verified retrieval. The common-stress layer reduces to the material point, conserves/damps heat correctly, closes both ledgers, and restarts exactly. An earlier unstable explicit-diffusion run was rejected and corrected by enforcing the Fourier bound.
- HPC3 grain-metric run `20260828T134840Z-5e8fabc-192eaf` (job `55637844`) passed ten tests with verified retrieval. Allocated labels, periodic topology components, resolved support, physical grains, and promoted recrystallized grains are distinct metrics; promotion requires resolved persistent support, lineage, and symmetry-reduced misorientation, while retirement preserves provenance. The preceding packaging-only failure omitted an imported module and executed no grain test.
- HPC3 multi-order run `20260828T135602Z-d1de48a-eb7ef1` (job `55637866`) passed all seventeen multi-order/grain tests with verified retrieval. The projected Onsager evolution preserves the pointwise simplex to roundoff, decreases the declared free energy, leaves an exactly pure parent unchanged, is label-permutation symmetric, recovers the analytical circular-nucleus growth signs, restarts exactly for its complete current state, and feeds the tracker without allocating labels.
- HPC3 stored-energy coupling run `20260828T140503Z-8fb6236-7dab82` (job `55637879`) passed fourteen tests with verified retrieval. The grain-growth driving force is the explicit line-energy/density difference; common density offsets do not alter binary dynamics; a pure parent neither resets nor heats; and every free-energy decrement closes between stored energy, interface/order energy, and heat.
- HPC3 candidate-decision run `20260828T141129Z-e01a0ce-969f27` (job `55637888`) passed six tests with verified retrieval. The classical cylindrical nucleus has the analytical stationary barrier at `R_c` and zero excess at `2R_c`; the Poisson event probability has the required temperature/driving monotonicity; and resolution, subcriticality, misorientation, and external-draw rejection remain distinct. The kernel never allocates a label.
- HPC3 coupled-ledger run `20260828T141957Z-c9708bb-f4aa93` (job `55637907`) passed all twenty-five material-point, phase, stored-energy, and coupled tests with verified retrieval. The binary aggregate reduces exactly to each isolated limit, uses one accepted interval, preserves a zero child, restarts exactly, and closes external work across elastic, stored, interface/order, and both heat channels without double counting. An earlier run exposed an over-strict temperature-roundoff gate and is retained as failed evidence.
- HPC3 spatial-coupled run `20260828T142706Z-3224ef1-bdf10f` (job `55638019`) passed fifteen tests with verified retrieval. Local temperature, two density fields, and two order parameters share a periodic 2-D grid under common stress; conduction, global energy closure, zero-child invariance, isolated-limit reduction, and exact restart pass.
- HPC3 localization-metric run `20260828T143238Z-ca7ab69-74cc1d` (job `55638898`) passed all six tests with verified retrieval. Classification now requires simultaneous plastic concentration, matched-control temperature excess, post-peak softening, resolved finite width, persistence, and joint onset/width refinement. The preceding run `20260828T143111Z-d608655-a7aea7` (job `55638861`) is retained as failed evidence: its negative fixture accidentally contained the required three consecutive qualifying states; the classifier behaved correctly and the fixture was corrected.
- HPC3 mechanism-ladder run `20260828T144000Z-c9dc4ab-7b33f9` (job `55640278`) passed seventeen tests with verified retrieval. Six declared common-equation variants plus thermal cases' isothermal twins preserve phase-disabled fields, route isothermal heat to an explicit bath, and close their ledgers. All six are nonlocalizing controls: minimum active fraction is about `0.907`, maximum matched-control temperature excess is `0.125 K`, and maximum softening is zero.
- HPC3 stability run `20260828T144457Z-535a0ff-662a0e` passed ten tests with verified retrieval. The marker-owning job is `55640502`; job `55640458` is an explicitly retained duplicate caused by a silent staging receipt during a `/pub` metadata stall. Analytical EXP-floor tangents, the finite-wavenumber thermal/storage Jacobian, conduction shift, nonlinear finite-difference closure, and impossible-storage rejection pass. The generic state has a positive `~4.807 s^-1` density-storage mode but strongly damped thermal diagonals; it is not thermal-ASB evidence.
- The user authorized reuse of the complete single-glider DDD constants as a generic fixture and an arbitrary analytical boundary, without matching a materials class. HPC3 run `20260828T151810Z-d31c6a4-d3af25` (job `55641106`) passed fourteen exact-mapping and upstream tests with verified retrieval.
- The arbitrary boundary is preregistered as `rho=rho_peak(T, rate)` from the independent EXP-floor law. At the DDD rate it spans `4.4117e15` to `8.5383e15 m^-2` over 1050 to 850 K. The driver's hard-coded `1e18 m^-2` field is excluded; the observed monotone DDD response through `3e16 m^-2` remains a structural mismatch, not a fitted correction.
- HPC3 run `20260828T152809Z-db81077-68e54d` (job `55641308`) passed all fifteen boundary and mapping tests with verified retrieval. The frozen surface covers 850--1050 K and `4.5`, `450`, and `45000 s^-1`; future spatial cases are preregistered at density ratios `0.5`, `1`, and `2` relative to the peak.
- The first boundary-spatial preflight `20260828T153642Z-c48cad2-f5273f` (job `55641810`) failed before simulation because one upstream test dependency was omitted from staging and an exact-equality assertion rejected a `~3e-16` relative roundoff difference. It is retained as failed packaging evidence; no equation or case definition changed.
- Corrected single-job smoke `20260828T154200Z-9d7ed90-e57dd1` (job `55642217`) passed 21 tests and fetched with verified checksums. At 950 K, `45000 s^-1`, and `rho/rho_peak=1`, 16²/32² final stress and maximum-temperature changes were `5.04e-9` and `1.10e-6`, below the 5% provisional target. The case reached 0.09 shear and `26.74 K` matched-control excess but remained spatially uniform (`f_q~0.999999`) and nonlocalizing.
- HPC3 run `20260828T155312Z-266dda8-d6b092` (job `55642641`) passed six isolated periodic-antiplane tests with verified retrieval. The Fourier projection recovers uniform common stress, relaxes longitudinal modes, produces equilibrated transverse/diagonal redistribution, and closes the exact midpoint work identity to `7.45e-9 J m^-3` (`~3.7e-15` relative). It is not yet constitutively coupled.
- Preceding antiplane run ID `20260828T154906Z-266dda8-f8fc4a` is retained as failed orchestration evidence: jobs `55642418` and `55642419` were both created after a metadata-stalled receipt and collided while finalizing the same remote path. All six tests passed in the shared log, but no scientific result is accepted from that run.
- First-pass legacy audit and candidate thermodynamic architecture drafted.
- Full evidence inventory completed: 33,358 files, 21,536,785,369 bytes, no hash errors.
- Campaign-specific HPC3 smoke job `55633650` completed, fetched, and checksum-verified.
- Exact-source HPC3 legacy controls completed and were fetched with verified checksums: v32 job `55633674`, v33 job `55633691`, and v34 job `55633694`.
- v32 reproduces all finite numerical diagnostics within `atol=1e-10`, `rtol=1e-9`; 18 degenerate zero-variance correlations differ only as `NaN` versus roundoff near `1e-17`.
- v33 reproduces the false-grain mechanism but not its exact trajectory: 165 hazard births and 177 allocated labels versus 12 unchanged topology components. These are rejected as physical grains.
- v34 produces zero active, new, or promotable candidates and zero births throughout. This reproduces the stored zero-candidate failure, not the requested candidate-without-promotion premise. Its detailed trajectory is not numerically reproducible from the supplied source/configuration.

## Active scientific boundaries

1. The analytical kernel and first optimizer-identifiability gate pass on HPC3. Physical fitting must not estimate both stress scale and attempt rate from strength peaks alone; it requires independent peak density or an authoritative fixed scale.
2. No materials-class calibration is sought in the present generic campaign; the authorized DDD fixture must not be described as a physical calibration.
3. The collective extension is an ablation, not baseline physics. Existing DDD histories show density-dependent clustering but do not resolve causal parentage or a nonzero feedback operator; higher-cadence evidence is required before reconsideration.
4. The periodic antiplane operator is now integrated with net EXP-floor flow, storage, recovery, heat, and phase evolution. The model is generic and isotropic; physical boundaries and material-scaled anisotropic properties are outside the present scope.
5. The full finite-wavenumber linearization and nonlinear timestep/grid refinement pass, but the v2 matrix supplies no candidate band. Onset and width convergence therefore cannot yet be claimed.
6. A stateful physical embryo gate and energy-release ledger pass, but automatic stochastic embryo sampling and phase-field allocation are absent because their attempt prefactor and orientation distribution are unconstrained.
7. The strict ASB classifier has evaluated all 27 coupled v2 trajectories. Every trajectory fails plastic concentration, so the result is a verified no-localization baseline rather than an ASB boundary.
8. The generic recovery parameters define an arbitrary analytical screen only. They must not be interpreted as material properties or a calibrated DRX/ASB transition.
9. A collective closure remains an ablation. Existing event histories cannot identify a signed stress-transfer kernel, memory time, or causal parentage.
10. A predictive campaign would still require a defensible finite-wavelength localization mechanism, physical embryo-rate inputs, uncertainty propagation, and external validation.

## Remaining interpretation limits

The generic campaign may proceed without selecting Fe, Cr, or another materials class. A predictive materials claim would still require an authoritative dataset. “Post-peak collective candidate” is deliberately broader than a particular transparent-junction mechanism; a specific transfer kernel and reset law remain research questions rather than boundary inputs.
