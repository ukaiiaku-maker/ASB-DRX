# Numerical verification record

Numerical work is restricted to HPC3. The material-agnostic analytical kernel has passed its first verification gate; no calibrated physical simulation has been run.

## EXP-floor analytical kernel

Status: passed, completed, fetched, and checksum-verified.

- Git commit: `9d9e7c4`
- Run ID / Slurm job: `20260828T120911Z-9d9e7c4-1bdf8a` / `55637582`
- Result: `COMPLETED`, exit `0:0`; application and finalization exit zero
- Environment: `anaconda/2025.12`, Python 3.13.9, SciPy 1.16.3; one CPU, 1 GB, no GPU
- Five tests passed: barrier endpoints and activation-volume derivative, forward-rate closure, local maximum, fixed-temperature strength/density rate scalings, peak-existence condition, and invalid-parameter rejection
- Fetched result: `hpc3-results/asb-drx-independent/20260828T120911Z-9d9e7c4-1bdf8a`
- Unit-test log SHA-256: `4b636918125f8f280b0e61ba6c7fda55df0a28c9360990fd2e47e8b3a74d5746`
- Input-inventory SHA-256: `6a11472cb7222b684cf5c83b8db2fdc89d0bf2a0c1eaa1ff3f2f0372321b10ef`
- Runner retrieval status: `verified`

This verifies implementation consistency with the declared equations. It does not validate a material parameterization.

## Analytical peak identifiability

Status: passed, completed, fetched, and checksum-verified. This uses exact synthetic fixtures and makes no physical fit.

- Git commit: `9caa154`
- Run ID / Slurm job: `20260828T131741Z-9caa154-ffbe2c` / `55637767`
- Result: `COMPLETED`, exit `0:0`; application and finalization exit zero
- Environment: `anaconda/2025.12`, Python 3.13.9, SciPy 1.16.3; one CPU, 2 GB, no GPU; 1 s wall clock and 190.47 MB peak memory
- Seven tests passed: the five analytical-law tests plus strength-only rank deficiency and strength-plus-density recovery
- Result archive SHA-256: `070814acca5b69a377421aef13bec665f38af3494129cb3f7e361c76b0a03eb1`
- Unit-test log SHA-256: `f1bbe99fda07d2a5dd2a29247abc9d9faa25ae236a4fae7059c6fe2dc82b02d3`
- Runner retrieval status: `verified`

Peak strength alone correctly fails the five-parameter local-rank gate because `tau_ref` and `eta0` compensate. Adding independently observed peak density restores full local rank and recovers the planted five scale/temperature parameters to the declared tolerance. Therefore a physical campaign must either measure peak density independently or fix one member of the scale pair from authoritative physics; stress peaks alone cannot justify both.

The preceding run `20260828T131623Z-7598a32-e67e73` / job `55637761` failed before numerical tests because its launcher omitted `PYTHONPATH`. It was fetched as incomplete, no scientific output was interpreted, and commit `9caa154` corrected the launcher environment only.

## Collective-context structural diagnostic

Status: passed, completed, fetched, and checksum-verified. This is a structural diagnostic of old DDD context and is explicitly excluded from production parameterization.

- Git commit: `1a147e0`
- Run ID / Slurm job: `20260828T130729Z-1a147e0-710710` / `55637740`
- Result: `COMPLETED`, exit `0:0`; application and finalization exit zero
- Environment: `anaconda/2025.12`, Python 3.13.9; one CPU, 2 GB, no GPU; 2 s wall clock and 103.27 MB peak memory
- Two tests passed: depinning-count clustering and native branching-response/hazard-crossing diagnostics
- All six staged DDD-context files match their audited SHA-256 hashes
- Fetched result: `hpc3-results/asb-drx-independent/20260828T130729Z-1a147e0-710710`
- Result archive SHA-256: `66cc05879596513d65dab9a789541eda5d507bc3e090cb7aebe3b9b254f690e7`
- Diagnostic JSON SHA-256: `ff9335351ab57fd3d1e0d62d9f47161d20eaace30d31502c369f117f85866628`
- Unit-test log SHA-256: `0bc3b563ba8ae1be2a1256d6af136dc858546b47a60c1ea19b28bbf2dad456da`
- Runner retrieval status: `verified`

The count histories show stronger multi-hit clustering at higher density. The native one-step contact operators have zero spectral-radius proxy, with only 11 survivor-redistribution samples at the densest condition. This does not identify a causal branching law; it leaves the collective extension as an ablation and the independent EXP-floor law as baseline physics.

An earlier packaging attempt, run `20260828T130634Z-810e232-8c908d` / job `55637733` at commit `810e232`, failed before scientific execution because the archive omitted `src/asb_drx/analytical.py`, which package initialization imports. The application exit was 1 and finalization exit 0. The partial archive was fetched and marked incomplete; no diagnostic result was interpreted. Commit `1a147e0` corrected only the source manifest before the successful rerun.

## Thermodynamic verification kernel

Status: passed, completed, fetched, and checksum-verified. This is a dimension/sign/conservation fixture, not a calibrated phase-field simulation.

- Git commit: `07d589a`
- Run ID / Slurm job: `20260828T132308Z-07d589a-1e7eac` / `55637784`
- Result: `COMPLETED`, exit `0:0`; application and finalization exit zero
- Environment: `anaconda/2025.12`, Python 3.13.9, NumPy 2.3.5; one CPU, 2 GB, no GPU; 1 s wall clock and 36.91 MB peak memory
- Six tests passed: discrete variational derivative, nonincreasing unloaded relaxation, conservative reservoir transfer/overdraw rejection, exact work-ledger closure/excess rejection, circular-nucleus signs/stationarity, and invalid-range rejection
- The 100-step accepted relaxation decreased free energy from `0.2820377174` to `0.2767598660 J m^-2` (relative change `-0.0187133`) with zero step halvings
- The fixture critical radius is `2.5e-7 m`; radius rates are negative below, zero at, and positive above the critical value
- Result archive SHA-256: `d4edda846b89713fc754eb630a2e80c2ecbda3621f7f2d77cc6ab50a7a9c992c`
- Verification JSON SHA-256: `c390cfbd67f2d35161f6d351aedbf93e73294e97a7ad6d880b47efd3610b1495`
- Unit-test log SHA-256: `6554142b35b394f08a34b9dbd6e8494a8a50487854cabfbf1203d20705e4db85`
- Runner retrieval status: `verified`

This passes the first executable thermodynamic gate. It does not yet verify coupled mechanical/thermal work closure, a diffuse 2-D nucleus, grid/timestep convergence, restart equivalence, orientation/grain invariants, or any material response.

## Diffuse 2-D nucleus, refinement, and limited-state restart

Status: passed, completed, fetched, and checksum-verified. This extends the generic thermodynamic fixture; it remains uncalibrated.

- Git commit: `a4a0bf0`
- Run ID / Slurm job: `20260828T132810Z-a4a0bf0-5c7bfe` / `55637801`
- Result: `COMPLETED`, exit `0:0`; application and finalization exit zero
- Nine tests passed, including diffuse 2-D subcritical shrinkage/supercritical growth, exact segmented restart for the complete current kernel state, and final grid/timestep refinement below 5%
- The derived diffuse-interface values are `gamma=0.4714045 J m^-2` and `R_c=2.3570226e-6 m`; the 128-grid subcritical radius change is `-3.3636e-9 m` and the supercritical change is `+1.1736e-9 m`
- Fixed-domain grid changes at 64/96/128 points give a final relative change of `0.003844` (0.384%)
- Fixed-time timestep changes at `2e-4`, `1e-4`, and `5e-5 s` give a final relative change of `2.0716e-5` (0.0021%)
- Result archive SHA-256: `6176c7c80e537d96e66f0b078d66c9b21538e29936e9a0ebfca000bc162e9695`
- Verification JSON SHA-256: `5a5cafc6c0c8c33a43fd4b4b6458bde2a936f2edc722a8765101f88456f47097`
- Unit-test log SHA-256: `8d3c5d35e4a9d15c393cadc5f04310f806cba65b75542b08377626a4c26166ad`
- Runner retrieval status: `verified`

The exact restart claim is deliberately limited to the current deterministic state (`eta`, time, accepted-step count). It does not satisfy the production restart gate, which must additionally cover mechanics, temperature, all density reservoirs, orientation, collective/embryo state if enabled, controllers, and RNG streams.

## Finite-loading thermomechanical material point

Status: passed, completed, fetched, and checksum-verified. The parameters are generic verification fixtures, not a material fit.

- Git commit: `6920914`
- Run ID / Slurm job: `20260828T133324Z-6920914-866ee2` / `55637814`
- Ten tests passed: the five analytical-law tests plus homogeneous-rate/finite-loading closure, exact incremental work partition, zero-storage heat limit, impossible-storage rejection, and exact restart of the complete current material-point state
- Over 100 steps, cumulative external work `3.0704347e6 J m^-3` equals elastic increase `1.5846880e6`, stored line energy `2.4164294e3`, and heat `1.4833302e6 J m^-3`; recorded closure error is exactly zero
- The fixture temperature rises from `1000` to `1000.4238086 K`; this is a bookkeeping response, not a prediction
- Result archive SHA-256: `bceff3420550918a82a604d2d058a716b08b5591a40cf6692c5b0328f2765c08`
- Verification JSON SHA-256: `d33ac83d33149cbc220cb1f6331416491bc0077d975e9fdf4ec8e059c955d219`
- Unit-test log SHA-256: `263fab5d67fb43c30d721e7dce6c98df48d1724cc561810b6f8e13ad39e29621`
- Runner retrieval status: `verified`

This proves the homogeneous discrete ledger and finite elastic-loading limit. It does not yet prove spatial equilibrium, conduction, thermoelasticity, multiple slip, recovery/annihilation heat, DRX coupling, localization, or ASB.

## Periodic common-stress thermomechanical shear layer

Status: passed, completed, fetched, and checksum-verified. This is a generic spatial mechanism/control fixture, not ASB or material validation.

- Git commit: `ae2fdf9`
- Run ID / Slurm job: `20260828T133943Z-ae2fdf9-d7f5b0` / `55637821`
- Fourteen tests passed: analytical and material-point regressions plus homogeneous reduction, global mechanical/thermal ledger closure, conductive damping/mean conservation, and exact current-state restart
- A 0.25 K sinusoidal perturbation decreases from `0.1767767` to `0.0889227 K` standard deviation while mean temperature rises from `1000` to `1000.4238086 K` through plastic heating
- Cumulative mechanical and thermal closure errors are `-8.27e-10` and `-2.86e-7 J m^-3`, respectively, against `3.0704346e6 J m^-3` external work
- Result archive SHA-256: `8f3b762c413171767cd3ce2dbe17e8087ae9a7edffbe3b7403a839c384ce9760`
- Verification JSON SHA-256: `88f9129aa54d910fbbc256f88f546e50af5b1219d3f0b5af917f34b795bc3ca0`
- Unit-test log SHA-256: `a6d7e0809fafb5091adf560a5a8e22253ebd7ac69cfe1f6a0fc23b568d2605ab`
- Runner retrieval status: `verified`

The preceding run `20260828T133809Z-b3a2116-395502` / job `55637819` failed its conduction control because the explicit diffusion Fourier number was about 17.9, above the 1-D stability bound of 0.5. Positivity had failed to detect the oscillatory instability. The failed archive was fetched and no spatial scientific result was interpreted. Commit `ae2fdf9` added the explicit CFL acceptance rule and used a stable fixture spacing before rerun.

The common-stress layer is the 1-D quasistatic simple-shear equilibrium limit. It does not include displacement-resolved multidimensional mechanics, thermoelasticity, physical boundaries, DRX coupling, or a calibrated instability/localization test.

## Physical-grain metric and lifecycle

Status: passed, completed, fetched, and checksum-verified. This verifies generic classification invariants; it is not a nucleation, orientation-evolution, or DRX kinetics model.

- Git commit: `5e8fabc`
- Run ID / Slurm job: `20260828T134840Z-5e8fabc-192eaf` / `55637844`
- Ten tests passed: empty allocated labels, sub-resolution support, disconnected islands, periodic component wrapping, persistence and promotion, growth history, symmetry-equivalent rejection, invalid-lineage rejection, provenance-preserving retirement, and exact tracker checkpoint round trip
- The demonstration has three allocated labels but only two resolved/physical grains; one valid persistent child is promoted, one empty label is not, and the promoted area fraction is `0.1111111`
- Result archive SHA-256: `ac45dff29a25defd70dc7e81496a40603c4a1e79339ed677f83db2b89498b325`
- Verification JSON SHA-256: `66e72cf9445ed0e3ff1407ebb2559ede831a72c52627382c81c77af2979d2a85`
- Unit-test log SHA-256: `9b7aeb431664dc30684fb9ac5710e09aa786098e222789030ae74e9445588b5a`
- Runner retrieval status: `verified`

The verified distinction is between allocated labels, periodic topology components, currently resolved labels, physical grains, and promoted recrystallized grains. The generic purity, area, persistence, misorientation, and scalar symmetry settings are test fixtures. They must be replaced by interface-resolution studies and the selected crystal/material definition before production use. This gate does not supply stochastic trials, nucleation energetics, an orientation manifold, multi-order-parameter dynamics, collisions, stored-energy relief, or DRX coupling.

The preceding run `20260828T134738Z-39eff44-ef427b` / job `55637837` failed before any grain test because its input manifest omitted `src/asb_drx/analytical.py`, which package initialization imports. It was fetched as incomplete and no scientific result was interpreted. Commit `5e8fabc` corrected only the explicit package dependency before the successful rerun.

## Constrained multi-order phase-field dynamics

Status: passed, completed, fetched, and checksum-verified. This is an isotropic variational baseline with generic fixtures, not a physical nucleation, DRX, material, or ASB result.

- Git commit: `d1de48a`
- Run ID / Slurm job: `20260828T135602Z-d1de48a-eb7ef1` / `55637866`
- Seventeen tests passed: all ten grain lifecycle regressions plus projected variational consistency, energy/simplex acceptance, pure-parent invariance, circular-nucleus growth signs, label-permutation symmetry, exact complete-current-state restart, and fixed-label tracker coupling
- The analytical binary boundary energy is `0.4714045208 J m^-2`, giving `R_c=2.357022604e-6 m` for the generic `Delta f=2e5 J m^-3` fixture
- After 200 accepted steps, the subcritical equivalent radius changes by `-1.4843269e-9 m` and the supercritical radius by `+4.2256725e-10 m`; both declared free energies decrease
- Maximum final pointwise simplex error is `1.1102230e-16`
- Result archive SHA-256: `9314bcf5424fe5a1c36e788f7479fbe4e4238d8db55992cf537c29803fcda584`
- Verification JSON SHA-256: `9546c97a9887d95584822edd819ab7b71bc0d6daa81d3face1b10b41c635e4a2`
- Unit-test log SHA-256: `75d4454d251d951a65de585c24b1b160e8448a103c3888063207dda1ca205e94`
- Runner retrieval status: `verified`

The endpoint-flat bulk interpolation is essential to the passed pure-parent invariant: a pre-existing lower-energy child can drive boundary motion, but its allocated zero field is not generated everywhere by a constant chemical force. The tracker sees only evolved physical support and never changes the two-label allocation. The exact restart claim covers the kernel's current fields, time, and accepted-step count.

This gate does not cover a crystallographic orientation manifold, anisotropic/misorientation-dependent GB energy or mobility, triple junctions, elastic and stored-density driving, stochastic trial allocation, energetic candidate acceptance, collision/coalescence, thermomechanical coupling, material-scale convergence, or external validation.

## Explicit stored-dislocation-energy grain-growth coupling

Status: passed, completed, fetched, and checksum-verified. This is a generic isolated relaxation ledger, not material, nucleation, DRX, or ASB validation.

- Git commit: `8fb6236`
- Run ID / Slurm job: `20260828T140503Z-8fb6236-7dab82` / `55637879`
- Fourteen tests passed: seven multi-order regressions plus explicit line-energy driving, pure-parent/no-reset behavior, lower-density child growth, exact free-energy/heat closure, common-density-offset invariance, continuous support-weighted stored energy, and exact complete-current-state restart
- The fixture uses `e_line=5e-9 J m^-1` and grain densities `(5e13, 1e13) m^-2`, yielding the explicit `Delta f=2e5 J m^-3`; these are generic values, not a fit
- Over 200 accepted steps, stored energy changes by `-5.5970269e-9 J m^-1`, interface/order energy by `+2.4351482e-9 J m^-1`, and heat by `+3.1618787e-9 J m^-1`; recorded closure error is zero
- The resulting fixture temperature changes from `1000` to `1000.0000035289 K`
- Result archive SHA-256: `8b01c3259e9fb88c008e69ea8b475d0ced40cbbf0bff301f3135b70ba7cc5437`
- Verification JSON SHA-256: `baeb9c55e318b47d206633e7fff66eb615a62502d61b124c068cc4c5705fbcdd`
- Unit-test log SHA-256: `2a0f11773dac77605ce2007b06fdd131a82286e1799ac640521dce4178996c7d`
- Runner retrieval status: `verified`

The isolated driving energy is no longer an independently prescribed phase offset: it is `e_line(rho_parent-rho_child)`. Adding a common density offset leaves the binary dynamics unchanged. The per-grain densities do not reset during phase motion; only their continuously interpolated volume support changes. The exact restart includes all current fields, temperature, time, and accepted-step count.

The current phase step has no simultaneous external mechanical work, deformation/storage evolution, recovery/annihilation products, conduction, temperature-dependent mobility, stochastic trial rate, or material calibration. Those channels must be coupled with one non-duplicated global ledger.

## Auditable nucleation candidate decision

Status: passed, completed, fetched, and checksum-verified. This verifies analytical identities and decision plumbing with generic fixtures; it is not a calibrated nucleation or DRX result.

- Git commit: `e01a0ce`
- Run ID / Slurm job: `20260828T141129Z-e01a0ce-969f27` / `55637888`
- Six tests passed: stationary critical barrier/escape radius, Poisson--Arrhenius expression, temperature/driving monotonicity, deterministic external-draw acceptance, distinct physical/numerical rejection reasons, and invalid-input rejection
- The generic fixture gives `R_c=1e-9 m`, zero-excess radius `2e-9 m`, `Delta G*=3.1415927e-19 J = 22.75446 k_B T` at `1000 K`, and event probability `1.3117753e-5`
- Result archive SHA-256: `a0c68aa8102fd527ae67620d0a005672d86add9eda36af402b26f8204a648f1e`
- Verification JSON SHA-256: `de2b4e3dd79caebde091f7644a14e0e7ca7125225fa2f4ba7d646d4e1d4dca3a`
- Unit-test log SHA-256: `abbaeaed03d1d067fe254815a4be9d523a727f7368b7653685f8a283d731e02c`
- Runner retrieval status: `verified`

This candidate barrier is the classical interfacial/stored-energy barrier and is explicitly distinct from the EXP-floor dislocation-slip barrier. The kernel accepts an externally supplied uniform draw so RNG lineage can be owned by a later full-state checkpoint. It reports a decision but cannot allocate a label or assert a physical grain.

The represented thickness and areal attempt rate in the fixture are uncalibrated. A physical model still requires independently justified values, mesh/time-step invariant eligible-event intensity, spatial site physics, overlap handling, energy accounting through barrier crossing, RNG-complete restart, and held-out DRX-onset validation.

## Shared thermomechanical/phase global ledger

Status: passed, completed, fetched, and checksum-verified. This is a generic binary aggregate with homogeneous mechanics and a periodic 2-D phase domain; it is not spatial mechanics, material, DRX, or ASB validation.

- Git commit: `c9708bb`
- Run ID / Slurm job: `20260828T141957Z-c9708bb-f4aa93` / `55637907`
- Twenty-five tests passed: five material-point, seven multi-order, seven stored-energy, and six coupled tests
- Coupled tests cover exact pure-parent/material-point reduction, exact zero-mechanics/phase reduction at the shared accepted interval, direct global closure, zero-child preservation, one accepted time increment, and exact complete-current-state restart
- Over 100 steps, external work is `4.578161157e6 J m^-3`; the channels are elastic `4.094594770e6`, stored `526.654694`, interface/order `0.476057`, mechanical heat `483038.626021`, and phase heat `0.630555 J m^-3`
- Cumulative global closure error is `1.4482e-9 J m^-3`; cumulative thermal roundoff is `-9.5612e-7 J m^-3`, about `2e-12` of total heat
- Result archive SHA-256: `eb8f125587e65538c11c0fb0060c3307b43e77ceca36614973bd82237cf860e6`
- Verification JSON SHA-256: `1429171ade780cf5b5626ffd98e9ac87bef7232e91c931a2eb50cc281b563daa`
- Unit-test log SHA-256: `e4bedcb2818e10e16709d29508fd2b8e6a206b385e84145bcce9696abf2c6cc0`
- Runner retrieval status: `verified`

The first run `20260828T141817Z-f40c3fd-ee4135` / job `55637902` passed the global mechanical balance and 23 of 25 tests but failed two thermal comparisons. The cause was a tolerance below the representable energy change when adding a tiny `Delta T` to a `1000 K` state; this could also cause unnecessary timestep halving. No scientific result was interpreted from that incomplete run. Commit `c9708bb` added a heat-capacity/temperature-scaled floating-point floor and made the phase-limit comparison use the actual shared accepted interval.

The passing kernel updates grain-wise EXP-floor plastic rates and dislocation storage under common stress, then relaxes the phase fields with the updated stored energies. Direct initial-to-final energy differences prevent storage consumed by boundary motion from being counted twice. Both heat channels update one temperature, and all current coupled state restarts bitwise exactly.

This remains an aggregate verification limit. It has no displacement-resolved stress field, heterogeneous temperature, conduction in the phase domain, recovery/annihilation, multiple slip, physical boundary conditions, calibrated GB kinetics, stochastic allocator, localization convergence, or external data comparison.

## Periodic 2-D spatial thermomechanical/phase control

Status: passed, completed, fetched, and checksum-verified; generic common-stress control, not localization, material, DRX, or ASB validation.

- Git commit/run/job: `3224ef1` / `20260828T142706Z-3224ef1-bdf10f` / `55638019`
- Fifteen tests passed: shear-layer and aggregate regressions plus homogeneous reduction, 2-D conduction, global spatial closure, zero-child invariance, and exact spatial restart
- Temperature standard deviation decreases from `0.1767767` to `0.0746919 K`; mean rises from `1000` to `1000.00011856 K`
- External work is `3960.8591 J m^-3`; cumulative global/thermal closure errors are `1.0718e-11` and `-3.8496e-8 J m^-3`
- Archive / JSON / test-log SHA-256: `6790e8d5b8bfaeee8170ff18b3cf9c283543d2a76bd2f7b34ee4317f659b2410`, `8283528107ea74f099c6b2e8fe610b76a9fbd67289df147741ea14fe8d1b3cf1`, `b8d775b1c90eb3e90f0475dc5c9def8cd2f9691c8715f129d102169eb88aa5e0`

The phase heat distribution uses a normalized local Onsager-dissipation proxy while preserving the exact global free-energy decrement. This is declared numerical closure. Physical ASB still requires sustained growth against controls, mesh/timestep-converged onset and band width, realistic boundaries, and justified mechanical equilibrium.

## Environment smoke

Status: passed, completed, fetched, and checksum-verified.

Purpose: verify staging, the discovered system Python, deterministic arithmetic, manifest emission, Slurm provenance, fetch, and checksums. This smoke is not a physical-model result and cannot pass any scientific gate.

- Run ID: `20260828T034723Z-a5dd798-f2bdfd`
- Slurm job: `55633650`, `COMPLETED`, exit `0:0`
- Node/time/resources: `hpc3-l18-05`, 11 s wall clock, one CPU, 12.37 MB peak memory, no GPU
- Environment: no modules; `/usr/bin/python3` 3.9.25, GCC 11.5 runtime, Linux 5.14/glibc 2.34
- Deterministic integral: 0.6321205588338278 versus 0.6321205588285577; absolute error 5.2701176755931556e-12 below 1e-10 tolerance
- Fetched result: `hpc3-results/asb-drx-independent/20260828T034723Z-a5dd798-f2bdfd`
- Result JSON SHA-256: `6e67b128dff1d9bf7b97f13379fc910db30d1fd1818ee95811524eddfcedee13`
- Source snapshot SHA-256: `e564f812a36125ffdbaeefb781d553b30a2642325cf9adde362442c1e87c4458`
- Exact inputs: `run.sh` `8d2c11c0...`, `smoke.py` `4e3088e8...`
- Runner retrieval status: `verified`

## Conjunctive localization metric gate

Status: passed, completed, fetched, and checksum-verified; generic acceptance-rule verification, not a localized simulation or ASB validation.

- Run `20260828T143238Z-ca7ab69-74cc1d`, commit `ca7ab69`, Slurm job `55638898`, Python `3.13.9`
- Six tests passed: homogeneous and band geometry, conjunctive criteria with persistent restart, matched-control temperature and running-peak softening, under-resolved rejection, and joint onset/width refinement
- Fixture criteria: active fraction `<= 0.4`, temperature excess over control `>= 20 K`, softening `>= 0.1`, width/interface ratio `>= 3`, persistence `3` accepted states, refinement tolerance `5%`
- Wall clock `0 s`; peak batch memory `36.74 MB`
- Archive SHA-256 `9590bf4de693359725d90d55bb680723b7a9a6bdb991108ac1c1a73a5cc96c9a`; test transcript `61294b45f6d2908c7aaeb9d7335132cf85d34cbd1b294dacea6aa76684699b47`; final marker `b3c3fd348a971bf7ef8e94d83f04bf01e6818b28fe0f68ecb87bdc63543590c8`

The preceding immutable run `20260828T143111Z-d608655-a7aea7` / job `55638861` failed one assertion because the supposed negative history contained three consecutive qualifying snapshots after an interruption. That is exactly the declared persistence condition. The failed run was fetched incomplete and retained; commit `ca7ab69` corrected the fixture to only two post-interruption snapshots while preserving an explicit positive restart case.

No simulated trajectory has yet passed this classifier. The thresholds are visible generic scaffolding and cannot be presented as material calibration or experimental validation.

## Common-equation mechanism ladder

Status: passed, completed, fetched, and checksum-verified; generic negative controls, not an instability-boundary, DRX, material, or ASB result.

- Run `20260828T144000Z-c9dc4ab-7b33f9`, commit `c9dc4ab`, Slurm job `55640278`; 17 tests passed in the application transcript
- Cases: unloaded isothermal relaxation; isothermal deformation with phase disabled; isothermal DRX; thermal high-rate with phase disabled; coupled intermediate-rate; coupled high-rate
- Every thermal case is compared against an automatically generated isothermal twin with identical rate and phase switch
- Phase-disabled fields remain exact invariants with zero phase heat; fixed-temperature cases route all generated heat to an explicit bath and close `Delta E_thermal + Q_bath = Q_mechanical + Q_phase`
- All six cases are nonlocalized: minimum active fraction `0.90733-1.0`, maximum matched-control temperature excess `0.12503 K`, maximum softening `0`, and effective width about `15.54-16 micrometers`
- Wall clock `33 s`; peak batch memory `37.79 MB`
- Archive / JSON / test transcript SHA-256: `5df65c272d353564e355cea7b615ced97e74c4ae7ba3fa4bc1a21149f01a0f7a`, `f5dfbd2c14dfe946baf70886321446a1fe179e5fbc933c020a3603dc7952f215`, `bd1f0e8804687cf707f7c1bb2309931e2e1ae8ad58bd0af207c70b0db64a5753`

These 16-by-16, 20-step fixtures verify attribution mechanics only. Their `10` and `1000 s^-1` rates, periodic boundaries, perturbation, and generic localization thresholds are not a physical sweep and may not be interpreted as locating an ASB or DRX boundary.

## Finite-wavenumber thermal/storage stability

Status: passed, completed, fetched, and checksum-verified; generic frozen-state tangent screening, not a material or nonlinear ASB prediction.

- Run `20260828T144457Z-535a0ff-662a0e`, commit `535a0ff`, marker-owning Slurm job `55640502`; ten analytical/stability tests passed
- Analytical temperature and density rate tangents match centered differences; the 2-by-2 Jacobian matches the nonlinear local right-hand side; conduction shifts only `J_TT` by `-alpha k^2`; impossible storage energy is rejected
- At `tau=300 MPa`, `rho=5e13 m^-2`, `T=1000 K`, the generic Taylor ratio implies an unphysical `120 GPa` local activation stress, so the numerical spectrum is not material evidence
- Modes 1, 2, 4, and 8 on a `16 micrometer` periodic domain have maximum real eigenvalues `4.8071479`, `4.8071403`, `4.8071384`, and `4.8071379 s^-1`; the near wavelength-independent positive branch is the forest-storage tangent, while thermal diagonals become increasingly negative with wavenumber
- Wall clock `1 s`; peak batch memory `36.47 MB`
- Archive / JSON / test transcript SHA-256: `4193778453024a1c69fe48cc54301d48e4d1e3f81f07cdc9fa681321024635a6`, `2cf9dd1efb0fe49f6c07065e1b92a7d01a055526fe8c640d6d165a07d0aa24af`, `ed550fdce19b249e57ecc2f45dde286a1cc6f9978e5cf0dfbbd2ec4060595430`

The first silent staging attempt scheduled job `55640458` even though the local journal remained `PREPARED`; manual recovery then scheduled `55640502` from the identical immutable source. Both completed, but only `55640502` owns the fetched final marker and result archive. The duplicate is a provenance incident, not an independent replicate.

## Required future records

Future records must extend the passed generic gates to production-complete restart including RNG/allocator state, material-scaled grain and localization criteria, multidimensional equilibrium, matched-control mechanism runs, localization convergence, parameter uncertainty, and external validation, with run IDs, commits, configurations, tolerances, and checksums.

## Legacy computations retained as context only

All controls used exact versioned source snapshots, `anaconda/2025.12` (Python 3.13.9), one CPU, 4 GB, account `SDILLON1`, partition `free`, QoS `low`, and no GPU. All completed with exit `0:0`; local retrieval is checksum-verified and remote copies remain intact. The machine-readable record is `evidence/legacy_controls/hpc3_regressions.json`.

| Control | Run / job | Wall / peak memory | Numerical reproduction | Conservative scientific result |
|---|---|---:|---|---|
| v32, 30,000 s^-1 | `20260828T035427Z-de0cbdb-e55043` / `55633674` | 29:12 / 429.95 MB | All finite values pass `atol=1e-10`, `rtol=1e-9`; 18 zero-variance correlations are `NaN` versus roundoff near zero | ASB-like regression retained, but physical ASB is not established without sustained localization and mesh-converged width |
| v33, 1,000 s^-1 | `20260828T040152Z-c988539-feadb5` / `55633691` | 45:47 / 613.20 MB | Structural, not trajectory-level: 165 hazard births versus reference 119 | Labels rise 12 to 177 while topology remains 12; all 165 label births are rejected as physical DRX |
| v34 coupled, 1,000 s^-1 | `20260828T040212Z-c988539-ee4dad` / `55633694` | 22:55 / 495.43 MB | Bookkeeping failure reproduced; detailed trajectory diverges | Candidate active/new/promotable/age maxima are all zero, as are births; therefore this is not a candidate-without-promotion case |

The v33/v34 stored reference folders lack an immutable source/environment record. Identical recorded parameters are insufficient for trajectory reproduction. This is a provenance failure in the legacy evidence, not a basis for parameter tuning.

These legacy computations do not calibrate, validate, or regression-gate the new model. Physical parameter optimization remains pending one authoritative target material and strength/rate/temperature dataset.
# Single-glider DDD generic-fixture gate

HPC3 run `20260828T151810Z-d31c6a4-d3af25` / job `55641106` passed all fourteen EXP-floor, stability, and fixture tests and was fetched with verified checksums. It verifies the exact entropy-bearing `H-k_B T S` barrier mapping, the DDD `q=2 b sqrt(rho)` and `q^4=16 b^4 rho^2` prefactor, forward/peak rate closure, and equality of the stored line energy to the DDD line tension `0.5 G b^2=2.46016e-9 J m^-1`.

The fixture predicts peak densities `[8.5383e15, 7.0442e15, 5.9295e15, 5.0773e15, 4.4117e15] m^-2` and strengths `[122.536, 112.835, 104.838, 98.150, 92.488] MPa` at `[850, 900, 950, 1000, 1050] K` and `4.5 s^-1`. These are generic governing-equation results, not material calibration. The run used Python 3.13.9, 62 s wall time, 1 s CPU, and 37.51 MB peak memory. Archive, report, and unit-test hashes are `b12365cd222825bbe9f91493c280c8f364999613cbf42ef19db4111cb65a1eb4`, `59e059d3c2a709155078a52c6b6d06d0f8ba57fe47d5b819d68113dc77796f53`, and `3f0561561b51f8af92fad8ded9f57b82ab91a37be4c106bb9fe09be61cd78702`.

## Analytical boundary-surface gate

HPC3 run `20260828T152809Z-db81077-68e54d` / job `55641308` passed all fifteen tests and was fetched with verified checksums. The prospective boundary was committed before execution and is exactly `rho=rho_peak(T, rate)`. The verified surface contains fifteen points over 850--1050 K and rates `4.5`, `450`, and `45000 s^-1`; its exact rate scaling is `rho_peak proportional to rate^(2/p)`. The DDD upper density `3e16 m^-2` lies at ratios 3.514--6.800 above the analytical peak across its temperature range and is labeled only `post_peak_collective_candidate`.

The run used Python 3.13.9, 183 s wall time, 1 s CPU, and 169.89 MB peak memory; the delay was environment/shared-filesystem startup. Archive, report, unit-test, inventory, checksum, final-marker, and source-archive hashes are `35a807237d8cfcdb9f883b0be7f7311207d5669f3f2023b67e59ec9a912b2c09`, `ceaa8f302e135f8fbdc0ba6688566e452c6361562737f93f80f4c814c78c1ba5`, `5d9004a01d61f586c460b05fdf4c0e8e5900769a70942eae27952f5e24254b3e`, `baf7abe2be9771fdeed9c205437c775f677145e28c82deca4301d7d0135edc60`, `adbc399948ab3a75319a90323d226c216f3cc6fcb2242a2b7f2a652ef03699bc`, `9dda03e3962f973699d2e882e2bc972c94b939a7df28a3171969e40356c6b5ea`, and `3221ee99a0d9a75a06080d7754672cfe01de861d2fd7b7b0886fab4309d7b393`.

## Analytical-boundary spatial smoke

Preflight run `20260828T153642Z-c48cad2-f5273f` / job `55641810` failed before simulation: the staged archive omitted `shear_layer.py`, required by an upstream test, and a new assertion required bitwise equality for two analytically equivalent stresses that differed by `~3e-16` relatively. The retrieved partial archive, unit-test log, checksum file, final marker, and source archive have hashes `0f609e4c44f92d6a7c75218cbab1269ca2531af669e877cf960e5d31d6d20b0e`, `b47490107a25cd04274ae14864589e84cd962097b8eff722931cfc7ba2df8199`, `5e8e814ac7afed98528dbe4f84b3b0bc0b22060b1b90f4a6aa2465fbbea09964`, `d8bbd11bb96379ea30507f6d0c1a44487ff2c8561a2b3150ded58fa2d228443f`, and `8126a82f29bc4383fe408dcd48727f45e39e212dee38ce8cdbdab31aaabf4e52`. Packaging and tolerance were corrected without changing the smoke equations or condition.

Corrected run `20260828T154200Z-9d7ed90-e57dd1` / job `55642217` passed all 21 tests and completed the single-job smoke at 950 K, `45000 s^-1`, and the exact analytical peak density `5.9295381e17 m^-2`. This rate is explicitly outside the source DDD rate and tests numerical coupling only. Both 16² and 32² grids reached 0.09 applied shear in 100 steps. The fine case ended at `1.011091871 GPa`, reached `976.7585 K`, and had `26.7399 K` maximum excess over its matched isothermal control. Stress and maximum-temperature refinement changes were `5.04e-9` and `1.10e-6`; maximum ledger errors were below `5.73e-7 J m^-3` globally and `3.76e-8 J m^-3` thermally.

The case is a nonlocalizing control, not ASB: minimum active fraction is `0.9999988`, maximum softening is `0.0356`, and the strict classifier fails plastic concentration and post-peak softening. This is a numerical smoke pass but a no-go for a sparse production array under frozen common stress. Archive, report, test, inventory, checksum, marker, and source hashes are `04ddcb8376aec3b608d138c97892f09b76602d2ec4d910cde1595f94765afed9`, `f71525a95bb3adb4042c05694190d1d892c40fcdee045330eb108af485d7fd8e`, `bd6cf4a31a4a691703d4ef408e9a2a27239973ed8de304b7673fedddb74b281c`, `528eeed7ea609245ce259aaa4a40be1580f6b8176e0ffc84931c690a6e92577f`, `d2750b3d992853c8d196f13a2900990c3b99026358a8b30bfb19f96980ea10a5`, `db7812602bd97ca115e9fe13aa5cb6da7da2962ebb76df9c891febbde864f16f`, and `73e0926f0a1999696b09d67771bfa6edcd0910b54e13b3bbba5546d396bc1242`.

## Periodic antiplane local-mechanics gate

HPC3 run `20260828T155312Z-266dda8-d6b092` / job `55642641` passed all six tests and was fetched with verified checksums. The exact Fourier transverse projection recovers common stress and elastic energy for uniform plastic shear, relaxes a longitudinal compatible mode, creates the correct local stress for a transverse band and diagonal mode, has zero reported spectral equilibrium residual in the 64² diagnostic, and closes the equilibrium-to-equilibrium midpoint identity `external work = plastic work + elastic-energy change`. The diagnostic closure is `7.4506e-9 J m^-3` against a `~2.015e6 J m^-3` elastic change.

The run used one CPU, 1 s wall time, and 21.32 MB peak memory. Archive, report, tests, inventory, checksum, marker, and source hashes are `4f87e4044e11973baafb23d4fdf64e4e4b83d7829363a74768d73fa475f7760e`, `121ce8aa887cc9c5f35492d3765ef395a6fadb86d1b2fda1fddb24ee6a4e92a8`, `8f1045096ffc36a34e7b1da13a617e9a60ce31c42153bc7e4e8de9aefba5da51`, `cb1702a7389b91d3004cf7f0be8628a1906e921646ae4f2f12c8346c6bcb0f73`, `cee60f6f3c8da5f0760775899c149a5fee93163ac1e093a7ccd37c0ad140ab3b`, `ea2f02755627ae3ca71e56e5de2be0067f3e12b50947c4e9dbabd1c0926918a3`, and `1f9f3bfbbc2ae9e3f9c919039f47c939df329bb67f5cc4737eb0347572c9fc59`.

The preceding run ID `20260828T154906Z-266dda8-f8fc4a` produced duplicate jobs `55642418` and `55642419` after a delayed submission receipt. Both ran concurrently against one remote directory; the six tests passed, but archive finalization collided and both jobs exited 1. It is rejected as scientific evidence and retained as an orchestration failure.
