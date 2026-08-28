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

## Required future records

Future records must extend the passed isolated gates to coupled multi-order-parameter evolution, full-state restart, material-scaled grain criteria, multidimensional equilibrium, localization convergence, parameter uncertainty, and external validation, with run IDs, commits, configurations, tolerances, and checksums.

## Legacy computations retained as context only

All controls used exact versioned source snapshots, `anaconda/2025.12` (Python 3.13.9), one CPU, 4 GB, account `SDILLON1`, partition `free`, QoS `low`, and no GPU. All completed with exit `0:0`; local retrieval is checksum-verified and remote copies remain intact. The machine-readable record is `evidence/legacy_controls/hpc3_regressions.json`.

| Control | Run / job | Wall / peak memory | Numerical reproduction | Conservative scientific result |
|---|---|---:|---|---|
| v32, 30,000 s^-1 | `20260828T035427Z-de0cbdb-e55043` / `55633674` | 29:12 / 429.95 MB | All finite values pass `atol=1e-10`, `rtol=1e-9`; 18 zero-variance correlations are `NaN` versus roundoff near zero | ASB-like regression retained, but physical ASB is not established without sustained localization and mesh-converged width |
| v33, 1,000 s^-1 | `20260828T040152Z-c988539-feadb5` / `55633691` | 45:47 / 613.20 MB | Structural, not trajectory-level: 165 hazard births versus reference 119 | Labels rise 12 to 177 while topology remains 12; all 165 label births are rejected as physical DRX |
| v34 coupled, 1,000 s^-1 | `20260828T040212Z-c988539-ee4dad` / `55633694` | 22:55 / 495.43 MB | Bookkeeping failure reproduced; detailed trajectory diverges | Candidate active/new/promotable/age maxima are all zero, as are births; therefore this is not a candidate-without-promotion case |

The v33/v34 stored reference folders lack an immutable source/environment record. Identical recorded parameters are insufficient for trajectory reproduction. This is a provenance failure in the legacy evidence, not a basis for parameter tuning.

These legacy computations do not calibrate, validate, or regression-gate the new model. Physical parameter optimization remains pending one authoritative target material and strength/rate/temperature dataset.
