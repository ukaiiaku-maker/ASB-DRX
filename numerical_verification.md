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

Closure material-point validation, relaxation/free-energy monotonicity, content/work ledgers, nucleus limits, homogeneous limits, timestep/grid convergence, exact restart, schema tests, and label/grain invariants will be appended with run IDs, commits, configs, tolerances, and checksums.

## Legacy computations retained as context only

All controls used exact versioned source snapshots, `anaconda/2025.12` (Python 3.13.9), one CPU, 4 GB, account `SDILLON1`, partition `free`, QoS `low`, and no GPU. All completed with exit `0:0`; local retrieval is checksum-verified and remote copies remain intact. The machine-readable record is `evidence/legacy_controls/hpc3_regressions.json`.

| Control | Run / job | Wall / peak memory | Numerical reproduction | Conservative scientific result |
|---|---|---:|---|---|
| v32, 30,000 s^-1 | `20260828T035427Z-de0cbdb-e55043` / `55633674` | 29:12 / 429.95 MB | All finite values pass `atol=1e-10`, `rtol=1e-9`; 18 zero-variance correlations are `NaN` versus roundoff near zero | ASB-like regression retained, but physical ASB is not established without sustained localization and mesh-converged width |
| v33, 1,000 s^-1 | `20260828T040152Z-c988539-feadb5` / `55633691` | 45:47 / 613.20 MB | Structural, not trajectory-level: 165 hazard births versus reference 119 | Labels rise 12 to 177 while topology remains 12; all 165 label births are rejected as physical DRX |
| v34 coupled, 1,000 s^-1 | `20260828T040212Z-c988539-ee4dad` / `55633694` | 22:55 / 495.43 MB | Bookkeeping failure reproduced; detailed trajectory diverges | Candidate active/new/promotable/age maxima are all zero, as are births; therefore this is not a candidate-without-promotion case |

The v33/v34 stored reference folders lack an immutable source/environment record. Identical recorded parameters are insufficient for trajectory reproduction. This is a provenance failure in the legacy evidence, not a basis for parameter tuning.

These legacy computations do not calibrate, validate, or regression-gate the new model. Physical parameter optimization remains pending one authoritative target material and strength/rate/temperature dataset.
