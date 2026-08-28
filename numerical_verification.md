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
