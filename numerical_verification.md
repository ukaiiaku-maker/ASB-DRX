# Numerical verification record

No production-model numerical verification has been run. Numerical work is restricted to HPC3.

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
