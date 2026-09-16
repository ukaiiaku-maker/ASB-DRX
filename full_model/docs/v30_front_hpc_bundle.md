# V30 front long-run bundle

This bundle is prepared but not submitted. Every scientific case explicitly
uses `coupled_bidirectional_v30`; the legacy phase-first afterburner and the
equal-state projection are disabled.

## Execution order

1. Submit `submit_v30_front_a1_anchor.sbatch` from a clean, pushed source SHA.
2. Postprocess completed and partial cases with
   `postprocess_v30_front_hpc.py`. Resume any `PARTIAL_RESTARTABLE` array
   elements using the same submission command.
3. Submit `submit_v30_front_a1_delta.sbatch` and postprocess again.
4. Submit `submit_v30_front_a2.sbatch` only when the postprocessor reports
   `a2_authorized: true`; set `V30_A2_AUTHORIZED=1` explicitly.

At submission, export the local and fetched remote identities as
`V30_EXPECTED_SOURCE_SHA` and `V30_EXPECTED_REMOTE_SHA`. The runner requires
both identities to equal its clean checkout. It records the source SHA, both
expected identities, job identity, manifest/config/restart
hashes, and every attempt in `case_status.json`.

Each case checkpoints every 100 physical steps and every 840 wall-clock
seconds. A scheduler signal leaves the case `PARTIAL_RESTARTABLE`; the next
invocation selects the highest valid checkpoint and does not replay completed
segments. Separate attempt directories preserve partial fields, logs, contour
history, and failed records.

## Matrix and preregistered decisions

Tier A1 contains 24 fault-isolated cases: 64/128 equal, mobility-off,
favorable, reversed, label-swapped favorable, advance-retreat-readvance, and
relative contrasts of ±1e-6, ±1e-3, and ±1e-2. Equal and driven anchors run
2000 steps with `dt=1e-7 s`, reaching 200 microseconds unless a declared
geometric terminal occurs. Cycle cases contain three 600-step pressure
segments. Tier A2 adds 192-grid equal/favorable/reversed cases at 1100 K and
128-grid equal/favorable/reversed cases at 900 and 1300 K.

The postprocessor enforces zero equal/off ledgers, 0.02-cell equal drift,
directional signs, label covariance, odd near-equal response, cycle revisit,
zero legacy calls, ledger closure, and the preregistered 5% 64/128 criterion.

## Resource estimate

Based on local production smoke timings and quadratic grid scaling:

- 64 square: approximately 0.5–1.5 CPU-hours per 2000-step anchor;
- 128 square: approximately 2–5 CPU-hours per 2000-step anchor;
- 192 square: approximately 5–10 CPU-hours per 2000-step confirmation;
- expected memory is below 8 GiB per case.

The A1 matrix is expected to consume roughly 30–60 CPU-hours. With the
preregistered four-case concurrency cap, expected elapsed time is 8–18 hours,
depending on filesystem and queue performance. A1 allocations are capped at
8 hours per element; A2 at 12 hours. Cases exceeding an allocation resume from
their latest ≤14-minute checkpoint.
