# V31 Mura Tier-B1 terminal audit

## Decision

Tier B1 is **not scientifically qualified**, and Tier B2 is **not
authorized**.  No parameter change, restart, resubmission, or enlarged matrix
was performed.

The terminal classification is
`TIER_B1_MECHANICAL_HETEROGENEITY_THERMODYNAMIC_ADMISSIBILITY_FAILURE`.
Slurm job `56070295` used source
`675d74183baf2043ad0f7c055fe6e3370435ae65`, exited `1:0`, and ran for
`04:15:49` of a `16:00:00` allocation.  The batch step used about 1.13 GiB of
32 GiB requested memory.  This was neither a wall-time nor memory failure.

## Terminal case state

The homogeneous and broadband-noise cases at both 64 and 128 completed 20%
strain.  Both mechanical-heterogeneity cases rejected a trial step with the
same explicit guard:

`RuntimeError: Mura line storage exceeds available plastic work`

The last accepted states were 4.05498% strain at 64 and 3.18866% strain at
128.  Their accepted-step Mura invariants remained valid.  The job duration
was set by the slow 128-grid controls that continued to 20%; it is not the
time at which the mechanical failure first occurred.

The `process_failure.json` files present in the four completed case trees are
stale restart artifacts.  They were written at 22:28:02 local time, before the
Slurm job began at 22:28:39, whereas the completed status and runner logs were
written later.  The V31 decision records these markers without treating them
as terminal failures.

## Matched-horizon comparison

Step 1000 at exactly 3% strain is the latest checkpoint shared by all six
cases.  At that horizon the mechanical 64/128 comparison exceeds the
provisional 5% threshold for dual Nye mismatch (11.67%), maximum signed
density (85.95%), orientation span (7.73%), and structure-factor peak
fraction (72.13%).  The broadband pair also lacks pattern convergence, with
50.31% alpha-RMS, 46.86% maximum-signed-density, and 71.79%
structure-factor-peak differences.  The homogeneous reduction is exact.

Thus the negative decision is independently supported by both the repeated
thermodynamic rejection and the matched-horizon refinement comparison.

## Evidence

- `full_model/verification/v31_mura_b1_decision.json`: machine-readable
  decision, terminal case states, matched-horizon metrics, and exact selected
  checkpoint hashes.
- `full_model/verification/v31_mura_b1_index.csv`: compact six-case index.
- `full_model/verification/v31_mura_b1_evidence_manifest.json`: hashes for all
  140 files in the preserved raw V30 tree.
- `full_model/verification/v31_mura_b1_figures/matched_horizon_fields.png`:
  signed density, authoritative Nye magnitude, and orientation at 3% strain.
- `full_model/verification/v31_mura_b1_figures/terminal_histories.png`:
  invariant and organization histories through each case's terminal horizon.

The immutable 498 MiB copy of the full remote tree is outside Git at
`/Users/sdillon/HPC3/v31-evidence/mura-b1-56070295/raw`.  Its full-file
manifest is retained externally and copied into the repository; the manifest
SHA-256 is
`64d2e76a5e97a87edf158b904668c3be0598365084efd1b5669a87468576690f`.
The independently calculated, path-sorted SHA-256-stream digest is
`27e2ede728b8f5cdd62ad7edf7e222cdb190ffe0aeac62ebdc3421055cc63fa5`
for both the remote tree and the preserved local copy.
