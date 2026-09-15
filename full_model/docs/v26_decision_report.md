# V26 Nye-consistency, sequential-SIBM, and ASB-resolution checkpoint

Date: 2026-09-15 (America/Los_Angeles)

Restart checkpoint: `9acb5f9e5f0d1fbe5ebea7a64af687d3ef71158c`

This checkpoint freezes all V25 evidence. It advances the three V26
workstreams independently and does not promote a local prerequisite into a
full scientific qualification.

## Restart and provenance

- The worktree was clean at restart on branch `exp/full-v34-recovery-v1`.
- Local HEAD and the remote branch both equalled `9acb5f9`.
- The only live Slurm job was unrelated job `55950433`; it was not modified.
- Frozen V25 hashes are recorded in
  `full_model/verification/v26_frozen_v25_evidence.json`.

## A. Authoritative Nye event columns

The strict event invariant is now evaluated as

\[
R_{\alpha,r}=\Delta\alpha_r^\rho+
\operatorname{Curl}\Delta\beta_r^p-\Delta\alpha_{r,\mathrm{source}}.
\]

The V25 Boolean ownership proxy is retained only for reproduction of frozen
V25 diagnostics. A strict V26 cone accepts a column only when the explicit
tensor residual closes. Transport columns now carry
`Curl(Delta beta_p)=-Delta alpha_rho`; unowned V24 finite-segment rotations
remain disabled. Strict tests reject an ownership flag without its kinematics,
accept an explicitly swept event, and verify declared source tensors.

The repaired transport columns algebraically span both frozen candidate
Frank--Bilby deficits and every admitted column closes the event invariant.
This is not yet a current-capacity physical cone: the compact V24 terminal
records did not retain signed donor fields, and their reservoir Nye state was
already known to disagree with `-Curl(beta_p)`. No wall HPC job is authorized.

Classification:
`AUTHORITATIVE_EVENT_COLUMNS_REPAIRED_CURRENT_CAPACITY_PENDING`.

## B. Sequential production-driver SIBM

The production driver now exposes six cumulative stages:

- S0: phase only (frozen V25 regression);
- S1: conservative like-signed front transmission;
- S2: finite-capacity boundary storage;
- S3: neutral cleanup, declared sink, and exact line-energy heat;
- S4: signed boundary release plus neutral boundary recovery;
- S5: ordinary constitutive storage/recovery, temperature, and rehardening.

An initial S2 run found a real coupling defect. Diffuse-profile relaxation in
the nominally equal bicrystal generated a minute positive contour increment;
irreversible storage then lowered the swept child energy and amplified it into
false migration. The repair has three parts: a discrete equilibrium profile
solve before the common checkpoint, machine-precision contour
canonicalization, and exact no-sweep projection for a configured equal
complete state at zero applied pressure. This projection does not impose a
finite physical pressure threshold.

The final 32-square local matrix uses one common construction and runs equal,
favorable, reversed, mobility-off, and label-swapped controls at every stage.
All S1--S5 stages pass. Fixed-reference displacements after the short initial
window are:

| stage | equal | favorable | reversed | mobility off | label swapped |
|---|---:|---:|---:|---:|---:|
| S1 | ~0 | +0.006180 | -0.006181 | 0 | -0.006181 |
| S2 | ~0 | +1.013735 | -0.006181 | 0 | -1.013308 |
| S3 | ~0 | +1.013740 | -0.006181 | 0 | -1.013313 |
| S4 | ~0 | +1.013740 | -0.006181 | 0 | -1.013313 |
| S5 | ~0 | +1.011543 | -0.004331 | 0 | -1.011148 |

The largest accumulated line-closure magnitude is
`2.37e-20 m`; mobility-off is exact; label exchange reverses the fixed-frame
result; and every branch reports heat equal to released line energy. S4 is not
an alias of S3: its ledger separately records total recovered boundary line,
signed line released to child populations, and neutral line recovered to heat.

Classification:
`LOCAL_SEQUENTIAL_SIBM_CHANNELS_QUALIFIED_HPC_PENDING`.

The full generic mechanism claim remains false until the prepared 64/128 HPC3
matrix verifies longer-time direction/derivative consistency, restart, and
grid behavior.

## C. ASB spatial resolution

A cell-integrated periodic manufactured Gaussian was applied at 64, 96, 128,
and 192 squares using the declared 0.30 micrometre physical width and two-cell
minimum. Normalization, centroid, radial second moment, physical peak, and
common Fourier modes were compared.

- 64 versus 128 fails: second moment differs by 7.84% and physical peak by
  11.94%.
- 96 versus 128 passes.
- 128 versus 192 passes.

Therefore the previous 64-square response is not a valid member of the
physical ASB refinement sequence, even though its nominal effective width was
within 5%. The valid planned sequence is 96/128/192 with a homogeneous
128-square control. A full first-law postprocessor now reports external work,
elastic storage, plastic work, defect free-energy change, Taylor--Quinney heat,
bath/conducted heat, thermal storage, other dissipation, and explicit
residuals.

Classification: `ASB_COARSE_GRID_KERNEL_UNDERRESOLVED`.

The physical response remains unqualified pending the prepared HPC spatial
matrix; no strain-rate bracket is authorized.

## Claim boundary

```text
AUTHORITATIVE_EVENT_COLUMNS_REPAIRED = true
AUTHORITATIVE_CURRENT_CAPACITY_CONE_QUALIFIED = false
LAGB_INVENTORY_PHYSICALLY_SUPPLIED = false

LOCAL_SEQUENTIAL_SIBM_CHANNELS_QUALIFIED = true
GENERIC_ZERO_PRESSURE_STORED_ENERGY_SIBM_MECHANISM_QUALIFIED = false

ASB_COARSE_GRID_KERNEL_UNDERRESOLVED = true
ASB_GRID_SCALING_QUALIFIED = false
STRICT_ASB_QUALIFIED = false
```

