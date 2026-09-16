# V27 local evidence closure and current-capacity decision

Date: 2026-09-15

## Decision summary

V27 closes the interrupted V26 evidence campaign without attributing later
source changes to the frozen `bf150ca` calculations. The three independent
decisions are:

| pathway | primary classification | fixture | scientific gate |
|---|---|---:|---:|
| zero-pressure SIBM | `SIBM_EQUAL_STATE_PROJECTION_DEPENDENT` | pass | fail |
| ASB spatial scaling | `ASB_PHYSICAL_RESPONSE_NOT_GRID_CONVERGED` | pass | fail |
| wall/current-capacity cone | `REACTION_CONE_INVALID_DUE_TO_DUAL_NYE_MISMATCH` | pass | fail |

No 192-square SIBM run, ASB strain-rate bracket, long wall calculation, or
grain allocation is authorized. These are scientific rejections of the tested
mechanisms, not conservation failures and not evidence for calibrated DRX or
ASB.

## Frozen V26 HPC records and local continuation

The SIBM bundle `20260915T232402Z-bf150ca-b4b6bc` (job `56055686`) ran on
`hpc3-14-00` from `2026-09-15T23:25:06Z` to `23:34:37Z` and failed with
application exit 1 after 9:34. Its archive finalized successfully. The failure
was a scale-dependent signed boundary-reservoir ownership invariant during
retreat, not a scheduler or packaging failure. The fetched source and result
archive SHA-256 values are `b286e40d...31fc78` and `984b86f5...baeacf`.

The ASB bundle `20260915T232425Z-bf150ca-5e84d7` (job `56055688`) began on
`hpc3-14-00` at `2026-09-15T23:25:06Z`. The completed 96- and 128-square cases
and the 192-square checkpoint at step 1250 were fetched before the job was
cancelled deliberately after 1:05:28 so the user-selected local campaign could
continue without consuming HPC3. Its fetched source archive SHA-256 is
`1426444d...ea7e63`. The 192-square trajectory was continued exactly from the
frozen checkpoint through step 4999 locally. The final checkpoint SHA-256 is
`b1bd6e4c...0ba6b`, and the monotone 501-row assembly record SHA-256 is
`1dc505d6...fb7e`.

Later commits repair the reservoir release rule, cumulative recovery ledger,
and translation/profile distinction. Those repairs are not attributed to the
V26 HPC source. The ASB production equations were unchanged by the SIBM-only
repairs. The current HPC3 queue contains only unrelated job `55950433`, which
was observed and not touched.

## SIBM decision

The 128-square interface has 4.048 points across its physical width; the
64-square member has only 2.024 and is retained solely as an underresolution
diagnostic. At 128 square, line closure remains within
`1.14e-19 m`, the mobility-off control is stationary, and heat equals released
line energy. Nevertheless, the essential signed controls fail:

| 128-square case | initial velocity (m/s) | cumulative signed contour area (m²) | material displacement / width | final complete pressure (MPa) |
|---|---:|---:|---:|---:|
| equal | 0.00108 | `3.30e-13` | 0.0604 | approximately 0 |
| favorable | 0.02175 | `2.85e-11` | 5.60 | +29.03 |
| reversed | 0.02028 | `2.93e-11` | 3.78 | +8.33 |
| mobility off | 0 | 0 | 0 | approximately 0 |
| label swapped | 0.01751 | `2.68e-11` | 3.91 | +26.59 |

The reversed-energy branch advances rather than retreating, and the complete
label swap does not reverse the response. Antisymmetric relative density
perturbations from `1e-10` through `1e-2` produce essentially the same positive
initial velocity; their velocity oddness residuals are approximately 2.0.
The exact equal branch invokes the projection at all 1000 updates and suppresses
absolute geometric volume equal to 7.55% of the represented domain. Although
it records zero virgin sweep, this is resolved profile motion rather than a
machine-roundoff correction. Thus the projection is not a machine-symmetry-only
invariant, and the generic stored-energy SIBM claim is rejected before a
192-square convergence study.

The complete force, area velocity, material displacement, and near-equal
response are plotted in `full_model/verification/v27_sibm_decision.png`; the
machine-readable record is
`full_model/verification/v27_sibm_local_decision.json`.

## ASB decision

The common-state audit passes: physical declarations are identical and the
sampled grain-edge mismatch versus 96 square is 2.94% at 128 and 4.66% at 192.
All four cases close the declared first law below 5%; the heterogeneous
relative residuals are below `4e-13`, and the homogeneous control residual is
0.803%. The 128-square heterogeneous trajectory exceeds its matched
homogeneous maximum temperature by 213.7 K.

The physical response does not converge between the decisive 128- and
192-square members:

| observable | relative difference |
|---|---:|
| final stress | 0.64% |
| maximum temperature | 4.30% |
| maximum plastic rate | 2.31% |
| peak stress | 6.53% |
| external work | 6.33% |
| plastic work | 10.19% |
| deposited Taylor-Quinney heat | 14.31% |
| active plastic fraction | 39.47% |
| effective band width | 29.84% |

Neither fine-grid case reaches the unchanged 20% post-peak-softening
threshold: 128 gives 15.47% and 192 gives 10.15%. Continuous conjunctive
persistence is therefore not established. The physical localization measures,
work, and heat fail the 5% requirement even though the first-law ledger itself
closes. A strain-rate bracket would conflate spatial error with rate response
and is not authorized.

The stress/temperature histories, fine-grid differences, and first-law
residuals are plotted in `full_model/verification/v27_asb_decision.png`; the
machine-readable record is
`full_model/verification/v27_asb_local_decision.json`.

## Wall/current-capacity decision

The regenerated 32-square transition-band replay retains current signed
reservoirs, line alignments, incoming flux exposure, transport capacities,
plastic distortion, and both Nye constructions. Every admitted transport
column satisfies the strict event identity exactly, and the diagnostic cone
can algebraically span the instantaneous Frank--Bilby deficit. This does not
qualify physical supply: the evolved reservoir Nye and authoritative
`-Curl(beta_p)` fields disagree by approximately 1.23 relative RMS, far above
the fixed 5% tolerance. The online cone therefore operates on an inconsistent
state and is invalid regardless of its small algebraic residual.

Finite-segment reorientation remains disabled, the optimizer changes no state,
and no phase or grain label can be allocated. The result is frozen in
`full_model/verification/v27_current_capacity_wall.json`.

## Claim hierarchy and next decisions

Retained:

- strict transport-event identities close;
- SIBM line, signed-Burgers, and heat ledgers close after the scoped software
  repairs;
- ASB common-state restriction and first-law accounting pass;
- the EXP-floor kinetic backbone remains unchanged.

Rejected or still false:

- `GENERIC_ZERO_PRESSURE_STORED_ENERGY_SIBM_MECHANISM_QUALIFIED`;
- `ASB_GRID_SCALING_QUALIFIED` and `STRICT_ASB_QUALIFIED`;
- `AUTHORITATIVE_CURRENT_CAPACITY_CONE_QUALIFIED`;
- `LAGB_INVENTORY_PHYSICALLY_SUPPLIED`;
- `FULL_MODEL_DRX_MECHANISM_SUPPORTED`.

The next scientifically admissible development work is to remove the SIBM
projection dependence and restore odd label/energy symmetry, reconcile the
single authoritative Nye state before any wall-cone rerun, and identify the
source of ASB localization-width/work sensitivity before changing strain rate.
