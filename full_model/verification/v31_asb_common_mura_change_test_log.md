# V31 Workstream C change and test log

Date: 2026-09-16

Starting checkpoint: `512b8f0bf00ff901aa34ed636345283c08bd48c2` on
`exp/full-v34-recovery-v1`.

## Physical ownership changes

- Added opt-in `v31_asb_common_mura_ledger` production mode. It promotes the
  accepted common Mura/wall state to the sole owner of signed reservoirs,
  plastic distortion, Nye, elastic-spin orientation, defect energy, and heat.
- Kept numerical compatibility penalties outside physical storage and disabled
  legacy scalar energy snapshots in this mode.
- Replaced the old reservoir mass-action split, which could run uphill relative
  to the declared free energy because it assumed an undeclared mixing entropy,
  with a bounded reversible Onsager exchange. It is stationary at equal
  declared affinity and satisfies minus-affinity times extent >= 0 pointwise.
- Exposed five mechanical/reaction dissipation channels term by term. Fourier
  conduction and bath exergy remain separately typed. No channel is assigned
  from a first-law residual.
- Corrected the heat comparison so conduction redistribution and bath export
  are not incorrectly counted as deposited mechanical heat.
- Added per-step and cumulative V31 closure, minimum-channel, restart, and
  classification output in `v31_common_mura_asb_ledger.json`.

## Verification

- Focused common-wall and energy-ledger suite after the exact wall-order
  gradient-energy check was added: 41 passed.
- Expanded ASB/Mura regression suite: 68 passed.
- Full repository suite (`PYTHONPATH=src python -m pytest -q tests`): 599
  passed in 215.75 s. An initial unscoped invocation was discarded because it
  collected duplicate archived HPC work trees and omitted the `src` import
  path; it did not execute model tests.
- Actual production driver, two accepted steps at 32, 64, and 128 grids:
  all three classified `ASB_COMMON_MURA_LEDGER_QUALIFIED`.
- Relative cumulative first-law residual: `1.2149076216939457e-05` on every
  production grid, below the provisional 5% threshold.
- Relative mechanical dissipation-to-deposited-heat residual:
  `4.394126749023172e-16` on every production grid.
- Continuous 8-step versus 3+5 step checkpoint/restart: bitwise identity for
  signed populations, defect partitions, temperature, orientation, plastic
  distortion, family Nye, junction state, last-step ledger, and cumulative
  ledger.

No long run or HPC job was launched. Unrelated front-workstream changes present
in the shared worktree were not included in this workstream's edits.
