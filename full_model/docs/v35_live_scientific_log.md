# V35 live scientific log

## 2026-09-16 restart audit

- Canonical branch: `exp/full-v34-recovery-v1`.
- Remote checkpoint at the V35 audit: `ae8644a85a0020202d5d12c1849c879d03688914`.
- Local checkpoint after recording the completed V34 full-law result:
  `56b68521a6ed123563246a794a2a467f5ed14e0b`.
- HPC3 queue was empty. Existing local thermal managers and their useful
  workers were preserved.
- `59f4d12` failed at step 1451. The exact `e7aa16e` repair resumed that
  checkpoint, remained invariant-valid through history step 4650, and then
  disappeared without a terminal record. Its latest exact restart is step
  4553. This is a valid interrupted partial, not a scientific terminal at
  step 1451 or 2971.

## Mura feasible-extent decision

Production source `fe4f44686e7ba41abbfd21a6db6817b6b3c0a455` adds a
separate `energy_limited_feasible_extents` mode. It samples the complete
fixed-strain candidate over extents `1, 1/2, ..., 1/2048`, keeps the largest
admissible extent on the constitutively proposed ray, and then reevaluates the
joint event. It does not maximize release, weaken the energy tolerance, carry
rejected debt, or advance an extra clock.

At the step-1451 checkpoint, families 2 and 3 are genuinely uphill throughout
the sampled interval. At step 2971, family 2 is still uphill, but family 3 is
downhill from small extent through `s=0.0625` and uphill at `s=0.125` and
above. The frozen V34 full-event Boolean therefore suppresses a feasible
partial event at that state. All bounded trials preserve nonnegative heat,
Mura/Nye compatibility, and no-projection invariants.

Source `0067da2d6b9db31a78cd5f58d33b43b68c829fc4` additionally makes short
production continuations end at the exact requested physical horizon. An
earlier exact-time launch with a mistyped full provenance SHA was stopped and
its partial directory quarantined; no result from it is used.

## Thermal branches still running

- The source-`0c45d03` full local-adiabatic case completed step 2500 and is
  invariant-valid. Its nominal selective cases remain
  `INVALID_CAUSAL_ABLATION_ROUTING`.
- The corrected source-`06de449` authoritative frozen-flow case is running to
  step 2500; the corrected recovery-temperature group follows sequentially.
- The source-`0c45d03` exact prescribed-temperature/thermostat case is useful
  and running; finite-conduction/no-bath follows it.
- No strict-ASB conclusion will be made until the valid common-time causal
  comparison is complete.

## Current interpretation flags

- `SINGLE_INTERVAL_FORCED_TRANSACTION_VERIFIED`
- `RECURRENT_STATE_REPRESENTATION_INCOMPLETE` (pending merge of the recurrent
  owner-moment implementation)
- `FRONT_PHYSICAL_HORIZON_INSUFFICIENT`
- `GENERIC_KINETICS_NOT_MATERIAL_CALIBRATED`
- `SCREENING_BOUNDED_KINETICS`
- `RUNNING_PHYSICAL_HORIZON`
