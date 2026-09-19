# V43 live scientific log

## 2026-09-19 audit

- Local checkpoint was clean at `1de2383`; the configured remote branch had
  been force-moved to unrelated older V33 commit `8b2f77c`.  No destructive
  reset or force push was performed.  V43 continues on recovery branch
  `exp/full-v34-recovery-v43-20260919` from the retained V42 lineage.
- Slurm job `56166842` completed in `00:18:38` with exit `0:0`.  The manager
  fetched archive SHA-256
  `2359d0e3c37cc19f082ab8db88405ef97f3f28961671fcf2c065ba38f6af4659`,
  verified it, and postprocessed the matched 5.01% continuation.  Classification:
  `VALID_NEGATIVE_AT_TESTED_CONDITION;NO_QUALIFIED_BOUNDARY`.
- The n128/n192 embedded source mismatch was inspected, not presumed benign.
  Between `ee4b805` and `17f63ee`, the only production dependency change is
  removal of unused legacy topology imports.  The recurrent driver,
  initialization, equations, parameters, and numerical updates are identical.
  The retained spatial pair is therefore scoped source-equivalent.

## Geometry and ownership

- Selected the smallest persistent geometric representation supported by the
  2-D production model: oriented periodic links as the exact boundary of
  retained swept plaquettes.
- Nonzero geometry, swept `beta_p`, family Nye, line length, scalar reservoirs,
  and moments now publish atomically.  The V42 geometry-neutral conversion is
  retained unchanged as the disabling comparator.
- Manufactured verification passes accepted event, rejected rollback, second
  evolved-state event, exact restart, translation symmetry, endpoint closure,
  moment realizability, and 16/32 segment refinement.
- Shared geometry-neutral ordering heat is now owned identically by on/off
  controls.  Rejected geometry emits zero heat.
- A bounded zero-event-work physical response accepts an initial advance.
  Low-barrier full-extent paths then become energy-pinned; slower finite-rate
  hypotheses accept several smaller increments.  This is a response family,
  not calibration or spontaneous DRX.

## Front and restart corrections

- The front artifact contains eight, not sixteen, rows per prepared state.
  Trial endpoint energy, accepted publication energy, no-op, and rate rejection
  are now separate.  The retained hold has eight evaluated uphill endpoints,
  zero publications, and eight rate-rejected no-ops.  The prepared state has
  eight evaluated downhill endpoints, four publications, and four rate-rejected
  no-ops.  The four publications remain independent counterfactual probes.
- The archived mechanical runner now requires explicit `initialize` or
  `restart` mode.  Restart validates path, SHA-256, schema/migration, fields,
  step, time, strain, RNG metadata, and source lineage before step one.  A
  staged `git archive` test loads identical content under an ordinary filename
  and rejects missing path, bad checksum, and bad lineage with zero checkpoints
  and no continuation status.

## Spatial investigation

- The original first macro is localized by actual stage fields.  Initial,
  first-Mura, and front stages agree in the selected common band.  The second
  Mura half produces 4.30% complex-coefficient error and 7.31% RMS-amplitude
  difference, while common-band power differs only 0.124%.  This identifies a
  growing phase/location error rather than a simple amplitude normalization.
- Both grids previously consumed each half-macro in one large constitutive
  step.  A bounded four-substep repeat from the matched first-macro checkpoint
  completed locally.  It worsens the interval-2 final curl comparison from
  12.316% to 13.683% in the common complex band and from 27.199% to 29.235% in
  RMS amplitude.  Ordered inventory is unchanged by the time refinement at
  either grid.  Time truncation is rejected as the repair; the mixed donor-cell
  scalar/spectral moment Mura seam is the next concrete discriminator.
